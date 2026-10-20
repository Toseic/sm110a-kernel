#!/usr/bin/env python3
"""FP8 shared FC1: fused vs two native GEMMs + PyTorch epilogue, cold L2."""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import (
    MMA_PEAK_TFLOPS,
    BenchmarkBase,
    SolReport,
    benchmark_with_stats,
    sol_bounds,
)
from sm110a_kernel import (
    Fp8GemmConfig,
    Fp8SharedFc1Config,
    compile_fp8_gemm,
    compile_fp8_shared_fc1,
    dequantize_fp8_rows,
    from_blocked_fp8_scales,
    pack_fp8_shared_fc1_weights,
    quantize_fp8_rows,
)


def epilogue(gate, up, limit):
    gate, up = gate.float(), up.float()
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return (F.silu(gate) * up).to(torch.bfloat16)


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


class Fp8SharedFc1Benchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--tokens", type=int, nargs="+", default=[128, 256, 512, 1024, 4096]
        )
        parser.add_argument("--k", type=int, default=4096)
        parser.add_argument("--inter-dim", type=int, default=2048)
        parser.add_argument("--swiglu-limit", type=float, default=10.0)
        parser.add_argument(
            "--baseline", choices=["unfused", "none"], default="unfused"
        )
        parser.add_argument("--no-accuracy", action="store_true")
        self.add_sol_arg(parser)
        self.add_timing_args(parser, warmup=5, iters=20, samples=5)
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if any(m <= 0 or m % 128 for m in args.tokens):
            raise SystemExit(
                "tokens must be positive multiples of 128 (physical GEMM rows)"
            )
        if args.k <= 0 or args.k % 64 or args.inter_dim <= 0 or args.inter_dim % 128:
            raise SystemExit("K must be a positive multiple of 64; inter-dim of 128")
        if not math.isfinite(args.swiglu_limit) or args.swiglu_limit < 0:
            raise SystemExit("swiglu-limit must be finite and non-negative")
        if not args.cold_l2:
            raise SystemExit("this comparison requires cold L2")

    def time(self, run, args):
        return benchmark_with_stats(
            run,
            args.warmup,
            args.iters,
            args.samples,
            cold_l2_flush=args._cold_l2_flush,
        )["median_ms"]

    @torch.inference_mode()
    def run_with_args(self, args):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        print(
            f"FP8 shared FC1: K={args.k} I={args.inter_dim} limit={args.swiglu_limit}; "
            "prequantized inputs; packing/quantization excluded; cold-L2 CUDA Graphs"
        )
        print(
            "Sweep fused: tile_n=128, tile_k=64/128, cluster_m=1/2, split_k=1. "
            "Baseline sweeps tile_n=128/256 with the same K/cluster options; no split-K/PDL."
        )
        print("M fused(us) unfused(us) speedup TF/s max_abs relative_RMSE config")
        if args.sol:
            print(
                "SOL: max(MMA, compulsory GMEM, tiled L2). FP8 peak=2*BF16 peak "
                f"({2 * MMA_PEAK_TFLOPS:g} TF/s before clock scaling); excludes "
                "launch/SFU/conversion/store-efficiency costs. Scales count physical bytes."
            )
        if args.sol:
            import common

            print(
                f"[SOL] L2 peak={common.L2_BW_GBPS:.3f} GB/s "
                "(1024 B/cycle at 1575 MHz reference LTS clock)"
            )
        report = SolReport()
        for m in args.tokens:
            k, inter = args.k, args.inter_dim
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn(
                inter, k, device="cuda", dtype=torch.bfloat16
            ) / math.sqrt(k)
            up = torch.randn_like(gate) / math.sqrt(k)
            a, sfa = quantize_fp8_rows(x)
            bg, sg = quantize_fp8_rows(gate)
            bu, su = quantize_fp8_rows(up)
            b, sfb = pack_fp8_shared_fc1_weights(
                bg,
                bu,
                from_blocked_fp8_scales(sg, inter, k),
                from_blocked_fp8_scales(su, inter, k),
            )
            out = torch.empty(m, inter, device="cuda", dtype=torch.bfloat16)
            best = None
            for cm in (1, 2):
                for tk in (64, 128):
                    if m % (128 * cm) or k % tk:
                        continue
                    c = Fp8SharedFc1Config(
                        tile_k=tk, cluster_m=cm, swiglu_limit=args.swiglu_limit
                    )
                    module = compile_fp8_shared_fc1(c)
                    graph, _ = capture(
                        lambda: module.fp8_shared_fc1_out(a, b, sfa, sfb, out)
                    )
                    ms = self.time(graph.replay, args)
                    if not args.no_accuracy:
                        if best is None:
                            lhs = dequantize_fp8_rows(a, sfa, m, k)
                            ref = epilogue(
                                (
                                    lhs @ dequantize_fp8_rows(bg, sg, inter, k).t()
                                ).bfloat16(),
                                (
                                    lhs @ dequantize_fp8_rows(bu, su, inter, k).t()
                                ).bfloat16(),
                                args.swiglu_limit,
                            )
                        graph.replay()
                        torch.testing.assert_close(out, ref, rtol=0.016, atol=0.002)
                    if best is None or ms < best[0]:
                        best = (ms, c, graph)
            ms, c, graph = best
            error = rmse = float("nan")
            if not args.no_accuracy:
                graph.replay()
                error = (out.float() - ref.float()).abs().max().item()
                rmse = (
                    (out.float() - ref.float()).square().mean().sqrt()
                    / ref.float().square().mean().sqrt()
                ).item()
                eager = out.clone()
                module = compile_fp8_shared_fc1(c)
                module.fp8_shared_fc1_out(a, b, sfa, sfb, out)
                torch.testing.assert_close(out, eager, rtol=0, atol=0)
                x_new = torch.randn_like(x)
                a_new, s_new = quantize_fp8_rows(x_new)
                a.copy_(a_new)
                sfa.copy_(s_new)
                graph.replay()
                replay_out = out.clone()
                module.fp8_shared_fc1_out(a, b, sfa, sfb, out)
                torch.testing.assert_close(out, replay_out, rtol=0, atol=0)
                lhs = dequantize_fp8_rows(a, sfa, m, k)
                new_ref = epilogue(
                    (lhs @ dequantize_fp8_rows(bg, sg, inter, k).t()).bfloat16(),
                    (lhs @ dequantize_fp8_rows(bu, su, inter, k).t()).bfloat16(),
                    args.swiglu_limit,
                )
                torch.testing.assert_close(out, new_ref, rtol=0.016, atol=0.002)
            # Restore the exact input used for fused timings before baseline timing.
            original_a, original_sfa = quantize_fp8_rows(x)
            a.copy_(original_a)
            sfa.copy_(original_sfa)
            baseline = None
            if args.baseline == "unfused":
                gout, uout = torch.empty_like(out), torch.empty_like(out)
                for cm in (1, 2):
                    for tk in (64, 128):
                        for tn in (128, 256):
                            if m % (128 * cm) or k % tk or inter % tn:
                                continue
                            gm = compile_fp8_gemm(
                                Fp8GemmConfig(tile_n=tn, tile_k=tk, cluster_m=cm)
                            )

                            def run_unfused():
                                gm.fp8_gemm_out(a, bg, sfa, sg, gout)
                                gm.fp8_gemm_out(a, bu, sfa, su, uout)
                                return epilogue(gout, uout, args.swiglu_limit)

                            baseline_graph, baseline_out = capture(run_unfused)
                            bms = self.time(baseline_graph.replay, args)
                            if not args.no_accuracy:
                                baseline_graph.replay()
                                torch.testing.assert_close(
                                    baseline_out, ref, rtol=0.016, atol=0.002
                                )
                            baseline = bms if baseline is None else min(baseline, bms)
            flops = 4 * m * inter * k
            print(
                f"{m} {ms * 1000:.3f} "
                + (
                    f"{baseline * 1000:.3f} {baseline / ms:.2f}x "
                    if baseline
                    else "- - "
                )
                + f"{flops / (ms * 1e9):.2f} {error:.6g} {rmse:.6g} "
                f"128x128x{c.tile_k}/c{c.cluster_m}",
                flush=True,
            )
            if args.sol:
                payload = sum(t.numel() * t.element_size() for t in (a, b, sfa, sfb))
                tile_reads = (
                    m * k * (2 * inter // 128)
                    + 2 * inter * k * (m // (128 * c.cluster_m))
                ) * 1.125
                bounds = sol_bounds(
                    flops,
                    payload + out.numel() * 2,
                    payload,
                    tile_reads,
                    mma_peak_tflops=2 * MMA_PEAK_TFLOPS,
                )
                report.add(bounds, ms, label=f"M={m}")
        if args.sol:
            report.print()
        if not args.no_accuracy:
            print(
                "Accuracy: all tested configs passed rtol=0.016 atol=0.002; "
                "eager/graph and replay after changing A/scales passed bitwise."
            )


if __name__ == "__main__":
    Fp8SharedFc1Benchmark().run()
