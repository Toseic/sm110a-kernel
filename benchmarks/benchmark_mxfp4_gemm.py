#!/usr/bin/env python3
"""Accuracy and performance benchmark for the native SM110A MXFP4 GEMM."""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepgemm_baseline import mxfp4_nt as deepgemm_mxfp4_nt

from common import (
    BenchmarkBase,
    SolReport,
    TablePrinter,
    benchmark_with_stats,
    quick_probe,
    sol_bounds,
    tflops_from_ms,
)
from sm110a_kernel import (
    from_blocked_mxfp4_scales,
    mxfp4_gemm_reference,
    quantize_mxfp4_gemm,
)
from sm110a_kernel.kernels.mxfp4_gemm import (
    Mxfp4GemmRunner,
    format_candidate,
)

DEFAULT_SHAPES = (
    # mxf4 and its blocked scale format start at M=128.  BF16's smaller
    # M={1,4,8,16,32} sweep is intentionally inapplicable here.
    (128, 4096, 4096),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 11008, 4096),
    (11008, 4096, 4096),
)

MXFP4_MMA_PEAK_TFLOPS = 4.0 * 250.0


def mxfp4_gemm_sol_bounds(m, n, k, config):
    """Model packed operands, block scales, and multicast full-N SFB."""
    cluster_tile_m = config.cluster_m * config.tile_m
    tile_count = math.ceil(m / cluster_tile_m) * math.ceil(n / config.tile_n)
    flops = 2 * m * n * k

    packed_operand_bytes = (m * k + n * k) / 2
    # Logical MXFP4 scales consume one byte per 32 values.  Their 128x4
    # blocked backing store pads every K=64 pair to 4 bytes per row.
    unique_scale_bytes = (m * k + n * k) / 32
    input_bytes = packed_operand_bytes + unique_scale_bytes
    compulsory_bytes = input_bytes + 2 * m * n

    packed_tile_bytes = (cluster_tile_m + config.tile_n) * k / 2
    sfa_tile_bytes = cluster_tile_m * k / 16
    # Load the complete N-wide SFB tile once per CTA cluster.  The two-CTA
    # configuration multicasts it into both CTAs' shared memory.
    staged_sfb_n = max(config.tile_n, 128)
    sfb_tile_bytes = staged_sfb_n * k / 16
    tile_read_bytes = tile_count * (packed_tile_bytes + sfa_tile_bytes + sfb_tile_bytes)
    return sol_bounds(
        flops,
        compulsory_bytes,
        input_bytes,
        tile_read_bytes,
        mma_peak_tflops=MXFP4_MMA_PEAK_TFLOPS,
    )


class Mxfp4GemmBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--m", type=int, default=0)
        parser.add_argument("--n", type=int, default=0)
        parser.add_argument("--k", type=int, default=0)
        self.add_timing_args(parser, warmup=5, iters=20, samples=7)
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument(
            "--baseline",
            choices=("dequant-bf16", "deepgemm"),
            default="dequant-bf16",
            help="reference implementation to benchmark",
        )
        parser.add_argument(
            "--sol",
            action="store_true",
            help="print analytical MXFP4 MMA/HBM/L2 SOL bounds",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="print per-config runner probe and timing details",
        )
        self.add_cache_args(parser)
        self.add_pre_parallel_jit_arg(parser)
        parser.add_argument(
            "--cache-path",
            default=None,
            help="autotune cache path (default: profiles/autotune_cache.json)",
        )
        return parser

    def validate_args(self, args):
        chosen = args.m or args.n or args.k
        if chosen and not (args.m and args.n and args.k):
            raise SystemExit("--m, --n, and --k must be specified together")
        self.validate_timing_args(args, allow_zero_warmup=True)
        shapes = ((args.m, args.n, args.k),) if chosen else DEFAULT_SHAPES
        for m, n, k in shapes:
            if m % 128 or n % 128 or k % 64:
                raise SystemExit("M and N must be multiples of 128, and K of 64")
        args.shapes = shapes
        return args

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        table = TablePrinter(
            (
                ("M", 8),
                ("N", 8),
                ("K", 8),
                ("Kernel(ms)", 14),
                ("Kernel TF/s", 15),
                ("Baseline(ms)", 14),
                ("Baseline TF/s", 15),
                ("Speedup", 10),
                ("MaxDiff", 12),
                ("BestConfig", 24),
            ),
            debug=args.debug,
        )
        if args.baseline == "deepgemm":
            print("[baseline] locally patched DeepGEMM FP4xFP4 NT GEMM")
        else:
            print(
                "[baseline] FP32 dequantize + BF16 GEMM reference (PyTorch has no MXFP4 _scaled_mm path)"
            )
        print("[runner] config selection is cached per workload and CUDA environment")
        table.print_header()
        sol_report = SolReport(debug=args.debug) if args.sol else None
        for m, n, k in args.shapes:
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
            aq, bq, sfa, sfb = quantize_mxfp4_gemm(a, b)
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            ref = mxfp4_gemm_reference(aq, bq, sfa, sfb)
            baseline_out = torch.empty_like(out)
            baseline_fn = (
                deepgemm_mxfp4_nt(
                    aq,
                    bq,
                    sfa,
                    sfb,
                    baseline_out,
                    from_blocked_scales=from_blocked_mxfp4_scales,
                )
                if args.baseline == "deepgemm"
                else lambda: mxfp4_gemm_reference(aq, bq, sfa, sfb)
            )
            baseline_stats = benchmark_with_stats(
                baseline_fn,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=args._cold_l2_flush,
            )
            selection = self.resolve_kernel(
                aq,
                bq,
                sfa,
                sfb,
                out,
                m,
                n,
                k,
                args,
            )
            kernel_stats = selection.stats
            max_diff = 0.0
            if not args.no_accuracy:
                selection.prepared.run()
                if args.baseline == "deepgemm":
                    baseline_fn()
                else:
                    ref = baseline_fn()
                torch.cuda.synchronize()
                max_diff = (out.float() - ref.float()).abs().max().item()
                if not torch.equal(out, ref):
                    raise AssertionError(
                        f"MXFP4 mismatch for {m}x{n}x{k}: max diff {max_diff}"
                    )
                if args.baseline == "deepgemm" and not torch.equal(baseline_out, ref):
                    baseline_diff = (
                        (baseline_out.float() - ref.float()).abs().max().item()
                    )
                    raise AssertionError(
                        f"DeepGEMM MXFP4 mismatch for {m}x{n}x{k}: max diff {baseline_diff}"
                    )
            kms = kernel_stats["median_ms"]
            bms = baseline_stats["median_ms"]
            table.print_row(
                m,
                n,
                k,
                f"{kms:.4f}",
                f"{tflops_from_ms(m, n, k, kms):.2f}",
                f"{bms:.4f}",
                f"{tflops_from_ms(m, n, k, bms):.2f}",
                f"{bms / kms:.2f}x",
                f"{max_diff:.6f}",
                format_candidate(selection.candidate),
            )
            if args.sol:
                sol_report.add(
                    mxfp4_gemm_sol_bounds(m, n, k, selection.candidate),
                    kernel_stats["median_ms"],
                    label=f"MXFP4 GEMM ({m}x{n}x{k})",
                )
        if sol_report:
            sol_report.print()

    def resolve_kernel(self, aq, bq, sfa, sfb, out, m, n, k, args):
        kernel_runner = Mxfp4GemmRunner(aq, bq, sfa, sfb, out)
        candidates = kernel_runner.compatible_candidates()

        options = self.make_tune_options(
            args,
            role="ours",
            pre_parallel_jit=args.pre_parallel_jit,
        )
        return kernel_runner.tune(
            options,
            candidates=candidates,
            benchmark=lambda prepared: benchmark_with_stats(
                prepared.run,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            ),
            probe=lambda prepared: quick_probe(
                prepared.run,
                args.iters,
                cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            ),
            metric_formatter=lambda _config, stats: (
                f"{tflops_from_ms(m, n, k, stats['median_ms']):.1f} TF/s"
            ),
        )


if __name__ == "__main__":
    Mxfp4GemmBenchmark().run()
