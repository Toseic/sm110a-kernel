#!/usr/bin/env python3
"""Runner-backed config and tokens-per-expert sweep for NVFP4 grouped-M GEMM."""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    BenchmarkBase,
    TablePrinter,
    benchmark_with_stats,
    print_sol_report,
    quick_probe,
    sol_bounds,
    tflops_from_ms,
)
from sm110a_kernel import (
    Nvfp4GroupedMConfig,
    nvfp4_grouped_m_gemm_out,
    prepare_nvfp4_grouped_m_metadata,
    quantize_nvfp4_rows,
)
from sm110a_kernel.runner import AutotuneRunner, PreparedKernel, TuneOptions


EXPERTS = 256
N = 4096
K = 6144
DEFAULT_TOKENS_PER_EXPERT = (128, 256, 512, 1024)
NVFP4_GROUPED_M_CONFIGS = tuple(
    Nvfp4GroupedMConfig(
        tile_n=tile_n,
        tile_k=tile_k,
        epilogue_tile_n=epilogue_tile_n,
        cluster_m=cluster_m,
        mode=mode,
    )
    for tile_n in (64, 128, 256)
    for tile_k in (64, 128, 256)
    for epilogue_tile_n in (32, 64, 128)
    for cluster_m in (1, 2)
    for mode in ( 1, 2)
    if tile_n % epilogue_tile_n == 0
)
_MODE_LABELS = {0: "np", 1: "pe", 2: "clc"}
NVFP4_MMA_PEAK_TFLOPS = 4.0 * 250.0


def compatible_candidates(n, k):
    return tuple(
        config for config in NVFP4_GROUPED_M_CONFIGS
        if n % config.tile_n == 0 and n % 128 == 0 and k % config.tile_k == 0
    )


def format_candidate(config):
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/e{config.epilogue_tile_n}/c{config.cluster_m}/{_MODE_LABELS[config.mode]}"
    )


def nvfp4_grouped_m_gemm_sol_bounds(tokens_per_expert, experts, n, k, config):
    """Model packed operands, blocked scales, and grouped tile reuse."""
    m = tokens_per_expert * experts
    cluster_tile_m = config.cluster_m * config.tile_m
    task_count = (
        experts
        * math.ceil(tokens_per_expert / cluster_tile_m)
        * math.ceil(n / config.tile_n)
    )
    flops = 2 * m * n * k

    packed_a_bytes = m * k / 2
    packed_b_bytes = experts * n * k / 2
    sfa_bytes = m * k / 16
    sfb_bytes = experts * n * k / 16
    input_bytes = packed_a_bytes + packed_b_bytes + sfa_bytes + sfb_bytes
    compulsory_bytes = input_bytes + 2 * m * n

    packed_tile_bytes = (cluster_tile_m + config.tile_n) * k / 2
    sfa_tile_bytes = cluster_tile_m * k / 16
    # SFB is stored in 128-row blocks. Tile-N=64 therefore stages a complete
    # 128-row scale tile. With cluster-M=2 rank 0 TMA-multicasts it to its
    # peer, so it is fetched once per cluster rather than once per CTA.
    staged_sfb_n = max(config.tile_n, 128)
    sfb_tile_bytes = staged_sfb_n * k / 16
    tile_read_bytes = task_count * (
        packed_tile_bytes + sfa_tile_bytes + sfb_tile_bytes
    )
    return sol_bounds(
        flops,
        compulsory_bytes,
        input_bytes,
        tile_read_bytes,
        mma_peak_tflops=NVFP4_MMA_PEAK_TFLOPS,
    )


def make_workload(tokens_per_expert, experts, n, k):
    m = tokens_per_expert * experts
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    aq, sfa = quantize_nvfp4_rows(a)
    del a

    # B is retained by the benchmark only in NVFP4 form.  Quantize one expert
    # at a time so the 256-expert default never creates full-size FP32
    # quantization temporaries for the entire [experts, N, K] matrix.
    bq = None
    sfb = None
    for expert in range(experts):
        b_expert = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        bq_expert, sfb_expert = quantize_nvfp4_rows(b_expert)
        if bq is None:
            bq = torch.empty(
                (experts, n, k // 2), device="cuda", dtype=bq_expert.dtype
            )
            sfb = torch.empty(
                (experts * sfb_expert.numel(),),
                device="cuda",
                dtype=sfb_expert.dtype,
            )
        bq[expert].copy_(bq_expert)
        offset = expert * sfb_expert.numel()
        sfb[offset : offset + sfb_expert.numel()].copy_(sfb_expert)
        del b_expert, bq_expert, sfb_expert
    layout = torch.arange(experts, device="cuda", dtype=torch.int32).repeat_interleave(
        tokens_per_expert
    )
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ref = torch.empty_like(out)
    return aq, bq, sfa, sfb, layout, out, ref


def run_baseline(aq, bq, sfa, sfb, tokens_per_expert, out):
    experts, n, packed_k = bq.shape
    k = packed_k * 2
    a_scale_bytes = tokens_per_expert * (k // 16)
    b_scale_bytes = n * (k // 16)
    for expert in range(experts):
        row = expert * tokens_per_expert
        torch._scaled_mm(
            aq[row : row + tokens_per_expert],
            bq[expert].t(),
            sfa[expert * a_scale_bytes : (expert + 1) * a_scale_bytes],
            sfb[expert * b_scale_bytes : (expert + 1) * b_scale_bytes],
            out_dtype=torch.bfloat16,
            out=out[row : row + tokens_per_expert],
        )


class Nvfp4GroupedMGemmBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--experts", type=int, default=EXPERTS)
        parser.add_argument("--n", type=int, default=N)
        parser.add_argument("--k", type=int, default=K)
        parser.add_argument(
            "--tokens", type=int, default=0,
            help="single tokens-per-expert value; default runs the standard sweep",
        )
        self.add_timing_args(parser, warmup=2, iters=5, samples=3)
        self.add_cache_args(parser)
        self.add_sol_arg(parser)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--seed", type=int, default=1234)
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if args.experts <= 0 or args.n <= 0 or args.k <= 0 or args.tokens < 0:
            raise SystemExit("experts, N and K must be positive; tokens must be non-negative")
        if args.n % 128 or args.k % 64:
            raise SystemExit("N must be a multiple of 128 and K a multiple of 64")
        values = (args.tokens,) if args.tokens else DEFAULT_TOKENS_PER_EXPERT
        if any(value <= 0 for value in values):
            raise SystemExit("tokens per expert must be positive")
        args.tokens_per_expert_values = values
        return args

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        print("[baseline] per-expert torch._scaled_mm NVFP4")
        print("[runner] config selection cached by experts/N/K/tokens-per-expert")
        table = TablePrinter((
            ("Tok/E", 8), ("M", 10), ("E", 6), ("N", 8), ("K", 8),
            ("Kernel(ms)", 14), ("Kernel TF/s", 15),
            ("Torch(ms)", 14), ("Torch TF/s", 15),
            ("Speedup", 10), ("MaxDiff", 12), ("BestConfig", 34),
        ))
        table.print_header()
        for tokens in args.tokens_per_expert_values:
            aq, bq, sfa, sfb, layout, out, ref = make_workload(
                tokens, args.experts, args.n, args.k
            )
            baseline = lambda: run_baseline(aq, bq, sfa, sfb, tokens, ref)
            baseline_stats = benchmark_with_stats(
                baseline, args.warmup, args.iters, args.samples,
                cold_l2_flush=args._cold_l2_flush,
            )
            selection = self.resolve_kernel(aq, bq, sfa, sfb, layout, out, tokens, args)
            diff = 0.0
            if not args.no_accuracy:
                selection.prepared.run(); baseline(); torch.cuda.synchronize()
                diff = (out.float() - ref.float()).abs().max().item()
                if not torch.equal(out, ref):
                    raise AssertionError(f"NVFP4 grouped mismatch: max diff {diff}")
            m = tokens * args.experts
            kms, bms = selection.stats["median_ms"], baseline_stats["median_ms"]
            table.print_row(
                tokens, m, args.experts, args.n, args.k,
                f"{kms:.4f}", f"{tflops_from_ms(m,args.n,args.k,kms):.2f}",
                f"{bms:.4f}", f"{tflops_from_ms(m,args.n,args.k,bms):.2f}",
                f"{bms/kms:.2f}x", f"{diff:.6f}", format_candidate(selection.candidate),
            )
            if args.sol:
                print_sol_report(
                    nvfp4_grouped_m_gemm_sol_bounds(
                        tokens,
                        args.experts,
                        args.n,
                        args.k,
                        selection.candidate,
                    ),
                    kms,
                    label="NVFP4 grouped-M GEMM",
                )

    def resolve_kernel(self, aq, bq, sfa, sfb, layout, out, tokens, args):
        candidates = compatible_candidates(args.n, args.k)
        if not candidates:
            raise RuntimeError(f"no NVFP4 grouped config supports N={args.n}, K={args.k}")

        def prepare(config):
            metadata = prepare_nvfp4_grouped_m_metadata(
                layout, args.experts, args.n, config=config
            )
            run = lambda: nvfp4_grouped_m_gemm_out(
                aq, bq, sfa, sfb, metadata, out, config=config
            )
            run(); torch.cuda.synchronize()
            return PreparedKernel(config, run)

        options = TuneOptions(
            warmup=args.warmup, iters=args.iters, samples=args.samples,
            cold_l2_flush=args._cold_l2_flush, cache_path=args.cache_path,
            force_sweep=args.force_sweep, debug=args.debug,
        )
        m = tokens * args.experts
        return AutotuneRunner(options).resolve(
            candidates,
            namespace="nvfp4_grouped_m_gemm",
            key=(args.experts, args.n, args.k, tokens),
            prepare=prepare,
            benchmark=lambda prepared: benchmark_with_stats(
                prepared.run, args.warmup, args.iters, args.samples,
                cold_l2_flush=args._cold_l2_flush,
            ),
            probe=lambda prepared: quick_probe(
                prepared.run, cold_l2_flush=args._cold_l2_flush
            ),
            format_candidate=format_candidate,
            metric_formatter=lambda _config, stats: (
                f"{tflops_from_ms(m,args.n,args.k,stats['median_ms']):.1f} TF/s"
            ),
        )


if __name__ == "__main__":
    Nvfp4GroupedMGemmBenchmark().run()
