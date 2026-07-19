#!/usr/bin/env python3
"""Accuracy and performance benchmark for the native SM110A NVFP4 GEMM."""

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
    CuteDslNvfp4GemmRunner,
    Nvfp4GemmConfig,
    compile_nvfp4_gemm,
    nvfp4_gemm_out,
    quantize_nvfp4_gemm,
)
from sm110a_kernel.runner import AutotuneRunner, PreparedKernel, TuneOptions


DEFAULT_SHAPES = (
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 11008, 4096),
    (11008, 4096, 4096),
)

NVFP4_MMA_PEAK_TFLOPS = 4.0 * 250.0

NVFP4_GEMM_CONFIGS = tuple(
    Nvfp4GemmConfig(
        tile_n=tile_n,
        tile_k=tile_k,
        cluster_m=cluster_m,
        epilogue_block_n=epilogue_block_n,
    )
    for cluster_m in (1, 2)
    for tile_n in (64, 128, 256)
    for tile_k in (64, 128, 256)
    for epilogue_block_n in (32, 64, 128)
    if tile_n % epilogue_block_n == 0
)


def compatible_candidates(m, n, k):
    return tuple(
        config
        for config in NVFP4_GEMM_CONFIGS
        if m % (config.tile_m * config.cluster_m) == 0
        and n % config.tile_n == 0
        and n % 128 == 0
        and k % config.tile_k == 0
    )


def format_candidate(config):
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/c{config.cluster_m}/e{config.epilogue_block_n}"
    )


def nvfp4_gemm_sol_bounds(m, n, k, config):
    """Model packed operands, block scales, and multicast full-N SFB."""
    cluster_tile_m = config.cluster_m * config.tile_m
    tile_count = math.ceil(m / cluster_tile_m) * math.ceil(n / config.tile_n)
    flops = 2 * m * n * k

    packed_operand_bytes = (m * k + n * k) / 2
    unique_scale_bytes = (m * k + n * k) / 16
    input_bytes = packed_operand_bytes + unique_scale_bytes
    compulsory_bytes = input_bytes + 2 * m * n

    packed_tile_bytes = (cluster_tile_m + config.tile_n) * k / 2
    sfa_tile_bytes = cluster_tile_m * k / 16
    # Load the complete N-wide SFB tile once per CTA cluster.  The two-CTA
    # configuration multicasts it into both CTAs' shared memory.
    staged_sfb_n = max(config.tile_n, 128)
    sfb_tile_bytes = staged_sfb_n * k / 16
    tile_read_bytes = tile_count * (
        packed_tile_bytes + sfa_tile_bytes + sfb_tile_bytes
    )
    return sol_bounds(
        flops,
        compulsory_bytes,
        input_bytes,
        tile_read_bytes,
        mma_peak_tflops=NVFP4_MMA_PEAK_TFLOPS,
    )


class Nvfp4GemmBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--m", type=int, default=0)
        parser.add_argument("--n", type=int, default=0)
        parser.add_argument("--k", type=int, default=0)
        self.add_timing_args(parser, warmup=5, iters=20, samples=7)
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument(
            "--baseline", choices=("torch", "cutedsl"), default="torch",
            help="reference implementation to benchmark",
        )
        parser.add_argument(
            "--sol", action="store_true",
            help="print analytical NVFP4 MMA/HBM/L2 SOL bounds",
        )
        parser.add_argument(
            "--debug", action="store_true",
            help="print per-config runner probe and timing details",
        )
        parser.add_argument(
            "--cutedsl-pre-parallel-jit",
            action="store_true",
            help=(
                "compile CuTE DSL baseline configurations in parallel before "
                "benchmarking (disabled by default because CuTE DSL JIT is "
                "not thread-safe)"
            ),
        )
        self.add_cache_args(parser)
        self.add_pre_parallel_jit_arg(parser)
        parser.add_argument(
            "--cache-path", default=None,
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
                raise SystemExit(
                    "M and N must be multiples of 128, and K of 64"
                )
        args.shapes = shapes
        return args

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        table = TablePrinter((
            ("M", 8), ("N", 8), ("K", 8),
            ("Kernel(ms)", 14), ("Kernel TF/s", 15),
            ("Baseline(ms)", 14), ("Baseline TF/s", 15),
            ("Speedup", 10), ("MaxDiff", 12), ("BestConfig", 24),
        ))
        if args.baseline == "cutedsl":
            print("[baseline] CuTE DSL persistent NVFP4 GEMM (autotuned)")
        else:
            print("[baseline] torch._scaled_mm NVFP4")
        print("[runner] config selection is cached per workload and CUDA environment")
        table.print_header()
        for m, n, k in args.shapes:
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
            aq, bq, sfa, sfb = quantize_nvfp4_gemm(a, b)
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            ref = torch.empty_like(out)
            if args.baseline == "cutedsl":
                cute_dsl_runner = CuteDslNvfp4GemmRunner(aq, bq, sfa, sfb, ref)
                cute_dsl_runner.autotune(TuneOptions(
                    warmup=args.warmup, iters=args.iters, samples=args.samples,
                    cold_l2_flush=args._cold_l2_flush, cache_path=args.cache_path,
                    force_sweep=args.force_sweep, debug=args.debug,
                    # Keep this separate from the local kernel's JIT option:
                    # CuTE DSL compilation has process-global session state.
                    pre_parallel_jit=args.cutedsl_pre_parallel_jit,
                ))
                baseline_fn = cute_dsl_runner.run
            else:
                baseline_fn = lambda: torch._scaled_mm(
                    aq, bq.t(), sfa, sfb, out_dtype=torch.bfloat16, out=ref
                )
            baseline_stats = benchmark_with_stats(
                baseline_fn, args.warmup, args.iters, args.samples,
                cold_l2_flush=args._cold_l2_flush,
            )
            selection = self.resolve_kernel(
                aq, bq, sfa, sfb, out, m, n, k, args,
            )
            kernel_stats = selection.stats
            max_diff = 0.0
            if not args.no_accuracy:
                selection.prepared.run()
                baseline_fn()
                torch.cuda.synchronize()
                max_diff = (out.float() - ref.float()).abs().max().item()
                if args.baseline == "torch" and not torch.equal(out, ref):
                    raise AssertionError(
                        f"NVFP4 mismatch for {m}x{n}x{k}: max diff {max_diff}"
                    )
            kms = kernel_stats["median_ms"]
            bms = baseline_stats["median_ms"]
            table.print_row(
                m, n, k, f"{kms:.4f}", f"{tflops_from_ms(m,n,k,kms):.2f}",
                f"{bms:.4f}", f"{tflops_from_ms(m,n,k,bms):.2f}",
                f"{bms/kms:.2f}x", f"{max_diff:.6f}",
                format_candidate(selection.candidate),
            )
            if args.sol:
                print_sol_report(
                    nvfp4_gemm_sol_bounds(m, n, k, selection.candidate),
                    kernel_stats["median_ms"],
                    label="NVFP4 GEMM",
                )

    def resolve_kernel(self, aq, bq, sfa, sfb, out, m, n, k, args):
        candidates = compatible_candidates(m, n, k)
        if not candidates:
            raise RuntimeError(f"no NVFP4 config supports M={m}, N={n}, K={k}")
        def prepare(config):
            run = lambda: nvfp4_gemm_out(
                aq, bq, sfa, sfb, out, config=config,
            )
            run()
            torch.cuda.synchronize()
            return PreparedKernel(config, run)

        options = TuneOptions(
            warmup=args.warmup,
            iters=args.iters,
            samples=args.samples,
            cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            cache_path=args.cache_path,
            force_sweep=args.force_sweep,
            debug=args.debug,
            pre_parallel_jit=args.pre_parallel_jit,
        )
        return AutotuneRunner(options).resolve(
            candidates,
            namespace="nvfp4_gemm",
            key=(m, n, k),
            prepare=prepare,
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
            format_candidate=format_candidate,
            metric_formatter=lambda _config, stats: (
                f"{tflops_from_ms(m, n, k, stats['median_ms']):.1f} TF/s"
            ),
            pre_jit=compile_nvfp4_gemm,
        )


if __name__ == "__main__":
    Nvfp4GemmBenchmark().run()
