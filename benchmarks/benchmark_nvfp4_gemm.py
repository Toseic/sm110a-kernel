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

from deepgemm_baseline import nvfp4_nt as deepgemm_nvfp4_nt

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
    CuteDslNvfp4GemmRunner,
    quantize_nvfp4_gemm,
)
from sm110a_kernel.kernels.nvfp4_gemm import (
    Nvfp4GemmRunner,
    format_candidate,
)

DEFAULT_SHAPES = (
    # mxf4nvf4 and its blocked scale format start at M=128.  BF16's smaller
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

NVFP4_MMA_PEAK_TFLOPS = 4.0 * 250.0


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
    tile_read_bytes = tile_count * (packed_tile_bytes + sfa_tile_bytes + sfb_tile_bytes)
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
            "--baseline",
            choices=("torch", "cutedsl", "deepgemm"),
            default="torch",
            help="reference implementation to benchmark",
        )
        parser.add_argument(
            "--sol",
            action="store_true",
            help="print analytical NVFP4 MMA/HBM/L2 SOL bounds",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
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
        parser.add_argument(
            "--cutedsl-sweep-scheduler",
            action="store_true",
            help=(
                "expand each CuTE DSL NVFP4 tactic over swizzle sizes "
                "1,2,4,8 and both CTA raster orders"
            ),
        )
        parser.add_argument(
            "--cutedsl-nvmmh",
            action="store_true",
            help="rank/prune CuTE DSL NVFP4 tactics with nvMatmulHeuristics",
        )
        parser.add_argument(
            "--cutedsl-nvmmh-fields",
            default="tile,cluster",
            help="comma-separated nvMatmulHeuristics knobs: tile,cluster,swizzle,cta_order",
        )
        parser.add_argument(
            "--cutedsl-nvmmh-max-tactics",
            type=int,
            default=5,
            help="maximum ranked tile/cluster/swap tactics to retain",
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
        if args.baseline == "cutedsl":
            print("[baseline] CuTE DSL persistent NVFP4 GEMM (autotuned)")
        elif args.baseline == "deepgemm":
            print("[baseline] locally patched DeepGEMM cuBLASLt NVFP4 GEMM")
        else:
            print("[baseline] torch._scaled_mm NVFP4")
        print("[runner] config selection is cached per workload and CUDA environment")
        table.print_header()
        sol_report = SolReport(debug=args.debug) if args.sol else None
        for m, n, k in args.shapes:
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
            aq, bq, sfa, sfb = quantize_nvfp4_gemm(a, b)
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            ref = torch.empty_like(out)
            if args.baseline == "cutedsl":
                cute_dsl_runner = CuteDslNvfp4GemmRunner(aq, bq, sfa, sfb, ref)
                cute_dsl_runner.autotune(
                    self.make_tune_options(
                        args,
                        role="baseline",
                        # Keep this separate from the local kernel's JIT option:
                        # CuTE DSL compilation has process-global session state.
                        pre_parallel_jit=args.cutedsl_pre_parallel_jit,
                        cutedsl_sweep_scheduler=args.cutedsl_sweep_scheduler,
                        cutedsl_nvmmh=args.cutedsl_nvmmh,
                        cutedsl_nvmmh_fields=args.cutedsl_nvmmh_fields,
                        cutedsl_nvmmh_max_tactics=args.cutedsl_nvmmh_max_tactics,
                    )
                )
                baseline_fn = cute_dsl_runner.run
            elif args.baseline == "deepgemm":
                baseline_fn = deepgemm_nvfp4_nt(aq, bq, sfa, sfb, ref)
            else:
                baseline_fn = lambda: torch._scaled_mm(
                    aq, bq.t(), sfa, sfb, out_dtype=torch.bfloat16, out=ref
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
                    nvfp4_gemm_sol_bounds(m, n, k, selection.candidate),
                    kernel_stats["median_ms"],
                    label=f"NVFP4 GEMM ({m}x{n}x{k})",
                )
        if sol_report:
            sol_report.print()

    def resolve_kernel(self, aq, bq, sfa, sfb, out, m, n, k, args):
        kernel_runner = Nvfp4GemmRunner(aq, bq, sfa, sfb, out)
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
    Nvfp4GemmBenchmark().run()
