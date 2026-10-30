#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepgemm_baseline import bf16_nt as deepgemm_bf16_nt

from common import (
    GemmBenchmarkBase,
    SolReport,
    benchmark_with_stats,
    benchmark_with_stats_batched,
    print_sol_report,
    quick_probe,
    sol_bounds,
    tflops_from_ms,
)
from sm110a_kernel import (
    cublaslt_gemm_out,
    cublaslt_gemm_out_batched,
)
from sm110a_kernel.kernels.bf16_gemm import (
    Bf16GemmRunner,
)
from sm110a_kernel.kernels.bf16_gemm import (
    format_bf16_candidate as format_candidate,
)

DEFAULT_SHAPES = [
    (1, 4096, 4096),
    (4, 4096, 4096),
    (8, 4096, 4096),
    (16, 4096, 4096),
    (32, 4096, 4096),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (2560, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 11008, 4096),
    (11008, 4096, 4096),
    (256, 32768, 4096),
]
ACCURACY_SHAPE = (512, 512, 512)
CUBLASLT_WORKSPACE_BYTES = 32 * 1024 * 1024


def gemm_sol_bounds(m, n, k, config, swap_ab=False):
    """SOL model for the selected standard-GEMM mainloop configuration."""
    if swap_ab:
        m, n = n, m
    cluster_tile_m = config.cluster_m * config.tile_m
    tile_count = math.ceil(m / cluster_tile_m) * math.ceil(n / config.tile_n)
    flops = 2 * m * n * k
    a_and_b_bytes = 2 * (m * k + n * k)
    compulsory_bytes = a_and_b_bytes + (4 if config.output_fp32 else 2) * m * n
    tile_read_bytes = 2 * tile_count * (cluster_tile_m + config.tile_n) * k
    return sol_bounds(flops, compulsory_bytes, a_and_b_bytes, tile_read_bytes)


def print_result(m, n, k, kernel_stats, baseline_stats, max_abs_diff, best_candidate):
    return (
        m,
        n,
        k,
        f"{kernel_stats['median_ms']:.4f}",
        f"{tflops_from_ms(m, n, k, kernel_stats['median_ms']):.3f}",
        f"{baseline_stats['median_ms']:.4f}",
        f"{tflops_from_ms(m, n, k, baseline_stats['median_ms']):.3f}",
        f"{baseline_stats['median_ms'] / kernel_stats['median_ms']:.2f}",
        f"{max_abs_diff:.6f}",
        format_candidate(best_candidate),
    )


def run_accuracy_check(args):
    print("[accuracy] small shape with correctness check")
    print("[runner] resolving kernel config")
    # Keep every command-line option (notably the local kernel's
    # ``pre_parallel_jit`` flag) and override only the short accuracy run's
    # timing/check settings.  A hand-maintained Namespace easily drifts when
    # new benchmark options are added.
    accuracy_options = vars(args).copy()
    accuracy_options.update(
        warmup=2,
        iters=20,
        samples=7,
        skip_check=False,
    )
    accuracy_args = argparse.Namespace(**accuracy_options)
    bench = Bf16GemmBenchmark()
    bench._debug_enabled = getattr(args, "debug", False)
    kernel_stats, baseline_stats, max_abs_diff, baseline_name, best_candidate = (
        bench.run_case(*ACCURACY_SHAPE, args=accuracy_args)
    )
    bench.make_result_table(baseline_name).print_header()
    bench.make_result_table(baseline_name).print_row(
        *print_result(
            *ACCURACY_SHAPE,
            kernel_stats,
            baseline_stats,
            max_abs_diff,
            best_candidate,
        )
    )
    if args.sol:
        config, swap_ab = best_candidate
        print_sol_report(
            gemm_sol_bounds(*ACCURACY_SHAPE, config, swap_ab),
            kernel_stats["median_ms"],
            label="GEMM",
            debug=getattr(args, "debug", False),
        )
    print()


class Bf16GemmBenchmark(GemmBenchmarkBase):
    default_shapes = DEFAULT_SHAPES

    def baseline_choices(self):
        return ("torch", "cublaslt", "cutedsl", "deepgemm")

    def add_arguments(self, parser):
        parser.add_argument(
            "--pdl", action="store_true", help="enable programmatic dependent launch"
        )
        parser.add_argument(
            "--output-fp32",
            action="store_true",
            help="store FP32 accumulators directly (currently requires --baseline torch)",
        )
        parser.add_argument(
            "--max-active-clusters",
            type=int,
            default=0,
            help=(
                "persistent clusters to launch (default: device SM count / "
                "the selected cluster size)"
            ),
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Print per-config sweep details",
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

    def validate_args(self, args):
        # Allow --m alone to filter default shapes by M dimension, instead of
        # requiring --m, --n, --k together for single-shape mode.
        if args.m > 0 and args.n == 0 and args.k == 0:
            args.single_shape = False
            args.m_filter = args.m
            self.validate_timing_args(args, allow_zero_warmup=True)
            self.validate_extra_args(args)
            return args
        return super().validate_args(args)

    def validate_extra_args(self, args):
        if args.output_fp32 and args.baseline != "torch":
            raise SystemExit("--output-fp32 currently requires --baseline torch")
        if args.max_active_clusters < 0:
            raise SystemExit("--max-active-clusters must be non-negative")
        return args

    def iter_shapes(self, args):
        if args.single_shape:
            yield (args.m, args.n, args.k)
            return
        m_filter = getattr(args, "m_filter", 0)
        if m_filter:
            for shape in self.default_shapes:
                if shape[0] == m_filter:
                    yield shape
            return
        yield from self.default_shapes

    def run_preflight(self, args):
        if getattr(args, "m_filter", 0):
            matched = sum(1 for s in self.default_shapes if s[0] == args.m_filter)
            print(
                f"[filter] M={args.m_filter}: {matched} matching shape(s) "
                f"out of {len(self.default_shapes)}"
            )
            print()
        if not args.single_shape and not args.no_accuracy:
            run_accuracy_check(args)

    def emit_runtime_banner(self, args, actual_blas):
        print(
            f"[output] {'FP32' if args.output_fp32 else 'BF16'} (independent autotune cache)"
        )
        if args.baseline == "cutedsl":
            print("[baseline] CuTE DSL persistent BF16 GEMM (autotuned)")
            return "CuTE DSL"
        if args.baseline == "deepgemm":
            print("[baseline] locally patched DeepGEMM BF16 NT GEMM")
            return "DeepGEMM"
        baseline_name = super().emit_runtime_banner(args, actual_blas)
        print("[runner] config selection is cached per workload and CUDA environment")
        print()
        return baseline_name

    def make_result_table(self, baseline_name):
        return self.make_table_printer(
            (
                ("M", 8),
                ("N", 8),
                ("K", 8),
                ("KernelMed(ms)", 16),
                ("Kernel TF/s", 16),
                (f"{baseline_name}Med(ms)", 16),
                (f"{baseline_name} TF/s", 16),
                (f"{baseline_name}Speedup", 14),
                ("Diff", 12),
                ("BestConfig", 40),
            )
        )

    def print_header(self, baseline_name):
        self.make_result_table(baseline_name).print_header()
        self._sol_report = (
            SolReport(debug=self._debug_enabled) if self._sol_enabled else None
        )

    def run_case(self, m, n, k, args):
        a = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.empty((k, n), device="cuda", dtype=torch.bfloat16)
        a.uniform_(-1.0, 1.0)
        b.uniform_(-1.0, 1.0)
        # Native TMA loads B as contiguous (N, K).  Pack once before all
        # measurements; the benchmark operation remains a(M,K) @ b(K,N).
        b_kernel = b.transpose(0, 1).contiguous()

        output_dtype = torch.float32 if args.output_fp32 else torch.bfloat16
        out_kernel = torch.empty((m, n), device="cuda", dtype=output_dtype)
        out_baseline = torch.empty((m, n), device="cuda", dtype=output_dtype)
        workspace = None
        cute_dsl_runner = None

        if args.baseline == "torch":
            baseline_name = "Torch"
            if args.output_fp32:
                baseline_fn = lambda: torch.mm(
                    a, b, out_dtype=torch.float32, out=out_baseline
                )
            else:
                baseline_fn = lambda: torch.mm(a, b, out=out_baseline)
            baseline_stats = benchmark_with_stats(
                baseline_fn,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            )
        elif args.baseline == "cublaslt":
            baseline_name = "Lt"
            workspace = torch.empty(
                (CUBLASLT_WORKSPACE_BYTES,), device="cuda", dtype=torch.uint8
            )
            baseline_fn = lambda iters: cublaslt_gemm_out_batched(
                a, b, out_baseline, workspace, iters
            )
            baseline_stats = benchmark_with_stats_batched(
                baseline_fn,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            )
        elif args.baseline == "cutedsl":
            from sm110a_kernel import CuteDslBf16GemmRunner

            cute_dsl_runner = CuteDslBf16GemmRunner(a, b, out_baseline)
            cute_dsl_runner.autotune(
                self.make_tune_options(
                    args,
                    role="baseline",
                    # CuTE DSL's compiler has process-global session state and
                    # cannot safely compile configurations concurrently.  Keep
                    # this separate from the local kernel's parallel JIT flag.
                    pre_parallel_jit=args.cutedsl_pre_parallel_jit,
                )
            )
            baseline_name = "CuTE DSL"
            baseline_fn = cute_dsl_runner.run
            baseline_stats = benchmark_with_stats(
                baseline_fn,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            )
        elif args.baseline == "deepgemm":
            baseline_name = "DeepGEMM"
            baseline_fn = deepgemm_bf16_nt(a, b_kernel, out_baseline)
            baseline_stats = benchmark_with_stats(
                baseline_fn,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            )
        else:
            raise SystemExit(f"unsupported --baseline value: {args.baseline}")

        selection = self.resolve_kernel(a, b_kernel, out_kernel, m, n, k, args)
        best_candidate, kernel_stats = selection.candidate, selection.stats

        max_abs_diff = 0.0
        if not args.skip_check:
            best_config, best_swap_ab = best_candidate
            selection.prepared.run()
            if args.baseline == "torch":
                baseline_fn()
            elif args.baseline == "cublaslt":
                cublaslt_gemm_out(a, b, out_baseline, workspace)
            elif args.baseline == "deepgemm":
                baseline_fn()
            else:
                cute_dsl_runner.run()
            max_abs_diff = (
                (out_kernel.float() - out_baseline.float()).abs().max().item()
            )
            if args.output_fp32:
                torch.testing.assert_close(
                    out_kernel, out_baseline, rtol=1e-4, atol=1e-3
                )

        return (
            kernel_stats,
            baseline_stats,
            max_abs_diff,
            baseline_name,
            best_candidate,
        )

    def resolve_kernel(self, a, b_kernel, out, m, n, k, args):
        kernel_runner = Bf16GemmRunner(
            a,
            b_kernel,
            out,
            max_active_clusters=args.max_active_clusters or None,
            pdl=args.pdl,
        )
        candidates = kernel_runner.compatible_candidates()
        options = self.make_tune_options(
            args, role="ours", pre_parallel_jit=args.pre_parallel_jit
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
            metric_formatter=lambda _candidate, stats: (
                f"{tflops_from_ms(m, n, k, stats['median_ms']):.1f} TF/s"
            ),
        )

    def print_result(self, shape, result):
        m, n, k = shape
        kernel_stats, baseline_stats, max_abs_diff, _, best_candidate = result
        self.make_result_table(result[3]).print_row(
            *print_result(
                m, n, k, kernel_stats, baseline_stats, max_abs_diff, best_candidate
            )
        )
        if self._sol_enabled:
            config, swap_ab = best_candidate
            self._sol_report.add(
                gemm_sol_bounds(m, n, k, config, swap_ab),
                kernel_stats["median_ms"],
                label=f"GEMM ({m}x{n}x{k})",
            )

    def finish_results(self):
        if self._sol_report:
            self._sol_report.print()


def main():
    Bf16GemmBenchmark().run()


if __name__ == "__main__":
    main()
