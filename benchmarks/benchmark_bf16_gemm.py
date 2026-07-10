#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    GemmBenchmarkBase,
    benchmark_with_stats,
    benchmark_with_stats_batched,
    quick_probe,
    tflops_from_ms,
)
from sm110a_kernel import (
    Bf16GemmConfig,
    bf16_gemm_out,
    cublaslt_gemm_out,
    cublaslt_gemm_out_batched,
)


DEFAULT_SHAPES = [
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
BF16_GEMM_CONFIGS = tuple(
    Bf16GemmConfig(
        tile_m=128,
        tile_n=tile_n,
        tile_k=tile_k,
        cluster_m=cluster_m,
        epilogue_block_n=epilogue_block_n,
    )
    for tile_n in (128, 256)
    for tile_k in (64, 128)
    for cluster_m in (1, 2)
    for epilogue_block_n in (32, 64, 128)
    if tile_n % epilogue_block_n == 0
    and (
        227 * 1024 - 2 * 128 * epilogue_block_n * 2
    ) // ((128 + tile_n // cluster_m) * tile_k * 2) >= 2
)


def compatible_candidates(m, n, k):
    return tuple(
        (config, swap_ab)
        for config in BF16_GEMM_CONFIGS
        for swap_ab in (False, True)
        if k % config.tile_k == 0
        and (
            (
                not swap_ab
                and m % (config.tile_m * config.cluster_m) == 0
                and n % config.tile_n == 0
            )
            or (
                swap_ab
                and n % (config.tile_m * config.cluster_m) == 0
                and m % config.tile_n == 0
            )
        )
    )


def format_candidate(candidate):
    config, swap_ab = candidate
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/c{config.cluster_m}"
        f"/e{config.epilogue_block_n}"
        f"/{'swap' if swap_ab else 'noswap'}"
    )


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
    print("[autotune] sweeping over configs")
    accuracy_args = argparse.Namespace(
        warmup=2,
        iters=20,
        samples=7,
        skip_check=False,
        baseline=args.baseline,
        debug=args.debug,
    )
    bench = Bf16GemmBenchmark()
    kernel_stats, baseline_stats, max_abs_diff, baseline_name, best_candidate = bench.run_case(
        *ACCURACY_SHAPE, args=accuracy_args
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
    print()


class Bf16GemmBenchmark(GemmBenchmarkBase):
    default_shapes = DEFAULT_SHAPES

    def add_arguments(self, parser):
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Print per-config sweep details",
        )

    def run_preflight(self, args):
        if not args.single_shape and not args.no_accuracy:
            run_accuracy_check(args)

    def emit_runtime_banner(self, args, actual_blas):
        baseline_name = super().emit_runtime_banner(args, actual_blas)
        print(
            "[autotune] configs: "
            + ", ".join(
                format_candidate((config, swap_ab))
                for config in BF16_GEMM_CONFIGS
                for swap_ab in (False, True)
            )
        )
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
                ("BestConfig", 28),
            )
        )

    def print_header(self, baseline_name):
        self.make_result_table(baseline_name).print_header()

    def run_case(self, m, n, k, args):
        a = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.empty((k, n), device="cuda", dtype=torch.bfloat16)
        a.uniform_(-1.0, 1.0)
        b.uniform_(-1.0, 1.0)

        out_kernel = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        out_baseline = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        workspace = None

        if args.baseline == "torch":
            baseline_name = "Torch"
            baseline_fn = lambda: torch.mm(a, b, out=out_baseline)
            baseline_stats = benchmark_with_stats(
                baseline_fn,
                args.warmup,
                args.iters,
                args.samples,
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
            )
        else:
            raise SystemExit(f"unsupported --baseline value: {args.baseline}")

        best_candidate, kernel_stats = self.benchmark_best_kernel(
            a, b, out_kernel, m, n, k, args
        )

        max_abs_diff = 0.0
        if not args.skip_check:
            best_config, best_swap_ab = best_candidate
            bf16_gemm_out(
                a, b, out_kernel, config=best_config, swap_ab=best_swap_ab
            )
            if args.baseline == "torch":
                baseline_fn()
            else:
                cublaslt_gemm_out(a, b, out_baseline, workspace)
            max_abs_diff = (
                (out_kernel.float() - out_baseline.float()).abs().max().item()
            )

        return (
            kernel_stats,
            baseline_stats,
            max_abs_diff,
            baseline_name,
            best_candidate,
        )

    def benchmark_best_kernel(self, a, b, out, m, n, k, args):
        candidates = compatible_candidates(m, n, k)
        if not candidates:
            raise RuntimeError(f"no JIT config supports M={m}, N={n}, K={k}")

        def run_kernel(candidate):
            config, swap_ab = candidate
            bf16_gemm_out(a, b, out, config=config, swap_ab=swap_ab)
            torch.cuda.synchronize()
            return benchmark_with_stats(
                lambda: bf16_gemm_out(
                    a, b, out, config=config, swap_ab=swap_ab
                ),
                args.warmup,
                args.iters,
                args.samples,
            )

        def probe_kernel(candidate):
            config, swap_ab = candidate
            bf16_gemm_out(a, b, out, config=config, swap_ab=swap_ab)
            torch.cuda.synchronize()
            return quick_probe(
                lambda: bf16_gemm_out(
                    a, b, out, config=config, swap_ab=swap_ab
                )
            )

        best_candidate, best_stats, _ = self.autotune_candidates(
            candidates,
            args=args,
            benchmark_candidate=run_kernel,
            format_candidate=format_candidate,
            probe_candidate=probe_kernel,
            metric_formatter=lambda _candidate, stats: (
                f"{tflops_from_ms(m, n, k, stats['median_ms']):.1f} TF/s"
            ),
            candidate_width=28,
        )
        return best_candidate, best_stats

    def print_result(self, shape, result):
        m, n, k = shape
        kernel_stats, baseline_stats, max_abs_diff, _, best_candidate = result
        self.make_result_table(result[3]).print_row(
            *print_result(
                m, n, k, kernel_stats, baseline_stats, max_abs_diff, best_candidate
            )
        )


def main():
    Bf16GemmBenchmark().run()


if __name__ == "__main__":
    main()
