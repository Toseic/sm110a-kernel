#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    benchmark_with_stats,
    summarize_samples,
    tflops_from_ms,
)
from sm110a_kernel import GroupedMConfig, bf16_grouped_m_gemm_out
from sm110a_kernel.jit import compile_bf16_grouped_m_gemm_configs

EXPERTS = 256
N = 6144
K = 4096
TOKENS_PER_EXPERT_VALUES = (32, 64, 128, 256, 512, 1024)
GROUPED_M_CONFIGS = tuple(
    GroupedMConfig(
        tile_m=128,
        tile_n=tile_n,
        tile_k=tile_k,
        epilogue_tile_n=epilogue_tile_n,
        cluster_m=cluster_m,
    )
    for tile_n in (128, 256)
    for tile_k in (64, 128)
    for epilogue_tile_n in (32, 64, 128)
    for cluster_m in (1, 2)
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print per-config performance after the benchmark",
    )
    parser.add_argument(
        "--torch-blas",
        choices=("default", "cublas", "cublaslt"),
        default="default",
    )
    args = parser.parse_args()

    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    return args


def configure_torch_blas(preferred_blas):
    if not hasattr(torch.backends.cuda, "preferred_blas_library"):
        if preferred_blas != "default":
            raise SystemExit(
                "This PyTorch build does not support selecting the CUDA BLAS backend"
            )
        return "unavailable"

    if preferred_blas == "default":
        torch.backends.cuda.preferred_blas_library("default")
    elif preferred_blas == "cublas":
        torch.backends.cuda.preferred_blas_library("cublas")
    elif preferred_blas == "cublaslt":
        torch.backends.cuda.preferred_blas_library("cublaslt")
    else:
        raise SystemExit(f"unsupported --torch-blas value: {preferred_blas}")

    return str(torch.backends.cuda.preferred_blas_library()).split(".")[-1]


def build_layout(tokens_per_expert, experts):
    layout = torch.arange(experts, device="cuda", dtype=torch.int32)
    return layout.repeat_interleave(tokens_per_expert)


def make_workload(tokens_per_expert, b):
    experts, n, k = b.shape
    layout = build_layout(tokens_per_expert, experts)
    valid_m = experts * tokens_per_expert
    a = torch.empty((valid_m, k), device="cuda", dtype=torch.bfloat16)
    a.uniform_(-1.0, 1.0)
    out_kernel = torch.empty((valid_m, n), device="cuda", dtype=torch.bfloat16)
    out_baseline = torch.empty((valid_m, n), device="cuda", dtype=torch.bfloat16)
    return a, layout, out_kernel, out_baseline


def run_grouped_baseline(a, b, tokens_per_expert, out):
    for expert in range(b.size(0)):
        offset = expert * tokens_per_expert
        a_slice = a[offset : offset + tokens_per_expert]
        out_slice = out[offset : offset + tokens_per_expert]
        torch.mm(a_slice, b[expert].t(), out=out_slice)


def benchmark_grouped_baseline(a, b, tokens_per_expert, out, args):
    return benchmark_with_stats(
        lambda: run_grouped_baseline(a, b, tokens_per_expert, out),
        args.warmup,
        args.iters,
        args.samples,
    )


def benchmark_grouped_kernel(a, b, layout, out, config, args):
    # Compile and launch once before recording CUDA events. This also rejects a
    # configuration that cannot launch on the current device.
    bf16_grouped_m_gemm_out(a, b, layout, out, config=config)
    torch.cuda.synchronize()
    return benchmark_with_stats(
        lambda: bf16_grouped_m_gemm_out(
            a, b, layout, out, config=config
        ),
        args.warmup,
        args.iters,
        args.samples,
    )


def compatible_configs(n, k):
    return tuple(
        config
        for config in GROUPED_M_CONFIGS
        if n % config.tile_n == 0 and k % config.tile_k == 0
    )


def benchmark_best_grouped_kernel(a, b, layout, out, args):
    configs = compatible_configs(b.size(1), a.size(1))
    if not configs:
        raise RuntimeError(
            f"no grouped M config supports N={b.size(1)}, K={a.size(1)}"
        )

    results = []
    for config in configs:
        stats = benchmark_grouped_kernel(
            a, b, layout, out, config, args
        )
        results.append((stats, config))
    best_stats, best_config = min(
        results, key=lambda result: result[0]["median_ms"]
    )
    return best_stats, best_config, results


def format_config(config):
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/e{config.epilogue_tile_n}/c{config.cluster_m}"
    )


def max_valid_abs_diff(ref, test):
    return (ref.float() - test.float()).abs().max().item()


def print_header():
    print(
        f"{'Tok/E':<8}{'Pad/E':<8}{'M':<10}{'E':<8}{'N':<8}{'K':<8}"
        f"{'KernelMed(ms)':<16}{'Kernel TF/s':<16}"
        f"{'BaselineMed(ms)':<16}{'Baseline TF/s':<16}"
        f"{'Speedup':<12}{'MaxDiff':<12}{'BestConfig':<28}"
    )


def print_result(
    tokens_per_expert,
    padded_tokens,
    experts,
    n,
    k,
    kernel_stats,
    baseline_stats,
    diff,
    best_config,
):
    valid_m = experts * tokens_per_expert
    kernel_tflops = tflops_from_ms(valid_m, n, k, kernel_stats["median_ms"])
    baseline_tflops = tflops_from_ms(
        valid_m, n, k, baseline_stats["median_ms"]
    )
    speedup = baseline_stats["median_ms"] / kernel_stats["median_ms"]
    print(
        f"{tokens_per_expert:<8}{padded_tokens:<8}{valid_m:<10}"
        f"{experts:<8}{n:<8}{k:<8}"
        f"{kernel_stats['median_ms']:<16.4f}{kernel_tflops:<16.3f}"
        f"{baseline_stats['median_ms']:<16.4f}{baseline_tflops:<16.3f}"
        f"{speedup:<12.2f}{diff:<12.6f}{format_config(best_config):<28}"
    )


def print_debug_results(results, experts, n, k):
    print()
    print("[debug] per-config performance")
    print(
        f"{'Tok/E':<8}{'Config':<28}{'Median(ms)':<14}"
        f"{'TFLOPS':<14}{'Best':<6}"
    )
    for tokens_per_expert, stats, config, is_best in results:
        valid_m = experts * tokens_per_expert
        tflops = tflops_from_ms(
            valid_m, n, k, stats["median_ms"]
        )
        print(
            f"{tokens_per_expert:<8}{format_config(config):<28}"
            f"{stats['median_ms']:<14.4f}{tflops:<14.3f}"
            f"{'*' if is_best else '':<6}"
        )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    args = parse_args()
    torch.manual_seed(1234)

    actual_blas = configure_torch_blas(args.torch_blas)
    print(f"[torch] preferred BLAS backend: {actual_blas}")
    print(
        f"[workload] experts={EXPERTS} n={N} k={K} "
        f"tokens_per_expert={TOKENS_PER_EXPERT_VALUES}"
    )
    print(
        "[autotune] configs="
        + ", ".join(format_config(config) for config in GROUPED_M_CONFIGS)
    )
    compile_bf16_grouped_m_gemm_configs(
        compatible_configs(N, K)
    )

    b = torch.empty((EXPERTS, N, K), device="cuda", dtype=torch.bfloat16)
    b.uniform_(-1.0, 1.0)
    print_header()
    debug_results = []

    for tokens_per_expert in TOKENS_PER_EXPERT_VALUES:
        a, layout, out_kernel, out_baseline = make_workload(
            tokens_per_expert, b
        )
        baseline_stats = benchmark_grouped_baseline(
            a,
            b,
            tokens_per_expert,
            out_baseline,
            args,
        )
        (
            kernel_stats,
            best_config,
            config_results,
        ) = benchmark_best_grouped_kernel(a, b, layout, out_kernel, args)
        if args.debug:
            debug_results.extend(
                (
                    tokens_per_expert,
                    stats,
                    config,
                    config == best_config,
                )
                for stats, config in config_results
            )
        grouped_m_tile = best_config.cluster_m * best_config.tile_m
        padded_tokens = (
            math.ceil(tokens_per_expert / grouped_m_tile) * grouped_m_tile
        )
        diff = 0.0
        if args.check:
            bf16_grouped_m_gemm_out(
                a,
                b,
                layout,
                out_kernel,
                config=best_config,
            )
            run_grouped_baseline(a, b, tokens_per_expert, out_baseline)
            diff = max_valid_abs_diff(out_baseline, out_kernel)
        print_result(
            tokens_per_expert,
            padded_tokens,
            EXPERTS,
            N,
            K,
            kernel_stats,
            baseline_stats,
            diff,
            best_config,
        )

    if args.debug:
        print_debug_results(debug_results, EXPERTS, N, K)


if __name__ == "__main__":
    main()
