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
    EARLY_STOP_RATIO,
    benchmark_once,
    benchmark_with_stats,
    benchmark_with_stats_batched,
    calibrate_iters,
    quick_probe,
    summarize_samples,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--debug", action="store_true",
                        help="Print per-config sweep details")
    parser.add_argument("--no-accuracy", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument(
        "--torch-blas",
        choices=("default", "cublas", "cublaslt"),
        default="default",
    )
    parser.add_argument(
        "--baseline",
        choices=("torch", "cublaslt"),
        default="cublaslt",
    )
    args = parser.parse_args()

    single_shape = args.m > 0 or args.n > 0 or args.k > 0
    if single_shape and not (args.m > 0 and args.n > 0 and args.k > 0):
        raise SystemExit("--m, --n, --k must all be provided together")
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    return args, single_shape


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


def run_case(m, n, k, args):
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

    # Autotune with early stopping
    candidates = compatible_candidates(m, n, k)
    if not candidates:
        raise RuntimeError(
            f"no JIT config supports M={m}, N={n}, K={k}"
        )

    best_candidate = None
    best_stats = None
    best_probe_ms = None

    for candidate in candidates:
        config, swap_ab = candidate
        # Compile + brief warmup
        bf16_gemm_out(a, b, out_kernel, config=config, swap_ab=swap_ab)
        torch.cuda.synchronize()

        # Quick probe for early-stop decision
        probe_ms = quick_probe(
            lambda: bf16_gemm_out(
                a, b, out_kernel, config=config, swap_ab=swap_ab
            )
        )

        if best_probe_ms is not None and probe_ms > best_probe_ms * EARLY_STOP_RATIO:
            if args.debug:
                print(f"  {format_candidate(candidate):<28} probe={probe_ms:.4f}ms"
                      f" — skip (>{best_probe_ms * EARLY_STOP_RATIO:.4f})")
            continue

        # Proper warmup before detailed measurement
        for _ in range(args.warmup):
            bf16_gemm_out(a, b, out_kernel, config=config, swap_ab=swap_ab)
        torch.cuda.synchronize()

        stats = benchmark_with_stats(
            lambda: bf16_gemm_out(
                a, b, out_kernel, config=config, swap_ab=swap_ab
            ),
            args.warmup,
            args.iters,
            args.samples,
        )
        tflops = tflops_from_ms(m, n, k, stats["median_ms"])
        marker = ""
        if best_stats is None or stats["median_ms"] < best_stats["median_ms"]:
            best_stats = stats
            best_candidate = candidate
            best_probe_ms = probe_ms
            marker = " *"
        if args.debug:
            print(f"  {format_candidate(candidate):<28} min={stats['min_ms']:.4f}ms"
                  f" median={stats['median_ms']:.4f}ms"
                  f" {tflops:.1f} TF/s{marker}")
    if args.debug:
        print(f"  -> best: {format_candidate(best_candidate)}")
    kernel_stats = best_stats

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

    return kernel_stats, baseline_stats, max_abs_diff, baseline_name, best_candidate


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


def print_header():
    print_header_for("Torch")


def print_header_for(baseline_name):
    print(
        f"{'M':<8}{'N':<8}{'K':<8}"
        f"{'KernelMed(ms)':<16}{'Kernel TF/s':<16}"
        f"{baseline_name + 'Med(ms)':<16}{baseline_name + ' TF/s':<16}"
        f"{baseline_name + 'Speedup':<14}{'Diff':<12}{'BestConfig':<28}"
    )


def print_result(m, n, k, kernel_stats, baseline_stats, max_abs_diff, best_candidate):
    kernel_tflops = tflops_from_ms(m, n, k, kernel_stats["median_ms"])
    baseline_tflops = tflops_from_ms(m, n, k, baseline_stats["median_ms"])
    baseline_speedup = baseline_stats["median_ms"] / kernel_stats["median_ms"]
    print(
        f"{m:<8}{n:<8}{k:<8}"
        f"{kernel_stats['median_ms']:<16.4f}{kernel_tflops:<16.3f}"
        f"{baseline_stats['median_ms']:<16.4f}{baseline_tflops:<16.3f}"
        f"{baseline_speedup:<14.2f}{max_abs_diff:<12.6f}"
        f"{format_candidate(best_candidate):<28}"
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
    kernel_stats, baseline_stats, max_abs_diff, baseline_name, best_candidate = run_case(
        *ACCURACY_SHAPE, args=accuracy_args
    )
    print_header_for(baseline_name)
    print_result(*ACCURACY_SHAPE, kernel_stats, baseline_stats, max_abs_diff, best_candidate)
    print()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    args, single_shape = parse_args()
    torch.manual_seed(1234)

    actual_blas = None
    if args.baseline == "torch":
        actual_blas = configure_torch_blas(args.torch_blas)

    if not single_shape and not args.no_accuracy:
        run_accuracy_check(args)

    if args.baseline == "torch":
        print(f"[torch] preferred BLAS backend: {actual_blas}")
        baseline_name = "Torch"
    else:
        print("[baseline] direct cuBLASLt binding")
        baseline_name = "Lt"

    print("[autotune] configs: "
          + ", ".join(
              format_candidate((config, swap_ab))
              for config in BF16_GEMM_CONFIGS
              for swap_ab in (False, True)
          ))
    print()

    print_header_for(baseline_name)
    if single_shape:
        kernel_stats, baseline_stats, max_abs_diff, _, best_candidate = run_case(
            args.m, args.n, args.k, args
        )
        print_result(
            args.m, args.n, args.k,
            kernel_stats, baseline_stats, max_abs_diff, best_candidate
        )
    else:
        for shape in DEFAULT_SHAPES:
            kernel_stats, baseline_stats, max_abs_diff, _, best_candidate = run_case(
                *shape, args
            )
            print_result(
                *shape, kernel_stats, baseline_stats, max_abs_diff, best_candidate
            )


if __name__ == "__main__":
    main()
