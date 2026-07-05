#!/usr/bin/env python3

import argparse
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sm110a_kernel import bf16_grouped_m_gemm_out


TARGET_SAMPLE_MS = 50.0
EXPERTS = 256
N = 6144
K = 4096
TOKENS_PER_EXPERT_VALUES = (32, 64, 128, 256, 512, 1024)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--check", action="store_true")
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


def tflops_from_ms(m, n, k, elapsed_ms):
    flops = 2.0 * m * n * k
    return flops / (elapsed_ms * 1.0e-3) / 1.0e12


def calibrate_iters(initial_ms, requested_iters):
    if initial_ms <= 0.0:
        return requested_iters
    total_ms = initial_ms * requested_iters
    if total_ms >= TARGET_SAMPLE_MS:
        return requested_iters
    scale = TARGET_SAMPLE_MS / total_ms
    return max(requested_iters, math.ceil(requested_iters * scale))


def summarize_samples(samples, measured_iters):
    samples = sorted(samples)
    return {
        "min_ms": samples[0],
        "median_ms": statistics.median(samples),
        "max_ms": samples[-1],
        "measured_iters": measured_iters,
    }


def benchmark_once(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_with_stats(fn, warmup, iters, samples):
    probe_ms = benchmark_once(fn, warmup, iters)
    measured_iters = calibrate_iters(probe_ms, iters)

    timings = [benchmark_once(fn, warmup, measured_iters)]
    for _ in range(1, samples):
        timings.append(benchmark_once(fn, 0, measured_iters))
    return summarize_samples(timings, measured_iters)


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


def benchmark_grouped_kernel(a, b, layout, out, args):
    return benchmark_with_stats(
        lambda: bf16_grouped_m_gemm_out(a, b, layout, out),
        args.warmup,
        args.iters,
        args.samples,
    )


def max_valid_abs_diff(ref, test):
    return (ref.float() - test.float()).abs().max().item()


def print_header():
    print(
        f"{'Tok/E':<8}{'Pad/E':<8}{'M':<10}{'E':<8}{'N':<8}{'K':<8}"
        f"{'KernelMed(ms)':<16}{'Kernel TF/s':<16}"
        f"{'BaselineMed(ms)':<16}{'Baseline TF/s':<16}"
        f"{'Speedup':<12}{'MaxDiff':<12}"
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
        f"{speedup:<12.2f}{diff:<12.6f}"
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

    b = torch.empty((EXPERTS, N, K), device="cuda", dtype=torch.bfloat16)
    b.uniform_(-1.0, 1.0)
    print_header()

    for tokens_per_expert in TOKENS_PER_EXPERT_VALUES:
        a, layout, out_kernel, out_baseline = make_workload(
            tokens_per_expert, b
        )
        cluster_m = 1 if tokens_per_expert <= 128 else 2
        grouped_m_tile = cluster_m * 128
        padded_tokens = (
            math.ceil(tokens_per_expert / grouped_m_tile) * grouped_m_tile
        )
        baseline_stats = benchmark_grouped_baseline(
            a,
            b,
            tokens_per_expert,
            out_baseline,
            args,
        )
        kernel_stats = benchmark_grouped_kernel(a, b, layout, out_kernel, args)
        diff = 0.0
        if args.check:
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
        )


if __name__ == "__main__":
    main()
