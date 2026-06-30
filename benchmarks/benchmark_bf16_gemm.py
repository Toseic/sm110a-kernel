#!/usr/bin/env python3

import argparse
from datetime import datetime
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sm110a_kernel import (
    bf16_gemm_out,
    bf16_gemm_profile,
    cublaslt_gemm_out,
    cublaslt_gemm_out_batched,
    make_profiler_buffer,
    write_trace,
)


DEFAULT_SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (2560, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 11008, 4096),
    (8192, 4096, 4096),
]
ACCURACY_SHAPE = (512, 512, 512)
TARGET_SAMPLE_MS = 50.0
CUBLASLT_WORKSPACE_BYTES = 32 * 1024 * 1024
DEFAULT_PROFILE_NUM_ENTRIES = 100_000
DEFAULT_PROFILE_NUM_BLOCKS = 20
DEFAULT_PROFILE_OUTPUT_DIR = ROOT / "profiles"
BF16_GEMM_PROFILER_TAGS = (
    "SETUP",
    "ISSUE_TMA",
    "ISSUE_MMA",
    "WAIT_TMA",
    "WAIT_MMA",
    "WAIT_MAINLOOP",
    "WAIT_EPILOGUE",
    "EPILOGUE",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
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
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-output", default=str(DEFAULT_PROFILE_OUTPUT_DIR))
    parser.add_argument("--profile-num-blocks", type=int, default=DEFAULT_PROFILE_NUM_BLOCKS)
    parser.add_argument(
        "--profile-num-entries", type=int, default=DEFAULT_PROFILE_NUM_ENTRIES
    )
    args = parser.parse_args()

    single_shape = args.m > 0 or args.n > 0 or args.k > 0
    if single_shape and not (args.m > 0 and args.n > 0 and args.k > 0):
        raise SystemExit("--m, --n, --k must all be provided together")
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.profile and not single_shape:
        raise SystemExit("--profile requires --m, --n, --k")
    if args.profile_num_blocks <= 0:
        raise SystemExit("--profile-num-blocks must be positive")
    if args.profile_num_entries <= 0:
        raise SystemExit("--profile-num-entries must be positive")
    return args, single_shape


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


def benchmark_once_batched(fn, warmup, iters):
    for _ in range(warmup):
        fn(1)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn(iters)
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


def benchmark_with_stats_batched(fn, warmup, iters, samples):
    probe_ms = benchmark_once_batched(fn, warmup, iters)
    measured_iters = calibrate_iters(probe_ms, iters)

    timings = [benchmark_once_batched(fn, warmup, measured_iters)]
    for _ in range(1, samples):
        timings.append(benchmark_once_batched(fn, 0, measured_iters))
    return summarize_samples(timings, measured_iters)


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

    kernel_stats = benchmark_with_stats(
        lambda: bf16_gemm_out(a, b, out_kernel),
        args.warmup,
        args.iters,
        args.samples,
    )
    max_abs_diff = 0.0
    if not args.skip_check:
        bf16_gemm_out(a, b, out_kernel)
        if args.baseline == "torch":
            baseline_fn()
        else:
            cublaslt_gemm_out(a, b, out_baseline, workspace)
        max_abs_diff = (
            (out_kernel.float() - out_baseline.float()).abs().max().item()
        )

    return kernel_stats, baseline_stats, max_abs_diff, baseline_name


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
        f"{baseline_name + 'Speedup':<14}{'Diff' + baseline_name:<14}"
    )


def print_result(m, n, k, kernel_stats, baseline_stats, max_abs_diff):
    kernel_tflops = tflops_from_ms(m, n, k, kernel_stats["median_ms"])
    baseline_tflops = tflops_from_ms(m, n, k, baseline_stats["median_ms"])
    baseline_speedup = kernel_stats["median_ms"] / baseline_stats["median_ms"]
    print(
        f"{m:<8}{n:<8}{k:<8}"
        f"{kernel_stats['median_ms']:<16.4f}{kernel_tflops:<16.3f}"
        f"{baseline_stats['median_ms']:<16.4f}{baseline_tflops:<16.3f}"
        f"{baseline_speedup:<14.2f}{max_abs_diff:<14.6f}"
    )


def run_accuracy_check(args):
    print("[accuracy] small shape with correctness check")
    accuracy_args = argparse.Namespace(
        warmup=2,
        iters=20,
        samples=7,
        skip_check=False,
        baseline=args.baseline,
    )
    kernel_stats, baseline_stats, max_abs_diff, baseline_name = run_case(
        *ACCURACY_SHAPE, args=accuracy_args
    )
    print_header_for(baseline_name)
    print_result(*ACCURACY_SHAPE, kernel_stats, baseline_stats, max_abs_diff)
    print()


def run_profile(args):
    output_path = Path(args.profile_output)
    if output_path.suffixes[-2:] != [".json", ".gz"] and output_path.suffix != ".json":
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path / (
            f"bf16_gemm_profile"
            f"_m{args.m}_n{args.n}_k{args.k}"
            f"_blocks{args.profile_num_blocks}"
            f"_entries{args.profile_num_entries}"
            f"_{timestamp}.json.gz"
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    a = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(args.k, args.n, device="cuda", dtype=torch.bfloat16)
    profiler = make_profiler_buffer(
        args.profile_num_blocks, args.profile_num_entries, device=a.device
    )

    for _ in range(args.warmup):
        bf16_gemm_profile(a, b, profiler, args.profile_num_entries)
    torch.cuda.synchronize()

    profiler.zero_()
    bf16_gemm_profile(a, b, profiler, args.profile_num_entries)
    torch.cuda.synchronize()
    write_trace(profiler, output_path, tags=BF16_GEMM_PROFILER_TAGS)
    print(f"[profile] wrote {output_path}")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    args, single_shape = parse_args()
    torch.manual_seed(1234)
    if args.profile:
        run_profile(args)
        return

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
    print_header_for(baseline_name)
    if single_shape:
        kernel_stats, baseline_stats, max_abs_diff, _ = run_case(args.m, args.n, args.k, args)
        print_result(args.m, args.n, args.k, kernel_stats, baseline_stats, max_abs_diff)
    else:
        for shape in DEFAULT_SHAPES:
            kernel_stats, baseline_stats, max_abs_diff, _ = run_case(*shape, args)
            print_result(*shape, kernel_stats, baseline_stats, max_abs_diff)


if __name__ == "__main__":
    main()
