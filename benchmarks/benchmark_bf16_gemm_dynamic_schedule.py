#!/usr/bin/env python3

import argparse
from datetime import datetime
import sys
from pathlib import Path

import torch

from common import (
    GemmBenchmarkBase,
    benchmark_with_stats,
    benchmark_with_stats_batched,
    tflops_from_ms,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sm110a_kernel import (
    bf16_gemm_dynamic_schedule_out,
    bf16_gemm_dynamic_schedule_profile,
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
    (11008, 4096, 4096),
    (256, 32768, 4096),
]
ACCURACY_SHAPE = (512, 512, 512)
TARGET_SAMPLE_MS = 50.0
CUBLASLT_WORKSPACE_BYTES = 32 * 1024 * 1024
DEFAULT_PROFILE_NUM_ENTRIES = 100_000
DEFAULT_PROFILE_NUM_BLOCKS = 20
DEFAULT_PROFILE_OUTPUT_DIR = ROOT / "profiles"
BF16_GEMM_DYNAMIC_SCHEDULE_PROFILER_TAGS = (
    "SETUP",
    "ISSUE_TMA",
    "ISSUE_MMA",
    "WAIT_TMA",
    "WAIT_MMA",
    "WAIT_MAINLOOP",
    "WAIT_EPILOGUE",
    "EPILOGUE",
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
    else:
        raise SystemExit(f"unsupported --baseline value: {args.baseline}")

    kernel_stats = benchmark_with_stats(
        lambda: bf16_gemm_dynamic_schedule_out(a, b, out_kernel),
        args.warmup,
        args.iters,
        args.samples,
        cold_l2_flush=getattr(args, "_cold_l2_flush", None),
    )
    max_abs_diff = 0.0
    if not args.skip_check:
        bf16_gemm_dynamic_schedule_out(a, b, out_kernel)
        if args.baseline == "torch":
            baseline_fn()
        else:
            cublaslt_gemm_out(a, b, out_baseline, workspace)
        max_abs_diff = (
            (out_kernel.float() - out_baseline.float()).abs().max().item()
        )

    return kernel_stats, baseline_stats, max_abs_diff, baseline_name


def print_result(m, n, k, kernel_stats, baseline_stats, max_abs_diff):
    return (
        m,
        n,
        k,
        f"{kernel_stats['median_ms']:.4f}",
        f"{tflops_from_ms(m, n, k, kernel_stats['median_ms']):.3f}",
        f"{baseline_stats['median_ms']:.4f}",
        f"{tflops_from_ms(m, n, k, baseline_stats['median_ms']):.3f}",
        f"{kernel_stats['median_ms'] / baseline_stats['median_ms']:.2f}",
        f"{max_abs_diff:.6f}",
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
    bench = Bf16GemmDynamicScheduleBenchmark()
    kernel_stats, baseline_stats, max_abs_diff, baseline_name = bench.run_case(
        *ACCURACY_SHAPE, args=accuracy_args
    )
    table = bench.make_result_table(baseline_name)
    table.print_header()
    table.print_row(
        *print_result(*ACCURACY_SHAPE, kernel_stats, baseline_stats, max_abs_diff)
    )
    print()


def run_profile(args):
    output_path = Path(args.profile_output)
    if output_path.suffixes[-2:] != [".json", ".gz"] and output_path.suffix != ".json":
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path / (
            f"bf16_gemm_dynamic_schedule_profile"
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
        bf16_gemm_dynamic_schedule_profile(a, b, profiler, args.profile_num_entries)
    torch.cuda.synchronize()

    profiler.zero_()
    bf16_gemm_dynamic_schedule_profile(a, b, profiler, args.profile_num_entries)
    torch.cuda.synchronize()
    write_trace(profiler, output_path, tags=BF16_GEMM_DYNAMIC_SCHEDULE_PROFILER_TAGS)
    print(f"[profile] wrote {output_path}")


class Bf16GemmDynamicScheduleBenchmark(GemmBenchmarkBase):
    default_shapes = DEFAULT_SHAPES

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
                (f"Diff{baseline_name}", 14),
            )
        )

    def add_arguments(self, parser):
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--profile-output",
            default=str(DEFAULT_PROFILE_OUTPUT_DIR),
        )
        parser.add_argument(
            "--profile-num-blocks",
            type=int,
            default=DEFAULT_PROFILE_NUM_BLOCKS,
        )
        parser.add_argument(
            "--profile-num-entries",
            type=int,
            default=DEFAULT_PROFILE_NUM_ENTRIES,
        )

    def validate_extra_args(self, args):
        if args.profile and not args.single_shape:
            raise SystemExit("--profile requires --m, --n, --k")
        if args.profile_num_blocks <= 0:
            raise SystemExit("--profile-num-blocks must be positive")
        if args.profile_num_entries <= 0:
            raise SystemExit("--profile-num-entries must be positive")

    def run_with_args(self, args):
        if args.profile:
            run_profile(args)
            return
        super().run_with_args(args)

    def run_preflight(self, args):
        if not args.single_shape and not args.no_accuracy:
            run_accuracy_check(args)

    def print_header(self, baseline_name):
        self.make_result_table(baseline_name).print_header()

    def run_case(self, m, n, k, args):
        return run_case(m, n, k, args)

    def print_result(self, shape, result):
        m, n, k = shape
        kernel_stats, baseline_stats, max_abs_diff, baseline_name = result
        self.make_result_table(baseline_name).print_row(
            *print_result(m, n, k, kernel_stats, baseline_stats, max_abs_diff)
        )


def main():
    Bf16GemmDynamicScheduleBenchmark().run()


if __name__ == "__main__":
    main()
