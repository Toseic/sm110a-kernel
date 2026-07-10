"""Shared benchmarking utilities for sm110a-kernel benchmarks."""

import argparse
import math
import statistics

import torch


TARGET_SAMPLE_MS = 50.0
PROBE_TARGET_MS = 10.0       # quick probe target — small enough to be fast
EARLY_STOP_RATIO = 1.3       # skip config if probe > best_probe * this


class TablePrinter:
    """Simple fixed-width table printer shared by benchmark scripts."""

    def __init__(self, columns):
        self.columns = tuple(columns)

    def print_header(self):
        print("".join(f"{title:<{width}}" for title, width in self.columns))

    def print_row(self, *values):
        if len(values) != len(self.columns):
            raise ValueError(
                f"expected {len(self.columns)} values, got {len(values)}"
            )
        print(
            "".join(
                f"{value:<{width}}"
                for value, (_, width) in zip(values, self.columns)
            )
        )


class BenchmarkBase:
    """Common CLI and runtime helpers for benchmark entry points."""

    seed = 1234

    def build_parser(self):
        return argparse.ArgumentParser()

    def parse_args(self):
        args = self.build_parser().parse_args()
        self.validate_args(args)
        return args

    def validate_args(self, args):
        return args

    def add_timing_args(self, parser, *, warmup=20, iters=20, samples=7):
        parser.add_argument("--warmup", type=int, default=warmup)
        parser.add_argument("--iters", type=int, default=iters)
        parser.add_argument("--samples", type=int, default=samples)

    def validate_timing_args(self, args, *, allow_zero_warmup=False):
        if allow_zero_warmup:
            if args.warmup < 0:
                raise SystemExit("--warmup must be non-negative")
        elif args.warmup <= 0:
            raise SystemExit("--warmup must be positive")
        if args.iters <= 0:
            raise SystemExit("--iters must be positive")
        if args.samples <= 0:
            raise SystemExit("--samples must be positive")

    def add_torch_blas_arg(self, parser, *, default="default"):
        parser.add_argument(
            "--torch-blas",
            choices=("default", "cublas", "cublaslt"),
            default=default,
        )

    def configure_torch_blas(self, preferred_blas):
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

    def initialize(self):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required")
        torch.manual_seed(self.seed)

    def run(self):
        args = self.parse_args()
        self.initialize()
        self.run_with_args(args)

    def run_with_args(self, args):
        raise NotImplementedError

    def make_table_printer(self, columns):
        return TablePrinter(columns)

    def autotune_candidates(
        self,
        candidates,
        *,
        args,
        benchmark_candidate,
        format_candidate,
        probe_candidate=None,
        metric_formatter=None,
        candidate_width=28,
        flush=False,
    ):
        candidates = tuple(candidates)
        if not candidates:
            raise RuntimeError("no compatible candidates")

        best_candidate = None
        best_stats = None
        best_probe_ms = None
        results = []
        debug = getattr(args, "debug", False)

        for candidate in candidates:
            probe_ms = None
            if probe_candidate is not None:
                probe_ms = probe_candidate(candidate)
                if (
                    best_probe_ms is not None
                    and probe_ms > best_probe_ms * EARLY_STOP_RATIO
                ):
                    if debug:
                        print(
                            f"  {format_candidate(candidate):<{candidate_width}}"
                            f" probe={probe_ms:.4f}ms"
                            f" - skip (>{best_probe_ms * EARLY_STOP_RATIO:.4f})",
                            flush=flush,
                        )
                    continue

            stats = benchmark_candidate(candidate)
            results.append((candidate, stats, probe_ms))

            is_best = (
                best_stats is None
                or stats["median_ms"] < best_stats["median_ms"]
            )
            marker = " *" if is_best else ""
            if is_best:
                best_candidate = candidate
                best_stats = stats
                if probe_ms is not None:
                    best_probe_ms = probe_ms

            if debug:
                metric_suffix = ""
                if metric_formatter is not None:
                    metric_suffix = f" {metric_formatter(candidate, stats)}"
                print(
                    f"  {format_candidate(candidate):<{candidate_width}}"
                    f" min={stats['min_ms']:.4f}ms"
                    f" median={stats['median_ms']:.4f}ms"
                    f"{metric_suffix}{marker}",
                    flush=flush,
                )

        if debug and best_candidate is not None:
            print(
                f"  -> best: {format_candidate(best_candidate)}",
                flush=flush,
            )
        return best_candidate, best_stats, results


class GemmBenchmarkBase(BenchmarkBase):
    """Base class for benchmarks that sweep M/N/K GEMM shapes."""

    default_shapes = ()

    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--m", type=int, default=0)
        parser.add_argument("--n", type=int, default=0)
        parser.add_argument("--k", type=int, default=0)
        self.add_timing_args(parser)
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--skip-check", action="store_true")
        self.add_torch_blas_arg(parser)
        parser.add_argument(
            "--baseline",
            choices=("torch", "cublaslt"),
            default="cublaslt",
        )
        self.add_arguments(parser)
        return parser

    def add_arguments(self, parser):
        return None

    def validate_args(self, args):
        single_shape = args.m > 0 or args.n > 0 or args.k > 0
        if single_shape and not (args.m > 0 and args.n > 0 and args.k > 0):
            raise SystemExit("--m, --n, --k must all be provided together")
        args.single_shape = single_shape
        self.validate_timing_args(args, allow_zero_warmup=True)
        self.validate_extra_args(args)
        return args

    def validate_extra_args(self, args):
        return args

    def iter_shapes(self, args):
        if args.single_shape:
            yield (args.m, args.n, args.k)
            return
        yield from self.default_shapes

    def run_with_args(self, args):
        actual_blas = None
        if args.baseline == "torch":
            actual_blas = self.configure_torch_blas(args.torch_blas)

        self.run_preflight(args)
        baseline_name = self.emit_runtime_banner(args, actual_blas)
        self.print_header(baseline_name)
        for shape in self.iter_shapes(args):
            result = self.run_case(*shape, args=args)
            self.print_result(shape, result)

    def run_preflight(self, args):
        return None

    def emit_runtime_banner(self, args, actual_blas):
        if args.baseline == "torch":
            print(f"[torch] preferred BLAS backend: {actual_blas}")
            return "Torch"
        print("[baseline] direct cuBLASLt binding")
        return "Lt"

    def print_header(self, baseline_name):
        raise NotImplementedError

    def run_case(self, m, n, k, args):
        raise NotImplementedError

    def print_result(self, shape, result):
        raise NotImplementedError


def tflops_from_ms(m, n, k, elapsed_ms):
    flops = 2.0 * m * n * k
    return flops / (elapsed_ms * 1.0e-3) / 1.0e12


def calibrate_iters(initial_ms, requested_iters, target_ms=TARGET_SAMPLE_MS):
    if initial_ms <= 0.0:
        return requested_iters
    total_ms = initial_ms * requested_iters
    if total_ms >= target_ms:
        return requested_iters
    scale = target_ms / total_ms
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


def quick_probe(fn):
    """Lightweight probe (~10 ms GPU time) for early-stop decisions."""
    ms = benchmark_once(fn, warmup=2, iters=5)
    iters = calibrate_iters(ms, 5, target_ms=PROBE_TARGET_MS)
    return benchmark_once(fn, warmup=0, iters=iters)
