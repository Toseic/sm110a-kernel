"""Shared benchmarking utilities for sm110a-kernel benchmarks."""

import argparse
import math
import statistics

import torch


EARLY_STOP_RATIO = 1.3       # skip config if probe > best_probe * this
MMA_PEAK_TFLOPS = 250.0
GMEM_BW_GBPS = 273.0
L2_BW_GBPS = 1.281 * 1000.0
BF16_BYTES = 2
_DECIMAL_MB = 1000 ** 2
_DECIMAL_GFLOP = 1000 ** 3

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

    def add_cache_args(self, parser):
        """Register the --force-sweep and --cache-path flags."""
        parser.add_argument(
            "--force-sweep",
            action="store_true",
            help="ignore the cached best config and re-run the full autotune sweep",
        )

    def add_sol_arg(self, parser):
        parser.add_argument(
            "--sol",
            action="store_true",
            help="print analytical SOL bounds and measured SOL attainment",
        )
        parser.add_argument(
            "--cache-path",
            default=None,
            help=(
                "path to the autotune cache JSON "
                "(default: <repo>/profiles/autotune_cache.json)"
            ),
        )

    def add_timing_args(self, parser, *, warmup=20, iters=20, samples=7):
        parser.add_argument("--warmup", type=int, default=warmup)
        parser.add_argument("--iters", type=int, default=iters)
        parser.add_argument("--samples", type=int, default=samples)
        parser.add_argument(
            "--no-cold-l2",
            dest="cold_l2",
            action="store_false",
            default=True,
            help="do not evict L2 before each timed kernel invocation",
        )

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

    def configure_cold_l2(self, args):
        """Prepare an L2-sized eviction buffer for the timing helpers."""
        args._cold_l2_flush = None
        if not getattr(args, "cold_l2", True):
            print("[cache] cold L2: disabled")
            return

        l2_bytes = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).L2_cache_size
        if l2_bytes <= 0:
            print("[cache] cold L2: unavailable (device did not report L2 size)")
            return

        # Writing more than the cache capacity evicts data touched by the
        # preceding warmup. Keep the tensor alive for the benchmark lifetime.
        self._cold_l2_buffer = torch.empty(
            (2 * l2_bytes,), device="cuda", dtype=torch.uint8
        )

        def flush_l2():
            self._cold_l2_buffer.zero_()

        args._cold_l2_flush = flush_l2
        print(
            "[cache] cold L2: enabled "
            f"({2 * l2_bytes / 2**20:.0f} MiB eviction buffer)"
        )

    def run(self):
        args = self.parse_args()
        self.initialize()
        self.configure_cold_l2(args)
        self.run_with_args(args)

    def run_with_args(self, args):
        raise NotImplementedError

    def make_table_printer(self, columns):
        return TablePrinter(columns)

class GemmBenchmarkBase(BenchmarkBase):
    """Base class for benchmarks that sweep M/N/K GEMM shapes."""

    default_shapes = ()

    def baseline_choices(self):
        return ("torch", "cublaslt")

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
            choices=self.baseline_choices(),
            default="cublaslt",
        )
        self.add_cache_args(parser)
        self.add_sol_arg(parser)
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
        self._sol_enabled = getattr(args, "sol", False)
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


def _ms_for_bytes(byte_count, bandwidth_gbps):
    return byte_count / (bandwidth_gbps * _DECIMAL_MB)


def sol_bounds(
    flops,
    compulsory_bytes,
    a_and_b_bytes,
    tile_read_bytes,
    mma_peak_tflops=MMA_PEAK_TFLOPS,
):
    """Return the shared GEMM SOL models for the requested MMA peak."""
    mma_ms = flops / (mma_peak_tflops * _DECIMAL_GFLOP)
    hbm_ms = _ms_for_bytes(compulsory_bytes, GMEM_BW_GBPS)
    gmem_ms = _ms_for_bytes(tile_read_bytes, GMEM_BW_GBPS)
    pure_l2_ms = _ms_for_bytes(tile_read_bytes, L2_BW_GBPS)
    best_l2_ms = (
        _ms_for_bytes(a_and_b_bytes, GMEM_BW_GBPS)
        + _ms_for_bytes(tile_read_bytes - a_and_b_bytes, L2_BW_GBPS)
    )
    return {
        "mma_ms": mma_ms,
        "hbm_ms": hbm_ms,
        "gmem_ms": gmem_ms,
        "pure_l2_ms": pure_l2_ms,
        "best_l2_ms": best_l2_ms,
        "compulsory_bytes": compulsory_bytes,
        "tile_read_bytes": tile_read_bytes,
        # HBM misses and L2 hits can be serviced concurrently. Their sum is a
        # useful serialized-traffic estimate, but not a hard lower bound.
        # The roofline is the maximum demand on the independent resources.
        "sol_ms": max(mma_ms, hbm_ms, pure_l2_ms),
    }


def print_sol_report(bounds, measured_ms, *, label="SOL"):
    """Print a readable, multi-line bound breakdown below a benchmark result."""
    attainment = 100.0 * bounds["sol_ms"] / measured_ms
    print(f"[sol] {label}")
    print(f"      mma:          {bounds['mma_ms']:.4f} ms")
    print(f"      hbm:          {bounds['hbm_ms']:.4f} ms")
    print(f"      gmem:         {bounds['gmem_ms']:.4f} ms")
    print(f"      pure-l2:      {bounds['pure_l2_ms']:.4f} ms")
    print(f"      mixed-serial:  {bounds['best_l2_ms']:.4f} ms")
    print(f"      selected:     {bounds['sol_ms']:.4f} ms")
    print(f"      attainment:   {attainment:.1f}%")
    print(f"      tile-read:    {bounds['tile_read_bytes'] / _DECIMAL_MB:.1f} MB")


def summarize_samples(samples, measured_iters):
    samples = sorted(samples)
    return {
        "min_ms": samples[0],
        "median_ms": statistics.median(samples),
        "max_ms": samples[-1],
        "measured_iters": measured_iters,
    }


def benchmark_once(fn, warmup, iters, cold_l2_flush=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if cold_l2_flush is None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iters

    # The eviction is deliberately outside each event pair, so every measured
    # kernel invocation sees a cold L2 without charging eviction time to it.
    events = []
    for _ in range(iters):
        cold_l2_flush()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        events.append((start, end))
    events[-1][1].synchronize()
    return sum(start.elapsed_time(end) for start, end in events) / iters


def benchmark_once_batched(fn, warmup, iters, cold_l2_flush=None):
    for _ in range(warmup):
        fn(1)
    torch.cuda.synchronize()

    if cold_l2_flush is None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(iters)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iters

    # A batched native call cannot flush L2 between its internal launches, so
    # issue one launch at a time when cold-L2 measurement is enabled.
    events = []
    for _ in range(iters):
        cold_l2_flush()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(1)
        end.record()
        events.append((start, end))
    events[-1][1].synchronize()
    return sum(start.elapsed_time(end) for start, end in events) / iters


def benchmark_with_stats(fn, warmup, iters, samples, cold_l2_flush=None):
    """Time exactly ``iters`` kernel launches per sample."""
    timings = [benchmark_once(fn, warmup, iters, cold_l2_flush)]
    for _ in range(1, samples):
        timings.append(benchmark_once(fn, 0, iters, cold_l2_flush))
    return summarize_samples(timings, iters)


def benchmark_with_stats_batched(fn, warmup, iters, samples, cold_l2_flush=None):
    """Time exactly ``iters`` batched kernel launches per sample."""
    timings = [benchmark_once_batched(fn, warmup, iters, cold_l2_flush)]
    for _ in range(1, samples):
        timings.append(benchmark_once_batched(fn, 0, iters, cold_l2_flush))
    return summarize_samples(timings, iters)


def quick_probe(fn, iters, cold_l2_flush=None):
    """Probe using exactly ``iters`` launches for early-stop decisions."""
    return benchmark_once(fn, warmup=2, iters=iters, cold_l2_flush=cold_l2_flush)
