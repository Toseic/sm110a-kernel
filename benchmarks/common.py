"""Shared benchmarking utilities for sm110a-kernel benchmarks."""

import argparse
import math
from pathlib import Path
import subprocess
import statistics
import sys

import torch


EARLY_STOP_RATIO = 1.3       # skip config if probe > best_probe * this
MMA_PEAK_TFLOPS = 250.0
# ``B300Sim`` changes this for the duration of one benchmark process.  Keep
# the bandwidth constants unchanged: its purpose is to vary compute/GMEM
# balance while EMC is explicitly pinned at maximum frequency.
MMA_PEAK_SCALE = 1.0
GMEM_BW_GBPS = 273.0
L2_BW_GBPS = 1.281 * 1000.0
BF16_BYTES = 2
_DECIMAL_MB = 1000 ** 2
_DECIMAL_GFLOP = 1000 ** 3


def powerlaw_router_scores(tokens, experts, alpha, *, device, dtype):
    """Sample router logits whose selected-expert load follows rank^-alpha.

    Gumbel-top-k produces weighted samples without replacement for every
    token.  Adding ``log(rank^-alpha)`` to the Gumbel noise therefore gives
    each expert a power-law routing prior while preserving valid, distinct
    top-k expert selections.
    """
    if tokens <= 0 or experts <= 0:
        raise ValueError("tokens and experts must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    ranks = torch.arange(1, experts + 1, device=device, dtype=torch.float32)
    log_weights = -alpha * ranks.log()
    # If x ~ Exp(1), -log(x) has the standard Gumbel distribution.
    gumbel = -torch.empty(
        (tokens, experts), device=device, dtype=torch.float32
    ).exponential_().log_()
    return (gumbel + log_weights).to(dtype)


class B300Sim:
    """Temporarily lock Thor clocks, optionally at B300 balance.

    Default mode locks GPC and EMC to their currently permitted maxima.  With
    ``--b300-sim``, it instead lowers GPC to the closest B300-balance point.
    The latter is a roofline-balance experiment, not a performance emulator.
    Original devfreq limits are restored on normal exit and Python exceptions.
    """

    GPU_PATH = Path("/sys/class/devfreq/gpu-gpc-0")
    EMC_PATH = Path("/sys/class/devfreq/bwmgr")
    TARGET_HZ = {"bf16": 466_000_000, "nvfp4": 700_000_000}

    def __init__(self, args):
        self.args = args
        self.b300_sim = getattr(args, "b300_sim", False)
        self.enabled = not getattr(args, "no_lock_clocks", False)
        requested_mode = getattr(args, "b300_sim_mode", "auto")
        self.mode = self._resolve_mode(requested_mode)
        self._original = {}
        self._applied = False
        self._previous_mma_peak_scale = None

    @staticmethod
    def _resolve_mode(requested_mode):
        if requested_mode != "auto":
            return requested_mode
        # All current NVFP4 entry points include this in their filename.  Keep
        # an explicit override for future mixed-precision benchmarks.
        benchmark_name = Path(sys.argv[0]).stem.lower()
        return "nvfp4" if ("nvfp4" in benchmark_name or "mxfp4" in benchmark_name) else "bf16"

    @staticmethod
    def _read(path):
        return int(path.read_text().strip())

    @staticmethod
    def _frequencies(path):
        return tuple(int(value) for value in (path / "available_frequencies").read_text().split())

    def _write(self, path, value):
        """Write a sysfs value, escalating only for that single write."""
        try:
            path.write_text(f"{value}\n")
            return
        except PermissionError:
            pass

        # First support a least-privilege sudoers rule which permits only
        # ``tee`` for these exact devfreq files.  Do not probe with ``sudo
        # -v`` first: that probe itself is not covered by such a rule.
        command = ("sudo", "-n", "tee", str(path))
        result = subprocess.run(
            command,
            input=f"{value}\n",
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return

        # No NOPASSWD rule is available. Authenticate once, then retry every
        # individual write non-interactively so restoration cannot prompt.
        if not getattr(self, "_sudo_validated", False):
            try:
                subprocess.run(("sudo", "-v"), check=True)
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                raise RuntimeError(
                    "--b300-sim needs permission to write devfreq sysfs nodes; "
                    "run from a terminal that can authenticate with sudo"
                ) from exc
            self._sudo_validated = True
        try:
            subprocess.run(
                command,
                input=f"{value}\n",
                text=True,
                stdout=subprocess.DEVNULL,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"unable to write {path} for --b300-sim") from exc

    def __enter__(self):
        if not self.enabled:
            return self
        required = (self.GPU_PATH, self.EMC_PATH)
        missing = [str(path) for path in required if not path.is_dir()]
        if missing:
            raise RuntimeError(
                "--b300-sim is supported only on Thor devfreq systems; missing "
                + ", ".join(missing)
            )

        self._original = {
            (self.GPU_PATH, "min_freq"): self._read(self.GPU_PATH / "min_freq"),
            (self.GPU_PATH, "max_freq"): self._read(self.GPU_PATH / "max_freq"),
            (self.EMC_PATH, "min_freq"): self._read(self.EMC_PATH / "min_freq"),
            (self.EMC_PATH, "max_freq"): self._read(self.EMC_PATH / "max_freq"),
        }
        if self.b300_sim:
            gpu_target = min(
                self._frequencies(self.GPU_PATH),
                key=lambda value: abs(value - self.TARGET_HZ[self.mode]),
            )
        else:
            gpu_target = self._original[(self.GPU_PATH, "max_freq")]
        emc_target = self._original[(self.EMC_PATH, "max_freq")]
        try:
            # Set the lower limit before the upper limit: this works whether
            # the previous GPC range is above or below the requested point.
            self._write(self.GPU_PATH / "min_freq", gpu_target)
            self._write(self.GPU_PATH / "max_freq", gpu_target)
            self._write(self.EMC_PATH / "min_freq", emc_target)
            self._write(self.EMC_PATH / "max_freq", emc_target)
            self._applied = True
        except Exception:
            self._restore()
            raise

        global MMA_PEAK_SCALE
        self._previous_mma_peak_scale = MMA_PEAK_SCALE
        MMA_PEAK_SCALE = gpu_target / self._original[(self.GPU_PATH, "max_freq")]
        self.args._sol_mma_peak_scale = MMA_PEAK_SCALE
        precision_multiplier = 4.0 if self.mode == "nvfp4" else 1.0
        precision_name = "NVFP4" if self.mode == "nvfp4" else "BF16"
        self.args._sol_mma_peak_tflops = (
            precision_multiplier * MMA_PEAK_TFLOPS * MMA_PEAK_SCALE
        )
        if self.b300_sim:
            print(
                f"[b300-sim] mode={self.mode} GPC={gpu_target / 1e6:.0f} MHz "
                f"(target {self.TARGET_HZ[self.mode] / 1e6:.0f} MHz), "
                f"EMC={emc_target / 1e6:.0f} MHz locked"
            )
            print(
                f"[b300-sim] SOL MMA peak scale={MMA_PEAK_SCALE:.4f}; "
                f"{precision_name} peak={self.args._sol_mma_peak_tflops:.3f} TF/s"
            )
        else:
            print(
                f"[clocks] GPC={gpu_target / 1e6:.0f} MHz and "
                f"EMC={emc_target / 1e6:.0f} MHz locked at maximum"
            )
        return self

    def _restore(self):
        if not self._original:
            return
        errors = []
        # Raise maxima before minima so restoring a wider range is valid.
        for path, name in ((self.GPU_PATH, "max_freq"), (self.EMC_PATH, "max_freq"),
                           (self.GPU_PATH, "min_freq"), (self.EMC_PATH, "min_freq")):
            try:
                self._write(path / name, self._original[(path, name)])
            except Exception as exc:  # Preserve the benchmark exception.
                errors.append(exc)
        self._applied = False
        if self._previous_mma_peak_scale is not None:
            global MMA_PEAK_SCALE
            MMA_PEAK_SCALE = self._previous_mma_peak_scale
            self._previous_mma_peak_scale = None
        if errors:
            raise RuntimeError("failed to restore devfreq limits after --b300-sim") from errors[0]

    def __exit__(self, exc_type, exc_value, traceback):
        if self._applied:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                # A failed CUDA operation must not prevent clock restoration.
                print("[b300-sim] WARNING: CUDA synchronize failed before restoration", file=sys.stderr)
            try:
                self._restore()
                print("[clocks] restored original GPC and EMC limits")
            except Exception:
                if exc_type is None:
                    raise
                print("[clocks] WARNING: failed to restore original devfreq limits", file=sys.stderr)
        return False

class TablePrinter:
    """Print benchmark records as tables, or as lines in debug mode."""

    def __init__(self, columns, *, debug=False):
        self.columns = tuple(columns)
        self.debug = debug

    def print_header(self):
        if self.debug:
            return
        print("".join(f"{title:<{width}}" for title, width in self.columns))

    def print_row(self, *values):
        if len(values) != len(self.columns):
            raise ValueError(
                f"expected {len(self.columns)} values, got {len(values)}"
            )
        if self.debug:
            print(" ".join(
                f"{title}={value}"
                for value, (title, _) in zip(values, self.columns)
            ))
        else:
            print("".join(
                f"{value:<{width}}"
                for value, (_, width) in zip(values, self.columns)
            ))


class BenchmarkBase:
    """Common CLI and runtime helpers for benchmark entry points."""

    seed = 1234
    def build_parser(self):
        return argparse.ArgumentParser()

    def parse_args(self):
        parser = self.build_parser()
        self.add_b300_sim_args(parser)
        args = parser.parse_args()
        self.validate_args(args)
        return args

    @staticmethod
    def add_b300_sim_args(parser):
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--b300-sim",
            action="store_true",
            help=(
                "lock Thor EMC at maximum and lower GPC to emulate B300's "
                "compute/GMEM roofline balance (requires sudo)"
            ),
        )
        group.add_argument(
            "--no-lock-clocks",
            action="store_true",
            help="leave GPC and EMC devfreq limits unchanged",
        )
        parser.add_argument(
            "--b300-sim-mode",
            choices=("auto", "bf16", "nvfp4"),
            default="auto",
            help=(
                "precision balance for --b300-sim; auto selects NVFP4 for "
                "NVFP4 benchmark filenames and BF16 otherwise"
            ),
        )

    def validate_args(self, args):
        return args

    def add_cache_args(self, parser):
        """Register cache controls shared by local and baseline autotuners."""
        parser.add_argument(
            "--force-sweep",
            action="store_true",
            help=(
                "ignore cached configs and re-run both the local-kernel and "
                "baseline autotune sweeps"
            ),
        )
        parser.add_argument(
            "--force-sweep-ours",
            action="store_true",
            help="ignore the cached local-kernel config and re-run its autotune sweep",
        )
        parser.add_argument(
            "--force-sweep-baseline",
            action="store_true",
            help="ignore the cached baseline config and re-run its autotune sweep",
        )

    @staticmethod
    def _b300_sim_cache_path(args):
        """Keep throttled autotune choices separate from normal-clock ones."""
        configured = getattr(args, "cache_path", None)
        base = (
            Path(configured)
            if configured
            else Path(__file__).resolve().parent.parent / "profiles" / "autotune_cache.json"
        )
        mode = B300Sim._resolve_mode(getattr(args, "b300_sim_mode", "auto"))
        suffix = base.suffix or ".json"
        return base.with_name(f"{base.stem}.b300-sim-{mode}{suffix}")

    @staticmethod
    def make_tune_options(args, *, role="ours", **overrides):
        """Build runner options for one side of a benchmark comparison.

        ``role`` keeps cache-bypass policy consistent across benchmarks.  The
        legacy ``--force-sweep`` remains a shorthand for forcing both sides.
        """
        if role not in {"ours", "baseline"}:
            raise ValueError(f"unsupported autotune role: {role}")
        from sm110a_kernel.runner import TuneOptions

        cache_path = getattr(args, "cache_path", None)
        if getattr(args, "b300_sim", False):
            cache_path = BenchmarkBase._b300_sim_cache_path(args)
        values = dict(
            warmup=args.warmup,
            iters=args.iters,
            samples=args.samples,
            cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            cache_path=cache_path,
            force_sweep=(
                getattr(args, "force_sweep", False)
                or getattr(args, f"force_sweep_{role}", False)
            ),
            debug=getattr(args, "debug", False),
        )
        values.update(overrides)
        return TuneOptions(**values)

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

    def add_pre_parallel_jit_arg(self, parser):
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--pre-parallel-jit",
            action="store_true",
            dest="pre_parallel_jit",
            default=True,
            help="compile all compatible JIT configurations in parallel before benchmarking",
        )
        group.add_argument(
            "--no-pre-parallel-jit",
            action="store_false",
            dest="pre_parallel_jit",
            help="disable parallel JIT precompilation",
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
        parser.add_argument(
            "--clean-l2",
            dest="clean_l2",
            action="store_true",
            default=False,
            help="flush L2 with reads instead of writes, leaving the "
            "eviction-buffer lines clean (no writeback tax on the timed "
            "kernel; default flush dirties L2)",
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
        # preceding warmup and leaves the resident lines dirty, so the timed
        # kernel pays writebacks for every line it allocates. --clean-l2
        # flushes with a read instead: lines stay clean and evictions are
        # free, which approximates a cold cache without the writeback tax.
        # Keep the tensor alive for the benchmark lifetime.
        self._cold_l2_buffer = torch.empty(
            (2 * l2_bytes,), device="cuda", dtype=torch.uint8
        )

        clean_l2 = getattr(args, "clean_l2", False)

        def flush_l2():
            if clean_l2:
                self._cold_l2_buffer.max()
            else:
                self._cold_l2_buffer.zero_()

        args._cold_l2_flush = flush_l2
        mode = "clean read-flush" if clean_l2 else "dirty write-flush"
        print(
            f"[cache] cold L2: enabled ({mode}, "
            f"{2 * l2_bytes / 2**20:.0f} MiB eviction buffer)"
        )

    def run(self):
        args = self.parse_args()
        self._debug_enabled = getattr(args, "debug", False)
        with B300Sim(args):
            self.initialize()
            self.configure_cold_l2(args)
            self.run_with_args(args)

    def run_with_args(self, args):
        raise NotImplementedError

    def make_table_printer(self, columns, *, debug=False):
        return TablePrinter(columns, debug=debug or getattr(self, "_debug_enabled", False))


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
        self.add_pre_parallel_jit_arg(parser)
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
        self.finish_results()

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

    def finish_results(self):
        """Emit any deferred sweep output after all result rows."""
        return None


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
    mma_ms = flops / (mma_peak_tflops * MMA_PEAK_SCALE * _DECIMAL_GFLOP)
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


def sol_table_columns(label_width=22):
    return (
        ("SOL", label_width), ("SOL mma(ms)", 12), ("SOL hbm(ms)", 12),
        ("SOL gmem(ms)", 14), ("SOL pure-l2(ms)", 16),
        ("SOL mixed(ms)", 14), ("SOL selected(ms)", 17),
        ("SOL attainment", 16), ("SOL tile-read(MB)", 18),
    )


def sol_table_values(bounds, measured_ms, *, label="SOL"):
    attainment = 100.0 * bounds["sol_ms"] / measured_ms
    return (
        label, f"{bounds['mma_ms']:.4f}", f"{bounds['hbm_ms']:.4f}",
        f"{bounds['gmem_ms']:.4f}", f"{bounds['pure_l2_ms']:.4f}",
        f"{bounds['best_l2_ms']:.4f}", f"{bounds['sol_ms']:.4f}",
        f"{attainment:.1f}%", f"{bounds['tile_read_bytes'] / _DECIMAL_MB:.1f}",
    )


def print_sol_report(bounds, measured_ms, *, label="SOL", debug=False):
    """Print one SOL record; prefer :class:`SolReport` for sweeps."""
    report = SolReport(debug=debug)
    report.add(bounds, measured_ms, label=label)
    report.print()


class SolReport:
    """Accumulate a benchmark sweep's SOL rows and render one stable table."""

    def __init__(self, *, debug=False):
        self.debug = debug
        self._rows = []

    def add(self, bounds, measured_ms, *, label="SOL"):
        self._rows.append((label, bounds, measured_ms))

    def print(self):
        if not self._rows:
            return
        label_width = max(22, *(len(label) + 1 for label, _, _ in self._rows))
        table = TablePrinter(sol_table_columns(label_width), debug=self.debug)
        table.print_header()
        for label, bounds, measured_ms in self._rows:
            table.print_row(*sol_table_values(bounds, measured_ms, label=label))


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
