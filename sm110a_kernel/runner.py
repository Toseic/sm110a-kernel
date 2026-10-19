"""Config selection and execution helpers for SM110A kernels.

Benchmarks use this module to resolve a kernel once, then time the returned
``PreparedKernel.run`` callable.  Consequently cache lookups and autotuning
never appear in the timed path.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Generic, TypeVar

import torch

T = TypeVar("T")
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE_PATH = _ROOT / "profiles" / "autotune_cache.json"
_SCHEMA_VERSION = 2
_EARLY_STOP_RATIO = 1.3


@dataclass(frozen=True)
class TuneOptions:
    warmup: int = 2
    iters: int = 3
    samples: int = 3
    cold_l2_flush: Callable[[], None] | None = None
    cache_path: Path | str | None = None
    force_sweep: bool = False
    debug: bool = False
    pre_parallel_jit: bool = False
    cutedsl_sweep_scheduler: bool = False
    cutedsl_nvmmh: bool = False
    cutedsl_nvmmh_fields: str = "tile,cluster"
    cutedsl_nvmmh_max_tactics: int = 5


@dataclass
class PreparedKernel(Generic[T]):
    candidate: T
    run: Callable[[], object]


@dataclass
class Selection(Generic[T]):
    candidate: T
    prepared: PreparedKernel[T]
    stats: dict
    cache_hit: bool


def _cache_key(key):
    return json.dumps(list(key) if isinstance(key, tuple) else key, sort_keys=True)


def _candidate_class(candidate):
    return type(candidate[0]) if isinstance(candidate, tuple) else type(candidate)


def _serialize(candidate, stats):
    if (
        isinstance(candidate, tuple)
        and len(candidate) == 2
        and dataclasses.is_dataclass(candidate[0])
    ):
        value = {
            "config": dataclasses.asdict(candidate[0]),
            "swap_ab": bool(candidate[1]),
        }
    elif dataclasses.is_dataclass(candidate):
        value = {"config": dataclasses.asdict(candidate)}
    else:
        raise TypeError(f"unsupported candidate type: {type(candidate)!r}")
    value.update(median_ms=float(stats["median_ms"]), min_ms=float(stats["min_ms"]))
    return value


def _deserialize(entry, config_cls):
    config = config_cls(**entry["config"])
    return (config, entry["swap_ab"]) if "swap_ab" in entry else config


def environment_fingerprint():
    """Return a stable cache partition for this binary and CUDA environment."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    payload = {
        "schema": _SCHEMA_VERSION,
        "device": str(getattr(props, "uuid", None) or props.name),
        "capability": [props.major, props.minor],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_arch": os.environ.get("CUDA_ARCH", "sm_110a"),
        "compute_arch": os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a"),
        "revision": revision,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:20]


class AutotuneRunner:
    """Resolve and cache the fastest prepared kernel for one workload."""

    def __init__(self, options: TuneOptions):
        self.options = options
        self.path = (
            Path(options.cache_path) if options.cache_path else _DEFAULT_CACHE_PATH
        )
        self.fingerprint = environment_fingerprint()

    def _load(self):
        try:
            value = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {"schema_version": _SCHEMA_VERSION, "environments": {}}
        if value.get("schema_version") != _SCHEMA_VERSION or not isinstance(
            value.get("environments"), dict
        ):
            return {"schema_version": _SCHEMA_VERSION, "environments": {}}
        return value

    def _store(self, namespace, key, candidate, stats):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+") as lock_file:
            with contextlib.suppress(ImportError):
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            cache = self._load()
            env = cache.setdefault("environments", {}).setdefault(self.fingerprint, {})
            env.setdefault(namespace, {})[_cache_key(key)] = _serialize(
                candidate, stats
            )
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(cache, indent=2, sort_keys=True))
            temporary.replace(self.path)
            with contextlib.suppress(ImportError):
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def resolve(
        self,
        candidates: Sequence[T],
        *,
        namespace: str,
        key,
        prepare: Callable[[T], PreparedKernel[T]],
        benchmark: Callable[[PreparedKernel[T]], dict],
        probe: Callable[[PreparedKernel[T]], float] | None,
        format_candidate: Callable[[T], str],
        metric_formatter: Callable[[T, dict], str] | None = None,
        pre_jit: Callable[[T], object] | None = None,
    ) -> Selection[T]:
        candidates = tuple(candidates)
        if not candidates:
            raise RuntimeError("no compatible candidates")
        config_cls = _candidate_class(candidates[0])
        key_string = _cache_key(key)
        candidate_width = max(
            len(format_candidate(candidate)) for candidate in candidates
        )

        def print_header():
            print(
                f"  {'Config':<{candidate_width}}  {'Status':<12}"
                f"{'Time (ms)':>12}  Metric  Best",
                flush=True,
            )

        def print_row(candidate, status, elapsed_ms, metric="", best=False):
            print(
                f"  {format_candidate(candidate):<{candidate_width}}  {status:<12}"
                f"{elapsed_ms:>12.4f}  {metric:<12}  {'*' if best else ''}",
                flush=True,
            )

        if self.options.pre_parallel_jit:
            if pre_jit is None:
                raise ValueError("pre_parallel_jit requires a pre_jit callback")
            workers = min(
                len(candidates),
                int(os.environ.get("SM110A_JIT_MAX_WORKERS", "10")),
            )
            if workers <= 0:
                raise ValueError("SM110A_JIT_MAX_WORKERS must be positive")
            if self.options.debug:
                print(
                    f"[jit] pre-parallel {namespace}: "
                    f"{len(candidates)} configs, {workers} workers",
                    flush=True,
                )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(pre_jit, candidate) for candidate in candidates
                ]
                if self.options.debug:
                    from tqdm import tqdm

                    with tqdm(
                        total=len(futures),
                        desc=f"pre-parallel {namespace}",
                        unit="config",
                    ) as progress:
                        for future in as_completed(futures):
                            future.result()
                            progress.update(1)
                else:
                    for future in futures:
                        future.result()
            if self.options.debug:
                print(
                    f"[jit] pre-parallel {namespace} complete: "
                    f"{len(candidates)} configs",
                    flush=True,
                )

        if not self.options.force_sweep:
            entry = (
                self._load()
                .get("environments", {})
                .get(self.fingerprint, {})
                .get(namespace, {})
                .get(key_string)
            )
            if entry is not None:
                try:
                    candidate = _deserialize(entry, config_cls)
                    if candidate not in candidates:
                        candidate = None
                    prepared = prepare(candidate) if candidate is not None else None
                except (TypeError, ValueError, KeyError):
                    candidate = None
                if candidate is not None:
                    stats = benchmark(prepared)
                    if self.options.debug:
                        print(
                            f"  [runner cache] hit {namespace}{key} -> {format_candidate(candidate)}",
                            flush=True,
                        )
                        print_header()
                        metric = (
                            metric_formatter(candidate, stats)
                            if metric_formatter
                            else ""
                        )
                        print_row(
                            candidate,
                            "cache hit",
                            stats["median_ms"],
                            metric,
                            best=True,
                        )
                    return Selection(candidate, prepared, stats, True)

        best_candidate = best_prepared = best_stats = None
        best_probe = None
        if self.options.debug:
            print_header()
        for candidate in candidates:
            prepared = prepare(candidate)
            probe_ms = probe(prepared) if probe else None
            if best_probe is not None and probe_ms > best_probe * _EARLY_STOP_RATIO:
                if self.options.debug:
                    print_row(candidate, "probe skip", probe_ms, "-")
                continue
            stats = benchmark(prepared)
            is_best = best_stats is None or stats["median_ms"] < best_stats["median_ms"]
            if is_best:
                best_candidate, best_prepared, best_stats, best_probe = (
                    candidate,
                    prepared,
                    stats,
                    probe_ms,
                )
            if self.options.debug:
                metric = metric_formatter(candidate, stats) if metric_formatter else ""
                print_row(
                    candidate, "measured", stats["median_ms"], metric, best=is_best
                )
        if best_candidate is None:
            raise RuntimeError("all compatible candidates were skipped")
        self._store(namespace, key, best_candidate, best_stats)
        return Selection(best_candidate, best_prepared, best_stats, False)


# Hold configurations, never workload tensors or graph allocations.
_SELECTIONS = {}


def clear_kernel_runner_cache():
    """Forget process-local selections without deleting the persistent cache."""
    _SELECTIONS.clear()


def measure_stats(run, options):
    for _ in range(options.warmup):
        run()
    timings = []
    for _ in range(options.samples):
        events = []
        for _ in range(options.iters):
            if options.cold_l2_flush is not None:
                options.cold_l2_flush()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            events.append((start, end))
        events[-1][1].synchronize()
        timings.append(sum(a.elapsed_time(b) for a, b in events) / options.iters)
    return {
        "min_ms": min(timings),
        "median_ms": median(timings),
        "max_ms": max(timings),
    }


class BaseKernelRunner(ABC):
    """Bind a workload, tune once, then launch its prepared callable.

    Subclasses own candidates, cache keys, compilation and argument binding.
    Benchmarks may supply their timing/report callbacks to ``tune``. ``run``
    automatically tunes on its first eager invocation; warm up every intended
    workload before CUDA Graph capture. New instances share configurations,
    while always binding their own current inputs and outputs.
    """

    namespace: str
    graph_timing = False

    def __init__(self, device):
        self.device = torch.device(device)
        self.selection = None
        self._prepared = None

    @abstractmethod
    def compatible_candidates(self): ...

    @property
    @abstractmethod
    def workload_key(self): ...

    @abstractmethod
    def prepare(self, candidate) -> PreparedKernel: ...

    @abstractmethod
    def pre_jit(self, candidate): ...

    @staticmethod
    def format_candidate(candidate):
        return str(candidate)

    def _benchmark(self, prepared, options):
        if not self.graph_timing:
            return measure_stats(prepared.run, options)
        current = torch.cuda.current_stream()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(max(2, options.warmup)):
                prepared.run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                prepared.run()
            stats = measure_stats(graph.replay, options)
        current.wait_stream(stream)
        return stats

    def tune(
        self,
        options=None,
        *,
        candidates=None,
        benchmark=None,
        probe=None,
        metric_formatter=None,
    ):
        """Return a Selection, honoring benchmark timing/cache/JIT options."""
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"warm up {self.namespace} eagerly before CUDA graph capture"
                )
            options = options or TuneOptions()
            if options.warmup < 0 or options.iters < 1 or options.samples < 1:
                raise ValueError(
                    "warmup must be non-negative; iters and samples must be positive"
                )
            if options.cold_l2_flush is None:
                l2 = torch.cuda.get_device_properties(self.device).L2_cache_size
                eviction = torch.empty(
                    max(2 * l2, 64 << 20), device=self.device, dtype=torch.uint8
                )
                options = replace(options, cold_l2_flush=lambda: eviction.zero_())
            selection = AutotuneRunner(options).resolve(
                self.compatible_candidates() if candidates is None else candidates,
                namespace=self.namespace,
                key=(self.workload_key, self._runtime_key()[3]),
                prepare=self.prepare,
                benchmark=benchmark
                or (lambda prepared: self._benchmark(prepared, options)),
                probe=probe,
                format_candidate=self.format_candidate,
                metric_formatter=metric_formatter,
                pre_jit=self.pre_jit,
            )
            self.selection = selection
            self._prepared = selection.prepared
            _SELECTIONS[self._runtime_key()] = selection.candidate
            return selection

    def _runtime_key(self):
        from .jit_runtime import pipeline_delay_variant

        return (
            self.namespace,
            self.device.index,
            self.workload_key,
            pipeline_delay_variant(),
            os.environ.get("CUDA_ARCH", "sm_110a"),
            os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a"),
        )

    def run(self):
        if self._prepared is None:
            candidate = _SELECTIONS.get(self._runtime_key())
            if candidate is None:
                self.tune()
            else:
                self._prepared = self.prepare(candidate)
        return self._prepared.run()


@dataclass(frozen=True)
class FixedKernelConfig:
    """The sole implementation of an operator with no tunable tactics."""


class FixedKernelRunner(BaseKernelRunner):
    """Uniform preparation for kernels with a single supported configuration."""

    def __init__(self, device, namespace, key, compile_fn, bind):
        super().__init__(device)
        self.namespace, self._key = namespace, key
        self._compile_fn, self._bind = compile_fn, bind

    @property
    def workload_key(self):
        return self._key

    def compatible_candidates(self):
        return (FixedKernelConfig(),)

    def pre_jit(self, candidate):
        return self._compile_fn()

    def prepare(self, candidate):
        return PreparedKernel(candidate, self._bind(self.pre_jit(candidate)))

    def run(self):
        if self._prepared is None:
            with torch.cuda.device(self.device):
                if (
                    self._runtime_key() not in _SELECTIONS
                    and torch.cuda.is_current_stream_capturing()
                ):
                    raise RuntimeError(
                        f"warm up {self.namespace} eagerly before CUDA graph capture"
                    )
                self._prepared = self.prepare(FixedKernelConfig())
                _SELECTIONS[self._runtime_key()] = FixedKernelConfig()
        return self._prepared.run()


def main():
    """Run a benchmark through its runner-backed CLI to prewarm a cache."""
    parser = argparse.ArgumentParser(description="SM110A autotune runner")
    scripts = {
        "bf16-gemm": "benchmark_bf16_gemm.py",
        "bf16-gemm-serving": "benchmark_bf16_gemm_serving.py",
        "fp8-gemm": "benchmark_fp8_gemm.py",
        "nvfp4-gemm": "benchmark_nvfp4_gemm.py",
        "mxfp4-gemm": "benchmark_mxfp4_gemm.py",
        "fp8mxfp4-gemm": "benchmark_fp8mxfp4_gemm.py",
        "grouped-m-gemm": "benchmark_bf16_grouped_m_gemm.py",
        "nvfp4-grouped-m-gemm": "benchmark_nvfp4_grouped_m_gemm.py",
        "fc1-swiglu": "benchmark_bf16_fc1_block_interleaved_swiglu.py",
        "nvfp4-fc1": "benchmark_nvfp4_fc1.py",
        "mxfp4-fc1": "benchmark_mxfp4_fc1.py",
        "nvfp4-fc1fc2": "benchmark_nvfp4_fc1fc2.py",
        "mxfp4-fc1fc2": "benchmark_mxfp4_fc1fc2.py",
        "fc2-finalize": "benchmark_bf16_fc2_finalize.py",
        "nvfp4-fc2": "benchmark_nvfp4_fc2.py",
        "mxfp4-fc2": "benchmark_mxfp4_fc2.py",
        "bf16-fa4-gqa": "benchmark_bf16_fa4_gqa.py",
        "bf16-fa4-mha": "benchmark_bf16_fa4_mha.py",
        "bf16-fa4-mla": "benchmark_bf16_fa4_mla.py",
        "bf16-fa4-mqa": "benchmark_bf16_fa4_mqa.py",
        "bf16-fa4-sparse-mla": "benchmark_bf16_fa4_sparse_mla_h64.py",
        "bf16-fa4-sparse-mla-prefill": "benchmark_bf16_fa4_sparse_mla_prefill_h64.py",
        "bf16tofp4-quant": "benchmark_bf16tofp4_quant.py",
        "bf16tofp8-quant": "benchmark_bf16tofp8_quant.py",
        "bf16tonvfp4-quant": "benchmark_bf16tonvfp4_quant.py",
        "fp8-act-quant": "benchmark_fp8_act_quant.py",
        "indexer-score-prefill": "benchmark_mxfp4_indexer_score_prefill.py",
        "indexer-score-decode": "benchmark_mxfp4_indexer_score_decode.py",
        "topk": "benchmark_topk.py",
        "nccl-lsa": "benchmark_nccl_lsa.py",
    }
    parser.add_argument(
        "workload", choices=tuple(scripts), help="workload to resolve and cache"
    )
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="arguments forwarded to the benchmark"
    )
    parsed = parser.parse_args()
    forwarded = parsed.args[1:] if parsed.args[:1] == ["--"] else parsed.args
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(_ROOT / "benchmarks" / scripts[parsed.workload]),
                *forwarded,
            ]
        )
    )


if __name__ == "__main__":
    main()
