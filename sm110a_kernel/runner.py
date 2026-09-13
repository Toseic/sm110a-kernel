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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Sequence, TypeVar

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
    if isinstance(candidate, tuple) and len(candidate) == 2 and dataclasses.is_dataclass(candidate[0]):
        value = {"config": dataclasses.asdict(candidate[0]), "swap_ab": bool(candidate[1])}
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
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL
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
        self.path = Path(options.cache_path) if options.cache_path else _DEFAULT_CACHE_PATH
        self.fingerprint = environment_fingerprint()

    def _load(self):
        try:
            value = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {"schema_version": _SCHEMA_VERSION, "environments": {}}
        if value.get("schema_version") != _SCHEMA_VERSION or not isinstance(value.get("environments"), dict):
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
            env.setdefault(namespace, {})[_cache_key(key)] = _serialize(candidate, stats)
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
        candidate_width = max(len(format_candidate(candidate)) for candidate in candidates)

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
                raise ValueError(
                    "pre_parallel_jit requires a pre_jit callback"
                )
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
                futures = [executor.submit(pre_jit, candidate) for candidate in candidates]
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
            entry = self._load().get("environments", {}).get(self.fingerprint, {}).get(namespace, {}).get(key_string)
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
                        print(f"  [runner cache] hit {namespace}{key} -> {format_candidate(candidate)}", flush=True)
                        print_header()
                        metric = metric_formatter(candidate, stats) if metric_formatter else ""
                        print_row(candidate, "cache hit", stats["median_ms"], metric, best=True)
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
                best_candidate, best_prepared, best_stats, best_probe = candidate, prepared, stats, probe_ms
            if self.options.debug:
                metric = metric_formatter(candidate, stats) if metric_formatter else ""
                print_row(candidate, "measured", stats["median_ms"], metric, best=is_best)
        if best_candidate is None:
            raise RuntimeError("all compatible candidates were skipped")
        self._store(namespace, key, best_candidate, best_stats)
        return Selection(best_candidate, best_prepared, best_stats, False)


def main():
    """Run a benchmark through its runner-backed CLI to prewarm a cache."""
    parser = argparse.ArgumentParser(description="SM110A autotune runner")
    parser.add_argument(
        "workload",
        choices=("bf16-gemm", "bf16-gemm-serving", "nvfp4-gemm", "grouped-m-gemm", "nvfp4-grouped-m-gemm", "fc1-swiglu", "nvfp4-fc1", "fc2-finalize", "nvfp4-fc2", "bf16-fa4-gqa"),
        help="workload to resolve and cache",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments forwarded to the benchmark")
    parsed = parser.parse_args()
    scripts = {
        "bf16-gemm": "benchmark_bf16_gemm.py",
        "bf16-gemm-serving": "benchmark_bf16_gemm_serving.py",
        "nvfp4-gemm": "benchmark_nvfp4_gemm.py",
        "grouped-m-gemm": "benchmark_bf16_grouped_m_gemm.py",
        "nvfp4-grouped-m-gemm": "benchmark_nvfp4_grouped_m_gemm.py",
        "fc1-swiglu": "benchmark_bf16_fc1_block_interleaved_swiglu.py",
        "nvfp4-fc1": "benchmark_nvfp4_fc1.py",
        "fc2-finalize": "benchmark_bf16_fc2_finalize.py",
        "nvfp4-fc2": "benchmark_nvfp4_fc2.py",
        "bf16-fa4-gqa": "benchmark_bf16_fa4_gqa.py",
    }
    forwarded = parsed.args[1:] if parsed.args[:1] == ["--"] else parsed.args
    raise SystemExit(subprocess.call([sys.executable, str(_ROOT / "benchmarks" / scripts[parsed.workload]), *forwarded]))


if __name__ == "__main__":
    main()
