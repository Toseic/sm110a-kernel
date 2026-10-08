"""MXFP4 indexer score for prefill and batched decoding."""

from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median

import torch

from ..jit_runtime import SplitCudaJit, source_digest
from ..runner import AutotuneRunner, PreparedKernel, TuneOptions

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/mxfp4_indexer_score/src"


class IndexerScoreModule:
    def __init__(self, binding, library, _config):
        self._binding = binding
        self._launcher = library.entry_address
        self._cluster_m_cache = {}

    def select_cluster_m(
        self,
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        out=None,
        *,
        causal=True,
        options=None,
    ):
        use_default_options = options is None
        key = (q.device.index, q.size(0), k.size(0), bool(causal))
        if use_default_options and key in self._cluster_m_cache:
            return self._cluster_m_cache[key]
        if out is None:
            out = torch.empty(
                (q.size(0), k.size(0)), device=q.device, dtype=torch.float32
            )
        configs = (IndexerScorePrefillConfig(1), IndexerScorePrefillConfig(2))
        if q.size(0) <= 2:
            return 1
        with torch.cuda.device(q.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "autotune indexer score prefill before CUDA graph capture"
                )
            if options is None:
                options = TuneOptions()
            if options.cold_l2_flush is None:
                l2_bytes = torch.cuda.get_device_properties(q.device).L2_cache_size
                eviction = (
                    torch.empty((2 * l2_bytes,), device=q.device, dtype=torch.uint8)
                    if l2_bytes > 0
                    else None
                )
                options = replace(
                    options,
                    cold_l2_flush=(lambda: eviction.zero_())
                    if eviction is not None
                    else None,
                )

            def prepare(config):
                def run():
                    return self._binding.indexer_score_prefill_out(
                        q,
                        k,
                        q_scales,
                        k_scales,
                        head_weight,
                        out,
                        config.cluster_m,
                        bool(causal),
                        self._launcher,
                    )

                return PreparedKernel(config, run)

            selection = AutotuneRunner(options).resolve(
                configs,
                namespace=_PREFILL_TUNE_NAMESPACE,
                key=(q.size(0), k.size(0), bool(causal)),
                prepare=prepare,
                benchmark=lambda prepared: _tune_stats(prepared.run, options),
                probe=None,
                format_candidate=lambda config: f"cluster_m={config.cluster_m}",
            )
        winner = selection.candidate.cluster_m
        if use_default_options:
            self._cluster_m_cache[key] = winner
        return winner

    def indexer_score_prefill(
        self, q, k, q_scales, k_scales, head_weight, cluster_m=None, *, causal=True
    ):
        if cluster_m is None:
            cluster_m = self.select_cluster_m(
                q, k, q_scales, k_scales, head_weight, causal=causal
            )
        return self._binding.indexer_score_prefill(
            q,
            k,
            q_scales,
            k_scales,
            head_weight,
            cluster_m,
            bool(causal),
            self._launcher,
        )

    def indexer_score_prefill_out(
        self, q, k, q_scales, k_scales, head_weight, out, cluster_m=None, *, causal=True
    ):
        if cluster_m is None:
            cluster_m = self.select_cluster_m(
                q, k, q_scales, k_scales, head_weight, out, causal=causal
            )
        return self._binding.indexer_score_prefill_out(
            q,
            k,
            q_scales,
            k_scales,
            head_weight,
            out,
            cluster_m,
            bool(causal),
            self._launcher,
        )


@dataclass(frozen=True)
class IndexerScorePrefillConfig:
    cluster_m: int


_PREFILL_TUNE_NAMESPACE = (
    "indexer_score_prefill_cluster_v1_"
    + source_digest((_SOURCE / "indexer_score_prefill_kernel.cu",), _ROOT)[:12]
)


_JIT = SplitCudaJit(
    name="sm110a_indexer_score_prefill",
    root=_ROOT,
    binding_sources=(_SOURCE / "indexer_score_prefill_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "indexer_score_prefill_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_indexer_score_prefill_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda _config, _variant: (),
    make_facade=IndexerScoreModule,
)


def compile_indexer_score_prefill() -> IndexerScoreModule:
    return _JIT.compile(None)


def select_indexer_score_prefill_cluster_m(
    q, k, q_scales, k_scales, head_weight, out=None, *, causal=True, options=None
):
    return compile_indexer_score_prefill().select_cluster_m(
        q, k, q_scales, k_scales, head_weight, out, causal=causal, options=options
    )


def indexer_score_prefill(
    q, k, q_scales, k_scales, head_weight, cluster_m=None, *, causal=True
):
    return compile_indexer_score_prefill().indexer_score_prefill(
        q, k, q_scales, k_scales, head_weight, cluster_m, causal=causal
    )


def indexer_score_prefill_out(
    q, k, q_scales, k_scales, head_weight, out, cluster_m=None, *, causal=True
):
    return compile_indexer_score_prefill().indexer_score_prefill_out(
        q, k, q_scales, k_scales, head_weight, out, cluster_m, causal=causal
    )


class IndexerScoreDecodeModule:
    def __init__(self, binding, library, _config):
        self._binding = binding
        self._launcher = library.entry_address
        self._split_kv_cache = {}

    def select_split_kv(
        self, q, k, q_scales, k_scales, head_weight, out=None, *, options=None
    ):
        """Measure valid split counts once per (device, B, KVs), then cache the winner."""
        use_default_options = options is None
        key = (q.device.index, q.size(0), k.size(1))
        if use_default_options and key in self._split_kv_cache:
            return self._split_kv_cache[key]
        if out is None:
            out = torch.empty(
                (q.size(0), k.size(1)), device=q.device, dtype=torch.float32
            )
        splits = (1, 2, 4, 8)
        configs = tuple(
            IndexerScoreDecodeConfig(split)
            for split in splits
            if split <= (k.size(1) + 255) // 256
        )
        if not configs:
            raise ValueError("KVs must be positive")

        with torch.cuda.device(q.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "autotune this decoding shape before CUDA graph capture"
                )
            if use_default_options:
                l2_bytes = torch.cuda.get_device_properties(q.device).L2_cache_size
                eviction = (
                    torch.empty((2 * l2_bytes,), device=q.device, dtype=torch.uint8)
                    if l2_bytes > 0
                    else None
                )
                options = TuneOptions(
                    cold_l2_flush=(lambda: eviction.zero_())
                    if eviction is not None
                    else None
                )

            def prepare(config):
                def run():
                    return self._binding.indexer_score_decode_out(
                        q,
                        k,
                        q_scales,
                        k_scales,
                        head_weight,
                        out,
                        config.split_kv,
                        self._launcher,
                    )

                return PreparedKernel(config, run)

            selection = AutotuneRunner(options).resolve(
                configs,
                namespace="indexer_score_decode_splitkv_v1",
                key=(q.size(0), k.size(1)),
                prepare=prepare,
                benchmark=lambda prepared: _tune_stats(prepared.run, options),
                probe=None,
                format_candidate=lambda config: f"split_kv={config.split_kv}",
            )
        winner = selection.candidate.split_kv
        if use_default_options:
            self._split_kv_cache[key] = winner
        return winner

    def indexer_score_decode(
        self, q, k, q_scales, k_scales, head_weight, split_kv=None
    ):
        if split_kv is None:
            split_kv = self.select_split_kv(q, k, q_scales, k_scales, head_weight)
        return self._binding.indexer_score_decode(
            q, k, q_scales, k_scales, head_weight, split_kv, self._launcher
        )

    def indexer_score_decode_out(
        self, q, k, q_scales, k_scales, head_weight, out, split_kv=None
    ):
        if split_kv is None:
            split_kv = self.select_split_kv(q, k, q_scales, k_scales, head_weight, out)
        return self._binding.indexer_score_decode_out(
            q, k, q_scales, k_scales, head_weight, out, split_kv, self._launcher
        )


@dataclass(frozen=True)
class IndexerScoreDecodeConfig:
    split_kv: int


def _tune_stats(run, options):
    for _ in range(options.warmup):
        run()
    torch.cuda.synchronize()
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
        timings.append(
            sum(start.elapsed_time(end) for start, end in events) / options.iters
        )
    return {
        "min_ms": min(timings),
        "median_ms": median(timings),
        "max_ms": max(timings),
    }


_DECODE_JIT = SplitCudaJit(
    name="sm110a_indexer_score_decode",
    root=_ROOT,
    binding_sources=(_SOURCE / "indexer_score_decode_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "indexer_score_decode_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_indexer_score_decode_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda _config, _variant: (),
    make_facade=IndexerScoreDecodeModule,
)


def compile_indexer_score_decode() -> IndexerScoreDecodeModule:
    return _DECODE_JIT.compile(None)


def select_indexer_score_decode_split_kv(
    q, k, q_scales, k_scales, head_weight, out=None, *, options=None
):
    return compile_indexer_score_decode().select_split_kv(
        q, k, q_scales, k_scales, head_weight, out, options=options
    )


def indexer_score_decode(q, k, q_scales, k_scales, head_weight, split_kv=None):
    return compile_indexer_score_decode().indexer_score_decode(
        q, k, q_scales, k_scales, head_weight, split_kv
    )


def indexer_score_decode_out(q, k, q_scales, k_scales, head_weight, out, split_kv=None):
    return compile_indexer_score_decode().indexer_score_decode_out(
        q, k, q_scales, k_scales, head_weight, out, split_kv
    )
