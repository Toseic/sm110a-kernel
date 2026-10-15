"""Exact unordered row top-k with CGA split-K selection."""

from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import AutotuneRunner, PreparedKernel, TuneOptions

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/topk/src"


def _check_scores(scores, k):
    if not scores.is_cuda or scores.dim() != 2 or not scores.is_contiguous():
        raise RuntimeError("scores must be contiguous CUDA [Qlen, KVlen]")
    if scores.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError("scores must be float32 or bfloat16")
    if k != 512 or scores.size(1) < 512:
        raise RuntimeError("k is fixed at 512 and KVlen must be >= 512")


@dataclass(frozen=True)
class TopKConfig:
    split_k: int


def _tune_stats(run, options):
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
    return {"min_ms": min(timings), "median_ms": median(timings)}


class TopKModule:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self._config = config
        self._split_k_cache = {}

    def select_split_k(self, scores, k=512, out=None, *, options=None):
        """Resolve 1/2/4/8 CTAs per row using cold-L2 timings and runner cache."""
        _check_scores(scores, k)
        if out is None:
            out = torch.empty(
                (scores.size(0), k), device=scores.device, dtype=torch.int32
            )
        key = (
            scores.device.index,
            tuple(scores.shape),
            str(scores.dtype),
            k,
            scores.data_ptr() % 16,
            self._config,
        )
        use_default = options is None
        if use_default and key in self._split_k_cache:
            return self._split_k_cache[key]
        with torch.cuda.device(scores.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "autotune this TopK shape before CUDA graph capture, or specify split_k"
                )
            self._binding.topk_out(scores, out, k, 1, self._launcher)
            if not scores.size(0):
                self._split_k_cache[key] = 1
                return 1
            if options is None:
                l2_bytes = torch.cuda.get_device_properties(scores.device).L2_cache_size
                eviction = torch.empty(
                    max(2 * l2_bytes, 64 << 20), device=scores.device, dtype=torch.uint8
                )
                options = TuneOptions(cold_l2_flush=lambda: eviction.zero_())
            configs = tuple(
                TopKConfig(split)
                for split in (1, 2, 4, 8)
                if split <= (scores.size(1) + 1023) // 1024
            )

            def prepare(config):
                return PreparedKernel(
                    config,
                    lambda: self._binding.topk_out(
                        scores, out, k, config.split_k, self._launcher
                    ),
                )

            selection = AutotuneRunner(options).resolve(
                configs,
                namespace="topk512_cga_mask_scan_v3",
                key=key,
                prepare=prepare,
                benchmark=lambda prepared: _tune_stats(prepared.run, options),
                probe=None,
                format_candidate=lambda config: f"split_k={config.split_k}",
            )
        winner = selection.candidate.split_k
        if use_default:
            self._split_k_cache[key] = winner
        return winner

    def topk(self, scores, k=512, *, split_k=None):
        if split_k is not None:
            return self._binding.topk(scores, k, split_k, self._launcher)
        _check_scores(scores, k)
        out = torch.empty((scores.size(0), k), device=scores.device, dtype=torch.int32)
        return self.topk_out(scores, out, k)

    def topk_out(self, scores, out, k=512, *, split_k=None):
        if split_k is None:
            split_k = self.select_split_k(scores, k, out)
        return self._binding.topk_out(scores, out, k, split_k, self._launcher)


_JIT = SplitCudaJit(
    name="sm110a_topk_radix",
    root=_ROOT,
    binding_sources=(_SOURCE / "topk_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "topk_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_topk_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda config, _variant: (
        f"-DSM110A_TOPK_STAGES={config[0]}",
        f"-DSM110A_TOPK_TMA={int(config[1] == 'tma')}",
    ),
    make_facade=TopKModule,
)


def compile_topk(*, stages=2, copy_engine="ldgsts") -> TopKModule:
    """Compile staging/copy choices and all four CGA split-K variants."""
    if isinstance(stages, bool) or not isinstance(stages, int) or not 2 <= stages <= 8:
        raise ValueError("stages must be an integer in [2, 8]")
    if copy_engine not in ("tma", "ldgsts"):
        raise ValueError("copy_engine must be 'tma' or 'ldgsts'")
    return _JIT.compile((stages, copy_engine))


def topk(scores, k=512, *, split_k=None):
    """Return unordered int32 [Qlen, 512] indices; KVlen must be >= 512.

    split_k=None autotunes 1/2/4/8. Ties choose smaller indices; NaNs rank
    above +inf. Masked -inf entries participate when necessary.
    """
    return compile_topk().topk(scores, k, split_k=split_k)


def topk_out(scores, out, k=512, *, split_k=None):
    """Reuse output; prewarm autotuning or fix split_k before graph capture."""
    return compile_topk().topk_out(scores, out, k, split_k=split_k)
