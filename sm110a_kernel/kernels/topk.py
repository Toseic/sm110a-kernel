"""Exact unordered row top-k, with one CTA per score row."""

from pathlib import Path

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/topk/src"


class TopKModule:
    def __init__(self, binding, library, _config):
        self._binding = binding
        self._launcher = library.entry_address

    def topk(self, scores, k):
        return self._binding.topk(scores, k, self._launcher)

    def topk_out(self, scores, out, k):
        return self._binding.topk_out(scores, out, k, self._launcher)


_JIT = SplitCudaJit(
    name="sm110a_topk_naive",
    root=_ROOT,
    binding_sources=(_SOURCE / "topk_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "topk_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_topk_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda _config, _variant: (),
    make_facade=TopKModule,
)


def compile_topk() -> TopKModule:
    return _JIT.compile(None)


def topk(scores, k):
    """Return int32 [Qlen, k] indices for contiguous CUDA FP32/BF16 scores.

    Indices are unordered. Ties choose smaller indices; NaNs rank above +inf.
    Masked -inf entries participate when fewer than k unmasked entries exist.
    """
    return compile_topk().topk(scores, k)


def topk_out(scores, out, k):
    """Allocation-free topk; compile before CUDA graph capture."""
    return compile_topk().topk_out(scores, out, k)
