"""H=64 sparse MLA decode with indexed BF16 KV and an attention sink."""

from __future__ import annotations

import math
from pathlib import Path

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16_fa4_sparse_mla/src"


class Bf16Fa4SparseMlaModule:
    def __init__(self, binding, library, _config):
        self._binding = binding
        self._launcher = library.entry_address

    def sparse_mla(self, q, kv, indices, sink, scale):
        return self._binding.sparse_mla(q, kv, indices, sink, self._launcher, scale)

    def sparse_mla_out(self, q, kv, indices, sink, out, scale):
        return self._binding.sparse_mla_out(
            q, kv, indices, sink, out, self._launcher, scale
        )


_JIT = SplitCudaJit(
    name="sm110a_bf16_fa4_sparse_mla_h64",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16_fa4_sparse_mla_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16_fa4_sparse_mla_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_bf16_fa4_sparse_mla_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda _config, _variant: (),
    make_facade=Bf16Fa4SparseMlaModule,
)

_PREFILL_JIT = SplitCudaJit(
    name="sm110a_bf16_fa4_sparse_mla_prefill_h64",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16_fa4_sparse_mla_prefill_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16_fa4_sparse_mla_prefill_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_bf16_fa4_sparse_mla_prefill_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda _config, _variant: (),
    make_facade=Bf16Fa4SparseMlaModule,
)


def compile_bf16_fa4_sparse_mla():
    return _JIT.compile(None)


def compile_bf16_fa4_sparse_mla_prefill():
    return _PREFILL_JIT.compile(None)


def _resolve_scale(scale):
    if scale is None:
        return 1.0 / math.sqrt(512)
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return scale


def bf16_fa4_sparse_mla(q, kv, indices, sink, *, scale=None):
    """Return H=64 decode output in BHSD layout.

    Q is [B,64,1,512], KV is [B,N,512], indices is [B,K], and sink is
    [64]. Entries of indices equal to -1 are excluded from softmax.
    """
    return compile_bf16_fa4_sparse_mla().sparse_mla(
        q, kv, indices, sink, _resolve_scale(scale)
    )


def bf16_fa4_sparse_mla_out(q, kv, indices, sink, out, *, scale=None):
    """Write H=64 sparse MLA decode output to contiguous BHSD ``out``."""
    return compile_bf16_fa4_sparse_mla().sparse_mla_out(
        q, kv, indices, sink, out, _resolve_scale(scale)
    )


def bf16_fa4_sparse_mla_prefill(q, kv, indices, sink, *, scale=None):
    """Return H=64 prefill output for Q [B,64,Q,512], KV [B,N,512], indices [B,Q,K]."""
    return compile_bf16_fa4_sparse_mla_prefill().sparse_mla(
        q, kv, indices, sink, _resolve_scale(scale)
    )


def bf16_fa4_sparse_mla_prefill_out(q, kv, indices, sink, out, *, scale=None):
    """Write H=64 sparse MLA prefill output to contiguous BHSD ``out``."""
    return compile_bf16_fa4_sparse_mla_prefill().sparse_mla_out(
        q, kv, indices, sink, out, _resolve_scale(scale)
    )


class Bf16Fa4SparseMlaRunner(FixedKernelRunner):
    def __init__(self, q, kv, indices, sink, out, *, scale=None, prefill=False):
        scale = _resolve_scale(scale)
        super().__init__(
            q.device,
            "bf16_fa4_sparse_mla",
            (tuple(q.shape), tuple(kv.shape), tuple(indices.shape), prefill, scale),
            compile_bf16_fa4_sparse_mla_prefill
            if prefill
            else compile_bf16_fa4_sparse_mla,
            lambda module: (
                lambda: module.sparse_mla_out(q, kv, indices, sink, out, scale)
            ),
        )


class Bf16Fa4SparseMlaPrefillRunner(Bf16Fa4SparseMlaRunner):
    def __init__(self, q, kv, indices, sink, out, *, scale=None):
        super().__init__(q, kv, indices, sink, out, scale=scale, prefill=True)
