"""Sparse MLA with BF16 Q and FP8 non-RoPE KV, dequantized by gather warps."""

from __future__ import annotations

import math
from pathlib import Path

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/fp8_kv_fa4_sparse_mla/src"


class Fp8KvFa4SparseMlaModule:
    def __init__(self, binding, library, _config):
        self._binding = binding
        self._launcher = library.entry_address

    def sparse_mla(self, q, kv, scales, rope, indices, sink, scale):
        return self._binding.sparse_mla(
            q, kv, scales, rope, indices, sink, self._launcher, scale
        )

    def sparse_mla_out(self, q, kv, scales, rope, indices, sink, out, scale):
        return self._binding.sparse_mla_out(
            q, kv, scales, rope, indices, sink, out, self._launcher, scale
        )


_JIT = SplitCudaJit(
    name="sm110a_fp8_kv_fa4_sparse_mla_h64",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "fp8_kv_fa4_sparse_mla_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "fp8_kv_fa4_sparse_mla_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_fp8_kv_fa4_sparse_mla_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda _config, _variant: (),
    make_facade=Fp8KvFa4SparseMlaModule,
)


def compile_fp8_kv_fa4_sparse_mla():
    return _JIT.compile(None)


def _resolve_scale(scale):
    scale = 512**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return scale


def fp8_kv_fa4_sparse_mla(q, kv, scales, rope, indices, sink, *, scale=None):
    """Return BF16 [B,64,Q,512] for E4M3 [B,N,448], scales [B,N,7], RoPE [B,N,64].

    Indices are int32 [B,Q,K], or [B,K] for Q=1 decode. Negative and
    out-of-range indices are masked. Scales are E8M0 or FP32, block size 64.
    """
    return compile_fp8_kv_fa4_sparse_mla().sparse_mla(
        q, kv, scales, rope, indices, sink, _resolve_scale(scale)
    )


def fp8_kv_fa4_sparse_mla_out(q, kv, scales, rope, indices, sink, out, *, scale=None):
    """Write packed-cache sparse MLA output to contiguous BHSD ``out``."""
    return compile_fp8_kv_fa4_sparse_mla().sparse_mla_out(
        q, kv, scales, rope, indices, sink, out, _resolve_scale(scale)
    )


class Fp8KvFa4SparseMlaRunner(FixedKernelRunner):
    def __init__(self, q, kv, scales, rope, indices, sink, out, *, scale=None):
        scale = _resolve_scale(scale)
        super().__init__(
            q.device,
            "fp8_kv_fa4_sparse_mla",
            (
                tuple(q.shape),
                tuple(kv.shape),
                str(scales.dtype),
                tuple(indices.shape),
                scale,
            ),
            compile_fp8_kv_fa4_sparse_mla,
            lambda module: (
                lambda: module.sparse_mla_out(
                    q, kv, scales, rope, indices, sink, out, scale
                )
            ),
        )
