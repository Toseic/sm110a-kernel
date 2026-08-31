"""Correctness-first BF16 MHA scaffold for future FA4 development."""

from __future__ import annotations

import math
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load_extension():
    from torch.utils.cpp_extension import load

    cuda_arch = os.environ.get("CUDA_ARCH", "sm_110a")
    compute_arch = os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a")
    return load(
        name=f"sm110a_bf16_fa4_mha_{cuda_arch}",
        sources=[
            str(_ROOT / "csrc/bf16_fa4/src/bf16_fa4_mha_torch_binding.cpp"),
            str(_ROOT / "csrc/bf16_fa4/src/bf16_fa4_mha_kernel.cu"),
        ],
        extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch={compute_arch},code={cuda_arch}"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        verbose=os.environ.get("SM110A_JIT_VERBOSE") == "1",
    )


_EXT = None


def get_bf16_fa4_mha_module():
    """Load the FA MHA CUDA extension on demand."""
    global _EXT
    if _EXT is None:
        _EXT = _load_extension()
    return _EXT


def _resolve_scale(q, scale):
    if scale is None:
        return 1.0 / math.sqrt(q.size(-1))
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return scale


def bf16_fa4_mha(q, k, v, *, causal=False, scale=None):
    """Compute BF16 MHA for contiguous `[B, H, S, D]` CUDA tensors."""
    return get_bf16_fa4_mha_module().fa_mha(
        q, k, v, bool(causal), _resolve_scale(q, scale)
    )


def bf16_fa4_mha_out(q, k, v, out, *, causal=False, scale=None):
    """Write BF16 FA MHA to a caller-provided contiguous output."""
    return get_bf16_fa4_mha_module().fa_mha_out(
        q, k, v, out, bool(causal), _resolve_scale(q, scale)
    )
