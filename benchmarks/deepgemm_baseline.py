"""Adapters for the locally patched DeepGEMM SM110 baseline.

Preparation and layout conversion deliberately happen before a benchmark's
timed region.  Each returned callable is one DeepGEMM launch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
_MODULE = None


def load_deepgemm():
    global _MODULE
    if _MODULE is None:
        source_root = ROOT / "baselinedir" / "DeepGEMM"
        if not source_root.is_dir():
            raise RuntimeError(
                f"DeepGEMM baseline is unavailable: expected {source_root}; "
                "clone and build the local DeepGEMM baseline first"
            )
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        try:
            import deep_gemm
        except ImportError as exc:
            raise RuntimeError(
                "the DeepGEMM baseline requires its local extension; run "
                f"{source_root}/develop.sh first"
            ) from exc
        _MODULE = deep_gemm
    return _MODULE


def bf16_nt(a: torch.Tensor, b_nt: torch.Tensor, out: torch.Tensor):
    """Return DeepGEMM's BF16 NT dense-GEMM launch."""
    deep_gemm = load_deepgemm()
    return lambda: deep_gemm.bf16_gemm_nt(a, b_nt, out)


def bf16_grouped_nt(a, b_nt, out, layout):
    """Return DeepGEMM's contiguous grouped-M BF16 GEMM launch."""
    deep_gemm = load_deepgemm()
    if layout.dtype != torch.int32 or layout.numel() != a.size(0):
        raise ValueError("DeepGEMM grouped-M baseline requires int32 per-row layout")
    grouped_layout = layout.contiguous()
    return lambda: deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        a, b_nt, out, grouped_layout
    )


def mxfp4_nt(a, b, sfa, sfb, out, *, from_blocked_scales):
    """Return DeepGEMM's FP4xFP4 NT GEMM with native MXFP4 operands."""
    deep_gemm = load_deepgemm()
    m, packed_k = a.shape
    n = b.size(0)
    k = packed_k * 2
    if k % 128:
        raise RuntimeError(
            f"the DeepGEMM FP4xFP4 baseline requires K divisible by 128; got K={k}"
        )
    a_scales = deep_gemm.transform_sf_into_required_layout(
        from_blocked_scales(sfa, m, k).float(), m, k, (1, 32)
    )
    b_scales = deep_gemm.transform_sf_into_required_layout(
        from_blocked_scales(sfb, n, k).float(), n, k, (1, 32)
    )
    a_pair = (a.view(torch.int8), a_scales)
    b_pair = (b.view(torch.int8), b_scales)
    return lambda: deep_gemm.fp8_fp4_gemm_nt(
        a_pair, b_pair, out, recipe_a=(1, 32), recipe_b=(1, 32)
    )


def nvfp4_nt(a, b, sfa, sfb, out):
    """Return DeepGEMM's cuBLASLt NVFP4 NT GEMM with native operands."""
    deep_gemm = load_deepgemm()
    a_pair = (a.view(torch.int8), sfa.view(torch.uint8))
    b_pair = (b.view(torch.int8), sfb.view(torch.uint8))
    return lambda: deep_gemm.cublaslt_nvfp4_gemm_nt(a_pair, b_pair, out)
