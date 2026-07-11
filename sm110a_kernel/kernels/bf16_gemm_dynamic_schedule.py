"""Precompiled dynamic-schedule BF16 GEMM and cuBLASLt baselines."""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load_extension():
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("sm110a_kernel requires PyTorch with CUDA support installed.") from exc
    cuda_arch = os.environ.get("CUDA_ARCH", "sm_110a")
    compute_arch = os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a")
    return load(
        name=f"sm110a_kernel_ext_{cuda_arch}",
        sources=[str(_ROOT / "csrc/bf16-gemm/src/torch_binding.cpp"), str(_ROOT / "csrc/bf16-gemm/src/bf16_gemm_dynamic_schedule.cu")],
        extra_include_paths=[str(_ROOT / "include")],
        extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch={compute_arch},code={cuda_arch}"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda", "-lcublas", "-lcublasLt"],
        verbose=False,
    )


_EXT = _load_extension()


def bf16_gemm_dynamic_schedule(a, b):
    return _EXT.bf16_gemm_dynamic_schedule(a, b)


def bf16_gemm_dynamic_schedule_out(a, b, out):
    return _EXT.bf16_gemm_dynamic_schedule_out(a, b, out)


def bf16_gemm_dynamic_schedule_profile(a, b, profiler, num_entries):
    return _EXT.bf16_gemm_dynamic_schedule_profile(a, b, profiler, num_entries)


def cublaslt_gemm(a, b):
    return _EXT.cublaslt_gemm(a, b)


def cublaslt_gemm_out(a, b, out, workspace):
    return _EXT.cublaslt_gemm_out(a, b, out, workspace)


def cublaslt_gemm_out_batched(a, b, out, workspace, iters):
    return _EXT.cublaslt_gemm_out_batched(a, b, out, workspace, iters)
