"""cuBLASLt BF16 GEMM baseline."""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load_extension():
    try:
        from torch.utils.cpp_extension import load
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sm110a_kernel requires PyTorch with CUDA support installed."
        ) from exc
    cuda_arch = os.environ.get("CUDA_ARCH", "sm_110a")
    compute_arch = os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a")
    return load(
        name=f"sm110a_cublaslt_ext_{cuda_arch}",
        sources=[
            str(_ROOT / "csrc/bf16_gemm/src/bf16_gemm_torch_binding.cpp"),
            str(_ROOT / "csrc/bf16_gemm/src/bf16_gemm_kernel.cu"),
        ],
        extra_include_paths=[str(_ROOT / "include")],
        extra_cuda_cflags=[
            "-O3", "-lineinfo", f"-gencode=arch={compute_arch},code={cuda_arch}"
        ],
        extra_cflags=["-O3", "-std=c++20"],
        extra_ldflags=["-lcuda", "-lcublas", "-lcublasLt"],
        verbose=False,
    )


_EXT = _load_extension()

cublaslt_gemm = _EXT.cublaslt_gemm
cublaslt_gemm_out = _EXT.cublaslt_gemm_out
cublaslt_gemm_out_batched = _EXT.cublaslt_gemm_out_batched
