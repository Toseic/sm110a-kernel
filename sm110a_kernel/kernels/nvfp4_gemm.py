"""SM110A native NVFP4 block-scaled GEMM."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from ..jit_runtime import jit_build_directory, source_digest

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/nvfp4_gemm/src"
_INCLUDE_DIR = _ROOT / "include"


@dataclass(frozen=True)
class Nvfp4GemmConfig:
    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 64
    cluster_m: int = 2
    epilogue_block_n: int = 32

    def validate(self) -> None:
        if (self.tile_m, self.tile_n, self.tile_k, self.cluster_m) != (128, 256, 64, 2):
            raise ValueError("the NVFP4 kernel currently requires tile=128x256x64 and cluster_m=2")
        if self.epilogue_block_n not in (32, 64, 128) or self.tile_n % self.epilogue_block_n:
            raise ValueError("epilogue_block_n must be 32, 64, or 128")


def _kernel_cflags(config: Nvfp4GemmConfig) -> tuple[str, ...]:
    return tuple(
        f"-DSM110A_NVFP4_{name}={value}"
        for name, value in (
            ("TILE_M", config.tile_m), ("TILE_N", config.tile_n),
            ("TILE_K", config.tile_k), ("CLUSTER_M", config.cluster_m),
            ("EPILOGUE_BLOCK_N", config.epilogue_block_n),
        )
    )


_MODULE_CACHE = {}
_CACHE_LOCK = threading.Lock()


def compile_nvfp4_gemm(config: Nvfp4GemmConfig | None = None):
    config = config or Nvfp4GemmConfig()
    if not isinstance(config, Nvfp4GemmConfig):
        raise TypeError("config must be an Nvfp4GemmConfig")
    config.validate()
    import torch
    from torch.utils.cpp_extension import load

    cuda_arch = os.environ.get("CUDA_ARCH", "sm_110a")
    compute_arch = os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a")
    paths = (
        _SOURCE_DIR / "nvfp4_gemm_jit_binding.cpp",
        _SOURCE_DIR / "nvfp4_gemm_kernel.cu",
        _INCLUDE_DIR / "nvfp4_gemm.h",
        _INCLUDE_DIR / "common.h",
        _INCLUDE_DIR / "profiler.h",
    )
    signature = repr((config, cuda_arch, compute_arch, torch.__version__,
                      torch.version.cuda, sys.implementation.cache_tag,
                      source_digest(paths, _ROOT)))
    name = "sm110a_nvfp4_gemm_jit_" + hashlib.sha256(signature.encode()).hexdigest()[:16]
    with _CACHE_LOCK:
        if name in _MODULE_CACHE:
            return _MODULE_CACHE[name]
        flags = list(_kernel_cflags(config))
        module = load(
            name=name,
            sources=[str(paths[0]), str(paths[1])],
            extra_include_paths=[str(_INCLUDE_DIR)],
            extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch={compute_arch},code={cuda_arch}", *flags],
            extra_cflags=["-O3", "-std=c++17", *flags],
            extra_ldflags=["-lcuda"],
            build_directory=jit_build_directory(name),
            verbose=os.environ.get("SM110A_JIT_VERBOSE", "0") == "1",
        )
        _MODULE_CACHE[name] = module
        return module


def to_blocked_nvfp4_scales(scales):
    """Convert logical [rows, K/16] scales to Blackwell SWIZZLE_32_4_4."""
    import torch

    if scales.ndim != 2 or scales.dtype != torch.float8_e4m3fn:
        raise TypeError("scales must be a 2D torch.float8_e4m3fn tensor")
    rows, cols = scales.shape
    if rows % 128 or cols % 4:
        raise ValueError("scale shape must be divisible by [128, 4]")
    blocks = scales.contiguous().view(rows // 128, 128, cols // 4, 4)
    blocks = blocks.permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


def from_blocked_nvfp4_scales(scales, rows: int, k: int):
    """Inverse of :func:`to_blocked_nvfp4_scales`, mainly for references."""
    cols = k // 16
    tiles = scales.view(rows // 128, cols // 4, 32, 4, 4)
    blocks = tiles.transpose(2, 3).reshape(rows // 128, cols // 4, 128, 4)
    return blocks.permute(0, 2, 1, 3).reshape(rows, cols).contiguous()


def _pack_e2m1(values):
    import torch

    midpoints = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
                             device=values.device, dtype=torch.float32)
    code = torch.bucketize(values.abs().float(), midpoints).to(torch.uint8)
    code |= (values < 0).to(torch.uint8) << 3
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def quantize_nvfp4_rows(matrix):
    """Quantize contiguous [rows,K] data to packed E2M1 plus blocked UE4M3."""
    import torch

    if matrix.ndim != 2 or matrix.size(0) % 128 or matrix.size(1) % 64:
        raise ValueError("matrix shape must be divisible by [128, 64]")
    x = matrix.float().contiguous()
    blocks = x.view(x.size(0), x.size(1) // 16, 16)
    scale = (blocks.abs().amax(dim=-1) / 6.0).to(torch.float8_e4m3fn)
    scale_f = scale.float()
    normalized = blocks / torch.where(scale_f == 0, torch.ones_like(scale_f), scale_f).unsqueeze(-1)
    packed = _pack_e2m1(normalized.clamp(-6.0, 6.0).reshape_as(x))
    return packed, to_blocked_nvfp4_scales(scale)


def quantize_nvfp4_gemm(a, b):
    """Quantize mathematical A[M,K] and B[K,N] for :func:`nvfp4_gemm`."""
    if a.ndim != 2 or b.ndim != 2 or a.size(1) != b.size(0):
        raise ValueError("expected compatible 2D GEMM operands")
    aq, sfa = quantize_nvfp4_rows(a)
    bq, sfb = quantize_nvfp4_rows(b.t().contiguous())
    return aq, bq, sfa, sfb


def nvfp4_gemm(a, b, sfa, sfb, config: Nvfp4GemmConfig | None = None):
    return compile_nvfp4_gemm(config).nvfp4_gemm(a, b, sfa, sfb)


def nvfp4_gemm_out(a, b, sfa, sfb, out, config: Nvfp4GemmConfig | None = None):
    return compile_nvfp4_gemm(config).nvfp4_gemm_out(a, b, sfa, sfb, out)
