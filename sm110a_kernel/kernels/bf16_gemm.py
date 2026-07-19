"""Configuration-specialized standard BF16 GEMM kernel."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..jit_runtime import jit_build_directory, source_digest

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16_gemm/src"
_INCLUDE_DIR = _ROOT / "include"


@dataclass(frozen=True)
class Bf16GemmConfig:
    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 128
    cluster_m: int = 2
    epilogue_block_n: int = 32

    def validate(self) -> None:
        if self.tile_m != 128:
            raise ValueError("tile_m must be 128 for the current four-warp mainloop")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.tile_n <= 0:
            raise ValueError("tile_n must be positive")
        if self.epilogue_block_n not in (32, 64, 128):
            raise ValueError("epilogue_block_n must be one of 32, 64, or 128")
        if self.tile_n % self.epilogue_block_n:
            raise ValueError("tile_n must be divisible by epilogue_block_n")
        if self.tile_n % self.cluster_m:
            raise ValueError("tile_n must be divisible by cluster_m")
        if self.tile_k <= 0 or self.tile_k % 64:
            raise ValueError("tile_k must be a positive multiple of 64")
        epilogue_smem = 2 * self.tile_m * self.epilogue_block_n * 2
        mainloop_stage_smem = (self.tile_m + self.tile_n // self.cluster_m) * self.tile_k * 2
        if (227 * 1024 - epilogue_smem) // mainloop_stage_smem < 2:
            raise ValueError("configuration leaves room for fewer than two mainloop stages")


def _default_config(a, b) -> Bf16GemmConfig:
    if a.ndim != 2 or b.ndim != 2:
        return Bf16GemmConfig()
    return Bf16GemmConfig(cluster_m=1 if a.size(0) <= 128 else 2)


def _kernel_cflags(config: Bf16GemmConfig, variant) -> tuple[str, ...]:
    return tuple(
        f"-DSM110A_BF16_{name}={value}"
        for name, value in (
            ("TILE_M", config.tile_m), ("TILE_N", config.tile_n),
            ("TILE_K", config.tile_k), ("CLUSTER_M", config.cluster_m),
            ("EPILOGUE_BLOCK_N", config.epilogue_block_n),
        )
    )


_MODULE_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _module_name(config, cuda_arch, torch):
    digest = source_digest(
        (_SOURCE_DIR / "bf16_gemm_jit_binding.cpp", _SOURCE_DIR / "bf16_gemm_kernel.cu",
         _INCLUDE_DIR / "bf16_gemm.h", _INCLUDE_DIR / "common.h", _INCLUDE_DIR / "profiler.h"),
        _ROOT,
    )
    signature = repr((config, cuda_arch, os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a"),
                      torch.__version__, torch.version.cuda, sys.implementation.cache_tag, digest))
    return "sm110a_bf16_gemm_jit_" + hashlib.sha256(signature.encode()).hexdigest()[:16]


def compile_bf16_gemm(config: Optional[Bf16GemmConfig] = None) -> Bf16GemmModule:
    config = config or Bf16GemmConfig()
    if not isinstance(config, Bf16GemmConfig):
        raise TypeError("config must be a Bf16GemmConfig")
    config.validate()
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("JIT compilation requires PyTorch with CUDA support installed.") from exc
    with _CACHE_LOCK:
        cuda_arch = os.environ.get("CUDA_ARCH", "sm_110a")
        compute_arch = os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a")
        name = _module_name(config, cuda_arch, torch)
        cached = _MODULE_CACHE.get(name)
        if cached is not None:
            return cached
        definitions = list(_kernel_cflags(config, None))
        module = load(
            name=name,
            sources=[str(_SOURCE_DIR / "bf16_gemm_jit_binding.cpp"), str(_SOURCE_DIR / "bf16_gemm_kernel.cu")],
            extra_include_paths=[str(_INCLUDE_DIR)],
            extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch={compute_arch},code={cuda_arch}", *definitions],
            extra_cflags=["-O3", "-std=c++17", *definitions],
            extra_ldflags=["-lcuda"],
            build_directory=jit_build_directory(name),
            verbose=os.environ.get("SM110A_JIT_VERBOSE", "0") == "1",
        )
        _MODULE_CACHE[name] = module
        return module


def get_bf16_gemm_module(a, b, config=None):
    return compile_bf16_gemm(config or _default_config(a, b))


def bf16_gemm(a, b, config=None, swap_ab=False):
    return get_bf16_gemm_module(a, b, config).bf16_gemm(a, b, swap_ab)


def bf16_gemm_out(a, b, out, config=None, swap_ab=False):
    return get_bf16_gemm_module(a, b, config).bf16_gemm_out(a, b, out, swap_ab)


def bf16_gemm_nt_out(a, b_nt, out, config=None, swap_ab=False):
    """Compute ``a @ b_nt.T`` for a contiguous, pretransposed ``b_nt(N, K)``."""
    return compile_bf16_gemm(config).bf16_gemm_nt_out(a, b_nt, out, swap_ab)
