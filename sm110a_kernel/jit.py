"""JIT compilation for configuration-specialized CUDA kernels."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional


@dataclass(frozen=True)
class GroupedMConfig:
    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 128
    epilogue_tile_n: int = 32
    cluster_m: int = 2

    def validate(self) -> None:
        if self.tile_m != 128:
            raise ValueError("tile_m must be 128 for the current four-warp mainloop")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.epilogue_tile_n <= 0:
            raise ValueError("epilogue_tile_n must be positive")
        if self.tile_n <= 0 or self.tile_n % self.epilogue_tile_n:
            raise ValueError("tile_n must be positive and divisible by epilogue_tile_n")
        if self.tile_n % self.cluster_m:
            raise ValueError("tile_n must be divisible by cluster_m")
        if self.tile_k <= 0 or self.tile_k % 64:
            raise ValueError("tile_k must be a positive multiple of 64")


_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_DIR = _ROOT / "csrc/bf16-grouped-m-gemm/src"
_INCLUDE_DIR = _ROOT / "include"
_MODULE_CACHE: dict[str, ModuleType] = {}
_CACHE_LOCK = threading.Lock()


def _default_config(a, b) -> GroupedMConfig:
    if a.ndim != 2 or b.ndim != 3 or b.size(0) <= 0:
        return GroupedMConfig()
    average_rows_per_expert = a.size(0) // b.size(0)
    return GroupedMConfig(cluster_m=1 if average_rows_per_expert <= 128 else 2)


def _source_digest() -> str:
    digest = hashlib.sha256()
    paths = [
        _SOURCE_DIR / "torch_binding.cpp",
        _SOURCE_DIR / "bf16_grouped_m_gemm.cu",
        _INCLUDE_DIR / "bf16_grouped_m_gemm.h",
        _INCLUDE_DIR / "common.h",
    ]
    for path in paths:
        digest.update(str(path.relative_to(_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _module_name(
    config: GroupedMConfig,
    cuda_arch: str,
    torch_version: str,
    torch_cuda_version: Optional[str],
) -> str:
    signature = repr(
        (
            asdict(config),
            cuda_arch,
            os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a"),
            torch_version,
            torch_cuda_version,
            sys.implementation.cache_tag,
            _source_digest(),
        )
    )
    suffix = hashlib.sha256(signature.encode()).hexdigest()[:16]
    return f"sm110a_grouped_m_jit_{suffix}"


def compile_bf16_grouped_m_gemm(
    config: Optional[GroupedMConfig] = None,
) -> ModuleType:
    """Compile or load one grouped-M kernel configuration."""
    config = config or GroupedMConfig()
    if not isinstance(config, GroupedMConfig):
        raise TypeError("config must be a GroupedMConfig")
    config.validate()

    try:
        import torch
        from torch.utils.cpp_extension import load
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "JIT compilation requires PyTorch with CUDA support installed."
        ) from exc

    with _CACHE_LOCK:
        cuda_arch = os.environ.get("CUDA_ARCH", "sm_110a")
        compute_arch = os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a")
        name = _module_name(
            config, cuda_arch, torch.__version__, torch.version.cuda
        )
        cached = _MODULE_CACHE.get(name)
        if cached is not None:
            return cached

        build_directory = None
        cache_root = os.environ.get("SM110A_JIT_CACHE_DIR")
        if cache_root:
            build_path = Path(cache_root).expanduser().resolve() / name
            build_path.mkdir(parents=True, exist_ok=True)
            build_directory = str(build_path)

        definitions = [
            f"-DSM110A_GROUPED_TILE_M={config.tile_m}",
            f"-DSM110A_GROUPED_TILE_N={config.tile_n}",
            f"-DSM110A_GROUPED_TILE_K={config.tile_k}",
            f"-DSM110A_GROUPED_EPILOGUE_TILE_N={config.epilogue_tile_n}",
            f"-DSM110A_GROUPED_CLUSTER_M={config.cluster_m}",
        ]
        module = load(
            name=name,
            sources=[
                str(_SOURCE_DIR / "torch_binding.cpp"),
                str(_SOURCE_DIR / "bf16_grouped_m_gemm.cu"),
            ],
            extra_include_paths=[str(_INCLUDE_DIR)],
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                f"-gencode=arch={compute_arch},code={cuda_arch}",
                *definitions,
            ],
            extra_cflags=["-O3", "-std=c++17", *definitions],
            extra_ldflags=["-lcuda"],
            build_directory=build_directory,
            verbose=os.environ.get("SM110A_JIT_VERBOSE", "0") == "1",
        )
        _MODULE_CACHE[name] = module
        return module


def get_bf16_grouped_m_gemm_module(
    a, b, config: Optional[GroupedMConfig] = None
) -> ModuleType:
    return compile_bf16_grouped_m_gemm(config or _default_config(a, b))
