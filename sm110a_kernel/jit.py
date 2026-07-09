"""JIT compilation for configuration-specialized CUDA kernels."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional

from .jit_runtime import (
    LoadedCudaLibrary,
    SplitCudaJit,
    jit_build_directory,
    source_digest,
)


@dataclass(frozen=True)
class GroupedMConfig:
    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 128
    epilogue_tile_n: int = 32
    cluster_m: int = 2
    mode: int = 0  # 0=non-persistent, 1=persistent, 2=CLC

    def validate(self) -> None:
        if self.tile_m != 128:
            raise ValueError("tile_m must be 128 for the current four-warp mainloop")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.epilogue_tile_n not in (32, 64, 128):
            raise ValueError("epilogue_tile_n must be 32, 64, or 128")
        mma_n_granularity = 8 * self.cluster_m
        if (
            self.tile_n < mma_n_granularity
            or self.tile_n > 256
            or self.tile_n % mma_n_granularity
        ):
            raise ValueError(
                "tile_n must be at most 256 and a positive multiple of "
                f"{mma_n_granularity} for cluster_m={self.cluster_m}"
            )
        if self.tile_n % self.epilogue_tile_n:
            raise ValueError("tile_n must be divisible by epilogue_tile_n")
        if self.tile_k <= 0 or self.tile_k % 64:
            raise ValueError("tile_k must be a positive multiple of 64")
        if self.mode not in (0, 1, 2):
            raise ValueError(
                "mode must be 0 (non-persistent), 1 (persistent), or 2 (CLC)"
            )


_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_DIR = _ROOT / "csrc/bf16-grouped-m-gemm/src"
_INCLUDE_DIR = _ROOT / "include"


class _GroupedMModule:
    """Config-specific facade backed by one shared Torch binding."""

    def __init__(
        self,
        binding: ModuleType,
        library: LoadedCudaLibrary,
        config: GroupedMConfig,
    ) -> None:
        self._binding = binding
        self._library = library
        self._launcher_address = library.entry_address
        self._config = config

    def bf16_grouped_m_gemm(
        self, a, b, grouped_layout, swap_ab: bool = False
    ):
        return self._binding.bf16_grouped_m_gemm(
            a,
            b,
            grouped_layout,
            self._launcher_address,
            self._config.tile_m,
            self._config.tile_n,
            self._config.tile_k,
            self._config.cluster_m,
            swap_ab,
        )

    def bf16_grouped_m_gemm_out(
        self, a, b, grouped_layout, out, swap_ab: bool = False
    ):
        return self._binding.bf16_grouped_m_gemm_out(
            a,
            b,
            grouped_layout,
            out,
            self._launcher_address,
            self._config.tile_m,
            self._config.tile_n,
            self._config.tile_k,
            self._config.cluster_m,
            swap_ab,
        )


def _default_config(a, b) -> GroupedMConfig:
    if a.ndim != 2 or b.ndim != 3 or b.size(0) <= 0:
        return GroupedMConfig()
    average_rows_per_expert = a.size(0) // b.size(0)
    return GroupedMConfig(cluster_m=1 if average_rows_per_expert <= 128 else 2)


def _grouped_m_kernel_cflags(
    config: GroupedMConfig,
    variant,
) -> tuple[str, ...]:
    return (
        f"-DSM110A_GROUPED_TILE_M={config.tile_m}",
        f"-DSM110A_GROUPED_TILE_N={config.tile_n}",
        f"-DSM110A_GROUPED_TILE_K={config.tile_k}",
        f"-DSM110A_GROUPED_EPILOGUE_TILE_N={config.epilogue_tile_n}",
        f"-DSM110A_GROUPED_CLUSTER_M={config.cluster_m}",
        f"-DSM110A_GROUPED_MODE={config.mode}",
    )


def _make_grouped_m_module(
    binding: ModuleType,
    library: LoadedCudaLibrary,
    config: GroupedMConfig,
) -> _GroupedMModule:
    return _GroupedMModule(binding, library, config)


_GROUPED_M_JIT = SplitCudaJit[GroupedMConfig, _GroupedMModule](
    name="sm110a_grouped_m",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16_grouped_m_gemm.cu",),
    kernel_dependencies=(
        _INCLUDE_DIR / "bf16_grouped_m_gemm.h",
        _INCLUDE_DIR / "common.h",
    ),
    kernel_symbol="sm110a_bf16_grouped_m_gemm_launch",
    config_key=lambda config: config,
    variant_key=lambda: None,
    kernel_cflags=_grouped_m_kernel_cflags,
    make_facade=_make_grouped_m_module,
    include_paths=(_INCLUDE_DIR,),
)


def compile_bf16_grouped_m_gemm(
    config: Optional[GroupedMConfig] = None,
) -> _GroupedMModule:
    """Compile or load one grouped-M kernel configuration."""
    config = config or GroupedMConfig()
    if not isinstance(config, GroupedMConfig):
        raise TypeError("config must be a GroupedMConfig")
    config.validate()
    return _GROUPED_M_JIT.compile(config)


def compile_bf16_grouped_m_gemm_configs(
    configs: Iterable[GroupedMConfig],
    max_workers: Optional[int] = None,
) -> tuple[_GroupedMModule, ...]:
    """Compile grouped-M configs and the shared binding concurrently."""
    configs = tuple(configs)
    if not configs:
        return ()
    for config in configs:
        if not isinstance(config, GroupedMConfig):
            raise TypeError("configs must contain GroupedMConfig values")
        config.validate()
    return _GROUPED_M_JIT.compile_many(configs, max_workers)


def get_bf16_grouped_m_gemm_module(
    a, b, config: Optional[GroupedMConfig] = None
) -> _GroupedMModule:
    return compile_bf16_grouped_m_gemm(config or _default_config(a, b))


# ---------------------------------------------------------------------------
# BF16 GEMM JIT (standard, non-grouped)
# ---------------------------------------------------------------------------

_BF16_SOURCE_DIR = _ROOT / "csrc/bf16-gemm/src"
_MODULE_CACHE: dict[str, ModuleType] = {}
_CACHE_LOCK = threading.Lock()


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
        epilogue_smem = (
            2 * self.tile_m * self.epilogue_block_n * 2
        )
        mainloop_stage_smem = (
            self.tile_m + self.tile_n // self.cluster_m
        ) * self.tile_k * 2
        if (227 * 1024 - epilogue_smem) // mainloop_stage_smem < 2:
            raise ValueError(
                "configuration leaves room for fewer than two mainloop stages"
            )


def _bf16_default_config(a, b) -> Bf16GemmConfig:
    """Auto-select cluster_m based on problem dimensions."""
    if a.ndim != 2 or b.ndim != 2:
        return Bf16GemmConfig()
    m = a.size(0)
    return Bf16GemmConfig(cluster_m=1 if m <= 128 else 2)


def _bf16_source_digest() -> str:
    return source_digest(
        (
            _BF16_SOURCE_DIR / "bf16_gemm_jit_binding.cpp",
            _BF16_SOURCE_DIR / "bf16_gemm.cu",
            _INCLUDE_DIR / "bf16_gemm.h",
            _INCLUDE_DIR / "common.h",
            _INCLUDE_DIR / "profiler.h",
        ),
        _ROOT,
    )


def _bf16_module_name(
    config: Bf16GemmConfig,
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
            _bf16_source_digest(),
        )
    )
    suffix = hashlib.sha256(signature.encode()).hexdigest()[:16]
    return f"sm110a_bf16_gemm_jit_{suffix}"


def compile_bf16_gemm(
    config: Optional[Bf16GemmConfig] = None,
) -> ModuleType:
    """Compile or load one BF16 GEMM kernel configuration."""
    config = config or Bf16GemmConfig()
    if not isinstance(config, Bf16GemmConfig):
        raise TypeError("config must be a Bf16GemmConfig")
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
        name = _bf16_module_name(
            config, cuda_arch, torch.__version__, torch.version.cuda
        )
        cached = _MODULE_CACHE.get(name)
        if cached is not None:
            return cached

        definitions = [
            f"-DSM110A_BF16_TILE_M={config.tile_m}",
            f"-DSM110A_BF16_TILE_N={config.tile_n}",
            f"-DSM110A_BF16_TILE_K={config.tile_k}",
            f"-DSM110A_BF16_CLUSTER_M={config.cluster_m}",
            f"-DSM110A_BF16_EPILOGUE_BLOCK_N={config.epilogue_block_n}",
        ]
        module = load(
            name=name,
            sources=[
                str(_BF16_SOURCE_DIR / "bf16_gemm_jit_binding.cpp"),
                str(_BF16_SOURCE_DIR / "bf16_gemm.cu"),
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
            build_directory=jit_build_directory(name),
            verbose=os.environ.get("SM110A_JIT_VERBOSE", "0") == "1",
        )
        _MODULE_CACHE[name] = module
        return module


def get_bf16_gemm_module(
    a, b, config: Optional[Bf16GemmConfig] = None
) -> ModuleType:
    return compile_bf16_gemm(config or _bf16_default_config(a, b))
