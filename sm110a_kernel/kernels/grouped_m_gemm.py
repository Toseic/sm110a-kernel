"""Configuration-specialized grouped-M BF16 GEMM kernel."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16-grouped-m-gemm/src"
_INCLUDE_DIR = _ROOT / "include"


@dataclass(frozen=True)
class GroupedMConfig:
    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 128
    epilogue_tile_n: int = 32
    cluster_m: int = 2
    mode: int = 0
    direct_gmem: bool = False

    def validate(self) -> None:
        if self.tile_m != 128:
            raise ValueError("tile_m must be 128 for the current four-warp mainloop")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.epilogue_tile_n not in (32, 64, 128):
            raise ValueError("epilogue_tile_n must be 32, 64, or 128")
        granularity = 8 * self.cluster_m
        if self.tile_n < granularity or self.tile_n > 256 or self.tile_n % granularity:
            raise ValueError(f"tile_n must be at most 256 and a positive multiple of {granularity} for cluster_m={self.cluster_m}")
        if self.tile_n % self.epilogue_tile_n:
            raise ValueError("tile_n must be divisible by epilogue_tile_n")
        if self.tile_k <= 0 or self.tile_k % 64:
            raise ValueError("tile_k must be a positive multiple of 64")
        if self.mode not in (0, 1, 2):
            raise ValueError("mode must be 0 (non-persistent), 1 (persistent), or 2 (CLC)")


@dataclass(frozen=True)
class GroupedMMetadata:
    """Device-resident routing schedule consumed by one grouped GEMM config."""

    offsets: object
    counts: object
    tasks: object
    state: object  # int32 {task_count, next_cluster_tile/reserved}


def _default_config(a, b):
    if a.ndim != 2 or b.ndim != 3 or b.size(0) <= 0:
        return GroupedMConfig()
    return GroupedMConfig(cluster_m=1 if a.size(0) // b.size(0) <= 128 else 2)


class GroupedMModule:
    def __init__(self, binding, library, config: GroupedMConfig):
        self._binding = binding
        self._library = library
        self._launcher_address = library.entry_address
        symbol = getattr(library.handle, "sm110a_bf16_grouped_m_metadata_prepare")
        self._metadata_launcher_address = ctypes.cast(symbol, ctypes.c_void_p).value
        self._config = config

    def prepare_metadata(self, grouped_layout, experts, n, swap_ab=False):
        offsets, counts, tasks, state = self._binding.prepare_grouped_m_metadata(
            grouped_layout, experts, n, self._metadata_launcher_address,
            self._config.tile_m, self._config.tile_n, self._config.cluster_m, swap_ab)
        return GroupedMMetadata(offsets, counts, tasks, state)

    def bf16_grouped_m_gemm(self, a, b, metadata: GroupedMMetadata, swap_ab=False):
        return self._binding.bf16_grouped_m_gemm(
            a, b, metadata.tasks, metadata.state, self._launcher_address,
            self._config.tile_m, self._config.tile_n, self._config.tile_k,
            self._config.cluster_m, swap_ab)

    def bf16_grouped_m_gemm_out(self, a, b, metadata: GroupedMMetadata, out, swap_ab=False):
        return self._binding.bf16_grouped_m_gemm_out(
            a, b, metadata.tasks, metadata.state, out, self._launcher_address,
            self._config.tile_m, self._config.tile_n, self._config.tile_k,
            self._config.cluster_m, swap_ab)


def _kernel_cflags(config: GroupedMConfig, variant):
    return (
        f"-DSM110A_GROUPED_TILE_M={config.tile_m}",
        f"-DSM110A_GROUPED_TILE_N={config.tile_n}",
        f"-DSM110A_GROUPED_TILE_K={config.tile_k}",
        f"-DSM110A_GROUPED_EPILOGUE_TILE_N={config.epilogue_tile_n}",
        f"-DSM110A_GROUPED_CLUSTER_M={config.cluster_m}",
        f"-DSM110A_GROUPED_MODE={config.mode}",
        f"-DSM110A_GROUPED_DIRECT_GMEM={int(config.direct_gmem)}",
    )


_JIT = SplitCudaJit(
    name="sm110a_grouped_m",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16_grouped_m_gemm.cu", _SOURCE_DIR / "metadata.cu"),
    kernel_dependencies=(_INCLUDE_DIR / "bf16_grouped_m_gemm.h", _INCLUDE_DIR / "common.h"),
    kernel_symbol="sm110a_bf16_grouped_m_gemm_launch",
    config_key=lambda config: config,
    kernel_cflags=_kernel_cflags,
    make_facade=GroupedMModule,
    include_paths=(_INCLUDE_DIR,),
)


def compile_bf16_grouped_m_gemm(config: Optional[GroupedMConfig] = None):
    config = config or GroupedMConfig()
    if not isinstance(config, GroupedMConfig):
        raise TypeError("config must be a GroupedMConfig")
    config.validate()
    return _JIT.compile(config)


def compile_bf16_grouped_m_gemm_configs(configs: Iterable[GroupedMConfig], max_workers=None):
    configs = tuple(configs)
    if not configs:
        return ()
    for config in configs:
        if not isinstance(config, GroupedMConfig):
            raise TypeError("configs must contain GroupedMConfig values")
        config.validate()
    return _JIT.compile_many(configs, max_workers)


def get_bf16_grouped_m_gemm_module(a, b, config=None):
    return compile_bf16_grouped_m_gemm(config or _default_config(a, b))


def prepare_grouped_m_metadata(grouped_layout, experts, n, config=None, swap_ab=False):
    module = compile_bf16_grouped_m_gemm(config or GroupedMConfig())
    return module.prepare_metadata(grouped_layout, experts, n, swap_ab)


def bf16_grouped_m_gemm(a, b, metadata, config=None, swap_ab=False):
    return get_bf16_grouped_m_gemm_module(a, b, config).bf16_grouped_m_gemm(a, b, metadata, swap_ab)


def bf16_grouped_m_gemm_out(a, b, metadata, out, config=None, swap_ab=False):
    return get_bf16_grouped_m_gemm_module(a, b, config).bf16_grouped_m_gemm_out(a, b, metadata, out, swap_ab)
