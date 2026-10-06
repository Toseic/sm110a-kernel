"""Configuration-specialized SM110A NVFP4 grouped-M GEMM."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..jit_runtime import SplitCudaJit
from .grouped_m_gemm import GroupedMMetadata

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "csrc/nvfp4_grouped_m_gemm/src"
_META = _ROOT / "csrc/bf16_grouped_m_gemm/src/bf16_grouped_m_gemm_metadata.cu"
_INC = _ROOT / "include"


@dataclass(frozen=True)
class Nvfp4GroupedMConfig:
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64
    epilogue_tile_n: int = 32
    cluster_m: int = 2
    mode: int = 0
    # Independent 8x8 macro-tile traversal switches.  N-major maximizes B/SFB
    # reuse; serpentine reverses the inner axis on every adjacent outer tile.
    macro_swizzle_n_major: bool = True
    macro_swizzle_serpentine: bool = False

    def validate(self):
        if self.tile_m != 128 or self.tile_n not in (64, 128, 256):
            raise ValueError("tile_m must be 128 and tile_n must be 64, 128, or 256")
        if self.tile_k not in (64, 128, 256) or self.cluster_m not in (1, 2):
            raise ValueError("invalid tile_k or cluster_m")
        if (
            self.epilogue_tile_n not in (32, 64, 128)
            or self.tile_n % self.epilogue_tile_n
        ):
            raise ValueError("invalid epilogue_tile_n")
        if self.mode not in (0, 1, 2):
            raise ValueError("mode must be 0, 1, or 2")
        if not isinstance(self.macro_swizzle_n_major, bool) or not isinstance(
            self.macro_swizzle_serpentine, bool
        ):
            raise ValueError(
                "macro_swizzle_n_major and macro_swizzle_serpentine must be bool"
            )


class Nvfp4GroupedMModule:
    def __init__(self, binding, library, config):
        self._binding, self._address, self.config = (
            binding,
            library.entry_address,
            config,
        )
        symbol = getattr(library.handle, "sm110a_bf16_grouped_m_metadata_prepare")
        self._metadata_address = ctypes.cast(symbol, ctypes.c_void_p).value

    def prepare_metadata(self, layout, experts, n):
        values = self._binding.prepare_grouped_m_metadata(
            layout,
            experts,
            n,
            self._metadata_address,
            self.config.tile_m,
            self.config.tile_n,
            self.config.cluster_m,
        )
        return GroupedMMetadata(*values)

    def nvfp4_grouped_m_gemm(self, a, b, sfa, sfb, metadata):
        return self._binding.nvfp4_grouped_m_gemm(
            a, b, sfa, sfb, metadata.tasks, metadata.state, self._address
        )

    def nvfp4_grouped_m_gemm_out(self, a, b, sfa, sfb, metadata, out):
        return self._binding.nvfp4_grouped_m_gemm_out(
            a, b, sfa, sfb, metadata.tasks, metadata.state, out, self._address
        )


def _flags(c, variant):
    return tuple(
        f"-DSM110A_NVFP4_GROUPED_{k}={v}"
        for k, v in (
            ("TILE_M", c.tile_m),
            ("TILE_N", c.tile_n),
            ("TILE_K", c.tile_k),
            ("EPILOGUE_TILE_N", c.epilogue_tile_n),
            ("CLUSTER_M", c.cluster_m),
            ("MODE", c.mode),
            # The grouped-M task builder is compiled into this NVFP4 module too.
            ("MACRO_SWIZZLE_N_MAJOR", int(c.macro_swizzle_n_major)),
            ("MACRO_SWIZZLE_SERPENTINE", int(c.macro_swizzle_serpentine)),
        )
    )


_JIT = SplitCudaJit(
    name="sm110a_nvfp4_grouped_m",
    root=_ROOT,
    binding_sources=(_SRC / "nvfp4_grouped_m_gemm_torch_binding.cpp",),
    kernel_sources=(_SRC / "nvfp4_grouped_m_gemm_kernel.cu", _META),
    kernel_dependencies=(_INC / "nvfp4_grouped_m_gemm.h", _INC / "common.h"),
    kernel_symbol="sm110a_nvfp4_grouped_m_gemm_launch",
    config_key=lambda c: c,
    kernel_cflags=_flags,
    make_facade=Nvfp4GroupedMModule,
    include_paths=(_INC,),
)


def compile_nvfp4_grouped_m_gemm(config=None):
    config = config or Nvfp4GroupedMConfig()
    config.validate()
    return _JIT.compile(config)


def compile_nvfp4_grouped_m_gemm_configs(
    configs: Iterable[Nvfp4GroupedMConfig], max_workers=None
):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Nvfp4GroupedMConfig):
            raise TypeError("configs must contain Nvfp4GroupedMConfig values")
        config.validate()
    return _JIT.compile_many(configs, max_workers) if configs else ()


def prepare_nvfp4_grouped_m_metadata(layout, experts, n, config=None):
    return compile_nvfp4_grouped_m_gemm(config).prepare_metadata(layout, experts, n)


def nvfp4_grouped_m_gemm(a, b, sfa, sfb, metadata, config=None):
    return compile_nvfp4_grouped_m_gemm(config).nvfp4_grouped_m_gemm(
        a, b, sfa, sfb, metadata
    )


def nvfp4_grouped_m_gemm_out(a, b, sfa, sfb, metadata, out, config=None):
    return compile_nvfp4_grouped_m_gemm(config).nvfp4_grouped_m_gemm_out(
        a, b, sfa, sfb, metadata, out
    )
