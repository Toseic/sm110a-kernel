"""NVFP4 routed FC1 with fused permutation and block-interleaved SwiGLU."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "csrc/nvfp4_moe/src"
_INC = _ROOT / "include"

@dataclass(frozen=True)
class Nvfp4Fc1Config:
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64
    epilogue_tile_n: int = 64
    # Zero selects the CUDA kernel's measured fixed-C=3 default policy.
    # Explicit values remain useful for focused pipeline experiments.
    num_c_stage: int = 0
    cluster_m: int = 1
    cluster_n: int = 1
    direct_gmem: bool = False
    macro_swizzle_n_major: bool = True

    def validate(self) -> None:
        if self.tile_m != 128 or self.tile_n not in (64, 128, 256):
            raise ValueError("tile_m must be 128 and tile_n must be 64, 128, or 256")
        if self.tile_k not in (64, 128, 256):
            raise ValueError("tile_k must be 64, 128, or 256")
        if self.epilogue_tile_n not in (64, 128) or self.tile_n % self.epilogue_tile_n:
            raise ValueError("invalid epilogue_tile_n")
        if self.num_c_stage not in (0, 3, 4, 5):
            raise ValueError("num_c_stage must be 0 (auto) or in [3, 5]")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.cluster_n != 1:
            raise ValueError("cluster_n must be 1")
        if not isinstance(self.macro_swizzle_n_major, bool):
            raise ValueError("macro_swizzle_n_major must be bool")


@dataclass(frozen=True)
class Nvfp4Fc1Metadata:
    """Device-resident routing and task state, aligned with BF16 FC1 metadata."""

    routing: object
    source_rows: object
    tasks: object
    state: object


class Nvfp4Fc1Module:
    def __init__(self, binding, library, config: Nvfp4Fc1Config):
        self._binding = binding
        self._address = library.entry_address
        self.config = config
        symbol = getattr(library.handle, "sm110a_nvfp4_fc1_tasks_prepare")
        self._metadata_address = ctypes.cast(symbol, ctypes.c_void_p).value
        routing_symbol = getattr(library.handle, "sm110a_nvfp4_fc1_routing_prepare")
        self._routing_address = ctypes.cast(routing_symbol, ctypes.c_void_p).value

    def prepare_metadata(self, router_scores, topk: int, n: int) -> Nvfp4Fc1Metadata:
        import torch

        if router_scores.ndim != 2 or router_scores.dtype != torch.bfloat16:
            raise TypeError("router_scores must be a 2D BF16 tensor")
        tokens, experts = router_scores.shape
        if tokens % 128 or topk <= 0 or topk > experts:
            raise ValueError("tokens must be divisible by 128 and topk must be in [1, experts]")
        if n <= 0 or n % (self.config.tile_n * self.config.cluster_n):
            raise ValueError("N is incompatible with the NVFP4 FC1 config")

        routing, source_rows, tasks, state = self._binding.prepare_nvfp4_fc1_routing(
            router_scores, topk, n, self.config.tile_m, self.config.tile_n,
            self.config.cluster_m, self._routing_address)
        return Nvfp4Fc1Metadata(routing, source_rows, tasks, state)

    def _max_active_clusters(self, device, value):
        if value is None:
            import torch
            value = torch.cuda.get_device_properties(device).multi_processor_count // (
                self.config.cluster_m * self.config.cluster_n)
        if value <= 0:
            raise ValueError("max_active_clusters must be positive")
        return int(value)

    def nvfp4_fc1_out(self, a, b, sfa, sfb, metadata: Nvfp4Fc1Metadata, out,
                      max_active_clusters=None):
        return self._binding.nvfp4_fc1_out(
            a, b, sfa, sfb, metadata.source_rows, metadata.tasks, metadata.state,
            out, self._address, self._max_active_clusters(a.device, max_active_clusters))

    def nvfp4_fc1(self, a, b, sfa, sfb, metadata: Nvfp4Fc1Metadata, max_active_clusters=None):
        import torch
        out = torch.empty(
            (metadata.source_rows.numel(), b.size(1) // 2),
            device=a.device, dtype=torch.bfloat16)
        return self.nvfp4_fc1_out(a, b, sfa, sfb, metadata, out, max_active_clusters)


def _flags(config: Nvfp4Fc1Config, variant):
    return tuple(f"-DSM110A_NVFP4_FC1_{key}={value}" for key, value in (
        ("TILE_M", config.tile_m), ("TILE_N", config.tile_n),
        ("TILE_K", config.tile_k), ("EPILOGUE_TILE_N", config.epilogue_tile_n),
        ("NUM_C_STAGE", config.num_c_stage),
        ("CLUSTER_M", config.cluster_m), ("CLUSTER_N", config.cluster_n),
        ("DIRECT_GMEM", int(config.direct_gmem)),
        ("MACRO_SWIZZLE_N_MAJOR", int(config.macro_swizzle_n_major))))


_JIT = SplitCudaJit(
    name="sm110a_nvfp4_fc1", root=_ROOT,
    binding_sources=(_SRC / "nvfp4_fc1_torch_binding.cpp",),
    kernel_sources=(_SRC / "nvfp4_fc1_kernel.cu", _SRC / "nvfp4_fc1_metadata.cu",
                    _SRC / "nvfp4_fc1_routing.cu"),
    kernel_dependencies=(_INC / "nvfp4_moe_fc1.h", _INC / "common.h", _SRC / "nvfp4_fc1_pipeline_delay.cuh"),
    kernel_symbol="sm110a_nvfp4_fc1_launch", config_key=lambda config: config,
    kernel_cflags=_flags, make_facade=Nvfp4Fc1Module, include_paths=(_INC,))


def compile_nvfp4_fc1(config: Optional[Nvfp4Fc1Config] = None):
    config = config or Nvfp4Fc1Config()
    if not isinstance(config, Nvfp4Fc1Config):
        raise TypeError("config must be an Nvfp4Fc1Config")
    config.validate()
    return _JIT.compile(config)


def compile_nvfp4_fc1_configs(configs: Iterable[Nvfp4Fc1Config], max_workers=None):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Nvfp4Fc1Config):
            raise TypeError("configs must contain Nvfp4Fc1Config values")
        config.validate()
    return _JIT.compile_many(configs, max_workers) if configs else ()


def prepare_nvfp4_fc1_metadata(router_scores, topk, n, config=None):
    return compile_nvfp4_fc1(config).prepare_metadata(router_scores, topk, n)


def nvfp4_fc1(a, b, sfa, sfb, metadata, config=None, max_active_clusters=None):
    return compile_nvfp4_fc1(config).nvfp4_fc1(a, b, sfa, sfb, metadata, max_active_clusters)


def nvfp4_fc1_out(a, b, sfa, sfb, metadata, out, config=None, max_active_clusters=None):
    return compile_nvfp4_fc1(config).nvfp4_fc1_out(a, b, sfa, sfb, metadata, out, max_active_clusters)
