"""Fused routed FC2 and token-finalize kernel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16_moe/src"
_INCLUDE_DIR = _ROOT / "include"


@dataclass(frozen=True)
class Fc2FinalizeConfig:
    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 128
    epilogue_tile_n: int = 32
    cluster_m: int = 2
    reduce_tile_n: int = 64

    def validate(self) -> None:
        if self.tile_m != 128:
            raise ValueError("tile_m must be 128")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.tile_n < 8 * self.cluster_m or self.tile_n > 256 or self.tile_n % (8 * self.cluster_m):
            raise ValueError("tile_n must be a supported tcgen05 tile width")
        if self.epilogue_tile_n not in (32, 64, 128) or self.tile_n % self.epilogue_tile_n:
            raise ValueError("epilogue_tile_n must divide tile_n and be 32, 64, or 128")
        if self.tile_k <= 0 or self.tile_k % 64:
            raise ValueError("tile_k must be a positive multiple of 64")
        if self.reduce_tile_n != 64:
            raise ValueError("the fused FC2 epilogue currently requires reduce_tile_n=64")
        if self.tile_n % self.reduce_tile_n:
            raise ValueError("tile_n must be divisible by reduce_tile_n")


class Fc2FinalizeModule:
    def __init__(self, binding, library, config: Fc2FinalizeConfig):
        self._binding = binding
        self._launcher_address = library.entry_address
        self._config = config

    def fc2_finalize(self, fc1, b, router_scores, topk):
        return self._binding.fc2_finalize(fc1, b, router_scores, topk, self._launcher_address,
                                          self._config.tile_n, self._config.tile_k)

    def fc2_finalize_out(self, fc1, b, router_scores, topk, out):
        return self._binding.fc2_finalize_out(fc1, b, router_scores, topk, out,
                                              self._launcher_address, self._config.tile_n,
                                              self._config.tile_k)

    def fc2_finalize_with_routing_out(self, fc1, b, topk, routing_weights, routing, grouped_metadata, out):
        return self._binding.fc2_finalize_with_routing_out(
            fc1, b, topk, routing_weights, routing, grouped_metadata.tasks, grouped_metadata.state, out, self._launcher_address,
            self._config.tile_n, self._config.tile_k)


def _kernel_cflags(config: Fc2FinalizeConfig, _variant):
    return (
        f"-DSM110A_GROUPED_TILE_M={config.tile_m}",
        f"-DSM110A_GROUPED_TILE_N={config.tile_n}",
        f"-DSM110A_GROUPED_TILE_K={config.tile_k}",
        f"-DSM110A_GROUPED_EPILOGUE_TILE_N={config.epilogue_tile_n}",
        f"-DSM110A_GROUPED_CLUSTER_M={config.cluster_m}",
        f"-DSM110A_FC2_REDUCE_TILE_N={config.reduce_tile_n}",
    )


_JIT = SplitCudaJit(
    name="sm110a_fc2_finalize",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16_fc2_finalize_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16_fc2_finalize_kernel.cu",),
    kernel_dependencies=(
        _INCLUDE_DIR / "bf16_moe_fc2_finalize.h",
        _INCLUDE_DIR / "common.h",
    ),
    kernel_symbol="sm110a_fc2_finalize_launch",
    config_key=lambda config: config,
    kernel_cflags=_kernel_cflags,
    make_facade=Fc2FinalizeModule,
    include_paths=(_INCLUDE_DIR,),
)


def compile_fc2_finalize(config: Optional[Fc2FinalizeConfig] = None):
    config = config or Fc2FinalizeConfig()
    if not isinstance(config, Fc2FinalizeConfig):
        raise TypeError("config must be a Fc2FinalizeConfig")
    config.validate()
    return _JIT.compile(config)


def compile_fc2_finalize_configs(
    configs: Iterable[Fc2FinalizeConfig], max_workers=None
):
    configs = tuple(configs)
    if not configs:
        return ()
    for config in configs:
        if not isinstance(config, Fc2FinalizeConfig):
            raise TypeError("configs must contain Fc2FinalizeConfig values")
        config.validate()
    return _JIT.compile_many(configs, max_workers)


def get_fc2_finalize_module(_fc1, _b, config=None):
    return compile_fc2_finalize(config)


def fc2_finalize(fc1, b, router_scores, topk, config=None):
    return get_fc2_finalize_module(fc1, b, config).fc2_finalize(fc1, b, router_scores, topk)


def fc2_finalize_out(fc1, b, router_scores, topk, out, config=None):
    return get_fc2_finalize_module(fc1, b, config).fc2_finalize_out(fc1, b, router_scores, topk, out)


def fc2_finalize_with_routing_out(fc1, b, topk, routing_weights, routing, grouped_metadata, out, config=None):
    """Run FC2 with ``routing_weights[t, rank] * expert_output[t, rank]`` reduction."""
    return get_fc2_finalize_module(fc1, b, config).fc2_finalize_with_routing_out(
        fc1, b, topk, routing_weights, routing, grouped_metadata, out)
