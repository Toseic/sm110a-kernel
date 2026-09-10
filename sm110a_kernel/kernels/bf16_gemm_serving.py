"""Configuration-specialized standard BF16 GEMM kernel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16_gemm_serving/src"
_INCLUDE_DIR = _ROOT / "include"


@dataclass(frozen=True)
class Bf16GemmServingConfig:
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


def _default_config(a, b) -> Bf16GemmServingConfig:
    if a.ndim != 2 or b.ndim != 2:
        return Bf16GemmServingConfig()
    return Bf16GemmServingConfig(cluster_m=1 if a.size(0) <= 128 else 2)


class Bf16GemmServingModule:
    def __init__(self, binding, library, config: Bf16GemmServingConfig):
        self._binding, self._launcher, self._config = binding, library.entry_address, config

    def _args(self, device, max_active_clusters=None):
        c = self._config
        if max_active_clusters is None:
            import torch
            sm_count = torch.cuda.get_device_properties(device).multi_processor_count
            max_active_clusters = sm_count // c.cluster_m
        if max_active_clusters <= 0:
            raise ValueError("max_active_clusters must be positive")
        return (self._launcher, c.tile_m, c.tile_n, c.tile_k, c.cluster_m,
                int(max_active_clusters))

    def bf16_gemm_serving(self, a, b, swap_ab=False, max_active_clusters=None):
        return self._binding.bf16_gemm_serving(a, b, *self._args(a.device, max_active_clusters), swap_ab)
    def bf16_gemm_serving_out(self, a, b, out, swap_ab=False, max_active_clusters=None):
        return self._binding.bf16_gemm_serving_out(a, b, out, *self._args(a.device, max_active_clusters), swap_ab)
    def bf16_gemm_serving_nt(self, a, b_nt, swap_ab=False, max_active_clusters=None):
        return self._binding.bf16_gemm_serving_nt(a, b_nt, *self._args(a.device, max_active_clusters), swap_ab)
    def bf16_gemm_serving_nt_out(self, a, b_nt, out, swap_ab=False, max_active_clusters=None):
        return self._binding.bf16_gemm_serving_nt_out(a, b_nt, out, *self._args(a.device, max_active_clusters), swap_ab)


def _kernel_cflags(config, _variant):
    return tuple(f"-DSM110A_BF16_{name}={value}" for name, value in (
        ("TILE_M", config.tile_m), ("TILE_N", config.tile_n),
        ("TILE_K", config.tile_k), ("CLUSTER_M", config.cluster_m),
        ("EPILOGUE_BLOCK_N", config.epilogue_block_n)))


_JIT = SplitCudaJit(
    name="sm110a_bf16_gemm_serving", root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16_gemm_serving_split_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16_gemm_serving_kernel.cu",),
    kernel_dependencies=(_INCLUDE_DIR / "bf16_gemm_serving.h", _INCLUDE_DIR / "common.h", _INCLUDE_DIR / "profiler.h"),
    kernel_symbol="sm110a_bf16_gemm_serving_launch", config_key=lambda config: config,
    kernel_cflags=_kernel_cflags, make_facade=Bf16GemmServingModule, include_paths=(_INCLUDE_DIR,))


def compile_bf16_gemm_serving(config: Optional[Bf16GemmServingConfig] = None) -> Bf16GemmServingModule:
    config = config or Bf16GemmServingConfig()
    if not isinstance(config, Bf16GemmServingConfig):
        raise TypeError("config must be a Bf16GemmServingConfig")
    config.validate()
    return _JIT.compile(config)


def compile_bf16_gemm_serving_configs(configs: Iterable[Bf16GemmServingConfig], max_workers=None):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Bf16GemmServingConfig):
            raise TypeError("configs must contain Bf16GemmServingConfig instances")
        config.validate()
    return _JIT.compile_many(configs, max_workers)


def get_bf16_gemm_serving_module(a, b, config=None):
    return compile_bf16_gemm_serving(config or _default_config(a, b))


def bf16_gemm_serving(a, b, config=None, swap_ab=False, max_active_clusters=None):
    return get_bf16_gemm_serving_module(a, b, config).bf16_gemm_serving(a, b, swap_ab, max_active_clusters)


def bf16_gemm_serving_out(a, b, out, config=None, swap_ab=False, max_active_clusters=None):
    return get_bf16_gemm_serving_module(a, b, config).bf16_gemm_serving_out(a, b, out, swap_ab, max_active_clusters)


def bf16_gemm_serving_nt_out(a, b_nt, out, config=None, swap_ab=False, max_active_clusters=None):
    """Compute ``a @ b_nt.T`` for a contiguous, pretransposed ``b_nt(N, K)``."""
    return compile_bf16_gemm_serving(config).bf16_gemm_serving_nt_out(
        a, b_nt, out, swap_ab, max_active_clusters)
