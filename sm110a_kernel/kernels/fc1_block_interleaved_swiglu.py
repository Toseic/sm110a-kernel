"""Configuration-specialized routed FC1 block-interleaved SwiGLU kernel."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..jit_runtime import SplitCudaJit
from ..runner import BaseKernelRunner, PreparedKernel

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16_moe/src"
_INCLUDE_DIR = _ROOT / "include"


@dataclass(frozen=True)
class Fc1BlockInterleavedSwiGLUConfig:
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 128
    cluster_m: int = 1
    cluster_n: int = 1
    direct_gmem: bool = False

    def validate(self) -> None:
        if self.tile_m != 128:
            raise ValueError("tile_m must be 128 for the current four-warp mainloop")
        if self.cluster_m not in (1, 2) or self.cluster_n not in (1, 2):
            raise ValueError("cluster_m and cluster_n must be 1 or 2")
        if self.cluster_n == 2 and self.cluster_m != 2:
            raise ValueError("cluster_n=2 requires cluster_m=2")
        granularity = 8 * self.cluster_m
        if self.tile_n < granularity or self.tile_n > 256 or self.tile_n % granularity:
            raise ValueError(
                f"tile_n must be at most 256 and a positive multiple of {granularity} for cluster_m={self.cluster_m}"
            )
        if self.tile_n % 2 or self.tile_n % 64 or self.tile_n < 64:
            raise ValueError("tile_n must be an even multiple of 64 and at least 64")
        if self.cluster_n == 2 and (self.tile_n // 2) % self.cluster_n:
            raise ValueError("tile_n / 2 must be divisible by cluster_n")
        if self.tile_k <= 0 or self.tile_k % 64:
            raise ValueError("tile_k must be a positive multiple of 64")


@dataclass(frozen=True)
class Fc1Metadata:
    routing: object
    source_rows: object
    tasks: object
    state: object


class Fc1BlockInterleavedSwiGLUModule:
    def __init__(self, binding, library, config: Fc1BlockInterleavedSwiGLUConfig):
        self._binding = binding
        self._launcher_address = library.entry_address
        self._metadata_address = ctypes.cast(
            getattr(library.handle, "sm110a_fc1_metadata_prepare"), ctypes.c_void_p
        ).value
        self._main_address = ctypes.cast(
            getattr(library.handle, "sm110a_fc1_main_launch"), ctypes.c_void_p
        ).value
        self._config = config

    def prepare_metadata(self, router_scores, topk, n):
        routing, source_rows, tasks, state = self._binding.prepare_fc1_metadata(
            router_scores,
            topk,
            n,
            self._metadata_address,
            self._config.tile_m,
            self._config.tile_n,
            self._config.cluster_m,
            self._config.cluster_n,
        )
        return Fc1Metadata(routing, source_rows, tasks, state)

    def fc1_block_interleaved_swiglu_out(
        self, a, b_interleaved, metadata: Fc1Metadata, out
    ):
        return self._binding.fc1_main_out(
            a,
            b_interleaved,
            metadata.source_rows,
            metadata.tasks,
            metadata.state,
            out,
            self._main_address,
        )

    def fc1_block_interleaved_swiglu(self, a, b_interleaved, metadata: Fc1Metadata):
        out = a.new_empty((metadata.source_rows.numel(), b_interleaved.size(1) // 2))
        return self.fc1_block_interleaved_swiglu_out(a, b_interleaved, metadata, out)


def _kernel_cflags(config: Fc1BlockInterleavedSwiGLUConfig, variant):
    return (
        f"-DSM110A_FC1_TILE_M={config.tile_m}",
        f"-DSM110A_FC1_TILE_N={config.tile_n}",
        f"-DSM110A_FC1_TILE_K={config.tile_k}",
        f"-DSM110A_FC1_CLUSTER_M={config.cluster_m}",
        f"-DSM110A_FC1_CLUSTER_N={config.cluster_n}",
        f"-DSM110A_FC1_DIRECT_GMEM={int(config.direct_gmem)}",
    )


_JIT = SplitCudaJit(
    name="sm110a_fc1_moe",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16_moe_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16_fc1_block_interleaved_swiglu_kernel.cu",),
    kernel_dependencies=(_INCLUDE_DIR / "bf16_moe_fc1.h", _INCLUDE_DIR / "common.h"),
    kernel_symbol="sm110a_fc1_main_launch",
    config_key=lambda config: config,
    kernel_cflags=_kernel_cflags,
    make_facade=Fc1BlockInterleavedSwiGLUModule,
    include_paths=(_INCLUDE_DIR,),
)


def compile_fc1_block_interleaved_swiglu(
    config: Fc1BlockInterleavedSwiGLUConfig | None = None,
):
    config = config or Fc1BlockInterleavedSwiGLUConfig()
    if not isinstance(config, Fc1BlockInterleavedSwiGLUConfig):
        raise TypeError("config must be a Fc1BlockInterleavedSwiGLUConfig")
    config.validate()
    return _JIT.compile(config)


def compile_fc1_block_interleaved_swiglu_configs(
    configs: Iterable[Fc1BlockInterleavedSwiGLUConfig], max_workers=None
):
    configs = tuple(configs)
    if not configs:
        return ()
    for config in configs:
        if not isinstance(config, Fc1BlockInterleavedSwiGLUConfig):
            raise TypeError(
                "configs must contain Fc1BlockInterleavedSwiGLUConfig values"
            )
        config.validate()
    return _JIT.compile_many(configs, max_workers)


def get_fc1_block_interleaved_swiglu_module(a, b_interleaved, config=None):
    return compile_fc1_block_interleaved_swiglu(config)


def prepare_fc1_metadata(router_scores, topk, n, config=None):
    return compile_fc1_block_interleaved_swiglu(config).prepare_metadata(
        router_scores, topk, n
    )


def fc1_block_interleaved_swiglu_out(a, b_interleaved, metadata, out, config=None):
    return get_fc1_block_interleaved_swiglu_module(
        a, b_interleaved, config
    ).fc1_block_interleaved_swiglu_out(a, b_interleaved, metadata, out)


def fc1_block_interleaved_swiglu(a, b_interleaved, metadata, config=None):
    return get_fc1_block_interleaved_swiglu_module(
        a, b_interleaved, config
    ).fc1_block_interleaved_swiglu(a, b_interleaved, metadata)


INTERLEAVE_BLOCK = 32
FC1_CONFIGS = tuple(
    Fc1BlockInterleavedSwiGLUConfig(
        tile_m=128,
        tile_n=tile_n,
        tile_k=tile_k,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        direct_gmem=direct_gmem,
    )
    for tile_n in (64, 128, 256)
    for tile_k in (64, 128)
    for cluster_m in (1, 2)
    for cluster_n in (1, 2)
    for direct_gmem in (False, True)
    if tile_n % (8 * cluster_m) == 0
    and tile_n % (2 * INTERLEAVE_BLOCK) == 0
    and (cluster_n == 1 or cluster_m == 2)
)


def compatible_candidates(n, k):
    return tuple(
        config
        for config in FC1_CONFIGS
        if n % (config.tile_n // 2) == 0 and k % config.tile_k == 0
    )


def format_candidate(config):
    epilogue = "direct-gmem" if config.direct_gmem else "tmem-smem-tma"
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}/c{config.cluster_m}n{config.cluster_n}/"
        f"{epilogue}"
    )


class Fc1BlockInterleavedSwiGLURunner(BaseKernelRunner):
    namespace = "fc1_block_interleaved_swiglu_runner_v1"
    format_candidate = staticmethod(format_candidate)

    def __init__(self, a, b, router, topk, out, *, dynamic_metadata=True):
        super().__init__(a.device)
        self.a, self.b, self.router, self.topk, self.out = a, b, router, topk, out
        self.dynamic_metadata = dynamic_metadata
        self.metadata_by_candidate = {}
        if dynamic_metadata:
            self.namespace += "_dynamic"
            self.graph_timing = True

    @property
    def workload_key(self):
        return (tuple(self.a.shape), tuple(self.b.shape), self.topk)

    def compatible_candidates(self):
        return compatible_candidates(self.b.size(1) // 2, self.a.size(1))

    def pre_jit(self, candidate):
        return compile_fc1_block_interleaved_swiglu(candidate)

    def prepare(self, candidate):
        module = self.pre_jit(candidate)

        def metadata():
            return module.prepare_metadata(self.router, self.topk, self.out.size(1))

        md = None if self.dynamic_metadata else metadata()
        if md is not None:
            self.metadata_by_candidate[candidate] = md

        def run():
            module.fc1_block_interleaved_swiglu_out(
                self.a, self.b, metadata() if self.dynamic_metadata else md, self.out
            )
            return self.out

        return PreparedKernel(candidate, run)
