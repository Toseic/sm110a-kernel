"""NVFP4 routed FC2 with token-major routing reduction."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..jit_runtime import SplitCudaJit
from ..runner import BaseKernelRunner, PreparedKernel

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "csrc/nvfp4_moe/src"
_META = _ROOT / "csrc/bf16_grouped_m_gemm/src/bf16_grouped_m_gemm_metadata.cu"
_INC = _ROOT / "include"


@dataclass(frozen=True)
class Nvfp4Fc2Config:
    tile_m: int = 128
    # FC2 is launched persistently.  A wider N tile and a 256-wide K stage
    # amortize the task scheduler, TMA, and scale-copy work while preserving
    # the intended one-CTA-per-SM execution model.
    tile_n: int = 256
    tile_k: int = 256
    epilogue_tile_n: int = 32
    cluster_m: int = 1
    reduce_tile_n: int = 64
    macro_swizzle_n_major: bool = True
    macro_swizzle_serpentine: bool = False
    macro_swizzle_m: int = 8
    macro_swizzle_n: int = 8

    # FC2 requires prepare_pdl_out before the producer launch.
    pdl: bool = False
    # before_a overlaps independent B/SFB loads before the producer wait.
    pdl_wait: str = "init"
    # Serial control: use the same pre-FC1 preparation without enabling PDL.
    external_prepare: bool = False

    def validate(self) -> None:
        if self.pdl_wait not in ("init", "before_a"):
            raise ValueError("pdl_wait must be init or before_a")
        if not isinstance(self.external_prepare, bool):
            raise ValueError("external_prepare must be bool")
        if not isinstance(self.pdl, bool):
            raise ValueError("pdl must be bool")
        if self.tile_m != 128 or self.tile_n not in (64, 128, 256):
            raise ValueError("tile_m must be 128 and tile_n must be 64, 128, or 256")
        if self.tile_k not in (64, 128, 256):
            raise ValueError("tile_k must be 64, 128, or 256")
        if (
            self.epilogue_tile_n not in (32, 64, 128)
            or self.tile_n % self.epilogue_tile_n
        ):
            raise ValueError("invalid epilogue_tile_n")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.reduce_tile_n % 8 or self.tile_n % self.reduce_tile_n:
            raise ValueError("reduce_tile_n must be a multiple of 8 dividing tile_n")
        if not isinstance(self.macro_swizzle_n_major, bool) or not isinstance(
            self.macro_swizzle_serpentine, bool
        ):
            raise ValueError("macro swizzle options must be bool")
        if self.macro_swizzle_m not in (1, 2, 4, 8, 16):
            raise ValueError("macro_swizzle_m must be 1, 2, 4, 8, or 16")
        if self.macro_swizzle_n not in (1, 2, 4, 8, 16):
            raise ValueError("macro_swizzle_n must be 1, 2, 4, 8, or 16")


@dataclass(frozen=True)
class Nvfp4Fc2Metadata:
    offsets: object
    counts: object
    tasks: object
    state: object


class Nvfp4Fc2Module:
    def __init__(self, binding, library, config: Nvfp4Fc2Config):
        self._binding = binding
        self._address = library.entry_address
        self.config = config
        self._metadata_address = ctypes.cast(
            getattr(library.handle, "sm110a_bf16_grouped_m_metadata_prepare"),
            ctypes.c_void_p,
        ).value

    def prepare_metadata(self, grouped_layout, experts: int, n: int):
        values = self._binding.prepare_nvfp4_fc2_metadata(
            grouped_layout,
            experts,
            n,
            self._metadata_address,
            self.config.tile_m,
            self.config.tile_n,
            self.config.cluster_m,
        )
        return Nvfp4Fc2Metadata(*values)

    def _max_active_clusters(self, device, value):
        if value is None:
            import torch

            value = (
                torch.cuda.get_device_properties(device).multi_processor_count
                // self.config.cluster_m
            )
        if value <= 0:
            raise ValueError("max_active_clusters must be positive")
        return int(value)

    def prepare_pdl_out(self, out, metadata):
        """Reset FC2 state/output on the current stream BEFORE launching FC1.

        Required once per PDL invocation (including each CUDA graph replay).
        Keep preparation, FC1 and FC2 on the same stream with adjacent kernels.
        """
        if not (self.config.pdl or self.config.external_prepare):
            raise ValueError(
                "prepare_pdl_out requires pdl=True or external_prepare=True"
            )
        self._binding.prepare_pdl_out(
            out, metadata.state, not getattr(self.config, "deterministic", False)
        )

    def nvfp4_fc2(
        self,
        a,
        b,
        sfa,
        sfb,
        routing,
        routing_weights,
        metadata,
        topk,
        max_active_clusters=None,
    ):
        if self.config.pdl or self.config.external_prepare:
            raise ValueError("PDL requires the out API and prepare_pdl_out before FC1")
        return self._binding.nvfp4_fc2(
            a,
            b,
            sfa,
            sfb,
            routing,
            routing_weights,
            metadata.tasks,
            metadata.state,
            topk,
            self._address,
            self._max_active_clusters(a.device, max_active_clusters),
        )

    def nvfp4_fc2_out(
        self,
        a,
        b,
        sfa,
        sfb,
        routing,
        routing_weights,
        metadata,
        topk,
        out,
        max_active_clusters=None,
    ):
        return self._binding.nvfp4_fc2_out(
            a,
            b,
            sfa,
            sfb,
            routing,
            routing_weights,
            metadata.tasks,
            metadata.state,
            topk,
            out,
            self._address,
            self._max_active_clusters(a.device, max_active_clusters),
        )


def _flags(config: Nvfp4Fc2Config, _variant):
    return tuple(
        f"-DSM110A_NVFP4_FC2_{key}={value}"
        for key, value in (
            ("PDL", int(config.pdl)),
            ("EXTERNAL_PREPARE", int(config.pdl or config.external_prepare)),
            ("PDL_WAIT_BEFORE_A", int(config.pdl_wait == "before_a")),
            ("TILE_M", config.tile_m),
            ("TILE_N", config.tile_n),
            ("TILE_K", config.tile_k),
            ("EPILOGUE_TILE_N", config.epilogue_tile_n),
            ("CLUSTER_M", config.cluster_m),
            ("REDUCE_TILE_N", config.reduce_tile_n),
        )
    ) + (
        # The grouped-M task builder is compiled into this module too; it keys
        # macro-tile traversal on the NVFP4_GROUPED flags.
        f"-DSM110A_NVFP4_GROUPED_MACRO_SWIZZLE_N_MAJOR={int(config.macro_swizzle_n_major)}",
        f"-DSM110A_NVFP4_GROUPED_MACRO_SWIZZLE_SERPENTINE={int(config.macro_swizzle_serpentine)}",
        f"-DSM110A_NVFP4_GROUPED_MACRO_SWIZZLE_M={config.macro_swizzle_m}",
        f"-DSM110A_NVFP4_GROUPED_MACRO_SWIZZLE_N={config.macro_swizzle_n}",
    )


_JIT = SplitCudaJit(
    name="sm110a_nvfp4_fc2",
    root=_ROOT,
    binding_sources=(_SRC / "nvfp4_fc2_torch_binding.cpp",),
    kernel_sources=(_SRC / "nvfp4_fc2_kernel.cu", _META),
    kernel_dependencies=(_INC / "nvfp4_moe_fc2.h", _INC / "common.h", _INC / "pdl.h"),
    kernel_symbol="sm110a_nvfp4_fc2_launch",
    config_key=lambda config: config,
    kernel_cflags=_flags,
    make_facade=Nvfp4Fc2Module,
    include_paths=(_INC,),
)


def compile_nvfp4_fc2(config: Nvfp4Fc2Config | None = None):
    config = config or Nvfp4Fc2Config()
    if not isinstance(config, Nvfp4Fc2Config):
        raise TypeError("config must be an Nvfp4Fc2Config")
    config.validate()
    return _JIT.compile(config)


def compile_nvfp4_fc2_configs(configs: Iterable[Nvfp4Fc2Config], max_workers=None):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Nvfp4Fc2Config):
            raise TypeError("configs must contain Nvfp4Fc2Config values")
        config.validate()
    return _JIT.compile_many(configs, max_workers) if configs else ()


def prepare_nvfp4_fc2_metadata(grouped_layout, experts, n, config=None):
    return compile_nvfp4_fc2(config).prepare_metadata(grouped_layout, experts, n)


def nvfp4_fc2(
    a,
    b,
    sfa,
    sfb,
    routing,
    routing_weights,
    metadata,
    topk,
    config=None,
    max_active_clusters=None,
):
    return compile_nvfp4_fc2(config).nvfp4_fc2(
        a,
        b,
        sfa,
        sfb,
        routing,
        routing_weights,
        metadata,
        topk,
        max_active_clusters,
    )


def nvfp4_fc2_out(
    a,
    b,
    sfa,
    sfb,
    routing,
    routing_weights,
    metadata,
    topk,
    out,
    config=None,
    max_active_clusters=None,
):
    return compile_nvfp4_fc2(config).nvfp4_fc2_out(
        a,
        b,
        sfa,
        sfb,
        routing,
        routing_weights,
        metadata,
        topk,
        out,
        max_active_clusters,
    )


NVFP4_FC2_CONFIGS = tuple(
    Nvfp4Fc2Config(
        tile_n=tile_n,
        tile_k=tile_k,
        epilogue_tile_n=epilogue_tile_n,
        cluster_m=cluster_m,
    )
    for tile_n in (64, 128, 256)
    for tile_k in (64, 128, 256)
    for epilogue_tile_n in (32, 64, 128)
    for cluster_m in (1, 2)
    if tile_n % epilogue_tile_n == 0
)


def format_candidate(config):
    order = "n" if config.macro_swizzle_n_major else "m"
    if config.macro_swizzle_serpentine:
        order += "s"
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/e{config.epilogue_tile_n}/c{config.cluster_m}/persistent"
        f"/r{config.reduce_tile_n}/{order}{config.macro_swizzle_m}x{config.macro_swizzle_n}"
    )


def nvfp4_fc2_candidates(n, k):
    return tuple(
        c
        for c in NVFP4_FC2_CONFIGS
        if n % c.tile_n == 0 and n % 128 == 0 and k % c.tile_k == 0
    )


class Nvfp4Fc2Runner(BaseKernelRunner):
    namespace = "nvfp4_fc2_runner_v1"
    format_candidate = staticmethod(format_candidate)

    def __init__(
        self,
        a,
        b,
        sfa,
        sfb,
        routing,
        routing_weights,
        layout,
        topk,
        out,
        *,
        dynamic_metadata=True,
    ):
        super().__init__(a.device)
        (
            self.a,
            self.b,
            self.sfa,
            self.sfb,
            self.routing,
            self.routing_weights,
            self.layout,
            self.topk,
            self.out,
        ) = a, b, sfa, sfb, routing, routing_weights, layout, topk, out
        self.dynamic_metadata = dynamic_metadata
        self.metadata_by_candidate = {}
        if dynamic_metadata:
            self.namespace += "_dynamic_v1"
            self.graph_timing = True

    @property
    def workload_key(self):
        return (tuple(self.a.shape), tuple(self.b.shape), self.topk)

    def compatible_candidates(self):
        return nvfp4_fc2_candidates(self.b.size(1), self.a.size(1) * 2)

    def pre_jit(self, candidate):
        return compile_nvfp4_fc2(candidate)

    def prepare(self, candidate):
        module = self.pre_jit(candidate)

        def metadata():
            return module.prepare_metadata(self.layout, self.b.size(0), self.b.size(1))

        md = None if self.dynamic_metadata else metadata()
        if md is not None:
            self.metadata_by_candidate[candidate] = md

        def run():
            module.nvfp4_fc2_out(
                self.a,
                self.b,
                self.sfa,
                self.sfb,
                self.routing,
                self.routing_weights,
                metadata() if self.dynamic_metadata else md,
                self.topk,
                self.out,
            )
            return self.out

        return PreparedKernel(candidate, run)
