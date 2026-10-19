"""MXFP4 (W4A8) routed FC2 with token-major routing reduction."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import BaseKernelRunner, PreparedKernel

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "csrc/mxfp4_moe/src"
_META = _ROOT / "csrc/bf16_grouped_m_gemm/src/bf16_grouped_m_gemm_metadata.cu"
_INC = _ROOT / "include"


@dataclass(frozen=True)
class Mxfp4Fc2Config:
    """Compile-time weighting and accumulation semantics.

    ``routing_weight_position='fc1'`` disables the FC2 router multiply.
    ``deterministic=True`` stores each assignment once in BF16 scratch and
    sums ranks in FP32 with one final BF16 rounding. It requires an extra
    ``tokens * topk * n * 2`` bytes and one finalize kernel. The default
    in-place BF16 atomic reduction has schedule-dependent rounding.
    """

    tile_m: int = 128
    # FC2 is launched persistently.  A wider N tile and a 128-wide K stage
    # amortize the task scheduler, TMA, and scale-copy work while preserving
    # the intended one-CTA-per-SM execution model.  K=128 is the FP8 maximum:
    # activations ride a 128B-swizzle UMMA atom.
    tile_n: int = 256
    tile_k: int = 128
    epilogue_tile_n: int = 32
    # The task metadata is specialized for the cluster shape.  The benchmark
    # prepares it per candidate, so both one- and two-CTA paths are valid.
    cluster_m: int = 1
    reduce_tile_n: int = 64
    macro_swizzle_n_major: bool = True
    macro_swizzle_serpentine: bool = False
    macro_swizzle_m: int = 8
    macro_swizzle_n: int = 8
    routing_weight_position: str = "fc2"
    deterministic: bool = False

    # FC2 requires prepare_pdl_out before the producer launch.
    pdl: bool = False
    # before_a overlaps independent B/SFB loads before the producer wait.
    pdl_wait: str = "init"
    # Serial control: use the same pre-FC1 preparation without enabling PDL.
    external_prepare: bool = False

    # SFB only: pack once with compact_mxfp4_weight_scales; SFA stays padded.
    compact_weight_scales: bool = False

    def validate(self) -> None:
        if not isinstance(self.compact_weight_scales, bool):
            raise ValueError("compact_weight_scales must be bool")
        if self.pdl_wait not in ("init", "before_a"):
            raise ValueError("pdl_wait must be init or before_a")
        if not isinstance(self.external_prepare, bool):
            raise ValueError("external_prepare must be bool")
        if not isinstance(self.pdl, bool):
            raise ValueError("pdl must be bool")
        if self.routing_weight_position not in ("fc1", "fc2"):
            raise ValueError("routing_weight_position must be 'fc1' or 'fc2'")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be bool")
        # TileN=256 is the only shape validated against the torch reference
        # for the W4A8 kernel (the 128-column TMEM scale-block offset of
        # narrower tiles is untested here).
        if self.tile_m != 128 or self.tile_n != 256:
            raise ValueError("tile_m must be 128 and tile_n must be 256")
        if self.tile_k != 128:
            raise ValueError("tile_k must be 128 (FP8 swizzle atom + b4x16_p64 box)")
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
class Mxfp4Fc2Metadata:
    offsets: object
    counts: object
    tasks: object
    state: object


class Mxfp4Fc2Module:
    def __init__(self, binding, library, config: Mxfp4Fc2Config):
        self._binding = binding
        self._address = library.entry_address
        self.config = config
        self._metadata_address = ctypes.cast(
            getattr(library.handle, "sm110a_bf16_grouped_m_metadata_prepare"),
            ctypes.c_void_p,
        ).value

    def prepare_metadata(self, grouped_layout, experts: int, n: int):
        values = self._binding.prepare_mxfp4_fc2_metadata(
            grouped_layout,
            experts,
            n,
            self._metadata_address,
            self.config.tile_m,
            self.config.tile_n,
            self.config.cluster_m,
        )
        return Mxfp4Fc2Metadata(*values)

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

    def mxfp4_fc2(
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
        return self._binding.mxfp4_fc2(
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
            self.config.deterministic,
            self.config.compact_weight_scales,
        )

    def mxfp4_fc2_out(
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
        return self._binding.mxfp4_fc2_out(
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
            self.config.deterministic,
            self.config.compact_weight_scales,
        )


def _flags(config: Mxfp4Fc2Config, _variant):
    return tuple(
        f"-DSM110A_MXFP4_FC2_{key}={value}"
        for key, value in (
            ("COMPACT_WEIGHT_SCALES", int(config.compact_weight_scales)),
            ("PDL", int(config.pdl)),
            ("EXTERNAL_PREPARE", int(config.pdl or config.external_prepare)),
            ("PDL_WAIT_BEFORE_A", int(config.pdl_wait == "before_a")),
            ("TILE_M", config.tile_m),
            ("TILE_N", config.tile_n),
            ("TILE_K", config.tile_k),
            ("EPILOGUE_TILE_N", config.epilogue_tile_n),
            ("CLUSTER_M", config.cluster_m),
            ("REDUCE_TILE_N", config.reduce_tile_n),
            ("WEIGHT_IN_FC1", int(config.routing_weight_position == "fc1")),
            ("DETERMINISTIC", int(config.deterministic)),
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
    name="sm110a_mxfp4_fc2",
    root=_ROOT,
    binding_sources=(_SRC / "mxfp4_fc2_torch_binding.cpp",),
    kernel_sources=(_SRC / "mxfp4_fc2_kernel.cu", _META),
    kernel_dependencies=(
        _INC / "mxfp4_moe_fc2.h",
        _INC / "common.h",
        _INC / "pdl.h",
        _SRC / "mxfp4_weight_scales.cuh",
    ),
    kernel_symbol="sm110a_mxfp4_fc2_launch",
    config_key=lambda config: config,
    kernel_cflags=_flags,
    kernel_ldflags=("-lcuda",),
    make_facade=Mxfp4Fc2Module,
    include_paths=(_INC,),
)


def compile_mxfp4_fc2(config: Mxfp4Fc2Config | None = None):
    config = config or Mxfp4Fc2Config()
    if not isinstance(config, Mxfp4Fc2Config):
        raise TypeError("config must be an Mxfp4Fc2Config")
    config.validate()
    return _JIT.compile(config)


def compile_mxfp4_fc2_configs(configs: Iterable[Mxfp4Fc2Config], max_workers=None):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Mxfp4Fc2Config):
            raise TypeError("configs must contain Mxfp4Fc2Config values")
        config.validate()
    return _JIT.compile_many(configs, max_workers) if configs else ()


def to_blocked_mxfp4_sfa(scales):
    """Pack row-major E8M0 ``[rows, K/128]`` scales for the W4A8 MMA.

    The ``mxf8f6f4`` block-scaled MMA applies one UE8M0 scale per 32 K
    elements, while activations are quantized one scale per 128 elements.
    Each scale is therefore replicated across the four K=32 sub-blocks it
    covers, then stored in the padded 512-byte SWIZZLE_32_4_4 tile layout
    consumed by ``tcgen05.cp`` (live byte in column 0, as in
    ``fp8mxfp4_gemm``).
    """

    from .fp8mxfp4_gemm import to_blocked_fp8mxfp4_scales

    if scales.ndim != 2 or scales.dtype != torch.float8_e8m0fnu:
        raise TypeError("scales must be a 2D torch.float8_e8m0fnu tensor")
    rows, blocks = scales.shape
    if rows % 128 or blocks == 0:
        raise ValueError("scale rows must be divisible by 128")
    expanded = scales.view(torch.uint8).repeat_interleave(4, dim=1)
    return to_blocked_fp8mxfp4_scales(expanded.view(torch.float8_e8m0fnu).contiguous())


def prepare_mxfp4_fc2_metadata(grouped_layout, experts, n, config=None):
    return compile_mxfp4_fc2(config).prepare_metadata(grouped_layout, experts, n)


def mxfp4_fc2(
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
    return compile_mxfp4_fc2(config).mxfp4_fc2(
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


def mxfp4_fc2_out(
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
    return compile_mxfp4_fc2(config).mxfp4_fc2_out(
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


def to_blocked_mxfp4_scales(scales):
    """Pack logical E8M0 ``[rows, K/32]`` scales into compact 512B tiles.

    Each 512-byte tile covers 128 rows x four K=32 slices (a K=128 span);
    all four SWIZZLE_32_4_4 columns are live. Storage is one byte per
    (row, K=32) scale — four times smaller than the padded fp8mxfp4 layout.
    """

    if scales.ndim != 2 or scales.dtype != torch.float8_e8m0fnu:
        raise TypeError("scales must be a 2D torch.float8_e8m0fnu tensor")
    rows, cols = scales.shape
    if rows % 128 or cols % 4:
        raise ValueError("scale shape must be divisible by [128, 4]")
    grouped = scales.contiguous().view(torch.uint8).view(rows, cols // 4, 4)
    blocks = grouped.view(rows // 128, 128, cols // 4, 4).permute(0, 2, 1, 3)
    return (
        blocks.reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1)
        .view(torch.float8_e8m0fnu)
        .contiguous()
    )


def from_blocked_mxfp4_scales(scales, rows: int, k: int):
    """Inverse of :func:`to_blocked_mxfp4_scales`, mainly for references."""

    if scales.dtype != torch.float8_e8m0fnu or rows % 128 or k % 128:
        raise ValueError("expected E8M0 scales with rows % 128 == 0 and K % 128 == 0")
    tiles = scales.view(rows // 128, k // 128, 32, 4, 4)
    blocks = (
        tiles.view(torch.uint8).transpose(2, 3).reshape(rows // 128, k // 128, 128, 4)
    )
    return (
        blocks.permute(0, 2, 1, 3)
        .reshape(rows, k // 32)
        .view(torch.float8_e8m0fnu)
        .contiguous()
    )


MXFP4_FC2_CONFIGS = tuple(
    Mxfp4Fc2Config(
        tile_k=128,
        epilogue_tile_n=epilogue_tile_n,
        cluster_m=cluster_m,
    )
    for epilogue_tile_n in (32, 64)
    for cluster_m in (1, 2)
)


def format_fc2_candidate(config):
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/e{config.epilogue_tile_n}/c{config.cluster_m}/persistent"
        f"/r{config.reduce_tile_n}/sf={'compact' if getattr(config, 'compact_weight_scales', False) else 'padded'}"
    )


class Mxfp4Fc2Runner(BaseKernelRunner):
    namespace = "mxfp4_fc2"
    format_candidate = staticmethod(format_fc2_candidate)

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
        config=None,
        dynamic_metadata=True,
    ):
        super().__init__(a.device)
        self.a, self.b, self.sfa, self.sfb = a, b, sfa, sfb
        self.routing, self.routing_weights, self.layout = (
            routing,
            routing_weights,
            layout,
        )
        self.topk, self.out = topk, out
        self.dynamic_metadata = dynamic_metadata
        if dynamic_metadata:
            self.namespace += "_dynamic_v1"
            self.graph_timing = True
        self.config = config or Mxfp4Fc2Config()
        if self.config.pdl or self.config.external_prepare:
            raise ValueError("use Mxfp4MoeRunner to tune the complete PDL chain")

    @property
    def workload_key(self):
        return (
            "v3-runner",
            self.out.size(0),
            self.b.size(0),
            self.topk,
            self.b.size(1),
            self.a.size(1),
            self.config.compact_weight_scales,
            self.config.routing_weight_position,
            self.config.deterministic,
        )

    def compatible_candidates(self):
        return tuple(
            replace(
                c,
                compact_weight_scales=self.config.compact_weight_scales,
                routing_weight_position=self.config.routing_weight_position,
                deterministic=self.config.deterministic,
            )
            for c in MXFP4_FC2_CONFIGS
        )

    def pre_jit(self, candidate):
        return compile_mxfp4_fc2(candidate)

    def prepare(self, candidate):
        module = self.pre_jit(candidate)

        def prepare_metadata():
            return module.prepare_metadata(self.layout, self.b.size(0), self.b.size(1))

        metadata = None if self.dynamic_metadata else prepare_metadata()

        def run():
            current_metadata = prepare_metadata() if self.dynamic_metadata else metadata
            module.mxfp4_fc2_out(
                self.a,
                self.b,
                self.sfa,
                self.sfb,
                self.routing,
                self.routing_weights,
                current_metadata,
                self.topk,
                self.out,
            )
            return self.out

        return PreparedKernel(candidate, run)
