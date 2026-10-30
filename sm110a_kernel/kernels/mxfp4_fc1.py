"""MXFP4 (W4A8) routed FC1 with fused permutation and block-interleaved SwiGLU.

Weights follow the NVFP4 FC1 convention: one interleaved ``[experts, 2N, K/2]``
packed-E2M1 tensor whose physical rows alternate ``[gate:64, up:64]`` blocks.
The official DeepSeek-V4-Flash checkpoint stores W1 (gate) and W3 (up) as
separate tensors; :func:`quantize_mxfp4_fc1_weights` interleaves and
quantizes them into the packed form (plus blocked UE8M0 K=32 scales) this
kernel consumes.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import BaseKernelRunner, PreparedKernel
from .pdl import PDL_TRIGGER_POINTS

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "csrc/mxfp4_moe/src"
_INC = _ROOT / "include"


@dataclass(frozen=True)
class Mxfp4Fc1Config:
    """Compile-time FC1 epilogue semantics.

    ``routing_weight_position='fc1'`` multiplies SwiGLU by the FP32 router
    weight before activation quantization; FC2 must use the same position.
    ``activation_bf16_rounding=True`` rounds gate/up before SwiGLU and the
    weighted hidden before FP8 quantization, matching the official model.
    Defaults retain the original unweighted FP32 SwiGLU epilogue.
    """

    tile_m: int = 128
    # TileN is fixed at 256 (128 logical output columns): one accumulator
    # stage must cover exactly one 128-column E8M0 output-scale block.
    tile_n: int = 256
    # FP8 activations ride a 128B-swizzle UMMA atom, so K is capped at 128.
    tile_k: int = 128
    epilogue_tile_n: int = 128
    cluster_m: int = 1
    cluster_n: int = 1
    direct_gmem: bool = False
    macro_swizzle_n_major: bool = True
    routing_weight_position: str = "fc2"
    activation_bf16_rounding: bool = False

    # FC2 requires prepare_pdl_out before the producer launch.
    pdl: bool = False
    pdl_trigger: str = "entry"
    # Incoming quantizer dependency. Task metadata must be prepared first;
    # resetting the scheduler is the caller's responsibility on every replay.
    pdl_input: bool = False

    # SFB only: pack once with compact_mxfp4_weight_scales; SFA stays padded.
    compact_weight_scales: bool = False

    def validate(self) -> None:
        if not isinstance(self.compact_weight_scales, bool):
            raise ValueError("compact_weight_scales must be bool")
        if self.pdl_trigger not in PDL_TRIGGER_POINTS:
            raise ValueError(f"pdl_trigger must be one of {tuple(PDL_TRIGGER_POINTS)}")
        if not isinstance(self.pdl_input, bool):
            raise ValueError("pdl_input must be bool")
        if not isinstance(self.pdl, bool):
            raise ValueError("pdl must be bool")
        if self.tile_m != 128 or self.tile_n != 256:
            raise ValueError("tile_m must be 128 and tile_n must be 256")
        if self.tile_k != 128:
            raise ValueError("tile_k must be 128 (FP8 swizzle atom + b4x16_p64 box)")
        if self.epilogue_tile_n != 128:
            raise ValueError("epilogue_tile_n must be 128")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.cluster_n != 1:
            raise ValueError("cluster_n must be 1")
        if not isinstance(self.macro_swizzle_n_major, bool):
            raise ValueError("macro_swizzle_n_major must be bool")
        if self.routing_weight_position not in ("fc1", "fc2"):
            raise ValueError("routing_weight_position must be 'fc1' or 'fc2'")
        if not isinstance(self.activation_bf16_rounding, bool):
            raise ValueError("activation_bf16_rounding must be bool")


@dataclass(frozen=True)
class Mxfp4Fc1Metadata:
    """Device-resident routing and task state, aligned with BF16 FC1 metadata."""

    routing: object
    source_rows: object
    tasks: object
    state: object


def quantize_mxfp4_fc1_weights(gate, up):
    """Interleave separate W1/W3 and quantize to the packed MXFP4 FC1 layout.

    ``gate`` and ``up`` are ``[experts, n, k]`` BF16 tensors. Returns the
    packed E2M1 ``[experts, 2n, k/2]`` weight (physical
    ``[gate:64, up:64]`` block order, low nibble = even element) and the
    blocked UE8M0 K=32 scale tensor consumed as SFB.
    """

    from .fp8mxfp4_gemm import quantize_fp8mxfp4_weights

    if gate.ndim != 3 or up.shape != gate.shape:
        raise ValueError("gate and up must be [experts, n, k] tensors of equal shape")
    experts, n, k = gate.shape
    if n % 64 or k % 32:
        raise ValueError("n must be divisible by 64 and k by 32")
    bq = None
    sfb_parts = []
    for expert in range(experts):
        b_expert = torch.stack(
            (gate[expert].view(n // 64, 64, k), up[expert].view(n // 64, 64, k)),
            dim=1,
        ).reshape(2 * n, k)
        bq_expert, sfb_expert = quantize_fp8mxfp4_weights(b_expert)
        if bq is None:
            bq = torch.empty(
                (experts, 2 * n, k // 2),
                device=gate.device,
                dtype=torch.float4_e2m1fn_x2,
            )
        bq[expert].copy_(bq_expert.view(torch.float4_e2m1fn_x2))
        sfb_parts.append(sfb_expert.reshape(-1))
        del b_expert, bq_expert
    sfb = (
        torch.cat(sfb_parts)
        if experts
        else torch.empty((0,), device=gate.device, dtype=torch.float8_e8m0fnu)
    )
    return bq, sfb


class Mxfp4Fc1Module:
    def __init__(self, binding, library, config: Mxfp4Fc1Config):
        self._binding = binding
        self._address = library.entry_address
        self.config = config
        symbol = getattr(library.handle, "sm110a_mxfp4_fc1_tasks_prepare")
        self._metadata_address = ctypes.cast(symbol, ctypes.c_void_p).value
        routing_symbol = getattr(library.handle, "sm110a_mxfp4_fc1_routing_prepare")
        self._routing_address = ctypes.cast(routing_symbol, ctypes.c_void_p).value
        self._ids_address = ctypes.cast(
            getattr(library.handle, "sm110a_mxfp4_fc1_routing_from_ids"),
            ctypes.c_void_p,
        ).value

    def prepare_metadata_from_ids(self, expert_ids, experts: int, n: int):
        """Build routing from valid expert IDs, retaining token/rank order.

        IDs must be distinct per token and in [0, experts); no host
        synchronization validates values.
        The small-token path supports E <= 1024 and K <= 32.
        """
        values = self._binding.prepare_mxfp4_fc1_routing_from_ids(
            expert_ids,
            experts,
            n,
            self.config.tile_m,
            self.config.tile_n,
            self.config.cluster_m,
            self._ids_address,
        )
        return Mxfp4Fc1Metadata(*values)

    def prepare_metadata(self, router_scores, topk: int, n: int) -> Mxfp4Fc1Metadata:

        if router_scores.ndim != 2 or router_scores.dtype != torch.bfloat16:
            raise TypeError("router_scores must be a 2D BF16 tensor")
        tokens, experts = router_scores.shape
        if tokens % 128 or topk <= 0 or topk > experts:
            raise ValueError(
                "tokens must be divisible by 128 and topk must be in [1, experts]"
            )
        # ``n`` is the physical (gate+up interleaved) output width, matching
        # the NVFP4 FC1 metadata convention.
        if n <= 0 or n % (self.config.tile_n * self.config.cluster_n):
            raise ValueError("N is incompatible with the MXFP4 FC1 config")

        routing, source_rows, tasks, state = self._binding.prepare_mxfp4_fc1_routing(
            router_scores,
            topk,
            n,
            self.config.tile_m,
            self.config.tile_n,
            self.config.cluster_m,
            self._routing_address,
        )
        return Mxfp4Fc1Metadata(routing, source_rows, tasks, state)

    def prepare_pdl_input(self, metadata):
        """Reset the scheduler before a quantizer -> FC1 PDL chain.

        Call once per invocation (inside graph capture). Fresh prepare_metadata
        already performs this reset, so no extra call is needed in that case.
        """
        if not self.config.pdl_input:
            raise ValueError("prepare_pdl_input requires pdl_input=True")
        metadata.state.narrow(0, 1, 1).zero_()

    def _max_active_clusters(self, device, value):
        if value is None:
            import torch

            value = torch.cuda.get_device_properties(device).multi_processor_count // (
                self.config.cluster_m * self.config.cluster_n
            )
        if value <= 0:
            raise ValueError("max_active_clusters must be positive")
        return int(value)

    def mxfp4_fc1_out(
        self,
        a,
        b,
        sfa,
        sfb,
        metadata: Mxfp4Fc1Metadata,
        out,
        out_scale,
        swiglu_limit: float = 0.0,
        max_active_clusters=None,
        *,
        routing_weights=None,
    ):
        if self.config.routing_weight_position == "fc1" and routing_weights is None:
            raise ValueError(
                "FC1 weighting requires FP32 routing_weights [tokens,topk]"
            )
        return self._binding.mxfp4_fc1_out(
            a,
            b,
            sfa,
            sfb,
            metadata.source_rows,
            metadata.tasks,
            metadata.state,
            out,
            out_scale,
            float(swiglu_limit),
            self._address,
            self._max_active_clusters(a.device, max_active_clusters),
            metadata.routing,
            routing_weights,
            self.config.compact_weight_scales,
        )

    def mxfp4_fc1(
        self,
        a,
        b,
        sfa,
        sfb,
        metadata: Mxfp4Fc1Metadata,
        swiglu_limit: float = 0.0,
        max_active_clusters=None,
        *,
        routing_weights=None,
    ):

        m = metadata.source_rows.numel()
        logical_n = b.size(1) // 2
        out = torch.empty((m, logical_n), device=a.device, dtype=torch.float8_e4m3fn)
        out_scale = torch.empty(
            (m * (logical_n // 32) * 4,), device=a.device, dtype=torch.float8_e8m0fnu
        )
        self.mxfp4_fc1_out(
            a,
            b,
            sfa,
            sfb,
            metadata,
            out,
            out_scale,
            swiglu_limit,
            max_active_clusters,
            routing_weights=routing_weights,
        )
        return out, out_scale


def _flags(config: Mxfp4Fc1Config, variant):
    return tuple(
        f"-DSM110A_MXFP4_FC1_{key}={value}"
        for key, value in (
            ("COMPACT_WEIGHT_SCALES", int(config.compact_weight_scales)),
            ("PDL", int(config.pdl)),
            ("PDL_INPUT", int(config.pdl_input)),
            ("PDL_TRIGGER", PDL_TRIGGER_POINTS[config.pdl_trigger]),
            ("TILE_M", config.tile_m),
            ("TILE_N", config.tile_n),
            ("TILE_K", config.tile_k),
            ("EPILOGUE_TILE_N", config.epilogue_tile_n),
            ("CLUSTER_M", config.cluster_m),
            ("CLUSTER_N", config.cluster_n),
            ("DIRECT_GMEM", int(config.direct_gmem)),
            ("MACRO_SWIZZLE_N_MAJOR", int(config.macro_swizzle_n_major)),
            ("WEIGHT_IN_FC1", int(config.routing_weight_position == "fc1")),
            ("ACTIVATION_BF16_ROUNDING", int(config.activation_bf16_rounding)),
        )
    )


_JIT = SplitCudaJit(
    name="sm110a_mxfp4_fc1",
    root=_ROOT,
    binding_sources=(_SRC / "mxfp4_fc1_torch_binding.cpp",),
    kernel_sources=(
        _SRC / "mxfp4_fc1_kernel.cu",
        _SRC / "mxfp4_fc1_metadata.cu",
        _SRC / "mxfp4_fc1_routing.cu",
    ),
    kernel_dependencies=(
        _INC / "mxfp4_moe_fc1.h",
        _INC / "common.h",
        _INC / "pdl.h",
        _SRC / "mxfp4_fc1_pipeline_delay.cuh",
        _SRC / "mxfp4_weight_scales.cuh",
    ),
    kernel_symbol="sm110a_mxfp4_fc1_launch",
    config_key=lambda config: config,
    kernel_cflags=_flags,
    kernel_ldflags=("-lcuda",),
    make_facade=Mxfp4Fc1Module,
    include_paths=(_INC,),
)


def compile_mxfp4_fc1(config: Mxfp4Fc1Config | None = None):
    config = config or Mxfp4Fc1Config()
    if not isinstance(config, Mxfp4Fc1Config):
        raise TypeError("config must be an Mxfp4Fc1Config")
    config.validate()
    return _JIT.compile(config)


def compile_mxfp4_fc1_configs(configs: Iterable[Mxfp4Fc1Config], max_workers=None):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Mxfp4Fc1Config):
            raise TypeError("configs must contain Mxfp4Fc1Config values")
        config.validate()
    return _JIT.compile_many(configs, max_workers) if configs else ()


def prepare_mxfp4_fc1_metadata(router_scores, topk, n, config=None):
    return compile_mxfp4_fc1(config).prepare_metadata(router_scores, topk, n)


def mxfp4_fc1(
    a,
    b,
    sfa,
    sfb,
    metadata,
    swiglu_limit=0.0,
    config=None,
    max_active_clusters=None,
    *,
    routing_weights=None,
):
    return compile_mxfp4_fc1(config).mxfp4_fc1(
        a,
        b,
        sfa,
        sfb,
        metadata,
        swiglu_limit,
        max_active_clusters,
        routing_weights=routing_weights,
    )


def mxfp4_fc1_out(
    a,
    b,
    sfa,
    sfb,
    metadata,
    out,
    out_scale,
    swiglu_limit=0.0,
    config=None,
    max_active_clusters=None,
    *,
    routing_weights=None,
):
    return compile_mxfp4_fc1(config).mxfp4_fc1_out(
        a,
        b,
        sfa,
        sfb,
        metadata,
        out,
        out_scale,
        swiglu_limit,
        max_active_clusters,
        routing_weights=routing_weights,
    )


MXFP4_FC1_CONFIGS = tuple(
    Mxfp4Fc1Config(
        tile_k=128,
        cluster_m=cluster_m,
        direct_gmem=direct_gmem,
    )
    for cluster_m in (1, 2)
    for direct_gmem in (False, True)
)


def format_fc1_candidate(config):
    epilogue = "dg" if config.direct_gmem else "tma"
    return "/".join(
        [
            f"{config.tile_m}x{config.tile_n}x{config.tile_k}",
            f"e{config.epilogue_tile_n}",
            f"c{config.cluster_m}",
            epilogue,
            "compact" if getattr(config, "compact_weight_scales", False) else "padded",
        ]
    )


class Mxfp4Fc1Runner(BaseKernelRunner):
    namespace = "mxfp4_fc1"
    format_candidate = staticmethod(format_fc1_candidate)

    def __init__(
        self,
        a,
        b,
        sfa,
        sfb,
        router,
        topk,
        out,
        out_scale,
        *,
        swiglu_limit=0.0,
        routing_weights=None,
        config=None,
        dynamic_metadata=True,
    ):
        super().__init__(a.device)
        self.a, self.b, self.sfa, self.sfb = a, b, sfa, sfb
        self.router, self.topk, self.out, self.out_scale = router, topk, out, out_scale
        self.swiglu_limit, self.routing_weights = swiglu_limit, routing_weights
        self.dynamic_metadata = dynamic_metadata
        if dynamic_metadata:
            self.namespace += "_dynamic_v1"
            self.graph_timing = True
        self.config = config or Mxfp4Fc1Config()
        if self.config.pdl or self.config.pdl_input:
            raise ValueError("use Mxfp4MoeRunner to tune the complete PDL chain")
        self.metadata_by_candidate = {}

    @property
    def workload_key(self):
        return (
            "v3-runner",
            self.a.size(0),
            self.b.size(0),
            self.topk,
            self.b.size(1) // 2,
            self.a.size(1),
            self.config.compact_weight_scales,
            self.config.routing_weight_position,
            self.config.activation_bf16_rounding,
            self.swiglu_limit,
        )

    def compatible_candidates(self):
        return tuple(
            replace(
                c,
                compact_weight_scales=self.config.compact_weight_scales,
                routing_weight_position=self.config.routing_weight_position,
                activation_bf16_rounding=self.config.activation_bf16_rounding,
            )
            for c in MXFP4_FC1_CONFIGS
        )

    def pre_jit(self, candidate):
        return compile_mxfp4_fc1(candidate)

    def prepare(self, candidate):
        module = self.pre_jit(candidate)

        def prepare_metadata():
            return module.prepare_metadata(self.router, self.topk, self.b.size(1))

        metadata = None if self.dynamic_metadata else prepare_metadata()
        if metadata is not None:
            self.metadata_by_candidate[candidate] = metadata

        def run():
            current_metadata = prepare_metadata() if self.dynamic_metadata else metadata
            module.mxfp4_fc1_out(
                self.a,
                self.b,
                self.sfa,
                self.sfb,
                current_metadata,
                self.out,
                self.out_scale,
                self.swiglu_limit,
                routing_weights=self.routing_weights,
            )
            return self.out, self.out_scale

        return PreparedKernel(candidate, run)
