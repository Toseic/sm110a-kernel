"""Configurable H, I=2048 MXFP4 W4A8 FC1/SwiGLU/FC2 fusion."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import BaseKernelRunner, PreparedKernel
from .bf16tofp8_quant import compile_bf16tofp8_quant
from .mxfp4_fc1 import MXFP4_FC1_CONFIGS, Mxfp4Fc1Config, compile_mxfp4_fc1
from .mxfp4_fc2 import MXFP4_FC2_CONFIGS, Mxfp4Fc2Config, compile_mxfp4_fc2

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "csrc/mxfp4_moe/src"
_INC = _ROOT / "include"


@dataclass(frozen=True)
class Mxfp4Fc1Fc2Config:
    """One/two CTA pipelines and numerical semantics.

    Row scheduling splits FC1-I and FC2-H among h_split resident clusters
    per row; additional row tasks execute in waves. Tiled scheduling keeps
    both phases independently occupied and requires h_split=1. Metadata
    uses the same cluster_m, tile_n=256 and macro_swizzle_n_major=True.
    """

    cluster_m: int = 2
    h_split: int = 1
    stages: int = 5
    accumulator_stages: int = 2
    gather_warps: int = 4
    tiled_schedule: bool = False
    fuse_routing: bool = False
    routing_weight_position: str = "fc2"
    activation_bf16_rounding: bool = False
    reduce_tile_n: int = 64
    compact_weight_scales: bool = False
    hidden_dim: int = 6144

    def validate(self):
        if self.reduce_tile_n not in (64, 128):
            raise ValueError("reduce_tile_n must be 64 or 128")
        if self.gather_warps not in (2, 4):
            raise ValueError("gather_warps must be 2 or 4")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.cluster_m == 1 and self.stages > 3:
            raise ValueError("cluster_m=1 supports stages=2 or 3")
        if self.tiled_schedule and self.h_split != 1:
            raise ValueError("tiled_schedule requires h_split=1")
        if self.h_split not in (1, 2, 4):
            raise ValueError("h_split must be 1, 2, or 4")
        if (
            not isinstance(self.hidden_dim, int)
            or self.hidden_dim <= 0
            or self.hidden_dim % (256 * self.h_split)
        ):
            raise ValueError(
                "hidden_dim must be positive and divisible by 256 * h_split"
            )
        if self.accumulator_stages not in (1, 2):
            raise ValueError("accumulator_stages must be 1 or 2")
        if self.stages not in (2, 3, 4, 5):
            raise ValueError("stages must be 2, 3, 4, or 5")
        if self.routing_weight_position not in ("fc1", "fc2"):
            raise ValueError("routing_weight_position must be 'fc1' or 'fc2'")
        for name in (
            "fuse_routing",
            "activation_bf16_rounding",
            "tiled_schedule",
            "compact_weight_scales",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")


class Mxfp4Fc1Fc2Module:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._address = library.entry_address
        self.config = config
        self.library_path = library.path
        self._workspaces = {}

    def _workspace(self, a, row_task_capacity):
        stream = torch.cuda.current_stream(a.device)
        key = (a.device.index, row_task_capacity, stream.cuda_stream)
        if key not in self._workspaces:
            self._workspaces[key] = torch.empty(
                row_task_capacity * 2 * 294912 + max(8, row_task_capacity * 12),
                dtype=torch.uint8,
                device=a.device,
            )
        return self._workspaces[key]

    def _validate_launch(self, a, row_tasks):
        capacity = (
            torch.cuda.get_device_properties(a.device).multi_processor_count
            // self.config.cluster_m
        )
        if a.size(1) != self.config.hidden_dim:
            raise ValueError("A hidden dimension must match the compiled config")
        if row_tasks <= 0 or self.config.h_split > capacity:
            raise ValueError(
                f"each row requires {self.config.h_split} resident clusters; capacity={capacity}, rows={row_tasks}"
            )

    def run(self, a, w1, sfa, sfw1, w2, sfw2, swiglu_limit=0.0):
        output = torch.empty(
            (a.size(0), self.config.hidden_dim), device=a.device, dtype=torch.bfloat16
        )
        return self.run_out(a, w1, sfa, sfw1, w2, sfw2, output, swiglu_limit)

    def run_out(self, a, w1, sfa, sfw1, w2, sfw2, output, swiglu_limit=0.0):
        if self.config.fuse_routing:
            raise ValueError("run_out requires fuse_routing=False")
        if a.size(0) <= 0 or a.size(0) % (128 * self.config.cluster_m):
            raise ValueError(
                f"M must be positive and divisible by {128 * self.config.cluster_m}"
            )
        self._validate_launch(a, a.size(0) // (128 * self.config.cluster_m))
        return self._binding.mxfp4_fc1fc2_out(
            a,
            w1,
            sfa,
            sfw1,
            w2,
            sfw2,
            output,
            self._workspace(a, a.size(0) // (128 * self.config.cluster_m)),
            float(swiglu_limit),
            self._address,
            self.config.compact_weight_scales,
            self.config.hidden_dim,
        )

    def run_routed_out(
        self,
        a,
        w1,
        sfa,
        sfw1,
        w2,
        sfw2,
        metadata,
        routing_weights,
        topk,
        output,
        swiglu_limit=0.0,
        *,
        row_task_count=None,
    ):
        if not self.config.fuse_routing:
            raise ValueError("run_routed_out requires fuse_routing=True")
        expected_dtype = (
            torch.float32
            if self.config.routing_weight_position == "fc1"
            else torch.bfloat16
        )
        if routing_weights.dtype != expected_dtype:
            raise TypeError(f"routing_weights must have dtype {expected_dtype}")
        # Launch a capacity-sized persistent grid; the kernel reads the live
        # count from device metadata. Routing may change on every graph replay.
        if row_task_count is None:
            row_task_count = metadata.tasks.size(0) // 16
        self._validate_launch(a, row_task_count)
        return self._binding.mxfp4_fc1fc2_routed_out(
            a,
            w1,
            sfa,
            sfw1,
            w2,
            sfw2,
            metadata.source_rows,
            metadata.routing,
            metadata.tasks,
            metadata.state,
            routing_weights,
            topk,
            row_task_count,
            output,
            self._workspace(a, metadata.tasks.size(0) // 16),
            float(swiglu_limit),
            self._address,
            self.config.compact_weight_scales,
            self.config.hidden_dim,
        )


_JIT = SplitCudaJit(
    name="sm110a_mxfp4_fc1fc2",
    root=_ROOT,
    binding_sources=(_SRC / "mxfp4_fc1fc2_torch_binding.cpp",),
    kernel_sources=(_SRC / "mxfp4_fc1fc2_kernel.cu",),
    kernel_dependencies=(
        _INC / "mxfp4_fc1fc2.h",
        _INC / "common.h",
        _SRC / "mxfp4_fc1fc2_scale.cuh",
        _SRC / "mxfp4_fc1fc2_tiled.cuh",
    ),
    kernel_symbol="sm110a_mxfp4_fc1fc2_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda config, _variant: tuple(
        f"-DSM110A_MXFP4_FC1FC2_{key}={value}"
        for key, value in (
            ("HIDDEN_DIM", config.hidden_dim),
            ("GATHER_WARPS", config.gather_warps),
            ("REDUCE_TILE_N", config.reduce_tile_n),
            ("COMPACT_WEIGHT_SCALES", int(config.compact_weight_scales)),
            ("TILED", int(config.tiled_schedule)),
            ("CLUSTER_M", config.cluster_m),
            ("H_SPLIT", config.h_split),
            ("STAGES", config.stages),
            ("ACCUMULATOR_STAGES", config.accumulator_stages),
            ("FUSE_ROUTING", int(config.fuse_routing)),
            ("WEIGHT_IN_FC1", int(config.routing_weight_position == "fc1")),
            ("ACTIVATION_BF16_ROUNDING", int(config.activation_bf16_rounding)),
        )
    ),
    make_facade=Mxfp4Fc1Fc2Module,
    include_paths=(_INC,),
)


def compact_mxfp4_weight_scales(scales: torch.Tensor) -> torch.Tensor:
    """Pack padded blocked weight scales once, without changing their bits.

    The returned flat E8M0 tensor uses one quarter of the padded storage.
    Pass it only to a module with compact_weight_scales=True; input activation
    scales keep their original format. Pack static W1/W2 scales before replay.
    """
    if (
        not scales.is_cuda
        or scales.dtype != torch.float8_e8m0fnu
        or not scales.is_contiguous()
        or scales.numel() == 0
        or scales.numel() % 512
    ):
        raise ValueError("scales must be contiguous CUDA E8M0 padded blocked storage")
    return (
        scales.view(torch.uint8)
        .reshape(-1, 4)[:, 0]
        .contiguous()
        .view(torch.float8_e8m0fnu)
    )


# Backward-compatible name for the original fused-only entry point.
compact_mxfp4_fc1fc2_weight_scales = compact_mxfp4_weight_scales


def compile_mxfp4_fc1fc2(config: Mxfp4Fc1Fc2Config | None = None):
    config = config or Mxfp4Fc1Fc2Config()
    if not isinstance(config, Mxfp4Fc1Fc2Config):
        raise TypeError("config must be Mxfp4Fc1Fc2Config")
    config.validate()
    return _JIT.compile(config)


def compile_mxfp4_fc1fc2_configs(
    configs: Iterable[Mxfp4Fc1Fc2Config], max_workers=None
):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Mxfp4Fc1Fc2Config):
            raise TypeError("configs must contain Mxfp4Fc1Fc2Config values")
        config.validate()
    return _JIT.compile_many(configs, max_workers) if configs else ()


def prepare_mxfp4_fc1fc2_metadata(router_scores, topk, config=None):
    """Prepare FC1 routing in the tile order consumed by the fused kernel."""
    from .mxfp4_fc1 import Mxfp4Fc1Config, prepare_mxfp4_fc1_metadata

    config = config or Mxfp4Fc1Fc2Config(fuse_routing=True)
    config.validate()
    return prepare_mxfp4_fc1_metadata(
        router_scores,
        topk,
        4096,
        Mxfp4Fc1Config(
            cluster_m=config.cluster_m,
            macro_swizzle_n_major=True,
            routing_weight_position=config.routing_weight_position,
            activation_bf16_rounding=config.activation_bf16_rounding,
        ),
    )


def _default_config(a, row_tasks, **kwargs):
    kwargs.setdefault("hidden_dim", a.size(1))
    capacity = torch.cuda.get_device_properties(
        a.device
    ).multi_processor_count // kwargs.get("cluster_m", 2)
    split = next(
        (
            split
            for split in (4, 2, 1)
            if kwargs["hidden_dim"] % (256 * split) == 0
            and (split == 1 or row_tasks * split <= capacity)
        ),
        1,
    )
    return Mxfp4Fc1Fc2Config(h_split=split, **kwargs)


def mxfp4_fc1fc2(a, w1, sfa, sfw1, w2, sfw2, swiglu_limit=0.0, config=None):
    if config is None:
        cluster = 2 if a.size(0) % 256 == 0 else 1
        config = _default_config(
            a,
            a.size(0) // (128 * cluster),
            cluster_m=cluster,
            stages=5 if cluster == 2 else 3,
        )
    return compile_mxfp4_fc1fc2(config).run(a, w1, sfa, sfw1, w2, sfw2, swiglu_limit)


def mxfp4_fc1fc2_out(a, w1, sfa, sfw1, w2, sfw2, output, swiglu_limit=0.0, config=None):
    if config is None:
        cluster = 2 if a.size(0) % 256 == 0 else 1
        config = _default_config(
            a,
            a.size(0) // (128 * cluster),
            cluster_m=cluster,
            stages=5 if cluster == 2 else 3,
        )
    return compile_mxfp4_fc1fc2(config).run_out(
        a, w1, sfa, sfw1, w2, sfw2, output, swiglu_limit
    )


def mxfp4_fc1fc2_routed_out(
    a,
    w1,
    sfa,
    sfw1,
    w2,
    sfw2,
    metadata,
    routing_weights,
    topk,
    output,
    swiglu_limit=0.0,
    config=None,
    *,
    row_task_count=None,
):
    if config is None:
        if row_task_count is None:
            row_task_count = metadata.tasks.size(0) // 16
        config = _default_config(a, row_task_count, fuse_routing=True)
    return compile_mxfp4_fc1fc2(config).run_routed_out(
        a,
        w1,
        sfa,
        sfw1,
        w2,
        sfw2,
        metadata,
        routing_weights,
        topk,
        output,
        swiglu_limit,
        row_task_count=row_task_count,
    )


def fused_candidates(
    m,
    hidden_dim,
    *,
    compact_formats=(False,),
    reductions=(64,),
    gathers=(4,),
    schedules=(False,),
    clusters=(1, 2),
    splits=(1, 2, 4),
    stages=(3, 5),
    accumulators=(2,),
    fuse_routing=False,
    routing_weight_position="fc2",
    activation_bf16_rounding=False,
):
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    return tuple(
        Mxfp4Fc1Fc2Config(
            hidden_dim=hidden_dim,
            compact_weight_scales=compact,
            cluster_m=cluster,
            tiled_schedule=tiled,
            gather_warps=gather,
            reduce_tile_n=reduce,
            h_split=split,
            stages=stage,
            accumulator_stages=accumulator,
            fuse_routing=fuse_routing,
            routing_weight_position=routing_weight_position,
            activation_bf16_rounding=activation_bf16_rounding,
        )
        for compact in compact_formats
        for reduce in reductions
        for gather in gathers
        for tiled in schedules
        for cluster in clusters
        for split in splits
        for stage in stages
        for accumulator in accumulators
        if (fuse_routing or m % (128 * cluster) == 0)
        and (cluster == 2 or stage <= 3)
        and split <= sms // cluster
        and hidden_dim % (256 * split) == 0
        and (not tiled or split == 1)
    )


class Mxfp4Fc1Fc2Runner(BaseKernelRunner):
    namespace = "mxfp4_fc1fc2_runner_v1"
    graph_timing = True

    def __init__(
        self,
        a,
        w1,
        sfa,
        sfw1,
        w2,
        sfw2,
        out,
        *,
        router=None,
        routing_weights=None,
        topk=1,
        swiglu_limit=0.0,
        compact_weight_scales=False,
        routing_weight_position="fc2",
        activation_bf16_rounding=False,
        input=None,
        dynamic_metadata=False,
        row_task_counts=None,
    ):
        super().__init__(a.device)
        self.a, self.w1, self.sfa, self.sfw1 = a, w1, sfa, sfw1
        self.w2, self.sfw2, self.out = w2, sfw2, out
        self.router, self.routing_weights, self.topk = router, routing_weights, topk
        self.swiglu_limit, self.compact = swiglu_limit, compact_weight_scales
        self.position, self.rounding = routing_weight_position, activation_bf16_rounding
        self.input, self.dynamic_metadata = input, dynamic_metadata
        self.row_task_counts = row_task_counts
        if input is not None or dynamic_metadata:
            self.namespace = "mxfp4_fc1fc2_pipeline_v1"

    @property
    def workload_key(self):
        return (
            tuple(self.a.shape),
            tuple(self.w1.shape),
            tuple(self.w2.shape),
            self.topk,
            self.router is not None,
            self.compact,
            self.position,
            self.rounding,
            self.swiglu_limit,
            self.input is not None,
            self.dynamic_metadata,
        )

    def compatible_candidates(self):
        return fused_candidates(
            self.a.size(0),
            self.a.size(1),
            compact_formats=(self.compact,),
            fuse_routing=self.router is not None,
            routing_weight_position=self.position,
            activation_bf16_rounding=self.rounding,
        )

    def pre_jit(self, candidate):
        module = compile_mxfp4_fc1fc2(candidate)
        if self.router is not None:
            compile_mxfp4_fc1(Mxfp4Fc1Config(cluster_m=candidate.cluster_m))
        if self.input is not None:
            compile_bf16tofp8_quant(blocked=True)
        return module

    def prepare(self, candidate):
        module = self.pre_jit(candidate)
        quant = (
            compile_bf16tofp8_quant(blocked=True) if self.input is not None else None
        )
        sfw1 = (
            self.sfw1[candidate.compact_weight_scales]
            if isinstance(self.sfw1, dict)
            else self.sfw1
        )
        sfw2 = (
            self.sfw2[candidate.compact_weight_scales]
            if isinstance(self.sfw2, dict)
            else self.sfw2
        )
        if self.router is not None:
            fc1 = compile_mxfp4_fc1(Mxfp4Fc1Config(cluster_m=candidate.cluster_m))

            def metadata():
                return fc1.prepare_metadata(self.router, self.topk, self.w1.size(1))

            static_metadata = None if self.dynamic_metadata else metadata()

        def run():
            md = (
                (metadata() if self.dynamic_metadata else static_metadata)
                if self.router is not None
                else None
            )
            if quant is not None:
                quant.quantize_out(self.input, self.a, self.sfa)
            if md is None:
                module.run_out(
                    self.a,
                    self.w1,
                    self.sfa,
                    sfw1,
                    self.w2,
                    sfw2,
                    self.out,
                    self.swiglu_limit,
                )
            else:
                module.run_routed_out(
                    self.a,
                    self.w1,
                    self.sfa,
                    sfw1,
                    self.w2,
                    sfw2,
                    md,
                    self.routing_weights,
                    self.topk,
                    self.out,
                    self.swiglu_limit,
                    row_task_count=None
                    if self.row_task_counts is None
                    else self.row_task_counts[candidate.cluster_m],
                )
            return self.out

        return PreparedKernel(candidate, run)


@dataclass(frozen=True)
class Mxfp4MoeConfig:
    cluster_m: int
    direct_gmem: bool
    epilogue_tile_n: int


class Mxfp4MoeRunner(BaseKernelRunner):
    namespace = "mxfp4_moe_pipeline_v1"
    graph_timing = True

    def __init__(
        self,
        x,
        w1,
        sfw1,
        w2,
        sfw2,
        router,
        routing_weights,
        topk,
        out,
        *,
        swiglu_limit=0.0,
        compact_weight_scales=False,
        routing_weight_position="fc2",
        activation_bf16_rounding=False,
        deterministic=False,
        pdl=False,
    ):
        super().__init__(x.device)
        self.x, self.w1, self.sfw1, self.w2, self.sfw2 = x, w1, sfw1, w2, sfw2
        self.router, self.routing_weights, self.topk, self.out = (
            router,
            routing_weights,
            topk,
            out,
        )
        self.limit, self.compact = swiglu_limit, compact_weight_scales
        self.position, self.rounding = routing_weight_position, activation_bf16_rounding
        self.deterministic, self.pdl = deterministic, pdl
        tokens, dim = x.shape
        inter = w1.size(1) // 2
        self.aq = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self.sfa = torch.empty(
            (tokens * dim // 8,), device=x.device, dtype=torch.float8_e8m0fnu
        )
        self.hidden = torch.empty(
            (tokens * topk, inter), device=x.device, dtype=torch.float8_e4m3fn
        )
        self.hidden_scale = torch.empty(
            (tokens * topk * inter // 8,), device=x.device, dtype=torch.float8_e8m0fnu
        )
        self.weights1 = (
            routing_weights.float().contiguous() if self.position == "fc1" else None
        )
        self.weights2 = routing_weights.to(torch.bfloat16).contiguous()

    @property
    def workload_key(self):
        return (
            tuple(self.x.shape),
            tuple(self.w1.shape),
            tuple(self.w2.shape),
            self.topk,
            self.limit,
            self.compact,
            self.position,
            self.rounding,
            self.deterministic,
            self.pdl,
        )

    def compatible_candidates(self):
        return tuple(
            Mxfp4MoeConfig(c1.cluster_m, c1.direct_gmem, c2.epilogue_tile_n)
            for c1 in MXFP4_FC1_CONFIGS
            for c2 in MXFP4_FC2_CONFIGS
            if c1.cluster_m == c2.cluster_m
        )

    def pre_jit(self, candidate):
        quant = compile_bf16tofp8_quant(blocked=True, pdl=self.pdl)
        fc1 = compile_mxfp4_fc1(
            Mxfp4Fc1Config(
                cluster_m=candidate.cluster_m,
                direct_gmem=candidate.direct_gmem,
                compact_weight_scales=self.compact,
                routing_weight_position=self.position,
                activation_bf16_rounding=self.rounding,
                pdl=self.pdl,
                pdl_input=self.pdl,
            )
        )
        fc2 = compile_mxfp4_fc2(
            Mxfp4Fc2Config(
                cluster_m=candidate.cluster_m,
                epilogue_tile_n=candidate.epilogue_tile_n,
                compact_weight_scales=self.compact,
                routing_weight_position=self.position,
                deterministic=self.deterministic,
                pdl=self.pdl,
                pdl_wait="before_a",
            )
        )
        return quant, fc1, fc2

    def prepare(self, candidate):
        quant, fc1, fc2 = self.pre_jit(candidate)

        def run():
            md1 = fc1.prepare_metadata(self.router, self.topk, self.w1.size(1))
            layout = md1.routing.narrow(
                0, 2 * self.w1.size(0), self.x.size(0) * self.topk
            )
            md2 = fc2.prepare_metadata(layout, self.w1.size(0), self.x.size(1))
            if self.pdl:
                fc2.prepare_pdl_out(self.out, md2)
            quant.quantize_out(self.x, self.aq, self.sfa)
            fc1.mxfp4_fc1_out(
                self.aq,
                self.w1,
                self.sfa,
                self.sfw1,
                md1,
                self.hidden,
                self.hidden_scale,
                self.limit,
                routing_weights=self.weights1,
            )
            fc2.mxfp4_fc2_out(
                self.hidden,
                self.w2,
                self.hidden_scale,
                self.sfw2,
                md1.routing,
                self.weights2,
                md2,
                self.topk,
                self.out,
            )
            return self.out

        return PreparedKernel(candidate, run)


@dataclass(frozen=True)
class Mxfp4StandaloneConfig:
    """Serializable pair of independently tuned baseline configurations."""

    fc1: Mxfp4Fc1Config
    fc2: Mxfp4Fc2Config

    def __post_init__(self):
        if isinstance(self.fc1, dict):
            object.__setattr__(self, "fc1", Mxfp4Fc1Config(**self.fc1))
        if isinstance(self.fc2, dict):
            object.__setattr__(self, "fc2", Mxfp4Fc2Config(**self.fc2))
        self.fc1.validate()
        self.fc2.validate()
        if (
            self.fc1.pdl
            or self.fc1.pdl_input
            or self.fc2.pdl
            or self.fc2.external_prepare
        ):
            raise ValueError("standalone runner requires serial FC1/FC2 configs")
        if self.fc1.routing_weight_position != self.fc2.routing_weight_position:
            raise ValueError("FC1 and FC2 routing weight positions must match")


class Mxfp4StandaloneRunner(BaseKernelRunner):
    """Tune adjacent serial FC1/FC2 launches with fixed routing metadata."""

    namespace = "mxfp4_fc1fc2_standalone_runner_v1"
    graph_timing = True

    def __init__(
        self,
        a,
        w1,
        sfa,
        sfw1,
        w2,
        sfw2,
        router,
        routing_weights,
        topk,
        hidden,
        hidden_scale,
        out,
        *,
        swiglu_limit=0.0,
        fc1_config=None,
        fc2_config=None,
    ):
        super().__init__(a.device)
        self.a, self.w1, self.sfa, self.sfw1 = a, w1, sfa, sfw1
        self.w2, self.sfw2, self.router = w2, sfw2, router
        self.weights, self.topk = routing_weights, topk
        self.weights2 = routing_weights.to(torch.bfloat16)
        self.hidden, self.hidden_scale, self.out = hidden, hidden_scale, out
        self.limit = swiglu_limit
        self.fc1_config = fc1_config or Mxfp4Fc1Config(direct_gmem=True)
        self.fc2_config = fc2_config or Mxfp4Fc2Config()
        Mxfp4StandaloneConfig(self.fc1_config, self.fc2_config)

    @property
    def workload_key(self):
        return (
            tuple(self.a.shape),
            tuple(self.w1.shape),
            tuple(self.w2.shape),
            self.topk,
            self.limit,
            repr(self.fc1_config),
            repr(self.fc2_config),
        )

    def compatible_candidates(self):
        return tuple(
            Mxfp4StandaloneConfig(
                replace(self.fc1_config, cluster_m=c1),
                replace(self.fc2_config, cluster_m=c2, epilogue_tile_n=epi),
            )
            for c1 in (1, 2)
            for c2 in (1, 2)
            for epi in (32, 64)
        )

    @staticmethod
    def format_candidate(candidate):
        return (
            f"fc1_c={candidate.fc1.cluster_m},fc2_c={candidate.fc2.cluster_m},"
            f"epi={candidate.fc2.epilogue_tile_n},"
            f"compact={int(candidate.fc1.compact_weight_scales)}"
        )

    def pre_jit(self, candidate):
        return compile_mxfp4_fc1(candidate.fc1), compile_mxfp4_fc2(candidate.fc2)

    def prepare(self, candidate):
        mod1, mod2 = self.pre_jit(candidate)
        md1 = mod1.prepare_metadata(self.router, self.topk, self.w1.size(1))
        layout = md1.routing.narrow(0, 2 * self.w1.size(0), self.a.size(0) * self.topk)
        md2 = mod2.prepare_metadata(layout, self.w1.size(0), self.out.size(1))

        def fc1():
            return mod1.mxfp4_fc1_out(
                self.a,
                self.w1,
                self.sfa,
                self.sfw1,
                md1,
                self.hidden,
                self.hidden_scale,
                self.limit,
                routing_weights=self.weights
                if candidate.fc1.routing_weight_position == "fc1"
                else None,
            )

        def fc2():
            return mod2.mxfp4_fc2_out(
                self.hidden,
                self.w2,
                self.hidden_scale,
                self.sfw2,
                md1.routing,
                self.weights2,
                md2,
                self.topk,
                self.out,
            )

        def run():
            fc1()
            return fc2()

        prepared = PreparedKernel(candidate, run)
        prepared.fc1, prepared.fc2 = fc1, fc2
        return prepared
