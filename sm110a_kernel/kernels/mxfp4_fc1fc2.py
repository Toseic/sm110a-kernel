"""Fixed H=6144, I=2048 MXFP4 W4A8 FC1/SwiGLU/FC2 fusion."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit

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
        if row_tasks <= 0 or self.config.h_split > capacity:
            raise ValueError(
                f"each row requires {self.config.h_split} resident clusters; capacity={capacity}, rows={row_tasks}"
            )

    def run(self, a, w1, sfa, sfw1, w2, sfw2, swiglu_limit=0.0):
        output = torch.empty((a.size(0), 6144), device=a.device, dtype=torch.bfloat16)
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
        # Read live state each eager launch; do not cache mutable metadata counts.
        # For CUDA Graph capture supply the fixed row_task_count explicitly.
        if row_task_count is None:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("CUDA Graph capture requires explicit row_task_count")
            row_task_count = int(metadata.state[0].item()) // 16
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


def compact_mxfp4_fc1fc2_weight_scales(scales: torch.Tensor) -> torch.Tensor:
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
    capacity = torch.cuda.get_device_properties(
        a.device
    ).multi_processor_count // kwargs.get("cluster_m", 2)
    split = next(
        (split for split in (4, 2, 1) if split == 1 or row_tasks * split <= capacity), 1
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
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("CUDA Graph capture requires explicit row_task_count")
            row_task_count = int(metadata.state[0].item()) // 16
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
