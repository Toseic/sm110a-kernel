"""Fixed-shape two-CTA NVFP4 FC1+FC2 streaming kernel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "csrc/nvfp4_moe/src"
_INC = _ROOT / "include"


@dataclass(frozen=True)
class Nvfp4Fc1Fc2Config:
    """Compile-time streaming-buffer configuration.

    ``buffer_i`` is the number of logical FC1 output columns retained in
    shared memory before switching to FC2.  It must tile the fixed I=2048
    dimension in 256-column mainloop stages. ``h_split`` partitions the FC2
    small-M work across independent clusters: they split FC1-I, exchange the
    complete intermediate through a reusable global workspace, then split
    FC2-H. ``cluster_pairs`` instead performs the exchange through distributed
    shared memory inside a larger cluster. ``fuse_routing`` consumes the
    original token-major activation using FC1 routing metadata and reduces
    routing-weighted FC2 results directly into token-major output.
    """

    buffer_i: int = 2048
    h_split: int = 1
    cluster_pairs: int = 1
    fuse_routing: bool = False
    cache_source_rows: int = 0

    def validate(self) -> None:
        if self.buffer_i not in (256, 512, 1024, 2048):
            raise ValueError("buffer_i must be one of 256, 512, 1024, or 2048")
        if self.h_split not in (1, 2, 3, 4):
            raise ValueError("h_split must be 1, 2, 3, or 4")
        if self.cluster_pairs not in (1, 2, 4):
            raise ValueError("cluster_pairs must be 1, 2, or 4")
        if self.h_split > 1 and self.cluster_pairs > 1:
            raise ValueError("h_split and cluster_pairs cannot both exceed one")
        if self.h_split > 1 and self.buffer_i != 2048:
            raise ValueError("h_split currently requires buffer_i=2048")
        if 48 % (self.h_split * self.cluster_pairs):
            raise ValueError("h_split * cluster_pairs must divide 48")
        if (self.buffer_i // 256) % self.cluster_pairs:
            raise ValueError("buffer_i / 256 must be divisible by cluster_pairs")
        if not isinstance(self.fuse_routing, bool):
            raise ValueError("fuse_routing must be bool")
        if self.cache_source_rows not in (0, 1, 2):
            raise ValueError("cache_source_rows must be 0, 1, or 2")
        if self.cache_source_rows != 0 and not self.fuse_routing:
            raise ValueError("cache_source_rows requires fuse_routing=True")
        if self.fuse_routing and self.buffer_i != 2048:
            raise ValueError("fuse_routing requires buffer_i=2048")


class Nvfp4Fc1Fc2Module:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._address = library.entry_address
        self.config = config
        self.library_path = library.path
        self._workspaces = {}
        self._routed_row_counts = {}

    def _workspace(self, a, row_task_capacity=None):
        stream = torch.cuda.current_stream(a.device)
        if row_task_capacity is None:
            row_task_capacity = a.size(0) // 256
        key = (a.device.index, a.size(0), row_task_capacity, stream.cuda_stream)
        workspace = self._workspaces.get(key)
        if workspace is None:
            # One raw 144-KiB intermediate image per 128-row CTA plus enough
            # words for either the resident-grid barrier or one completion
            # counter per row plus one owner flag per h_split=2 task.
            workspace = torch.empty(
                row_task_capacity * 2 * 147456 +
                max(8, row_task_capacity * 3 * 4),
                dtype=torch.uint8, device=a.device
            )
            self._workspaces[key] = workspace
        return workspace

    def _validate_launch(self, a):
        if self.config.h_split == 1:
            return
        row_tasks = a.size(0) // 256
        tasks = row_tasks * self.config.h_split
        if self.config.h_split == 3 and row_tasks == 4:
            tasks = 10
        properties = torch.cuda.get_device_properties(a.device)
        resident_clusters = properties.multi_processor_count // (
            2 * self.config.cluster_pairs
        )
        if (self.config.h_split > 1 and tasks > resident_clusters and
                self.config.h_split != 2):
            raise ValueError(
                "h_split requires all logical tasks to be simultaneously "
                f"resident (tasks={tasks}, capacity={resident_clusters})"
            )

    def run(self, a, w1, sfa, sfw1, w2, sfw2):
        if self.config.fuse_routing:
            raise ValueError("run requires fuse_routing=False")
        self._validate_launch(a)
        return self._binding.nvfp4_fc1fc2(
            a, w1, sfa, sfw1, w2, sfw2, self._workspace(a), self._address
        )

    def run_out(self, a, w1, sfa, sfw1, w2, sfw2, output):
        if self.config.fuse_routing:
            raise ValueError("run_out requires fuse_routing=False")
        self._validate_launch(a)
        return self._binding.nvfp4_fc1fc2_out(
            a, w1, sfa, sfw1, w2, sfw2, output,
            self._workspace(a), self._address
        )

    def run_routed_out(
        self, a, w1, sfa, sfw1, w2, sfw2, metadata,
        routing_weights, topk, output
    ):
        if not self.config.fuse_routing:
            raise ValueError("run_routed_out requires fuse_routing=True")
        # FC1 metadata contains one task per physical FC1-N tile. There are
        # 4096 / 128 = 32 such tasks for every fused expert-row tile.
        row_task_capacity = metadata.tasks.size(0) // 32
        state_key = (metadata.state.device.index, metadata.state.data_ptr())
        row_task_count = self._routed_row_counts.get(state_key)
        if row_task_count is None:
            row_task_count = int(metadata.state[0].item()) // 32
            self._routed_row_counts[state_key] = row_task_count
        tasks = row_task_count * self.config.h_split
        if self.config.h_split == 3 and row_task_count == 4:
            tasks = 10
        properties = torch.cuda.get_device_properties(a.device)
        resident_clusters = properties.multi_processor_count // (
            2 * self.config.cluster_pairs
        )
        if (self.config.h_split > 1 and tasks > resident_clusters and
                self.config.h_split != 2):
            raise ValueError(
                "routed h_split requires all logical tasks to be resident "
                f"(tasks={tasks}, capacity={resident_clusters})"
            )
        return self._binding.nvfp4_fc1fc2_routed_out(
            a, w1, sfa, sfw1, w2, sfw2,
            metadata.source_rows, metadata.routing, metadata.tasks,
            metadata.state, routing_weights, topk, row_task_count, output,
            self._workspace(a, row_task_capacity), self._address,
        )


_JIT = SplitCudaJit(
    name="sm110a_nvfp4_fc1fc2",
    root=_ROOT,
    binding_sources=(_SRC / "nvfp4_fc1fc2_torch_binding.cpp",),
    kernel_sources=(_SRC / "nvfp4_fc1fc2_kernel.cu",),
    kernel_dependencies=(
        _INC / "nvfp4_fc1fc2.h", _INC / "common.h",
        _INC / "pipeline_delay.h", _SRC / "nvfp4_fc1fc2_pipeline_delay.cuh",
    ),
    kernel_symbol="sm110a_nvfp4_fc1fc2_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda config, _variant: (
        f"-DSM110A_NVFP4_FC1FC2_BUFFER_I={config.buffer_i}",
        f"-DSM110A_NVFP4_FC1FC2_H_SPLIT={config.h_split}",
        f"-DSM110A_NVFP4_FC1FC2_CLUSTER_PAIRS={config.cluster_pairs}",
        f"-DSM110A_NVFP4_FC1FC2_FUSE_ROUTING={int(config.fuse_routing)}",
        f"-DSM110A_NVFP4_FC1FC2_CACHE_SOURCE_ROWS="
        f"{config.cache_source_rows}",
    ),
    make_facade=Nvfp4Fc1Fc2Module,
    include_paths=(_INC,),
)


def compile_nvfp4_fc1fc2(config: Optional[Nvfp4Fc1Fc2Config] = None):
    config = config or Nvfp4Fc1Fc2Config()
    if not isinstance(config, Nvfp4Fc1Fc2Config):
        raise TypeError("config must be an Nvfp4Fc1Fc2Config")
    config.validate()
    return _JIT.compile(config)


def compile_nvfp4_fc1fc2_configs(
    configs: Iterable[Nvfp4Fc1Fc2Config], max_workers=None
):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Nvfp4Fc1Fc2Config):
            raise TypeError("configs must contain Nvfp4Fc1Fc2Config values")
        config.validate()
    return _JIT.compile_many(configs, max_workers) if configs else ()


def nvfp4_fc1fc2(a, w1, sfa, sfw1, w2, sfw2, config=None):
    if config is None:
        config = Nvfp4Fc1Fc2Config(
            h_split=4 if a.size(0) == 512 else (3 if a.size(0) == 1024 else 1)
        )
    return compile_nvfp4_fc1fc2(config).run(a, w1, sfa, sfw1, w2, sfw2)


def nvfp4_fc1fc2_out(a, w1, sfa, sfw1, w2, sfw2, output, config=None):
    if config is None:
        config = Nvfp4Fc1Fc2Config(
            h_split=4 if a.size(0) == 512 else (3 if a.size(0) == 1024 else 1)
        )
    return compile_nvfp4_fc1fc2(config).run_out(
        a, w1, sfa, sfw1, w2, sfw2, output
    )


def nvfp4_fc1fc2_routed_out(
    a, w1, sfa, sfw1, w2, sfw2, metadata, routing_weights, topk, output,
    config=None,
):
    config = config or Nvfp4Fc1Fc2Config(fuse_routing=True)
    return compile_nvfp4_fc1fc2(config).run_routed_out(
        a, w1, sfa, sfw1, w2, sfw2, metadata, routing_weights, topk, output
    )
