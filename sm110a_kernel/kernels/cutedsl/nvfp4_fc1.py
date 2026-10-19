"""Autotuned TensorRT-LLM CuTE DSL NVFP4 FC1 baseline with fused gather."""

from __future__ import annotations

import statistics
import sys
import types
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ...runner import AutotuneRunner, PreparedKernel, Selection, TuneOptions

_POINTER_DEPRECATION = "Use explicit `struct.scalar.ptr` for pointer instead."
_PROBE_WARMUP = 2
_ROOT = Path(__file__).resolve().parents[3]
_TRTLLM_PACKAGE_ROOT = _ROOT / "baselinedir" / "TensorRT-LLM" / "tensorrt_llm"


def _namespace_package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules.setdefault(name, module)


# Import only CuTE DSL sources.  Importing TensorRT-LLM's top-level package
# initializes MPI in this development environment, which a standalone kernel
# benchmark neither needs nor can assume is available.
_namespace_package("sm110a_trtllm", _TRTLLM_PACKAGE_ROOT)
_namespace_package("sm110a_trtllm._torch", _TRTLLM_PACKAGE_ROOT / "_torch")
_utils = types.ModuleType("sm110a_trtllm._torch.utils")


class _ActivationType(IntEnum):
    Swiglu = 5
    Relu2 = 8
    SiTu = 9


_utils.ActivationType = _ActivationType
_utils.is_gated_activation = lambda activation: (
    activation
    in (
        _ActivationType.Swiglu,
        _ActivationType.SiTu,
    )
)
sys.modules.setdefault("sm110a_trtllm._torch.utils", _utils)

from sm110a_trtllm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
    ActivationType,
    BlockScaledContiguousGatherGroupedGemmKernel,
)


@contextmanager
def _suppress_cutlass_pointer_deprecation():
    showwarning = warnings.showwarning

    def filtered(message, category, filename, lineno, file=None, line=None):
        if category is DeprecationWarning and str(message) == _POINTER_DEPRECATION:
            return
        showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = filtered
    try:
        yield
    finally:
        warnings.showwarning = showwarning


@dataclass(frozen=True)
class CuteDslNvfp4Fc1Config:
    mma_tile_m: int = 128
    mma_tile_n: int = 128
    cluster_m: int = 1
    cluster_n: int = 1

    def validate(self) -> None:
        if self.mma_tile_m not in (128, 256):
            raise ValueError("mma_tile_m must be 128 or 256")
        if self.mma_tile_n not in (64, 128, 256):
            raise ValueError("mma_tile_n must be 64, 128, or 256")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.cluster_n not in (1, 2, 4):
            raise ValueError("cluster_n must be 1, 2, or 4")
        if self.cluster_m * self.cluster_n > 8:
            raise ValueError("cluster size must be ≤ 8")
        if self.mma_tile_m == 256 and self.cluster_m % 2 != 0:
            raise ValueError("cluster_m must be even when mma_tile_m is 256")


DEFAULT_CONFIGS = tuple(
    CuteDslNvfp4Fc1Config(tile_m, tile_n, tile_m // 128, cluster_n)
    for tile_m in (128, 256)
    for tile_n in (128, 256)
    for cluster_n in (1, 2, 4)
)


def format_config(config):
    return f"{config.mma_tile_m}x{config.mma_tile_n}/c{config.cluster_m}x{config.cluster_n}"


# ---------------------------------------------------------------------------
# Custom CuTE JIT launch for TensorRT-LLM's fused gather FC1 kernel with its
# built-in NVFP4 output quantization enabled.
# ---------------------------------------------------------------------------
@cute.jit
def _launch_fc1(
    gemm_op: cutlass.Constexpr,
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    a_sf_ptr: cute.Pointer,
    b_sf_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    c_sf_ptr: cute.Pointer,
    norm_const_ptr: cute.Pointer,
    alpha_ptr: cute.Pointer,
    tile_idx_to_expert_idx_ptr: cute.Pointer,
    tile_idx_to_mn_limit_ptr: cute.Pointer,
    token_id_mapping_ptr: cute.Pointer,
    num_non_exiting_tiles_ptr: cute.Pointer,
    orig_m: cutlass.Int64,
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    l: cutlass.Int64,  # noqa: E741
    tile_size: cutlass.Constexpr,
    scaling_vector_size: cutlass.Constexpr,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """Construct tensors and invoke gather + grouped GEMM + SwiGLU."""
    scale_k = k // scaling_vector_size
    interm_size = n // 2  # SwiGLU collapses gate/up pairs
    num_tiles = m // tile_size

    a = cute.make_tensor(
        a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
    )
    b = cute.make_tensor(
        b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
    )
    a_sf = cute.make_tensor(
        a_sf_ptr,
        layout=cute.make_ordered_layout((orig_m, scale_k, 1), order=(1, 0, 2)),
    )
    b_sf = cute.make_tensor(
        b_sf_ptr,
        layout=cute.make_ordered_layout(
            (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
        ),
    )
    c = cute.make_tensor(
        c_ptr, layout=cute.make_ordered_layout((m, interm_size, 1), order=(1, 0, 2))
    )
    c_sf = cute.make_tensor(
        c_sf_ptr,
        layout=cute.make_ordered_layout(
            (m, interm_size // scaling_vector_size, 1), order=(1, 0, 2)
        ),
    )
    norm_const = cute.make_tensor(norm_const_ptr, layout=cute.make_layout((1,)))
    alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))
    tile_idx_to_expert_idx = cute.make_tensor(
        tile_idx_to_expert_idx_ptr, layout=cute.make_layout((num_tiles,))
    )
    tile_idx_to_mn_limit = cute.make_tensor(
        tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
    )
    token_id_mapping = cute.make_tensor(
        token_id_mapping_ptr, layout=cute.make_layout((m,))
    )
    num_non_exiting_tiles = cute.make_tensor(
        num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
    )

    gemm_op(
        a,
        b,
        c,
        a_sf,
        b_sf,
        c_sf,
        norm_const,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        cutlass.Int32(0),  # Unused by the explicit tile-metadata scheduling path.
        max_active_clusters,
        stream,
    )


def _measure(run, iters, cold_l2_flush):
    if cold_l2_flush is None:
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(iters):
            run()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iters
    events = []
    for _ in range(iters):
        cold_l2_flush()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        run()
        end.record()
        events.append((start, end))
    events[-1][1].synchronize()
    return sum(start.elapsed_time(end) for start, end in events) / iters


def _stats(run, options):
    for _ in range(options.warmup):
        run()
    torch.cuda.synchronize()
    samples = sorted(
        _measure(run, options.iters, options.cold_l2_flush)
        for _ in range(options.samples)
    )
    return {
        "min_ms": samples[0],
        "median_ms": statistics.median(samples),
        "max_ms": samples[-1],
    }


def _quick_probe(run, options):
    for _ in range(_PROBE_WARMUP):
        run()
    torch.cuda.synchronize()
    return _measure(run, options.iters, options.cold_l2_flush)


class CuteDslNvfp4Fc1Runner:
    """Prepare, autotune, and run CuTE DSL FC1 with fused A/SFA gather.

    Accepts the same tensor interface as the native ``nvfp4_fc1`` plus
    metadata (``source_rows``, ``tasks``, ``state``) from
    ``prepare_nvfp4_fc1_metadata``.
    """

    def __init__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        sfa: torch.Tensor,
        sfb: torch.Tensor,
        out: torch.Tensor,
        out_scale: torch.Tensor,
        source_rows: torch.Tensor,
        tasks: torch.Tensor,
        state: torch.Tensor,
    ):
        if not all(
            t.is_cuda
            for t in (a, b, sfa, sfb, out, out_scale, source_rows, tasks, state)
        ):
            raise ValueError("all CuTE DSL NVFP4 FC1 tensors must be CUDA tensors")
        if a.dtype != torch.float4_e2m1fn_x2 or b.dtype != torch.float4_e2m1fn_x2:
            raise ValueError("a and b must be packed torch.float4_e2m1fn_x2 tensors")
        if sfa.dtype != torch.float8_e4m3fn or sfb.dtype != torch.float8_e4m3fn:
            raise ValueError("expected E4M3 scale factors")
        if out.dtype != torch.float4_e2m1fn_x2:
            raise ValueError("expected packed float4_e2m1fn_x2 output")
        if out_scale.dtype != torch.float8_e4m3fn:
            raise ValueError("expected E4M3 output scales")
        if a.ndim != 2 or b.ndim != 3 or out.ndim != 2:
            raise ValueError(
                "expected a[tokens,K/2], b[experts,2*N,K/2], out[permuted_m,N/2]"
            )
        if (
            source_rows.dtype != torch.int32
            or tasks.dtype != torch.int32
            or state.dtype != torch.int32
        ):
            raise ValueError("metadata tensors must be int32")

        self.tokens, packed_k = a.shape
        self.experts, double_n, packed_k_b = b.shape
        self.k = packed_k * 2
        self.n = double_n  # gate+up interleaved (2 * N)
        if packed_k != packed_k_b:
            raise ValueError("K dimension mismatch between a and b")
        if self.n % 128 != 0:
            raise ValueError("B dim 1 (2*N) must be divisible by 128")
        # SFA/SFB may be flat (1D blocked layout) or 2D — accept either.
        expected_sfa_elems = self.tokens * (self.k // 16)
        if sfa.numel() != expected_sfa_elems:
            raise ValueError(
                f"sfa has {sfa.numel()} elements, expected {expected_sfa_elems}"
            )
        expected_sfb_elems = self.experts * self.n * (self.k // 16)
        if sfb.numel() != expected_sfb_elems:
            raise ValueError(
                f"sfb has {sfb.numel()} elements, expected {expected_sfb_elems}"
            )

        self.logical_m = source_rows.numel()  # permuted rows = tokens * topk
        if out.shape != (self.logical_m, self.n // 4):
            raise ValueError(
                f"out shape mismatch: {out.shape} vs ({self.logical_m},{self.n // 4})"
            )
        if out_scale.numel() != self.logical_m * (self.n // 32):
            raise ValueError("out_scale has invalid blocked scale storage")

        # ExpertTileTask is {row_start, row_count, tile_row, expert, tile_n}.
        # The native list contains one entry per M/N tile; retain just the
        # first entry of each expert range.
        if state.numel() < 1:
            raise ValueError("state tensor must contain the native task count")
        if tasks.numel() % 5:
            raise ValueError("tasks tensor must contain 5 int32 values per task")
        self.num_tasks = int(state[0].item())
        if self.num_tasks < 0 or tasks.numel() < self.num_tasks * 5:
            raise ValueError("tasks tensor is too small for the declared task count")
        task_rows = tasks.reshape(-1, 5)[: self.num_tasks].cpu().tolist()
        native_ranges = tuple(
            sorted(
                (row_start, row_count, expert)
                for row_start, row_count, tile_row, expert, tile_n in task_rows
                if (
                    tile_row == 0
                    and tile_n == 0
                    and row_count > 0
                    and 0 <= expert < self.experts
                    and 0 <= row_start < self.logical_m
                    and row_start + row_count <= self.logical_m
                )
            )
        )
        if not native_ranges:
            raise ValueError("native FC1 metadata contains no expert ranges")
        expected_row_start = 0
        for row_start, row_count, _expert in native_ranges:
            if row_start != expected_row_start:
                raise ValueError(
                    "native FC1 expert ranges do not contiguously cover routed rows"
                )
            expected_row_start += row_count
        if expected_row_start != self.logical_m:
            raise ValueError("native FC1 expert ranges do not cover every routed row")

        # Pad groups to 256 rows.  The token mapping uses -1 for these entries,
        # which the gather kernel predicates away; no routed activation is
        # materialized before the timed launch.
        padded_source_chunks = []
        valid_padded_chunks = []
        valid_output_chunks = []
        padded_ranges = []
        padded_start = 0
        for row_start, row_count, expert in native_ranges:
            padding = (-row_count) % 256
            rows = source_rows[row_start : row_start + row_count]
            if padding:
                rows = torch.cat(
                    (
                        rows,
                        torch.full(
                            (padding,), -1, device=rows.device, dtype=rows.dtype
                        ),
                    )
                )
            padded_source_chunks.append(rows)
            valid_padded_chunks.append(
                torch.arange(
                    padded_start,
                    padded_start + row_count,
                    device=source_rows.device,
                    dtype=torch.long,
                )
            )
            valid_output_chunks.append(
                torch.arange(
                    row_start,
                    row_start + row_count,
                    device=source_rows.device,
                    dtype=torch.long,
                )
            )
            padded_ranges.append((padded_start, row_count + padding, expert, row_count))
            padded_start += row_count + padding
        padded_source_rows = torch.cat(padded_source_chunks)
        self._valid_padded_rows = torch.cat(valid_padded_chunks)
        self._valid_output_rows = torch.cat(valid_output_chunks)
        self._expert_ranges = tuple(padded_ranges)
        self.m = padded_start
        self.out_target = out
        self.out_scale_target = out_scale

        # The fused-gather kernel consumes logical A/SFA rows, retaining the
        # native physical scale layout only at this API boundary.
        from sm110a_kernel.kernels.nvfp4_gemm import (
            from_blocked_nvfp4_scales,
            to_blocked_nvfp4_scales,
        )

        sfa_logical_2d = from_blocked_nvfp4_scales(sfa, self.tokens, self.k)
        sfb_logical_2d = from_blocked_nvfp4_scales(sfb, self.experts * self.n, self.k)

        self.a_storage = a
        self.sfa_storage = sfa_logical_2d.contiguous()
        # TensorRT-LLM maps a permuted row to an *expanded* token index and
        # divides by top-k in the LDGSTS producer.  Our native metadata stores
        # the token row directly, so lift it into that expanded-index domain.
        self.topk = self.logical_m // self.tokens
        self.token_id_mapping = (padded_source_rows * self.topk).contiguous()
        # Native FC1 stores [gate:64, up:64]; CuTE's gather epilogue consumes
        # [up:64, gate:64]. Reorder B and SFB together.
        swiglu_blocks = self.n // 128
        b_bytes = b.view(torch.uint8).reshape(
            self.experts, swiglu_blocks, 2, 64, self.k // 2
        )
        self.b_storage = (
            b_bytes[:, :, (1, 0)]
            .reshape_as(b.view(torch.uint8))
            .contiguous()
            .view(torch.float4_e2m1fn_x2)
            .reshape_as(b)
        )
        sfb_cute_logical_2d = sfb_logical_2d.reshape(
            self.experts, swiglu_blocks, 2, 64, self.k // 16
        )[:, :, (1, 0)].reshape_as(sfb_logical_2d)
        self.sfb_storage = to_blocked_nvfp4_scales(sfb_cute_logical_2d.contiguous())
        self.out_storage = torch.empty(
            (self.m, self.n // 4), device=out.device, dtype=out.dtype
        )
        self.out_scale_storage = torch.empty(
            (self.m * (self.n // 32),),
            device=out.device,
            dtype=torch.float8_e4m3fn,
        )

        # One alpha per expert (multiplicative identity).
        self.alpha = torch.ones((self.experts,), device=a.device, dtype=torch.float32)
        self.norm_const = torch.ones((1,), device=a.device, dtype=torch.float32)

        self.stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.selection: Selection | None = None

    def _cute_tile_metadata(self, config: CuteDslNvfp4Fc1Config):
        """Return gather kernel's per-M-tile expert and valid-row metadata."""
        cta_m = config.mma_tile_m
        if self.m % cta_m:
            raise ValueError(
                f"permuted M={self.m} must be divisible by CuTE CTA M={cta_m}"
            )

        tile_experts, tile_mn_limits = [], []
        for start, count, expert, valid_count in self._expert_ranges:
            valid_end = start + valid_count
            for row in range(start, start + count, cta_m):
                tile_experts.append(expert)
                tile_mn_limits.append(min(row + cta_m, valid_end))

        tile_expert_idx = torch.tensor(
            tile_experts, device=self.a_storage.device, dtype=torch.int32
        )
        tile_idx_to_mn_limit = torch.tensor(
            tile_mn_limits, device=self.a_storage.device, dtype=torch.int32
        )
        num_non_exiting = torch.tensor(
            [len(tile_experts)], device=self.a_storage.device, dtype=torch.int32
        )
        return tile_expert_idx, tile_idx_to_mn_limit, num_non_exiting

    def compatible_configs(self):
        """Return configs that pass ``can_implement`` for this problem shape."""
        valid = []
        for config in DEFAULT_CONFIGS:
            config.validate()
            if self.m % config.mma_tile_m != 0:
                continue
            if self.n % config.mma_tile_n != 0:
                continue
            if not BlockScaledContiguousGatherGroupedGemmKernel.can_implement(
                ab_dtype=cutlass.Float4E2M1FN,
                sf_dtype=cutlass.Float8E4M3FN,
                sf_vec_size=16,  # NVFP4
                c_dtype=cutlass.Float4E2M1FN,
                mma_tiler_mn=(config.mma_tile_m, config.mma_tile_n),
                cluster_shape_mn=(config.cluster_m, config.cluster_n),
                m=self.m,
                n=self.n,
                k=self.k,
                l=self.experts,
                a_major="k",
                b_major="k",
                c_major="n",
            ):
                continue
            valid.append(config)
        return tuple(valid)

    def _compile(self, config: CuteDslNvfp4Fc1Config):
        tile_expert_idx, tile_mn_limits, num_non_exiting = self._cute_tile_metadata(
            config
        )
        gemm = BlockScaledContiguousGatherGroupedGemmKernel(
            sf_vec_size=16,
            mma_tiler_mn=(config.mma_tile_m, config.mma_tile_n),
            cluster_shape_mn=(config.cluster_m, config.cluster_n),
            vectorized_f32=True,
            topk=self.topk,
            activation_type=ActivationType.Swiglu,
        )
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            config.cluster_m * config.cluster_n
        )

        a_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float4E2M1FN,
            self.a_storage.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
            assumed_align=32,
        )
        b_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float4E2M1FN,
            self.b_storage.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
            assumed_align=32,
        )
        a_sf_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float8E4M3FN,
            self.sfa_storage.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
            assumed_align=16,
        )
        b_sf_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float8E4M3FN,
            self.sfb_storage.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
            assumed_align=16,
        )
        c_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float4E2M1FN,
            self.out_storage.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
            assumed_align=16,
        )
        c_sf_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float8E4M3FN,
            self.out_scale_storage.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
            assumed_align=16,
        )
        norm_const_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float32, self.norm_const.data_ptr(), cutlass.cute.AddressSpace.gmem
        )
        alpha_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float32, self.alpha.data_ptr(), cutlass.cute.AddressSpace.gmem
        )
        tile_expert_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Int32, tile_expert_idx.data_ptr(), cutlass.cute.AddressSpace.gmem
        )
        tile_mn_limit_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Int32, tile_mn_limits.data_ptr(), cutlass.cute.AddressSpace.gmem
        )
        token_mapping_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Int32,
            self.token_id_mapping.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
        )
        num_tiles_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Int32, num_non_exiting.data_ptr(), cutlass.cute.AddressSpace.gmem
        )

        compile_args = [
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            c_ptr,
            c_sf_ptr,
            norm_const_ptr,
            alpha_ptr,
            tile_expert_ptr,
            tile_mn_limit_ptr,
            token_mapping_ptr,
            num_tiles_ptr,
            self.tokens,
            self.m,
            self.n,
            self.k,
            self.experts,
        ]

        fake_stream = make_fake_stream()
        with _suppress_cutlass_pointer_deprecation():
            return cute.compile(
                _launch_fc1,
                gemm,
                *compile_args,
                tile_size=config.mma_tile_m,
                scaling_vector_size=16,
                max_active_clusters=max_active_clusters,
                stream=fake_stream,
                options="--opt-level 2",
            )

    def _prepare(self, config: CuteDslNvfp4Fc1Config):
        compiled = self._compile(config)
        tile_expert_idx, tile_mn_limits, num_non_exiting = self._cute_tile_metadata(
            config
        )

        a_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float4E2M1FN,
            self.a_storage.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
            assumed_align=32,
        )
        b_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float4E2M1FN,
            self.b_storage.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
            assumed_align=32,
        )
        a_sf_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float8E4M3FN,
            self.sfa_storage.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
            assumed_align=16,
        )
        b_sf_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float8E4M3FN,
            self.sfb_storage.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
            assumed_align=16,
        )
        c_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float4E2M1FN,
            self.out_storage.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
            assumed_align=16,
        )
        c_sf_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float8E4M3FN,
            self.out_scale_storage.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
            assumed_align=16,
        )
        norm_const_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float32, self.norm_const.data_ptr(), cutlass.cute.AddressSpace.gmem
        )
        alpha_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Float32, self.alpha.data_ptr(), cutlass.cute.AddressSpace.gmem
        )
        tile_expert_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Int32, tile_expert_idx.data_ptr(), cutlass.cute.AddressSpace.gmem
        )
        tile_mn_limit_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Int32, tile_mn_limits.data_ptr(), cutlass.cute.AddressSpace.gmem
        )
        token_mapping_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Int32,
            self.token_id_mapping.data_ptr(),
            cutlass.cute.AddressSpace.gmem,
        )
        num_tiles_ptr = cutlass.cute.runtime.make_ptr(
            cutlass.Int32, num_non_exiting.data_ptr(), cutlass.cute.AddressSpace.gmem
        )

        def run_kernel():
            # Keep metadata tensors alive for the lifetime of this closure:
            # ``compiled`` only receives their raw device pointers.
            _ = tile_expert_idx, tile_mn_limits, num_non_exiting
            compiled(
                a_ptr,
                b_ptr,
                a_sf_ptr,
                b_sf_ptr,
                c_ptr,
                c_sf_ptr,
                norm_const_ptr,
                alpha_ptr,
                tile_expert_ptr,
                tile_mn_limit_ptr,
                token_mapping_ptr,
                num_tiles_ptr,
                self.tokens,
                self.m,
                self.n,
                self.k,
                self.experts,
                stream=self.stream,
            )

        run_kernel()
        torch.cuda.synchronize()
        return PreparedKernel(config, run_kernel)

    def _scatter_output(self):
        """Materialize the logical output after an untimed CuTE launch."""
        self.out_target.view(torch.uint8).index_copy_(
            0,
            self._valid_output_rows,
            self.out_storage.view(torch.uint8).index_select(0, self._valid_padded_rows),
        )
        from sm110a_kernel.kernels.nvfp4_gemm import (
            from_blocked_nvfp4_scales,
            to_blocked_nvfp4_scales,
        )

        logical_scales = from_blocked_nvfp4_scales(
            self.out_scale_storage, self.m, self.n // 2
        ).index_select(0, self._valid_padded_rows)
        self.out_scale_target.copy_(to_blocked_nvfp4_scales(logical_scales))

    def autotune(self, options: TuneOptions | None = None):
        options = options or TuneOptions()
        configs = self.compatible_configs()
        if not configs:
            raise RuntimeError(
                f"no CuTE DSL FC1 config supports "
                f"tokens={self.tokens}, N={self.n // 2}, K={self.k}, "
                f"experts={self.experts}"
            )
        self.selection = AutotuneRunner(options).resolve(
            configs,
            namespace="cute_dsl_nvfp4_fc1",
            # v7 enables fused NVFP4 output quantization and measures the CuTE
            # launch alone.  The output scatter is an
            # independent PyTorch kernel and must not influence tactic choice.
            key=(
                "v7-gather64-fp4-pure",
                self.tokens,
                self.experts,
                self.m,
                self.n // 2,
                self.k,
            ),
            prepare=self._prepare,
            benchmark=lambda prepared: _stats(prepared.run, options),
            probe=lambda prepared: _quick_probe(prepared.run, options),
            format_candidate=format_config,
            metric_formatter=lambda _c, stats: (
                # FC1 computes both gate and up projections: 4MNK FLOPs.
                f"{4.0 * self.m * (self.n // 2) * self.k / (stats['median_ms'] * 1e9):.1f} TF/s"
            ),
            pre_jit=self._compile,
        )
        return self.selection

    def run_kernel(self):
        """Launch only the fused CuTE DSL gather/GEMM/SwiGLU kernel."""
        if self.selection is None:
            self.autotune()
        self.selection.prepared.run()

    def run(self):
        """Launch the kernel, then restore its padded output to logical order."""
        self.run_kernel()
        self._scatter_output()


def cute_dsl_nvfp4_fc1_out(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    out: torch.Tensor,
    out_scale: torch.Tensor,
    source_rows: torch.Tensor,
    tasks: torch.Tensor,
    state: torch.Tensor,
    options: TuneOptions | None = None,
):
    """Autotuned CuTE DSL NVFP4 FC1 with fused NVFP4 output quantization.

    Pre-gathers A and SFA via ``source_rows`` on the Python side, then
    delegates to the persistent grouped-GEMM SwiGLU kernel.

    Args:
        a: Packed E2M1 activations ``(tokens, K/2)``.
        b: Packed E2M1 weights ``(experts, 2*N, K/2)`` (interleaved gate/up).
        sfa: E4M3 activation scales ``(tokens, K/16)``.
        sfb: E4M3 weight scales ``(experts*2*N, K/16)``, flat.
        out: Packed E2M1 output ``(tokens*topk, N/2)`` (pre-allocated).
        out_scale: Blocked E4M3 scales, one per 16 logical output values.
        source_rows: int32 map from permuted row → source token.
        tasks: int32 flat ExpertTileTask array.
        state: int32 ``[num_tasks, 0]``.
        options: Autotune options.
    """
    runner = CuteDslNvfp4Fc1Runner(
        a, b, sfa, sfb, out, out_scale, source_rows, tasks, state
    )
    runner.autotune(options)
    runner.run()
    return runner
