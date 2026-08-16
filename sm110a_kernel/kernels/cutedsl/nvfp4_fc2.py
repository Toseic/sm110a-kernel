"""Autotuned TensorRT-LLM CuTE DSL NVFP4 routed FC2 baseline.

This directly calls TRT-LLM's
``Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel`` (grouped NVFP4
GEMM with token-major routing reduction fused into the epilogue), mirroring the
FC1 CuTE DSL baseline wiring.
"""
from __future__ import annotations

import statistics
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_ptr

from ...runner import AutotuneRunner, PreparedKernel, Selection, TuneOptions

_POINTER_DEPRECATION = "Use explicit `struct.scalar.ptr` for pointer instead."
_PROBE_WARMUP = 2
_ROOT = Path(__file__).resolve().parents[3]
_KERNEL_ROOT = (
    _ROOT
    / "baselinedir"
    / "TensorRT-LLM"
    / "tensorrt_llm"
    / "_torch"
    / "cute_dsl_kernels"
)
if str(_KERNEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_KERNEL_ROOT))

from blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import (  # noqa: E402
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
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
class CuteDslNvfp4Fc2Config:
    mma_tile_m: int = 128
    mma_tile_n: int = 128
    cluster_m: int = 1
    cluster_n: int = 1
    raster_along_m: bool = False

    def validate(self) -> None:
        if self.mma_tile_m not in (128, 256):
            raise ValueError("mma_tile_m must be 128 or 256")
        if self.mma_tile_n not in (64, 128, 192, 256):
            raise ValueError("mma_tile_n must be 64, 128, 192, or 256")
        if self.cluster_m != self.mma_tile_m // 128:
            raise ValueError("cluster_m must be mma_tile_m // 128")
        if self.cluster_n not in (1, 2, 4):
            raise ValueError("cluster_n must be 1, 2, or 4")
        if self.cluster_m * self.cluster_n > 8:
            raise ValueError("cluster size must be <= 8 on Thor")
        if not isinstance(self.raster_along_m, bool):
            raise TypeError("raster_along_m must be a bool")


DEFAULT_CONFIGS = tuple(
    CuteDslNvfp4Fc2Config(tile_m, tile_n, tile_m // 128, cluster_n)
    for tile_m in (128, 256)
    for tile_n in (128, 256)
    for cluster_n in (1, 2, 4)
)


def format_config(config: CuteDslNvfp4Fc2Config) -> str:
    return (
        f"{config.mma_tile_m}x{config.mma_tile_n}"
        f"/c{config.cluster_m}x{config.cluster_n}"
        f"/{'mmaj' if config.raster_along_m else 'nmaj'}"
    )


def _measure(run, iters, cold_l2_flush):
    if cold_l2_flush is None:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iters
    events = []
    for _ in range(iters):
        cold_l2_flush()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
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


class CuteDslNvfp4Fc2Runner:
    """Prepare, autotune, and run the TRT-LLM fused finalize FC2 kernel.

    The kernel consumes expert-grouped packed A/SFA and writes the
    token-major, routing-weighted reduction directly into ``out``.  A/SFA
    groups are padded to 256 rows so both 128 and 256 MMA tile-M tactics are
    valid; padding rows are materialized once in the runner and are never
    written to the output.
    """

    def __init__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        sfa: torch.Tensor,
        sfb: torch.Tensor,
        routing: torch.Tensor,
        routing_weights: torch.Tensor,
        out: torch.Tensor,
        topk: int,
    ):
        if not all(t.is_cuda for t in (a, b, sfa, sfb, routing, routing_weights, out)):
            raise ValueError("all CuTE DSL NVFP4 FC2 tensors must be CUDA tensors")
        if a.dtype != torch.float4_e2m1fn_x2 or b.dtype != torch.float4_e2m1fn_x2:
            raise ValueError("a and b must be packed torch.float4_e2m1fn_x2 tensors")
        if sfa.dtype != torch.float8_e4m3fn or sfb.dtype != torch.float8_e4m3fn:
            raise ValueError("expected E4M3 scale factors")
        if out.dtype != torch.bfloat16:
            raise ValueError("expected BF16 output")
        if a.ndim != 2 or b.ndim != 3 or out.ndim != 2:
            raise ValueError("expected a[M,K/2], b[experts,N,K/2], and out[tokens,N]")
        if routing.dtype != torch.int32 or routing.ndim != 1:
            raise ValueError("routing must be a contiguous CUDA int32 metadata vector")

        self.m, packed_k = a.shape
        self.experts, self.n, packed_k_b = b.shape
        self.k = packed_k * 2
        self.tokens, n_out = out.shape
        self.topk = int(topk)

        if packed_k != packed_k_b:
            raise ValueError("K dimension mismatch between a and b")
        if n_out != self.n:
            raise ValueError("out shape does not match B")
        if self.m != self.tokens * self.topk:
            raise ValueError("M must equal tokens * topk")
        if self.m % 128:
            raise ValueError("M must be divisible by 128 for blocked scales")
        if self.n % 128 or self.k % 64:
            raise ValueError("N must be divisible by 128 and K by 64")
        if self.topk <= 0 or routing_weights.shape != (self.tokens, self.topk):
            raise ValueError("invalid topk/routing_weights shape")
        if sfa.numel() != self.m * (self.k // 16):
            raise ValueError(
                f"sfa has {sfa.numel()} elements, expected {self.m * (self.k // 16)}"
            )
        if sfb.numel() != self.experts * self.n * (self.k // 16):
            raise ValueError(
                f"sfb has {sfb.numel()} elements, "
                f"expected {self.experts * self.n * (self.k // 16)}"
            )
        if routing.numel() != 2 * self.experts + 3 * self.m:
            raise ValueError("routing has an invalid packed metadata size")

        routing = routing.contiguous()
        sfa = sfa.contiguous()
        sfb = sfb.contiguous()
        self._grouped_layout = routing[2 * self.experts : 2 * self.experts + self.m]
        self._grouped_assignments = routing[
            2 * self.experts + self.m : 2 * self.experts + 2 * self.m
        ]

        # Native FC2 routing is already grouped by expert.  Recover the expert
        # ranges on the host once; only these ranges are needed to build CuTE
        # tile metadata and to pad A/SFA for this kernel.
        layout_cpu = self._grouped_layout.cpu()
        counts = torch.bincount(layout_cpu.to(torch.long), minlength=self.experts).tolist()
        native_ranges = []
        src_start = 0
        for expert, count in enumerate(counts):
            if count:
                native_ranges.append((src_start, count, expert))
                src_start += count
        if src_start != self.m:
            raise ValueError("grouped layout does not cover every grouped row")

        from sm110a_kernel.kernels.nvfp4_gemm import (
            from_blocked_nvfp4_scales,
            to_blocked_nvfp4_scales,
        )

        sfa_logical = from_blocked_nvfp4_scales(sfa, self.m, self.k).contiguous()
        chunks_a = []
        chunks_sfa = []
        chunks_perm = []
        self._padded_ranges = []
        padded_start = 0
        for src_start, count, expert in native_ranges:
            padding = (-count) % 256
            padded_count = count + padding
            self._padded_ranges.append((padded_start, padded_count, expert, count))
            chunks_a.append(a[src_start : src_start + count].contiguous())
            chunks_sfa.append(sfa_logical[src_start : src_start + count].contiguous())
            chunks_perm.append(
                self._grouped_assignments[src_start : src_start + count].contiguous()
            )
            if padding:
                chunks_a.append(
                    torch.zeros(
                        (padding, packed_k), device=a.device, dtype=torch.uint8
                    ).view(torch.float4_e2m1fn_x2)
                )
                chunks_sfa.append(
                    torch.zeros(
                        (padding, self.k // 16), device=a.device, dtype=sfa.dtype
                    )
                )
                chunks_perm.append(
                    torch.zeros((padding,), device=a.device, dtype=torch.int32)
                )
            padded_start += padded_count

        self.m_padded = padded_start
        self.a_storage = torch.cat(chunks_a).contiguous()
        self.sfa_storage = to_blocked_nvfp4_scales(torch.cat(chunks_sfa).contiguous())
        self._permuted_idx_to_expanded_idx = torch.cat(chunks_perm).contiguous()
        self.sfb_storage = sfb.contiguous()
        self.b_storage = b.contiguous()
        self.alpha = torch.ones(
            (self.experts,), device=a.device, dtype=torch.float32
        )
        self.token_final_scales = routing_weights.float().contiguous()
        self.out = out

        self.stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self._max_active_clusters = {
            config.cluster_m * config.cluster_n: cutlass.utils.HardwareInfo().get_max_active_clusters(
                config.cluster_m * config.cluster_n
            )
            for config in DEFAULT_CONFIGS
        }
        self.selection: Optional[Selection[CuteDslNvfp4Fc2Config]] = None

    def _cute_tile_metadata(self, config: CuteDslNvfp4Fc2Config):
        """Return per-MMA-tile expert and valid-row metadata."""
        cta_m = config.mma_tile_m
        tile_experts = []
        tile_mn_limits = []
        for start, count, expert, valid_count in self._padded_ranges:
            valid_end = start + valid_count
            for row in range(start, start + count, cta_m):
                tile_experts.append(expert)
                tile_mn_limits.append(min(row + cta_m, valid_end))

        tile_idx_to_group_idx = torch.tensor(
            tile_experts, device=self.a_storage.device, dtype=torch.int32
        )
        tile_idx_to_mn_limit = torch.tensor(
            tile_mn_limits, device=self.a_storage.device, dtype=torch.int32
        )
        num_non_exiting_tiles = torch.tensor(
            [len(tile_experts)], device=self.a_storage.device, dtype=torch.int32
        )
        return tile_idx_to_group_idx, tile_idx_to_mn_limit, num_non_exiting_tiles

    def compatible_configs(self):
        """Return configs that pass ``can_implement`` for this problem shape."""
        valid = []
        for config in DEFAULT_CONFIGS:
            config.validate()
            if self.m_padded % config.mma_tile_m:
                continue
            if self.n % config.mma_tile_n:
                continue
            # Launching more cluster CTAs than tiles causes out-of-bounds
            # accesses in the CuTE DSL scheduler.
            if (
                -(-self.m_padded // config.mma_tile_m) < config.cluster_m
                or -(-self.n // config.mma_tile_n) < config.cluster_n
            ):
                continue
            if not Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
                ab_dtype=cutlass.Float4E2M1FN,
                sf_dtype=cutlass.Float8E4M3FN,
                sf_vec_size=16,  # NVFP4
                out_dtype=cutlass.BFloat16,
                mma_tiler_mn=(config.mma_tile_m, config.mma_tile_n),
                cluster_shape_mn=(config.cluster_m, config.cluster_n),
                m=self.m_padded,
                n=self.n,
                k=self.k,
                l=self.experts,
                a_major="k",
                b_major="k",
                out_major="n",
            ):
                continue
            valid.append(config)
        return tuple(valid)

    def _compile(self, config: CuteDslNvfp4Fc2Config):
        tile_idx_to_group_idx, tile_idx_to_mn_limit, num_non_exiting_tiles = (
            self._cute_tile_metadata(config)
        )
        gemm = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
            sf_vec_size=16,
            mma_tiler_mn=(config.mma_tile_m, config.mma_tile_n),
            cluster_shape_mn=(config.cluster_m, config.cluster_n),
            raster_along_m=config.raster_along_m,
        )
        max_active_clusters = self._max_active_clusters[
            config.cluster_m * config.cluster_n
        ]

        a_ptr = make_ptr(
            cutlass.Float4E2M1FN,
            self.a_storage.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        b_ptr = make_ptr(
            cutlass.Float4E2M1FN,
            self.b_storage.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        a_sf_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            self.sfa_storage.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        b_sf_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            self.sfb_storage.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        c_ptr = make_ptr(
            cutlass.BFloat16,
            self.out.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        alpha_ptr = make_ptr(
            cutlass.Float32,
            self.alpha.data_ptr(),
            cute.AddressSpace.gmem,
        )
        tile_idx_to_group_idx_ptr = make_ptr(
            cutlass.Int32,
            tile_idx_to_group_idx.data_ptr(),
            cute.AddressSpace.gmem,
        )
        tile_idx_to_mn_limit_ptr = make_ptr(
            cutlass.Int32,
            tile_idx_to_mn_limit.data_ptr(),
            cute.AddressSpace.gmem,
        )
        permuted_idx_to_expanded_idx_ptr = make_ptr(
            cutlass.Int32,
            self._permuted_idx_to_expanded_idx.data_ptr(),
            cute.AddressSpace.gmem,
        )
        num_non_exiting_tiles_ptr = make_ptr(
            cutlass.Int32,
            num_non_exiting_tiles.data_ptr(),
            cute.AddressSpace.gmem,
        )
        token_final_scales_ptr = make_ptr(
            cutlass.Float32,
            self.token_final_scales.data_ptr(),
            cute.AddressSpace.gmem,
        )

        compile_args = [
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            c_ptr,
            alpha_ptr,
            tile_idx_to_group_idx_ptr,
            tile_idx_to_mn_limit_ptr,
            permuted_idx_to_expanded_idx_ptr,
            num_non_exiting_tiles_ptr,
            token_final_scales_ptr,
            self.m_padded,
            self.n,
            self.k,
            self.experts,
            self.tokens,
            self.topk,
        ]

        with _suppress_cutlass_pointer_deprecation():
            compiled = cute.compile(
                gemm.wrapper,
                *compile_args,
                tile_size=config.mma_tile_m,
                scaling_vector_size=16,
                max_active_clusters=max_active_clusters,
                stream=make_fake_stream(),
                options="--opt-level 2",
            )
        return compiled, tile_idx_to_group_idx, tile_idx_to_mn_limit, num_non_exiting_tiles

    def _prepare(self, config: CuteDslNvfp4Fc2Config):
        compiled, tile_idx_to_group_idx, tile_idx_to_mn_limit, num_non_exiting_tiles = (
            self._compile(config)
        )
        a_ptr = make_ptr(
            cutlass.Float4E2M1FN,
            self.a_storage.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        b_ptr = make_ptr(
            cutlass.Float4E2M1FN,
            self.b_storage.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        a_sf_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            self.sfa_storage.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        b_sf_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            self.sfb_storage.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        c_ptr = make_ptr(
            cutlass.BFloat16,
            self.out.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        alpha_ptr = make_ptr(
            cutlass.Float32,
            self.alpha.data_ptr(),
            cute.AddressSpace.gmem,
        )
        tile_idx_to_group_idx_ptr = make_ptr(
            cutlass.Int32,
            tile_idx_to_group_idx.data_ptr(),
            cute.AddressSpace.gmem,
        )
        tile_idx_to_mn_limit_ptr = make_ptr(
            cutlass.Int32,
            tile_idx_to_mn_limit.data_ptr(),
            cute.AddressSpace.gmem,
        )
        permuted_idx_to_expanded_idx_ptr = make_ptr(
            cutlass.Int32,
            self._permuted_idx_to_expanded_idx.data_ptr(),
            cute.AddressSpace.gmem,
        )
        num_non_exiting_tiles_ptr = make_ptr(
            cutlass.Int32,
            num_non_exiting_tiles.data_ptr(),
            cute.AddressSpace.gmem,
        )
        token_final_scales_ptr = make_ptr(
            cutlass.Float32,
            self.token_final_scales.data_ptr(),
            cute.AddressSpace.gmem,
        )

        def run():
            # Keep metadata tensors alive for the lifetime of this closure;
            # ``compiled`` only receives their raw device pointers.
            _ = tile_idx_to_group_idx, tile_idx_to_mn_limit, num_non_exiting_tiles
            compiled(
                a_ptr,
                b_ptr,
                a_sf_ptr,
                b_sf_ptr,
                c_ptr,
                alpha_ptr,
                tile_idx_to_group_idx_ptr,
                tile_idx_to_mn_limit_ptr,
                permuted_idx_to_expanded_idx_ptr,
                num_non_exiting_tiles_ptr,
                token_final_scales_ptr,
                self.m_padded,
                self.n,
                self.k,
                self.experts,
                self.tokens,
                self.topk,
                stream=self.stream,
            )

        # The finalize epilogue scatter-adds into ``out``.  Start every tactic
        # from a zeroed destination; timed iterations then accumulate on top of
        # this initial value without adding a memset to the measured path.
        self.out.zero_()
        run()
        torch.cuda.synchronize()
        return PreparedKernel(config, run)

    def autotune(self, options: TuneOptions | None = None):
        options = options or TuneOptions()
        # This TRT-LLM kernel enters CuTe's global MLIR Location context while
        # generating its launch.  CUTLASS DSL 4.6 does not make that context
        # thread-safe, so compiling it from ThreadPoolExecutor workers can fail.
        # Use the same safe serial fallback as the grouped-M CuTE DSL runner.
        if options.pre_parallel_jit:
            if options.debug:
                print(
                    "[jit] CuTE DSL FC2 pre-parallel JIT is unsupported; "
                    "using serial compilation",
                    flush=True,
                )
            options = replace(options, pre_parallel_jit=False)
        configs = self.compatible_configs()
        if not configs:
            raise RuntimeError(
                "no CuTE DSL FC2 config supports "
                f"tokens={self.tokens}, experts={self.experts}, N={self.n}, K={self.k}"
            )
        self.selection = AutotuneRunner(options).resolve(
            configs,
            namespace="cute_dsl_nvfp4_fc2",
            key=("v1", self.tokens, self.experts, self.topk, self.n, self.k),
            prepare=self._prepare,
            benchmark=lambda prepared: _stats(prepared.run, options),
            probe=lambda prepared: _quick_probe(prepared.run, options),
            format_candidate=format_config,
            metric_formatter=lambda _config, stats: (
                f"{2.0 * self.m * self.n * self.k / (stats['median_ms'] * 1e9):.1f} TF/s"
            ),
            pre_jit=lambda config: self._compile(config)[0],
        )
        return self.selection

    def run(self):
        if self.selection is None:
            self.autotune()
        self.selection.prepared.run()


def cute_dsl_nvfp4_fc2_out(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    routing: torch.Tensor,
    routing_weights: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    options: TuneOptions | None = None,
):
    """Autotune then write the fused finalize FC2 result to ``out``."""
    runner = CuteDslNvfp4Fc2Runner(
        a, b, sfa, sfb, routing, routing_weights, out, topk
    )
    runner.autotune(options)
    out.zero_()
    runner.run()
    return runner
