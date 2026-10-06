"""Autotuned TensorRT-LLM CuTE DSL NVFP4 contiguous grouped-M GEMM baseline."""

from __future__ import annotations

import statistics
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from ...runner import AutotuneRunner, PreparedKernel, Selection, TuneOptions
from ._baseline_loader import install_baseline_packages

install_baseline_packages()

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm import (  # noqa: E402
    Sm100BlockScaledContiguousGroupedGemmKernel,
)

_POINTER_DEPRECATION = "Use explicit `struct.scalar.ptr` for pointer instead."
_PROBE_WARMUP = 2


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
class CuteDslNvfp4GroupedMGemmConfig:
    mma_tile_m: int
    mma_tile_n: int
    cluster_m: int
    cluster_n: int
    raster_along_m: bool = False


DEFAULT_CONFIGS = tuple(
    CuteDslNvfp4GroupedMGemmConfig(tile_m, tile_n, cluster_m, cluster_n, raster_along_m)
    for tile_m in (128, 256)
    for tile_n in (64, 128, 192, 256)
    for cluster_m in (1, 2, 4)
    for cluster_n in (1, 2, 4)
    for raster_along_m in (False, True)
    if (tile_m != 256 or cluster_m % 2 == 0) and cluster_m * cluster_n <= 8
)


def format_config(config: CuteDslNvfp4GroupedMGemmConfig) -> str:
    return (
        f"{config.mma_tile_m}x{config.mma_tile_n}"
        f"/c{config.cluster_m}x{config.cluster_n}"
        f"/{'mmaj' if config.raster_along_m else 'nmaj'}"
    )


@cute.jit
def _launch(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    sfa: cute.Tensor,
    sfb: cute.Tensor,
    tile_idx_to_group_idx: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    alpha: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    gemm_op(
        a,
        b,
        c,
        sfa,
        sfb,
        tile_idx_to_group_idx,
        num_non_exiting_tiles,
        alpha,
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


class CuteDslNvfp4GroupedMGemmRunner:
    """Run TRT-LLM's persistent contiguous grouped GEMM for equal-M experts.

    ``a`` and ``out`` are contiguous ``[experts * tokens_per_expert, K]``
    tensors whose expert ranges are consecutive.  ``b`` is ``[experts, N, K]``.
    This is the same grouped-M representation used by the local benchmark.
    """

    def __init__(self, a, b, sfa, sfb, out, tokens_per_expert: int):
        if not all(t.is_cuda for t in (a, b, sfa, sfb, out)):
            raise ValueError(
                "all CuTE DSL NVFP4 grouped GEMM tensors must be CUDA tensors"
            )
        if a.dtype != torch.float4_e2m1fn_x2 or b.dtype != torch.float4_e2m1fn_x2:
            raise ValueError("a and b must be packed torch.float4_e2m1fn_x2 tensors")
        if sfa.dtype != torch.float8_e4m3fn or sfb.dtype != torch.float8_e4m3fn:
            raise ValueError("expected torch.float8_e4m3fn scale tensors")
        if out.dtype != torch.bfloat16:
            raise ValueError("expected BF16 output")
        if a.ndim != 2 or b.ndim != 3 or out.ndim != 2:
            raise ValueError("expected a[M,K/2], b[experts,N,K/2], and out[M,N]")
        if tokens_per_expert <= 0:
            raise ValueError("tokens_per_expert must be positive")

        self.m, packed_k = a.shape
        self.experts, self.n, packed_k_b = b.shape
        self.k = packed_k * 2
        self.tokens_per_expert = int(tokens_per_expert)
        if packed_k != packed_k_b or self.m != self.experts * self.tokens_per_expert:
            raise ValueError(
                "A rows must be contiguous equal-sized expert groups matching B"
            )
        if out.shape != (self.m, self.n):
            raise ValueError("out must have shape [M, N]")
        if self.n % 128 or self.k % 64:
            raise ValueError("N must be divisible by 128 and K by 64")
        expected_sfa = self.m * self.k // 16
        expected_sfb = self.experts * self.n * self.k // 16
        if sfa.numel() != expected_sfa or sfb.numel() != expected_sfb:
            raise ValueError(
                "scale tensor sizes are incompatible with the packed operands"
            )

        # The TRT-LLM kernel's logical tensors are A[M,K,1], B[N,K,E], and
        # C[M,N,1].  These are zero-copy views over our packed operands; B's
        # expert-major allocation already has the desired K,N,E strides.
        self.a_storage = a.contiguous().as_strided(
            (*a.shape, 1), (*a.stride(), a.numel())
        )
        self.b_storage = b.permute(1, 2, 0)
        self.c_storage = out.as_strided((*out.shape, 1), (*out.stride(), out.numel()))
        self.sfa_storage, self.sfb_storage = sfa.contiguous(), sfb.contiguous()
        self.alpha_storage = torch.ones(
            (self.experts,), device=a.device, dtype=torch.float32
        )
        self.a = from_dlpack(self.a_storage, assumed_align=16)
        self.b = from_dlpack(self.b_storage, assumed_align=16)
        self.c = from_dlpack(self.c_storage, assumed_align=16)
        self.sfa = from_dlpack(self.sfa_storage, assumed_align=16)
        self.sfb = from_dlpack(self.sfb_storage, assumed_align=16)
        self.alpha = from_dlpack(self.alpha_storage).mark_layout_dynamic()
        for tensor in (self.a, self.b):
            tensor.mark_compact_shape_dynamic(
                mode=1, stride_order=(2, 0, 1), divisibility=2
            )
        self.c.mark_compact_shape_dynamic(
            mode=1, stride_order=(2, 0, 1), divisibility=1
        )
        # CuTe compilation can safely run in the runner's parallel JIT pool,
        # but CUDA allocation/context initialization cannot.  Build the small
        # per-tile-size mapping tensors on the caller thread once and retain
        # both their torch storage and CuTe views for every compiled tactic.
        self._tile_metadata = {}
        for tile_m in {config.mma_tile_m for config in DEFAULT_CONFIGS}:
            if self.tokens_per_expert % tile_m:
                continue
            tiles_per_expert = self.tokens_per_expert // tile_m
            mapping_storage = torch.arange(
                self.experts, device=a.device, dtype=torch.int32
            ).repeat_interleave(tiles_per_expert)
            valid_tiles_storage = torch.tensor(
                (mapping_storage.numel(),), device=a.device, dtype=torch.int32
            )
            self._tile_metadata[tile_m] = (
                mapping_storage,
                valid_tiles_storage,
                from_dlpack(mapping_storage).mark_layout_dynamic(),
                from_dlpack(valid_tiles_storage).mark_layout_dynamic(),
            )
        self._max_active_clusters = {
            config.cluster_m
            * config.cluster_n: cutlass.utils.HardwareInfo().get_max_active_clusters(
                config.cluster_m * config.cluster_n
            )
            for config in DEFAULT_CONFIGS
        }
        self.stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.selection: Selection[CuteDslNvfp4GroupedMGemmConfig] | None = None

    def compatible_configs(self):
        compatible = []
        for config in DEFAULT_CONFIGS:
            # This baseline deliberately avoids materializing padding in A/C.
            # The native kernel supports padded groups, but the benchmark's
            # equal-M groups are already tile aligned for these configurations.
            if self.tokens_per_expert % config.mma_tile_m:
                continue
            if not Sm100BlockScaledContiguousGroupedGemmKernel.can_implement(
                cutlass.Float4E2M1FN,
                cutlass.Float8E4M3FN,
                16,
                cutlass.BFloat16,
                (config.mma_tile_m, config.mma_tile_n),
                (config.cluster_m, config.cluster_n),
                self.m,
                self.n,
                self.k,
                self.experts,
                "k",
                "k",
                "n",
            ):
                continue
            compatible.append(config)
        return tuple(compatible)

    def _metadata(self, config):
        try:
            _, _, tile_idx_to_group_idx, num_non_exiting_tiles = self._tile_metadata[
                config.mma_tile_m
            ]
        except KeyError as error:
            raise ValueError(
                f"tokens_per_expert={self.tokens_per_expert} is not aligned to "
                f"tile M={config.mma_tile_m}"
            ) from error
        return tile_idx_to_group_idx, num_non_exiting_tiles

    def _compile(self, config):
        tile_idx_to_group_idx, num_non_exiting_tiles = self._metadata(config)
        gemm = Sm100BlockScaledContiguousGroupedGemmKernel(
            16,
            (config.mma_tile_m, config.mma_tile_n),
            (config.cluster_m, config.cluster_n),
            config.raster_along_m,
        )
        active_clusters = self._max_active_clusters[config.cluster_m * config.cluster_n]
        with _suppress_cutlass_pointer_deprecation():
            compiled = cute.compile(
                _launch,
                gemm,
                self.a,
                self.b,
                self.c,
                self.sfa,
                self.sfb,
                tile_idx_to_group_idx,
                num_non_exiting_tiles,
                self.alpha,
                active_clusters,
                make_fake_stream(),
                options="--opt-level 2",
            )
        return compiled, tile_idx_to_group_idx, num_non_exiting_tiles

    def _prepare(self, config):
        compiled, tile_idx_to_group_idx, num_non_exiting_tiles = self._compile(config)
        run = lambda: compiled(
            self.a,
            self.b,
            self.c,
            self.sfa,
            self.sfb,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            self.alpha,
            self.stream,
        )
        run()
        torch.cuda.synchronize()
        return PreparedKernel(config, run)

    def autotune(self, options: TuneOptions | None = None):
        options = options or TuneOptions()
        # This TRT-LLM grouped kernel enters CuTe's global MLIR Location
        # context while generating its launch.  CUTLASS DSL 4.6 does not make
        # that context thread-safe: compiling it from ThreadPoolExecutor
        # workers raises ``No current Location`` and can abort nanobind.  Keep
        # the public parallel-JIT option harmless by safely falling back to
        # the normal serial compile-and-tune path for this kernel only.
        if options.pre_parallel_jit:
            if options.debug:
                print(
                    "[jit] CuTE DSL grouped-M pre-parallel JIT is unsupported; "
                    "using serial compilation",
                    flush=True,
                )
            options = replace(options, pre_parallel_jit=False)
        configs = self.compatible_configs()
        if not configs:
            raise RuntimeError(
                "no CuTE DSL grouped config supports "
                f"tokens/expert={self.tokens_per_expert}, N={self.n}, K={self.k}"
            )
        self.selection = AutotuneRunner(options).resolve(
            configs,
            namespace="cute_dsl_nvfp4_grouped_m_gemm",
            key=(self.experts, self.tokens_per_expert, self.n, self.k),
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


def cute_dsl_nvfp4_grouped_m_gemm_out(
    a,
    b,
    sfa,
    sfb,
    out,
    tokens_per_expert: int,
    options: TuneOptions | None = None,
):
    """Autotune then write the contiguous grouped NVFP4 GEMM result to ``out``."""
    runner = CuteDslNvfp4GroupedMGemmRunner(a, b, sfa, sfb, out, tokens_per_expert)
    runner.autotune(options)
    runner.run()
    return runner
