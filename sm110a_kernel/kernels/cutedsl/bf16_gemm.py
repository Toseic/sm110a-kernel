"""Autotuned external runner for TensorRT-LLM's persistent CuTE DSL BF16 GEMM."""

from __future__ import annotations

import statistics
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from ...runner import AutotuneRunner, PreparedKernel, Selection, TuneOptions


# CUTLASS DSL 4.5 emits this warning for its own generated pointer plumbing.
# The TensorRT-LLM kernel currently requires that compatibility path; suppress
# only this known message so benchmark/autotune output remains readable.
_POINTER_DEPRECATION = "Use explicit `struct.scalar.ptr` for pointer instead."
_PROBE_WARMUP = 2


@contextmanager
def _suppress_cutlass_pointer_deprecation():
    """Suppress CUTLASS's forced-always pointer compatibility warning only."""
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


_ROOT = Path(__file__).resolve().parents[3]
_KERNEL_ROOT = _ROOT / "baselinedir" / "TensorRT-LLM" / "tensorrt_llm" / "_torch" / "cute_dsl_kernels"
if str(_KERNEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_KERNEL_ROOT))

from blackwell.dense_gemm_persistent import PersistentDenseGemmKernel


@dataclass(frozen=True)
class CuteDslBf16GemmConfig:
    mma_tile_m: int
    mma_tile_n: int
    cluster_m: int
    cluster_n: int
    use_2cta_instrs: bool = False


# Keep both the ordinary 1-CTA path and the 2-CTA/M-major cluster paths in
# the sweep.  The latter are important on Thor and are also used by the local
# kernel implementation.  ``compatible_configs`` below remains the authority
# for shape/alignment support.
DEFAULT_CONFIGS = (
    CuteDslBf16GemmConfig(64, 128, 1, 1),
    CuteDslBf16GemmConfig(128, 128, 1, 1),
    CuteDslBf16GemmConfig(128, 128, 1, 2),
    CuteDslBf16GemmConfig(128, 128, 1, 4),
    CuteDslBf16GemmConfig(128, 256, 1, 1),
    CuteDslBf16GemmConfig(128, 256, 1, 2),
    CuteDslBf16GemmConfig(128, 128, 2, 1),
    CuteDslBf16GemmConfig(128, 128, 2, 2),
    CuteDslBf16GemmConfig(128, 256, 2, 1),
    CuteDslBf16GemmConfig(128, 256, 2, 2),
    CuteDslBf16GemmConfig(128, 128, 2, 1, True),
    CuteDslBf16GemmConfig(128, 128, 2, 2, True),
    CuteDslBf16GemmConfig(128, 256, 2, 1, True),
    CuteDslBf16GemmConfig(128, 256, 2, 2, True),
    CuteDslBf16GemmConfig(256, 128, 2, 1, True),
    CuteDslBf16GemmConfig(256, 128, 2, 2, True),
    CuteDslBf16GemmConfig(256, 256, 2, 1, True),
    CuteDslBf16GemmConfig(256, 256, 2, 2, True),
)


def format_config(config: CuteDslBf16GemmConfig) -> str:
    return (
        f"{config.mma_tile_m}x{config.mma_tile_n}"
        f"/c{config.cluster_m}x{config.cluster_n}"
        f"/{'2cta' if config.use_2cta_instrs else '1cta'}"
    )


@cute.jit
def _launch_dense_gemm(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    gemm_op(a, b, c, max_active_clusters, stream)


def _measure(run, iters: int, cold_l2_flush) -> float:
    """Measure launches with the same cold-L2 semantics as benchmark helpers."""
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


def _stats(run, options: TuneOptions) -> dict:
    for _ in range(options.warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(options.samples):
        samples.append(_measure(run, options.iters, options.cold_l2_flush))
    samples.sort()
    return {"min_ms": samples[0], "median_ms": statistics.median(samples), "max_ms": samples[-1]}


def _quick_probe(run, options: TuneOptions) -> float:
    """Return a fixed-iteration timing used to prune slow configurations."""
    for _ in range(_PROBE_WARMUP):
        run()
    torch.cuda.synchronize()

    return _measure(run, options.iters, options.cold_l2_flush)


class CuteDslBf16GemmRunner:
    """Prepare, autotune, cache, and run CuTE DSL GEMM for ``out = a @ b``.

    ``a`` is contiguous ``(M, K)`` BF16 and ``b`` is contiguous ``(K, N)`` BF16.
    The runner owns a transposed ``(N, K)`` storage view required by the
    TensorRT-LLM kernel and keeps all tensors alive for compiled calls.
    """

    def __init__(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
        if not (a.is_cuda and b.is_cuda and out.is_cuda):
            raise ValueError("a, b, and out must be CUDA tensors")
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise ValueError("CuTE DSL BF16 GEMM requires bfloat16 tensors")
        if a.ndim != 2 or b.ndim != 2 or out.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("expected a(M,K), b(K,N), and out(M,N)")
        if out.shape != (a.shape[0], b.shape[1]):
            raise ValueError("out shape must be (M, N)")
        self.m, self.k, self.n = a.shape[0], a.shape[1], b.shape[1]
        self.a_storage = a.contiguous().unsqueeze(-1)
        self.b_storage = b.transpose(0, 1).contiguous().unsqueeze(-1)
        self.c_storage = out.unsqueeze(-1)
        self.a = from_dlpack(self.a_storage, assumed_align=16)
        self.b = from_dlpack(self.b_storage, assumed_align=16)
        self.c = from_dlpack(self.c_storage, assumed_align=16)
        for tensor in (self.a, self.b, self.c):
            tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=1)
        self.stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.selection: Selection[CuteDslBf16GemmConfig] | None = None

    def compatible_configs(self):
        return tuple(
            config for config in DEFAULT_CONFIGS
            if PersistentDenseGemmKernel.can_implement(
                cutlass.BFloat16, cutlass.Float32, cutlass.BFloat16,
                config.use_2cta_instrs,
                (config.mma_tile_m, config.mma_tile_n), (config.cluster_m, config.cluster_n),
                self.m, self.n, self.k, 1, "k", "k", "n",
            )
        )

    def _prepare(self, config: CuteDslBf16GemmConfig) -> PreparedKernel[CuteDslBf16GemmConfig]:
        gemm = PersistentDenseGemmKernel(
            cutlass.Float32, config.use_2cta_instrs,
            (config.mma_tile_m, config.mma_tile_n),
            (config.cluster_m, config.cluster_n), use_tma_store=True,
        )
        active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            config.cluster_m * config.cluster_n
        )
        with _suppress_cutlass_pointer_deprecation():
            compiled = cute.compile(
                _launch_dense_gemm, gemm, self.a, self.b, self.c, active_clusters,
                make_fake_stream(), options="--opt-level 2",
            )
        run = lambda: compiled(self.a, self.b, self.c, self.stream)
        run()
        torch.cuda.synchronize()
        return PreparedKernel(config, run)

    def autotune(self, options: TuneOptions | None = None) -> Selection[CuteDslBf16GemmConfig]:
        options = options or TuneOptions()
        configs = self.compatible_configs()
        if not configs:
            raise RuntimeError(f"no CuTE DSL config supports M={self.m}, N={self.n}, K={self.k}")
        self.selection = AutotuneRunner(options).resolve(
            configs, namespace="cute_dsl_bf16_gemm", key=(self.m, self.n, self.k),
            prepare=self._prepare, benchmark=lambda prepared: _stats(prepared.run, options),
            probe=lambda prepared: _quick_probe(prepared.run, options),
            format_candidate=format_config,
            metric_formatter=lambda _config, stats: (
                f"{2.0 * self.m * self.n * self.k / (stats['median_ms'] * 1.0e9):.1f} TF/s"
            ),
        )
        return self.selection

    def run(self) -> None:
        if self.selection is None:
            self.autotune()
        self.selection.prepared.run()


def cute_dsl_bf16_gemm_out(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, options: TuneOptions | None = None
) -> CuteDslBf16GemmRunner:
    """Autotune on first use, write ``a @ b`` to ``out``, and return its runner."""
    runner = CuteDslBf16GemmRunner(a, b, out)
    runner.autotune(options)
    runner.run()
    return runner
