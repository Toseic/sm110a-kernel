"""Autotuned TensorRT-LLM CuTE DSL NVFP4 GEMM baseline."""
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

_POINTER_DEPRECATION = "Use explicit `struct.scalar.ptr` for pointer instead."
_PROBE_WARMUP = 2
_ROOT = Path(__file__).resolve().parents[3]
_KERNEL_ROOT = _ROOT / "baselinedir" / "TensorRT-LLM" / "tensorrt_llm" / "_torch" / "cute_dsl_kernels"
if str(_KERNEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_KERNEL_ROOT))
from blackwell.dense_blockscaled_gemm_persistent import Sm100BlockScaledPersistentDenseGemmKernel

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
class CuteDslNvfp4GemmConfig:
    mma_tile_m: int
    mma_tile_n: int
    cluster_m: int
    cluster_n: int

DEFAULT_CONFIGS = tuple(
    CuteDslNvfp4GemmConfig(tile_m, tile_n, cluster_m, cluster_n)
    for tile_m in (128, 256) for tile_n in (64, 128, 256)
    for cluster_m in (1, 2, 4) for cluster_n in (1, 2, 4)
    if (tile_m != 256 or cluster_m % 2 == 0) and cluster_m * cluster_n <= 16
)

def format_config(config):
    return f"{config.mma_tile_m}x{config.mma_tile_n}/c{config.cluster_m}x{config.cluster_n}"

@cute.jit
def _launch(gemm_op: cutlass.Constexpr, a: cute.Tensor, b: cute.Tensor, sfa: cute.Tensor, sfb: cute.Tensor, c: cute.Tensor, alpha: cute.Tensor, max_active_clusters: cutlass.Constexpr, stream: cuda.CUstream):
    gemm_op(a, b, sfa, sfb, c, alpha, max_active_clusters, stream)

def _measure(run, iters, cold_l2_flush):
    if cold_l2_flush is None:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters): run()
        end.record(); end.synchronize()
        return start.elapsed_time(end) / iters
    events = []
    for _ in range(iters):
        cold_l2_flush()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); run(); end.record(); events.append((start, end))
    events[-1][1].synchronize()
    return sum(start.elapsed_time(end) for start, end in events) / iters

def _stats(run, options):
    for _ in range(options.warmup): run()
    torch.cuda.synchronize()
    samples = sorted(_measure(run, options.iters, options.cold_l2_flush) for _ in range(options.samples))
    return {"min_ms": samples[0], "median_ms": statistics.median(samples), "max_ms": samples[-1]}

def _quick_probe(run, options):
    for _ in range(_PROBE_WARMUP): run()
    torch.cuda.synchronize()
    return _measure(run, options.iters, options.cold_l2_flush)

class CuteDslNvfp4GemmRunner:
    """Prepare, autotune, cache, and run CuTE DSL NVFP4 ``out = a @ b``."""
    def __init__(self, a, b, sfa, sfb, out):
        if not all(t.is_cuda for t in (a, b, sfa, sfb, out)):
            raise ValueError("all CuTE DSL NVFP4 GEMM tensors must be CUDA tensors")
        if a.dtype != torch.float4_e2m1fn_x2 or b.dtype != torch.float4_e2m1fn_x2:
            raise ValueError("a and b must be packed torch.float4_e2m1fn_x2 tensors")
        if sfa.dtype != torch.float8_e4m3fn or sfb.dtype != torch.float8_e4m3fn or out.dtype != torch.bfloat16:
            raise ValueError("expected E4M3 scales and BF16 output")
        if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
            raise ValueError("expected 2D packed inputs and BF16 output")
        self.m, packed_k = a.shape; self.n, packed_k_b = b.shape; self.k = packed_k * 2
        if packed_k != packed_k_b or out.shape != (self.m, self.n):
            raise ValueError("expected a(M,K/2), b(N,K/2), and out(M,N)")
        if sfa.numel() != self.m * self.k // 16 or sfb.numel() != self.n * self.k // 16:
            raise ValueError("scale tensor sizes are incompatible with packed operands")
        # TensorRT-LLM's block-scaled persistent kernel uses the batched CuTE
        # ABI even for a single GEMM: its scale-factor layouts have modes
        # (MN, K, L).  Keep L=1 as a trailing dimension instead of dropping
        # it, otherwise CUTLASS cannot tile the scale-factor atom layout.
        # These are views, so A/B/C retain their original storage and callers
        # continue to observe the 2D output tensor passed to this runner.
        def batch1_view(tensor):
            # ``unsqueeze`` gives its size-one dimension stride 1.  CuTE's
            # DLPack layout validator needs the non-overlapping batch stride
            # used by TensorRT-LLM, even when that dimension has extent one.
            return tensor.as_strided(
                (*tensor.shape, 1), (*tensor.stride(), tensor.numel())
            )

        self.a_storage = batch1_view(a.contiguous())
        self.b_storage = batch1_view(b.contiguous())
        self.c_storage = batch1_view(out)
        self.sfa_storage, self.sfb_storage = sfa.contiguous(), sfb.contiguous()
        self.alpha_storage = torch.ones((1,), device=a.device, dtype=torch.float32)
        self.a, self.b = from_dlpack(self.a_storage, assumed_align=16), from_dlpack(self.b_storage, assumed_align=16)
        self.sfa, self.sfb = from_dlpack(self.sfa_storage, assumed_align=16), from_dlpack(self.sfb_storage, assumed_align=16)
        self.c, self.alpha = from_dlpack(self.c_storage, assumed_align=16), from_dlpack(self.alpha_storage).mark_layout_dynamic()
        for tensor in (self.a, self.b):
            tensor.mark_compact_shape_dynamic(
                mode=1, stride_order=(2, 0, 1), divisibility=2
            )
        self.c.mark_compact_shape_dynamic(
            mode=1, stride_order=(2, 0, 1), divisibility=1
        )
        self.stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.selection = None

    def compatible_configs(self):
        # Thor accepts the 1/2/4 CTA dimensions used here, but a 4x4 cluster
        # fails at launch (the TensorRT-LLM Blackwell kernel permits 16 CTAs).
        return tuple(c for c in DEFAULT_CONFIGS if c.cluster_m * c.cluster_n <= 8 and Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
            cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16, cutlass.BFloat16,
            (c.mma_tile_m, c.mma_tile_n), (c.cluster_m, c.cluster_n), self.m, self.n, self.k, 1, "k", "k", "n"))

    def _compile(self, config):
        gemm = Sm100BlockScaledPersistentDenseGemmKernel(16, (config.mma_tile_m, config.mma_tile_n), (config.cluster_m, config.cluster_n))
        active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(config.cluster_m * config.cluster_n)
        with _suppress_cutlass_pointer_deprecation():
            return cute.compile(_launch, gemm, self.a, self.b, self.sfa, self.sfb, self.c, self.alpha, active_clusters, make_fake_stream(), options="--opt-level 2")

    def _prepare(self, config):
        compiled = self._compile(config)
        run = lambda: compiled(self.a, self.b, self.sfa, self.sfb, self.c, self.alpha, self.stream)
        run(); torch.cuda.synchronize()
        return PreparedKernel(config, run)

    def autotune(self, options: TuneOptions | None = None):
        options = options or TuneOptions(); configs = self.compatible_configs()
        if not configs: raise RuntimeError(f"no CuTE DSL config supports M={self.m}, N={self.n}, K={self.k}")
        self.selection = AutotuneRunner(options).resolve(
            configs, namespace="cute_dsl_nvfp4_gemm", key=(self.m, self.n, self.k), prepare=self._prepare,
            benchmark=lambda prepared: _stats(prepared.run, options), probe=lambda prepared: _quick_probe(prepared.run, options),
            format_candidate=format_config, metric_formatter=lambda _c, stats: f"{2.0 * self.m * self.n * self.k / (stats['median_ms'] * 1e9):.1f} TF/s", pre_jit=self._compile)
        return self.selection

    def run(self):
        if self.selection is None: self.autotune()
        self.selection.prepared.run()

def cute_dsl_nvfp4_gemm_out(a, b, sfa, sfb, out, options: TuneOptions | None = None):
    runner = CuteDslNvfp4GemmRunner(a, b, sfa, sfb, out)
    runner.autotune(options); runner.run()
    return runner
