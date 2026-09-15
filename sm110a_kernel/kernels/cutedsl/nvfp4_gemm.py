"""Autotuned TensorRT-LLM CuTE DSL NVFP4 GEMM baseline."""
from __future__ import annotations

import statistics
import warnings
from functools import lru_cache
from contextlib import contextmanager
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from ...runner import AutotuneRunner, PreparedKernel, Selection, TuneOptions

from ._baseline_loader import install_baseline_packages

install_baseline_packages()

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import (
    Sm100BlockScaledPersistentDenseGemmKernel,
)

_POINTER_DEPRECATION = "Use explicit `struct.scalar.ptr` for pointer instead."
_PROBE_WARMUP = 2


@lru_cache(maxsize=None)
def _nvmmh_interface():
    """Create the optional NVIDIA Matmul Heuristics interface once."""
    import nvMatmulHeuristics as nvmmh
    return nvmmh, nvmmh.NvMatmulHeuristicsInterface(
        nvmmh.NvMatmulHeuristicsTarget.CUTLASS3,
        precision="F4F4T",
        flags=nvmmh.NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING,
    )


def _rank_nvmmh_configs(m, n, k, count):
    """Return TRT-LLM-compatible heuristic records, or no records on failure."""
    try:
        nvmmh, interface = _nvmmh_interface()
        layout = nvmmh.NvMatmulHeuristicsMatmulLayout.TN_ROW_MAJOR
        try:
            interface.loadInternalDiscoverySet(layout, None)
        except Exception:
            pass
        records = interface.get_with_mnk(int(m), int(n), int(k), layout, int(count), None)
        return sorted(records, key=lambda record: record.get("runtime", float("inf")))
    except Exception as error:  # Optional dependency/query must not break tuning.
        warnings.warn(
            f"nvMatmulHeuristics unavailable for CuTE DSL NVFP4: {error}; "
            "using the full tactic sweep",
            RuntimeWarning,
            stacklevel=2,
        )
        return ()

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
    # Keep the complete TRT-LLM tactic in the cache key.  ``swap_ab`` is a
    # mathematically equivalent C^T = B @ A^T launch that changes the output
    # layout and is frequently better for small M.  Prefetch and scheduler
    # knobs are compile-time kernel parameters too.
    swap_ab: bool = False
    use_prefetch: bool = False
    swizzle_size: int = 1
    raster_along_m: bool = True

DEFAULT_CONFIGS = tuple(
    CuteDslNvfp4GemmConfig(
        tile_m, tile_n, cluster_m, cluster_n, swap_ab, use_prefetch
    )
    for tile_m in (128, 256) for tile_n in (64, 128, 192, 256)
    for cluster_m in (1, 2, 4) for cluster_n in (1, 2, 4)
    for swap_ab in (False, True) for use_prefetch in (False, True)
    if (tile_m != 256 or cluster_m % 2 == 0) and cluster_m * cluster_n <= 16
)
_SCHEDULER_SWEEP = tuple(
    (swizzle_size, raster_along_m)
    for swizzle_size in (1, 2, 4, 8)
    for raster_along_m in (True, False)
)

def format_config(config):
    return (
        f"{config.mma_tile_m}x{config.mma_tile_n}"
        f"/c{config.cluster_m}x{config.cluster_n}"
        f"/{'swap' if config.swap_ab else 'noswap'}"
        f"/{'prefetch' if config.use_prefetch else 'noprefetch'}"
        f"/sw{config.swizzle_size}"
        f"/{'mmaj' if config.raster_along_m else 'nmaj'}"
    )

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
        # The swap tactic computes C^T = B @ A^T.  A/B remain K-major while
        # the transposed C view is M-major in the swapped GEMM coordinate
        # system.  This is the same no-copy layout transformation used by
        # TensorRT-LLM's CuTE DSL NVFP4 runner.
        self.c_swapped_storage = batch1_view(out.transpose(0, 1))
        self.sfa_storage, self.sfb_storage = sfa.contiguous(), sfb.contiguous()
        self.alpha_storage = torch.ones((1,), device=a.device, dtype=torch.float32)
        self.a, self.b = from_dlpack(self.a_storage, assumed_align=16), from_dlpack(self.b_storage, assumed_align=16)
        self.sfa, self.sfb = from_dlpack(self.sfa_storage, assumed_align=16), from_dlpack(self.sfb_storage, assumed_align=16)
        self.c = from_dlpack(self.c_storage, assumed_align=16)
        self.c_swapped = from_dlpack(self.c_swapped_storage, assumed_align=16)
        self.alpha = from_dlpack(self.alpha_storage).mark_layout_dynamic()
        for tensor in (self.a, self.b):
            tensor.mark_compact_shape_dynamic(
                mode=1, stride_order=(2, 0, 1), divisibility=2
            )
        self.c.mark_compact_shape_dynamic(
            mode=1, stride_order=(2, 0, 1), divisibility=1
        )
        self.c_swapped.mark_compact_shape_dynamic(
            mode=0, stride_order=(2, 1, 0), divisibility=1
        )
        self.stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.selection = None

    def compatible_configs(self, options: TuneOptions | None = None):
        # Thor accepts the 1/2/4 CTA dimensions used here, but a 4x4 cluster
        # fails at launch (the TensorRT-LLM Blackwell kernel permits 16 CTAs).
        compatible = []
        for config in DEFAULT_CONFIGS:
            # The upstream kernel accepts 4x4 clusters, but they fail at
            # launch on Thor.  Retain the hardware-specific <=8 safeguard.
            if config.cluster_m * config.cluster_n > 8:
                continue
            kernel_m, kernel_n = (
                (self.n, self.m) if config.swap_ab else (self.m, self.n)
            )
            c_major = "m" if config.swap_ab else "n"
            if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
                cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16,
                cutlass.BFloat16, (config.mma_tile_m, config.mma_tile_n),
                (config.cluster_m, config.cluster_n), kernel_m, kernel_n,
                self.k, 1, "k", "k", c_major,
            ):
                continue
            # Match TRT-LLM's prefetch pruning: only pay the extra JIT/tuning
            # cost when the workload has an incomplete CTA wave or a large K.
            ctas_m = -(-kernel_m // config.mma_tile_m)
            ctas_n = -(-kernel_n // config.mma_tile_n)
            ctas_m = -(-ctas_m // config.cluster_m) * config.cluster_m
            ctas_n = -(-ctas_n // config.cluster_n) * config.cluster_n
            cta_wave_ratio = (
                ctas_m * ctas_n /
                torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            )
            if config.use_prefetch and not (0.5 < cta_wave_ratio < 1.0 or self.k >= 8192):
                continue
            compatible.append(config)
        configs = tuple(compatible)
        if options and options.cutedsl_sweep_scheduler:
            configs = tuple(
                CuteDslNvfp4GemmConfig(
                    config.mma_tile_m, config.mma_tile_n, config.cluster_m,
                    config.cluster_n, config.swap_ab, config.use_prefetch,
                    swizzle_size, raster_along_m,
                )
                for config in configs
                for swizzle_size, raster_along_m in _SCHEDULER_SWEEP
            )
        return self._rank_prune_configs(configs, options)

    def _rank_prune_configs(self, configs, options):
        """Apply TRT-LLM's optional nvMatmulHeuristics tactic pruning."""
        if not options or not options.cutedsl_nvmmh or not configs:
            return configs
        fields = {
            {"cta_tile": "tile", "mma_tiler": "tile"}.get(field.strip(), field.strip())
            for field in options.cutedsl_nvmmh_fields.split(",") if field.strip()
        }
        fields &= {"tile", "cluster", "swizzle", "cta_order"}
        if not fields:
            return configs
        fields.update({"tile", "cluster"} if fields & {"tile", "cluster"} else set())
        max_tactics = max(1, options.cutedsl_nvmmh_max_tactics)
        use_scheduler = bool(fields & {"swizzle", "cta_order"})
        prune_tile_cluster = bool(fields & {"tile", "cluster"})
        valid_keys = {
            (config.mma_tile_m, config.mma_tile_n, config.cluster_m,
             config.cluster_n, config.swap_ab)
            for config in configs
        }
        prefer_swap = self.m <= 128
        swaps = [prefer_swap] if any(config.swap_ab == prefer_swap for config in configs) else [False, True]
        ranked, scheduler = {}, {}
        for swap_ab in swaps:
            kernel_m, kernel_n = (self.n, self.m) if swap_ab else (self.m, self.n)
            for record in _rank_nvmmh_configs(kernel_m, kernel_n, self.k, max(max_tactics * 4, 16)):
                kernel = record.get("kernel")
                try:
                    cta_m, cta_n = int(kernel.cta_tile_m), int(kernel.cta_tile_n)
                    cluster_m, cluster_n = int(kernel.cluster_m), int(kernel.cluster_n)
                except (AttributeError, TypeError, ValueError):
                    continue
                # Match TRT-LLM's 2-SM encoding: a cluster-M of two denotes
                # a 256-M CuTE MMA tile when the N tile meets its alignment.
                if cluster_m == 2 and cta_n % (32 if cta_n > 256 else 16) == 0:
                    mma_tile_m = 2 * cta_m
                else:
                    mma_tile_m = cta_m
                key = (mma_tile_m, cta_n, cluster_m, cluster_n, swap_ab)
                if key not in valid_keys or key in ranked:
                    continue
                ranked[key] = len(ranked)
                scheduler[key] = (
                    max(1, int(kernel.swizzle_factor)) if "swizzle" in fields else 1,
                    bool(int(kernel.cta_order) != 0) if "cta_order" in fields else True,
                )
        if prune_tile_cluster:
            matches = valid_keys & ranked.keys()
            if not matches:
                return configs
            kept = set(sorted(matches, key=ranked.__getitem__)[:max_tactics])
        else:
            kept = valid_keys
        selected = []
        for config in configs:
            key = (config.mma_tile_m, config.mma_tile_n, config.cluster_m,
                   config.cluster_n, config.swap_ab)
            if key not in kept:
                continue
            swizzle_size, raster_along_m = scheduler.get(key, (1, True))
            if use_scheduler:
                config = CuteDslNvfp4GemmConfig(
                    config.mma_tile_m, config.mma_tile_n, config.cluster_m,
                    config.cluster_n, config.swap_ab, config.use_prefetch,
                    swizzle_size, raster_along_m,
                )
            selected.append(config)
        if options.debug:
            print(
                f"[nvmmh] {len(configs)} valid -> {len(selected)} tactics; "
                f"fields={','.join(sorted(fields))}", flush=True,
            )
        return tuple(selected) or configs

    def _compile(self, config):
        gemm = Sm100BlockScaledPersistentDenseGemmKernel(
            16, (config.mma_tile_m, config.mma_tile_n),
            (config.cluster_m, config.cluster_n), config.use_prefetch,
            config.swizzle_size, config.raster_along_m,
        )
        active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(config.cluster_m * config.cluster_n)
        with _suppress_cutlass_pointer_deprecation():
            a, b = (self.b, self.a) if config.swap_ab else (self.a, self.b)
            sfa, sfb = (self.sfb, self.sfa) if config.swap_ab else (self.sfa, self.sfb)
            c = self.c_swapped if config.swap_ab else self.c
            return cute.compile(_launch, gemm, a, b, sfa, sfb, c, self.alpha,
                                active_clusters, make_fake_stream(),
                                options="--opt-level 2")

    def _prepare(self, config):
        compiled = self._compile(config)
        a, b = (self.b, self.a) if config.swap_ab else (self.a, self.b)
        sfa, sfb = (self.sfb, self.sfa) if config.swap_ab else (self.sfa, self.sfb)
        c = self.c_swapped if config.swap_ab else self.c
        run = lambda: compiled(a, b, sfa, sfb, c, self.alpha, self.stream)
        run(); torch.cuda.synchronize()
        return PreparedKernel(config, run)

    def autotune(self, options: TuneOptions | None = None):
        options = options or TuneOptions(); configs = self.compatible_configs(options)
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
