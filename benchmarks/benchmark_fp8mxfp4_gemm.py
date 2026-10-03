#!/usr/bin/env python3
"""Accuracy and timing benchmark for native E4M3 x K=32 MXFP4 GEMM."""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import benchmark_with_stats, tflops_from_ms
from sm110a_kernel import (
    Fp8Mxfp4GemmConfig, compile_fp8mxfp4_gemm, fp8mxfp4_gemm_reference,
    from_blocked_fp8mxfp4_scales, quantize_fp8mxfp4_gemm,
)

DEFAULT_SHAPES = (
    (128, 4096, 4096),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 11008, 4096),
    (11008, 4096, 4096),
)

_MODELS_MXFP4_MODULE = None
_DEEPGEMM_MODULE = None


def load_models_mxfp4_kernel():
    """Load DeepSeek V4.1 Flash's TileLang FP8 x MXFP4 kernel on demand."""
    global _MODELS_MXFP4_MODULE
    if _MODELS_MXFP4_MODULE is None:
        source = ROOT / 'models/deepseek_v41_flash/inference/kernel.py'
        spec = importlib.util.spec_from_file_location(
            '_sm110a_models_deepseek_v41_mxfp4', source,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f'cannot load models MXFP4 baseline from {source}')
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            if exc.name == 'tilelang':
                raise RuntimeError(
                    'the models-mxfp4 baseline requires TileLang; install '
                    'models/deepseek_v41_flash/inference/requirements.txt'
                ) from exc
            raise
        _MODELS_MXFP4_MODULE = module
    return _MODELS_MXFP4_MODULE


def models_mxfp4_baseline(x8, w4, sfb, out, *, compact_b_gmem):
    """Prepare the model MXFP4 kernel for the native FP8xMXFP4 operands.

    The model kernel consumes conventional packed FP4 bytes and logical E8M0
    scales, whereas the native kernel accepts either compact or padded B
    storage and a TMA-swizzled scale buffer.  Convert these layouts once,
    outside the timed region.
    """
    model_kernel = load_models_mxfp4_kernel()
    m, k = x8.shape
    n = w4.size(0)
    if compact_b_gmem:
        packed_b = w4
    else:
        packed_b = w4.view(n, k // 16, 16)[..., :8].reshape(n, k // 2).contiguous()
    packed_b = packed_b.view(torch.float4_e2m1fn_x2)
    # The model kernel supports 128-wide activation scaling when possible,
    # and its 32-wide variant covers the native kernel's K=64 padded path.
    act_block_size = 128 if k % 128 == 0 else 32
    sfa = torch.full((m, k // act_block_size), 127, device=x8.device,
                     dtype=torch.uint8).view(torch.float8_e8m0fnu)
    logical_sfb = from_blocked_fp8mxfp4_scales(sfb, n, k)
    kernel = model_kernel.fp4_gemm_kernel(
        n, k, act_block_size=act_block_size, scale_dtype=model_kernel.FE8M0,
    )

    def run():
        kernel(x8, packed_b, out, sfa, logical_sfb)

    return run


def load_deepgemm():
    """Load the locally patched DeepGEMM build used as an SM110 baseline."""
    global _DEEPGEMM_MODULE
    if _DEEPGEMM_MODULE is None:
        source_root = ROOT / 'baselinedir/DeepGEMM'
        if not source_root.is_dir():
            raise RuntimeError(
                f'DeepGEMM baseline is unavailable: expected {source_root}; '
                'clone and build the local DeepGEMM baseline first'
            )
        sys.path.insert(0, str(source_root))
        try:
            import deep_gemm
        except ImportError as exc:
            raise RuntimeError(
                'the DeepGEMM baseline requires its local extension; run '
                f'{source_root}/develop.sh first'
            ) from exc
        _DEEPGEMM_MODULE = deep_gemm
    return _DEEPGEMM_MODULE


def deepgemm_mxfp4_baseline(x8, w4, sfb, out, *, compact_b_gmem):
    """Prepare DeepGEMM's SM110 FP8 x MXFP4 dense GEMM baseline.

    DeepGEMM takes K-major packed E2M1 and TMA-aligned packed UE8M0 scale
    factors.  The native benchmark already provides the former, while its B
    scale buffer is hardware-swizzled; normalize and repack scales once before
    timing.  A is raw E4M3, so its per-32 scale is the E8M0 identity.
    """
    deep_gemm = load_deepgemm()
    m, k = x8.shape
    n = w4.size(0)
    if k % 128:
        raise RuntimeError(
            'the DeepGEMM FP8 x FP4 baseline requires K divisible by 128; '
            f'got K={k}'
        )
    if compact_b_gmem:
        packed_b = w4
    else:
        packed_b = w4.view(n, k // 16, 16)[..., :8].reshape(n, k // 2).contiguous()
    packed_b = packed_b.view(torch.int8)

    sfa_logical = torch.ones((m, k // 32), device=x8.device, dtype=torch.float32)
    sfb_logical = from_blocked_fp8mxfp4_scales(sfb, n, k).float()
    sfa = deep_gemm.transform_sf_into_required_layout(sfa_logical, m, k, (1, 32))
    sfb = deep_gemm.transform_sf_into_required_layout(sfb_logical, n, k, (1, 32))

    def run():
        deep_gemm.fp8_fp4_gemm_nt(
            (x8, sfa), (packed_b, sfb), out,
            recipe_a=(1, 32), recipe_b=(1, 32),
        )

    return run

def config_for(m, n, k, compact_b_gmem=True):
    return Fp8Mxfp4GemmConfig(cluster_m=1 if m % 256 else 2,
                              tile_n=256 if n % 256 == 0 else 128,
                              tile_k=128 if k % 128 == 0 else 64,
                              compact_b_gmem=compact_b_gmem and k % 128 == 0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m', type=int, default=0); p.add_argument('--n', type=int, default=0); p.add_argument('--k', type=int, default=0)
    p.add_argument('--warmup', type=int, default=5); p.add_argument('--iters', type=int, default=20); p.add_argument('--samples', type=int, default=7)
    p.add_argument('--no-accuracy', action='store_true')
    p.add_argument('--compact-b-gmem', dest='compact_b_gmem', action='store_true', default=True)
    p.add_argument('--padded-b-gmem', dest='compact_b_gmem', action='store_false')
    p.add_argument('--baseline', choices=('none', 'models-mxfp4', 'deepgemm-mxfp4'), default='none',
                   help='optional baseline: TileLang models kernel or locally patched DeepGEMM')
    p.add_argument('--seed', type=int, default=1234)
    a = p.parse_args()
    selected = a.m or a.n or a.k
    if selected and not (a.m and a.n and a.k): p.error('--m, --n, and --k must be specified together')
    shapes = ((a.m, a.n, a.k),) if selected else DEFAULT_SHAPES
    torch.manual_seed(a.seed)
    if a.baseline != 'none':
        label = ('DeepSeek V4.1 Flash TileLang FP8 x MXFP4 GEMM'
                 if a.baseline == 'models-mxfp4'
                 else 'DeepGEMM SM110 FP8 x MXFP4 GEMM')
        print(f'[baseline] {label}')
        print(' M       N       K    Kernel(ms)     TF/s  Baseline(ms)     TF/s  Speedup  MaxDiff BaselineDiff  Config')
    else:
        print(' M       N       K    Kernel(ms)     TF/s    MaxDiff  Config')
    for m, n, k in shapes:
        if m % 128 or n % 128 or k % 64: p.error('M,N must be multiples of 128 and K a multiple of 64')
        x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        w = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
        compact_b_gmem = a.compact_b_gmem and k % 128 == 0
        x8, w4, sfb = quantize_fp8mxfp4_gemm(x, w, compact_b_gmem=compact_b_gmem)
        out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        config = config_for(m, n, k, compact_b_gmem)
        module = compile_fp8mxfp4_gemm(config)
        run = lambda: module.fp8mxfp4_gemm_out(x8, w4, sfb, out)
        run(); torch.cuda.synchronize()
        stats = benchmark_with_stats(run, a.warmup, a.iters, a.samples)
        baseline_out = None
        baseline_stats = None
        baseline_run = None
        if a.baseline == 'models-mxfp4':
            baseline_out = torch.empty_like(out)
            baseline_run = models_mxfp4_baseline(
                x8, w4, sfb, baseline_out, compact_b_gmem=compact_b_gmem,
            )
        elif a.baseline == 'deepgemm-mxfp4':
            baseline_out = torch.empty_like(out)
            baseline_run = deepgemm_mxfp4_baseline(
                x8, w4, sfb, baseline_out, compact_b_gmem=compact_b_gmem,
            )
        if baseline_run is not None:
            baseline_run(); torch.cuda.synchronize()
            baseline_stats = benchmark_with_stats(
                baseline_run, a.warmup, a.iters, a.samples,
            )
        diff = 0.0
        baseline_diff = 0.0
        if not a.no_accuracy:
            ref = fp8mxfp4_gemm_reference(x8, w4, sfb); run(); torch.cuda.synchronize()
            diff = (out.float() - ref.float()).abs().max().item()
            if not torch.equal(out, ref): raise AssertionError(f'mismatch {m}x{n}x{k}: {diff}')
            if baseline_run is not None:
                baseline_run(); torch.cuda.synchronize()
                baseline_diff = (baseline_out.float() - ref.float()).abs().max().item()
        ms = stats['median_ms']
        if baseline_stats is None:
            print(f'{m:6d} {n:7d} {k:7d} {ms:12.4f} {tflops_from_ms(m,n,k,ms):8.2f} {diff:10.6f} {config}')
        else:
            baseline_ms = baseline_stats['median_ms']
            print(f'{m:6d} {n:7d} {k:7d} {ms:12.4f} {tflops_from_ms(m,n,k,ms):8.2f} '
                  f'{baseline_ms:10.4f} {tflops_from_ms(m,n,k,baseline_ms):8.2f} '
                  f'{baseline_ms/ms:7.2f}x {diff:8.6f} {baseline_diff:9.6f} {config}')

if __name__ == '__main__':
    main()
