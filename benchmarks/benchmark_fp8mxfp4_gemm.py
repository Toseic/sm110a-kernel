#!/usr/bin/env python3
"""Accuracy and timing benchmark for native E4M3 x K=32 MXFP4 GEMM."""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import benchmark_with_stats, tflops_from_ms
from sm110a_kernel import (
    Fp8Mxfp4GemmConfig, fp8mxfp4_gemm_out, fp8mxfp4_gemm_reference,
    quantize_fp8mxfp4_gemm,
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
    p.add_argument('--seed', type=int, default=1234)
    a = p.parse_args()
    selected = a.m or a.n or a.k
    if selected and not (a.m and a.n and a.k): p.error('--m, --n, and --k must be specified together')
    shapes = ((a.m, a.n, a.k),) if selected else DEFAULT_SHAPES
    torch.manual_seed(a.seed)
    print(' M       N       K    Kernel(ms)     TF/s    MaxDiff  Config')
    for m, n, k in shapes:
        if m % 128 or n % 128 or k % 64: p.error('M,N must be multiples of 128 and K a multiple of 64')
        x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        w = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
        compact_b_gmem = a.compact_b_gmem and k % 128 == 0
        x8, w4, sfb = quantize_fp8mxfp4_gemm(x, w, compact_b_gmem=compact_b_gmem)
        out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        config = config_for(m, n, k, compact_b_gmem)
        run = lambda: fp8mxfp4_gemm_out(x8, w4, sfb, out, config=config)
        run(); torch.cuda.synchronize()
        stats = benchmark_with_stats(run, a.warmup, a.iters, a.samples)
        diff = 0.0
        if not a.no_accuracy:
            ref = fp8mxfp4_gemm_reference(x8, w4, sfb); run(); torch.cuda.synchronize()
            diff = (out.float() - ref.float()).abs().max().item()
            if not torch.equal(out, ref): raise AssertionError(f'mismatch {m}x{n}x{k}: {diff}')
        ms = stats['median_ms']
        print(f'{m:6d} {n:7d} {k:7d} {ms:12.4f} {tflops_from_ms(m,n,k,ms):8.2f} {diff:10.6f} {config}')

if __name__ == '__main__':
    main()
