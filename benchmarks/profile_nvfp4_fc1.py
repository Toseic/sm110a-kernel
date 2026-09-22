#!/usr/bin/env python3
"""Single-configuration NVFP4 FC1 driver for Nsight Compute."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sm110a_kernel import (
    Nvfp4Fc1Config,
    compile_nvfp4_fc1,
    prepare_nvfp4_fc1_metadata,
    quantize_nvfp4_rows,
)
from sm110a_kernel.kernels.cutedsl.nvfp4_fc1 import (
    CuteDslNvfp4Fc1Config,
    CuteDslNvfp4Fc1Runner,
)


def interleaved_weights(experts, n, k):
    q_weights, scales = [], []
    for _ in range(experts):
        gate = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        up = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        packed, scale = quantize_nvfp4_rows(
            torch.stack((gate.view(n // 64, 64, k), up.view(n // 64, 64, k)), dim=1)
            .reshape(2 * n, k)
        )
        q_weights.append(packed)
        scales.append(scale.reshape(-1))
    return torch.stack(q_weights), torch.cat(scales)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("ours", "cute"), required=True)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--direct-gmem", action="store_true")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(1234)
    a_q, sfa = quantize_nvfp4_rows(
        torch.randn((args.tokens, args.k), device="cuda", dtype=torch.bfloat16)
    )
    b_q, sfb = interleaved_weights(args.experts, args.n, args.k)
    router = torch.randn(
        (args.tokens, args.experts), device="cuda", dtype=torch.bfloat16
    )
    out = torch.empty(
        (args.tokens * args.topk, args.n // 2), device="cuda",
        dtype=torch.float4_e2m1fn_x2,
    )
    out_scale = torch.empty(
        (args.tokens * args.topk * (args.n // 16),), device="cuda",
        dtype=torch.float8_e4m3fn,
    )

    native_config = Nvfp4Fc1Config(
        tile_n=args.tile_n,
        tile_k=256,
        epilogue_tile_n=128,
        cluster_m=args.cluster_m,
        direct_gmem=args.direct_gmem,
    )
    metadata = prepare_nvfp4_fc1_metadata(
        router, args.topk, 2 * args.n, config=native_config
    )
    if args.impl == "ours":
        module = compile_nvfp4_fc1(native_config)
        run = lambda: module.nvfp4_fc1_out(
            a_q, b_q, sfa, sfb, metadata, out, out_scale
        )
    else:
        runner = CuteDslNvfp4Fc1Runner(
            a_q, b_q, sfa, sfb, out, out_scale,
            metadata.source_rows, metadata.tasks, metadata.state
        )
        config = CuteDslNvfp4Fc1Config(
            mma_tile_m=128 * args.cluster_m,
            mma_tile_n=args.tile_n,
            cluster_m=args.cluster_m,
            cluster_n=1,
        )
        run = runner._prepare(config).run

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    print(f"PROFILE_READY impl={args.impl}", flush=True)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run()
    end.record()
    end.synchronize()
    print(f"PROFILE_MS impl={args.impl} ms={start.elapsed_time(end) / args.iters:.4f}", flush=True)


if __name__ == "__main__":
    main()
