#!/usr/bin/env python3
"""Fixed-shape correctness/performance test for the fused FC1+FC2 kernel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sm110a_kernel import (  # noqa: E402
    Nvfp4Fc1Config,
    Nvfp4Fc2Config,
    compile_nvfp4_fc1,
    compile_nvfp4_fc1fc2,
    compile_nvfp4_fc2,
    nvfp4_fc1_out,
    nvfp4_fc1fc2_out,
    nvfp4_fc2_out,
    prepare_nvfp4_fc1_metadata,
    prepare_nvfp4_fc2_metadata,
    quantize_nvfp4_rows,
)


def quantize_fc1_weight(n, k):
    gate = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    up = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    interleaved = torch.stack(
        (gate.view(n // 64, 64, k), up.view(n // 64, 64, k)), dim=1
    ).reshape(2 * n, k)
    packed, scales = quantize_nvfp4_rows(interleaved)
    return packed.unsqueeze(0), scales


def time_ms(call, warmup, iterations):
    for _ in range(warmup):
        call()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--no-accuracy", action="store_true")
    args = parser.parse_args()
    if args.m <= 0 or args.m % 512:
        parser.error("--m must be a positive multiple of 512")
    if args.warmup < 0 or args.iters <= 0:
        parser.error("invalid warmup/iteration count")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    h, intermediate = 6144, 2048
    torch.manual_seed(1234)
    aq, sfa = quantize_nvfp4_rows(
        torch.randn((args.m, h), device="cuda", dtype=torch.bfloat16)
    )
    w1q, sfw1 = quantize_fc1_weight(intermediate, h)
    w2q, sfw2 = quantize_nvfp4_rows(
        torch.randn((h, intermediate), device="cuda", dtype=torch.bfloat16)
    )
    w2q = w2q.unsqueeze(0)

    fused = compile_nvfp4_fc1fc2()
    output = torch.empty((args.m, h), device="cuda", dtype=torch.bfloat16)

    fc1_config = Nvfp4Fc1Config(
        tile_n=128, tile_k=256, epilogue_tile_n=128,
        cluster_m=2, direct_gmem=True,
    )
    fc2_config = Nvfp4Fc2Config(
        tile_n=128, tile_k=256, epilogue_tile_n=64, cluster_m=2,
    )
    compile_nvfp4_fc1(fc1_config)
    compile_nvfp4_fc2(fc2_config)
    router = torch.ones((args.m, 1), device="cuda", dtype=torch.bfloat16)
    fc1_metadata = prepare_nvfp4_fc1_metadata(
        router, 1, 2 * intermediate, fc1_config
    )
    grouped_layout = fc1_metadata.routing.narrow(0, 2, args.m)
    fc2_metadata = prepare_nvfp4_fc2_metadata(
        grouped_layout, 1, h, fc2_config
    )
    intermediate_packed = torch.empty(
        (args.m, intermediate // 2), device="cuda",
        dtype=torch.float4_e2m1fn_x2,
    )
    intermediate_scales = torch.empty(
        (args.m * (intermediate // 16),), device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    reference = torch.empty_like(output)
    routing_weights = torch.ones((args.m, 1), device="cuda", dtype=torch.bfloat16)

    def run_fused():
        return nvfp4_fc1fc2_out(
            aq, w1q, sfa, sfw1, w2q, sfw2, output
        )

    def run_standalone():
        nvfp4_fc1_out(
            aq, w1q, sfa, sfw1, fc1_metadata,
            intermediate_packed, intermediate_scales,
            config=fc1_config,
        )
        return nvfp4_fc2_out(
            intermediate_packed, w2q, intermediate_scales, sfw2,
            fc1_metadata.routing, routing_weights, fc2_metadata, 1,
            reference, config=fc2_config,
        )

    if not args.no_accuracy:
        run_fused()
        run_standalone()
        torch.cuda.synchronize()
        difference = output.float() - reference.float()
        print(
            f"[accuracy] max_abs={difference.abs().max().item():.6g}; "
            f"mean_abs={difference.abs().mean().item():.6g}; "
            f"rmse={difference.square().mean().sqrt().item():.6g}; "
            f"mismatches={int((output != reference).sum().item())}"
        )

    fused_ms = time_ms(run_fused, args.warmup, args.iters)
    standalone_ms = time_ms(run_standalone, args.warmup, args.iters)
    print(f"[fused] {fused_ms:.6f} ms")
    print(f"[fc1+fc2] {standalone_ms:.6f} ms")
    print(f"[ratio] {standalone_ms / fused_ms:.4f}x")


if __name__ == "__main__":
    main()
