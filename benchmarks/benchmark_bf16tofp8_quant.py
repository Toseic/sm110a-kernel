#!/usr/bin/env python3
"""Benchmark the 8-warp, one-thread-per-token BF16 -> block FP8 quantizer."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import BenchmarkBase, SolReport, benchmark_with_stats, sol_bounds
from sm110a_kernel import bf16tofp8_quant_out, compile_bf16tofp8_quant


TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
BLOCK_SIZE = 128
E4M3_MAX = 448.0


def reference(input):
    blocks = input.float().view(input.size(0), input.size(1) // BLOCK_SIZE,
                                BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1)
    # Build E8M0 bits explicitly: PyTorch's generic dtype conversion has a
    # different rounding mode, while the CUDA kernel uses round-positive-inf.
    exponent = torch.ceil(torch.log2(amax / E4M3_MAX)).to(torch.int32) + 127
    scales = exponent.clamp(0, 254).to(torch.uint8).view(torch.float8_e8m0fnu)
    scale_values = torch.exp2(scales.view(torch.uint8).float() - 127)
    output = (blocks / scale_values.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return output.reshape_as(input).contiguous(), scales.contiguous()


def quant_sol_bounds(tokens, d):
    """Bandwidth-only SOL for one BF16->E4M3/E8M0 quantization launch."""
    # One BF16 read, one E4M3 write, and one E8M0 byte per 128 values.
    traffic_bytes = tokens * d * (2 + 1 + 1 / BLOCK_SIZE)
    return sol_bounds(
        flops=0,
        compulsory_bytes=traffic_bytes,
        a_and_b_bytes=traffic_bytes,
        tile_read_bytes=traffic_bytes,
    )


def compile_tilelang_baseline(d):
    """Compile the DeepSeek v4.1 TileLang block-FP8 quantizer once per D."""
    from models.deepseek_v41_flash.inference.kernel import FE8M0, act_quant_kernel

    return act_quant_kernel(
        d, BLOCK_SIZE, scale_dtype=FE8M0, round_scale=True, inplace=False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=4096)
    parser.add_argument("--tokens", type=int, nargs="*", default=TOKENS)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument(
        "--no-cold-l2", dest="cold_l2", action="store_false", default=True,
        help="do not evict L2 before each timed kernel invocation",
    )
    parser.add_argument("--no-accuracy", action="store_true")
    parser.add_argument(
        "--baseline", choices=("tilelang", "none"), default="tilelang",
        help="baseline implementation (default: models/deepseek_v41_flash TileLang kernel)",
    )
    parser.add_argument("--sol", action="store_true",
                        help="print bandwidth SOL bounds and attainment")
    args = parser.parse_args()
    if args.d <= 0 or args.d % BLOCK_SIZE:
        parser.error("--d must be a positive multiple of 128")
    if not args.tokens or any(tokens <= 0 for tokens in args.tokens):
        parser.error("--tokens values must be positive")
    if args.warmup < 0 or args.iters <= 0 or args.samples <= 0:
        parser.error("invalid timing arguments")

    # Match the repository-wide benchmark contract: eviction is ordered before
    # each timed launch but excluded from its CUDA-event duration.
    cache_control = BenchmarkBase()
    cache_control.configure_cold_l2(args)
    compile_bf16tofp8_quant()
    tilelang_kernel = compile_tilelang_baseline(args.d) if args.baseline == "tilelang" else None
    if tilelang_kernel is not None:
        print("tokens   ours(ms)  TL(ms)  TL/ours    ours GB/s  TL GB/s  ours mismatch  TL mismatch")
    else:
        print("tokens   ours(ms)  ours GB/s  ours mismatch")
    sol_report = SolReport() if args.sol else None
    for tokens in args.tokens:
        input = torch.randn((tokens, args.d), device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
        scales = torch.empty((tokens, args.d // BLOCK_SIZE), device="cuda",
                             dtype=torch.float8_e8m0fnu)
        stats = benchmark_with_stats(
            lambda: bf16tofp8_quant_out(input, output, scales), args.warmup,
            args.iters, args.samples, cold_l2_flush=args._cold_l2_flush)
        if tilelang_kernel is not None:
            tilelang_output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
            tilelang_scales = torch.empty((tokens, args.d // BLOCK_SIZE), device="cuda",
                                          dtype=torch.float8_e8m0fnu)
            tilelang_stats = benchmark_with_stats(
                lambda: tilelang_kernel(input, tilelang_output, tilelang_scales),
                args.warmup, args.iters, args.samples,
                cold_l2_flush=args._cold_l2_flush)
        if args.no_accuracy:
            ours_mismatches = tilelang_mismatches = "-"
        else:
            expected_output, expected_scales = reference(input)
            ours_mismatches = (
                int((output.view(torch.uint8) != expected_output.view(torch.uint8)).sum().item()),
                int((scales.view(torch.uint8) != expected_scales.view(torch.uint8)).sum().item()),
            )
            if tilelang_kernel is not None:
                tilelang_mismatches = (
                    int((tilelang_output.view(torch.uint8) != expected_output.view(torch.uint8)).sum().item()),
                    int((tilelang_scales.view(torch.uint8) != expected_scales.view(torch.uint8)).sum().item()),
                )
        # Input read + one FP8 output + one scale byte per 128 inputs.
        bytes_moved = tokens * args.d * (2 + 1 + 1 / BLOCK_SIZE)
        gbps = bytes_moved / (stats["median_ms"] * 1e-3) / 1e9
        if tilelang_kernel is not None:
            tilelang_gbps = bytes_moved / (tilelang_stats["median_ms"] * 1e-3) / 1e9
            print(f"{tokens:6d}  {stats['median_ms']:8.4f}  {tilelang_stats['median_ms']:6.4f}"
                  f"  {tilelang_stats['median_ms'] / stats['median_ms']:7.2f}"
                  f"  {gbps:10.2f}  {tilelang_gbps:7.2f}"
                  f"  {str(ours_mismatches):>14}  {str(tilelang_mismatches):>13}")
        else:
            print(f"{tokens:6d}  {stats['median_ms']:8.4f}  {gbps:10.2f}"
                  f"  {str(ours_mismatches):>14}")
        if sol_report is not None:
            bounds = quant_sol_bounds(tokens, args.d)
            sol_report.add(bounds, stats["median_ms"], label=f"ours T={tokens}")
            if tilelang_kernel is not None:
                sol_report.add(bounds, tilelang_stats["median_ms"], label=f"TileLang T={tokens}")
    if sol_report is not None:
        sol_report.print()


if __name__ == "__main__":
    main()
