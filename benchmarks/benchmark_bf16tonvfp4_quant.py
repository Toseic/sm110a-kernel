#!/usr/bin/env python3
"""Benchmark the warp-specialized TMA BF16 -> NVFP4 quantizer.

Correctness is checked bitwise against the pure-PyTorch reference
``quantize_nvfp4_rows`` (packed E2M1 bytes and blocked UE4M3 scale bytes).
Timing is cold-L2 by default; the PyTorch baseline is timed on its production
path, which allocates its outputs internally.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import BenchmarkBase, SolReport, benchmark_with_stats, sol_bounds
from sm110a_kernel import (
    bf16tonvfp4_quant_out,
    compile_bf16tonvfp4_quant,
    quantize_nvfp4_rows,
)


TOKENS = (128, 256, 512, 1024, 2048, 4096, 8192)
DIMS = (4096, 7168)
ROW_MULTIPLE = 128
BLOCK_SIZE = 16


def quant_sol_bounds(tokens, d):
    """Bandwidth-only SOL for one BF16->NVFP4 quantization launch."""
    # One BF16 read, one packed E2M1 nibble write, and one UE4M3 scale byte
    # per 16 values; the blocked scale layout adds no extra bytes, only
    # store scatter.
    traffic_bytes = tokens * d * (2 + 0.5 + 1 / BLOCK_SIZE)
    return sol_bounds(
        flops=0,
        compulsory_bytes=traffic_bytes,
        a_and_b_bytes=traffic_bytes,
        tile_read_bytes=traffic_bytes,
    )


class QuantBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--d", type=int, nargs="*", default=list(DIMS))
        parser.add_argument("--tokens", type=int, nargs="*", default=list(TOKENS))
        self.add_timing_args(parser, warmup=30, iters=100, samples=9)
        parser.add_argument("--no-accuracy", action="store_true")
        self.add_sol_arg(parser)
        return parser

    def validate_args(self, args):
        if not args.d or any(d <= 0 or d % 64 for d in args.d):
            raise SystemExit("--d values must be positive multiples of 64")
        if not args.tokens or any(
                tokens <= 0 or tokens % ROW_MULTIPLE for tokens in args.tokens):
            raise SystemExit("--tokens values must be positive multiples of 128")
        self.validate_timing_args(args, allow_zero_warmup=True)

    def initialize(self):
        compile_bf16tonvfp4_quant()

    def run_with_args(self, args):
        print("tokens       d  ours(ms)  torch(ms)  torch/ours    ours GB/s"
              "  torch GB/s  mismatch(packed,scales)")
        sol_report = SolReport() if args.sol else None
        for d in args.d:
            for tokens in args.tokens:
                input = torch.randn((tokens, d), device="cuda",
                                    dtype=torch.bfloat16)
                packed = torch.empty((tokens, d // 2), device="cuda",
                                     dtype=torch.float4_e2m1fn_x2)
                scales = torch.empty(tokens * d // BLOCK_SIZE, device="cuda",
                                     dtype=torch.float8_e4m3fn)
                stats = benchmark_with_stats(
                    lambda: bf16tonvfp4_quant_out(input, packed, scales),
                    args.warmup, args.iters, args.samples,
                    cold_l2_flush=args._cold_l2_flush)
                torch_stats = benchmark_with_stats(
                    lambda: quantize_nvfp4_rows(input), args.warmup,
                    args.iters, args.samples,
                    cold_l2_flush=args._cold_l2_flush)
                if args.no_accuracy:
                    mismatches = "-"
                else:
                    expected_packed, expected_scales = quantize_nvfp4_rows(input)
                    mismatches = (
                        int((packed.view(torch.uint8) !=
                             expected_packed.view(torch.uint8)).sum().item()),
                        int((scales.view(torch.uint8) !=
                             expected_scales.view(torch.uint8)).sum().item()),
                    )
                # Input read + packed nibble write + one scale byte per 16
                # inputs.
                bytes_moved = tokens * d * (2 + 0.5 + 1 / BLOCK_SIZE)
                gbps = bytes_moved / (stats["median_ms"] * 1e-3) / 1e9
                torch_gbps = bytes_moved / (torch_stats["median_ms"] * 1e-3) / 1e9
                print(f"{tokens:6d}  {d:5d}  {stats['median_ms']:8.4f}"
                      f"  {torch_stats['median_ms']:8.4f}"
                      f"  {torch_stats['median_ms'] / stats['median_ms']:9.2f}"
                      f"  {gbps:10.2f}  {torch_gbps:9.2f}"
                      f"  {str(mismatches):>25}")
                if sol_report is not None:
                    bounds = quant_sol_bounds(tokens, d)
                    sol_report.add(bounds, stats["median_ms"],
                                   label=f"ours T={tokens} d={d}")
                    sol_report.add(bounds, torch_stats["median_ms"],
                                   label=f"torch T={tokens} d={d}")
                del input, packed, scales
        if sol_report is not None:
            sol_report.print()


if __name__ == "__main__":
    QuantBenchmark().run()
