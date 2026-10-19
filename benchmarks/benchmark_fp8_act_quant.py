#!/usr/bin/env python3
"""Benchmark the warp-specialized TMA FP8 activation quantizer.

Sweeps the four variants the generalized kernel covers: block size
{64, 128} x scale mode {E8M0 power-of-two, plain FP32}, against the TileLang
``act_quant_kernel`` used by the DeepSeek-V4-Flash model.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import BenchmarkBase, SolReport, benchmark_with_stats, sol_bounds
from sm110a_kernel import Fp8ActQuantRunner, compile_fp8_act_quant

TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
BLOCK_SIZES = (64, 128)
SCALE_MODES = ("ue8m0", "fp32")
E4M3_MAX = 448.0


def reference(input, block_size, scale):
    tokens, d = input.shape
    blocks = input.float().view(tokens, d // block_size, block_size)
    amax = blocks.abs().amax(dim=-1)
    if scale == "ue8m0":
        # Build E8M0 bits explicitly: the kernel computes amax * (1/448) and
        # rounds the exponent up, clamped to the finite [0, 254] range.
        values = amax * (1.0 / E4M3_MAX)
        exponent = torch.ceil(torch.log2(values)).to(torch.int32) + 127
        scales = exponent.clamp(0, 254).to(torch.uint8).view(torch.float8_e8m0fnu)
        scale_values = torch.exp2(scales.view(torch.uint8).float() - 127)
        output = (blocks / scale_values.unsqueeze(-1)).to(torch.float8_e4m3fn)
    else:
        # Plain FP32 scales, no power-of-two rounding; the kernel normalizes
        # with the same fp32 reciprocal multiply.
        scales = torch.clamp_min(amax, 1e-4) * (1.0 / E4M3_MAX)
        reciprocals = 1.0 / scales
        output = (blocks * reciprocals.unsqueeze(-1)).clamp_(-E4M3_MAX, E4M3_MAX)
        output = output.to(torch.float8_e4m3fn)
    return output.reshape_as(input).contiguous(), scales.contiguous()


def quant_sol_bounds(tokens, d, block_size, scale):
    """Bandwidth-only SOL for one activation quantization launch."""
    # One BF16 read, one E4M3 write, plus one scale element per block.
    scale_bytes = 1 if scale == "ue8m0" else 4
    traffic_bytes = tokens * d * (2 + 1 + scale_bytes / block_size)
    return sol_bounds(
        flops=0,
        compulsory_bytes=traffic_bytes,
        a_and_b_bytes=traffic_bytes,
        tile_read_bytes=traffic_bytes,
    )


def compile_tilelang_baseline(d, block_size, scale):
    """Compile the DeepSeek-V4-Flash TileLang block-FP8 quantizer per combo."""
    from models.deepseek_v4_flash.inference.kernel import (
        FE8M0,
        act_quant_kernel,
    )

    return act_quant_kernel(
        d,
        block_size,
        scale_dtype=FE8M0 if scale == "ue8m0" else "float32",
        round_scale=scale == "ue8m0",
        inplace=False,
    )


class Fp8ActQuantBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--d", type=int, default=4096)
        parser.add_argument("--tokens", type=int, nargs="*", default=TOKENS)
        parser.add_argument(
            "--bs", type=int, nargs="*", default=BLOCK_SIZES, choices=BLOCK_SIZES
        )
        parser.add_argument(
            "--scale", nargs="*", default=list(SCALE_MODES), choices=SCALE_MODES
        )
        self.add_timing_args(parser, warmup=30, iters=100, samples=9)
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument(
            "--baseline",
            choices=("tilelang", "none"),
            default="tilelang",
            help="baseline implementation (default: models/deepseek_v4_flash "
            "TileLang kernel)",
        )
        self.add_sol_arg(parser)
        return parser

    def validate_args(self, args):
        if args.d <= 0 or any(args.d % bs for bs in args.bs):
            raise SystemExit("--d must be a positive multiple of every --bs value")
        if not args.tokens or any(tokens <= 0 for tokens in args.tokens):
            raise SystemExit("--tokens values must be positive")
        self.validate_timing_args(args, allow_zero_warmup=True)

    def initialize(self):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required")
        torch.manual_seed(self.seed)

    def run_with_args(self, args):
        combos = [(bs, scale) for bs in args.bs for scale in args.scale]
        # Fail fast on JIT issues before any timing.
        for block_size, scale in combos:
            compile_fp8_act_quant(block_size, scale)
        tilelang_kernels = {}
        if args.baseline == "tilelang":
            for block_size, scale in combos:
                tilelang_kernels[(block_size, scale)] = compile_tilelang_baseline(
                    args.d, block_size, scale
                )
        sol_report = SolReport() if args.sol else None
        for block_size, scale in combos:
            tilelang_kernel = tilelang_kernels.get((block_size, scale))
            scale_dtype = torch.float8_e8m0fnu if scale == "ue8m0" else torch.float32
            print(f"\nblock={block_size} scale={scale}")
            if tilelang_kernel is not None:
                print(
                    "tokens   ours(ms)  TL(ms)  TL/ours    ours GB/s  TL GB/s"
                    "  ours mismatch  TL mismatch"
                )
            else:
                print("tokens   ours(ms)  ours GB/s  ours mismatch")
            for tokens in args.tokens:
                input = torch.randn(
                    (tokens, args.d), device="cuda", dtype=torch.bfloat16
                )
                output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
                scales = torch.empty(
                    (tokens, args.d // block_size), device="cuda", dtype=scale_dtype
                )
                kernel_runner = Fp8ActQuantRunner(
                    input, output, scales, block_size, scale
                )
                kernel_runner.run()
                stats = benchmark_with_stats(
                    kernel_runner.run,
                    args.warmup,
                    args.iters,
                    args.samples,
                    cold_l2_flush=args._cold_l2_flush,
                )
                tilelang_stats = None
                if tilelang_kernel is not None:
                    tilelang_output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
                    tilelang_scales = torch.empty(
                        (tokens, args.d // block_size), device="cuda", dtype=scale_dtype
                    )
                    tilelang_stats = benchmark_with_stats(
                        lambda: tilelang_kernel(
                            input, tilelang_output, tilelang_scales
                        ),
                        args.warmup,
                        args.iters,
                        args.samples,
                        cold_l2_flush=args._cold_l2_flush,
                    )
                    # Join all device work (TileLang may use its own
                    # streams) before any accuracy buffer is read.
                    torch.cuda.synchronize()
                if args.no_accuracy:
                    ours_mismatches = "-"
                # Correctness note: with --baseline tilelang, runs at large
                # token counts can show rare (1-2 elements in ~17M) single
                # fp8-byte diffs against the reference below.  The native
                # kernel matches the reference bit-exactly: it normalizes
                # with the same fp32 reciprocal multiply, so exact e4m3
                # rounding ties round identically.  TileLang instead divides
                # by the stored scale, which lands 1 ulp away at those ties
                # and rounds the other way; with --baseline none the same
                # shapes and inputs report zero mismatches repeatedly.
                # Treat ours' mismatch counts as the correctness gate; a
                # nonzero TL count at T >= 2048 is this known artifact.
                else:
                    expected_output, expected_scales = reference(
                        input, block_size, scale
                    )
                    ours_mismatches = (
                        int(
                            (
                                output.view(torch.uint8)
                                != expected_output.view(torch.uint8)
                            )
                            .sum()
                            .item()
                        ),
                        int(
                            (
                                scales.view(torch.uint8)
                                != expected_scales.view(torch.uint8)
                            )
                            .sum()
                            .item()
                        ),
                    )
                    if tilelang_kernel is not None:
                        tilelang_mismatches = (
                            int(
                                (
                                    tilelang_output.view(torch.uint8)
                                    != expected_output.view(torch.uint8)
                                )
                                .sum()
                                .item()
                            ),
                            int(
                                (
                                    tilelang_scales.view(torch.uint8)
                                    != expected_scales.view(torch.uint8)
                                )
                                .sum()
                                .item()
                            ),
                        )
                # Input read + one FP8 output + one scale element per block.
                scale_bytes = 1 if scale == "ue8m0" else 4
                bytes_moved = tokens * args.d * (2 + 1 + scale_bytes / block_size)
                gbps = bytes_moved / (stats["median_ms"] * 1e-3) / 1e9
                if tilelang_kernel is not None:
                    tilelang_gbps = (
                        bytes_moved / (tilelang_stats["median_ms"] * 1e-3) / 1e9
                    )
                    print(
                        f"{tokens:6d}  {stats['median_ms']:8.4f}  {tilelang_stats['median_ms']:6.4f}"
                        f"  {tilelang_stats['median_ms'] / stats['median_ms']:7.2f}"
                        f"  {gbps:10.2f}  {tilelang_gbps:7.2f}"
                        f"  {str(ours_mismatches):>14}  {str(tilelang_mismatches):>13}"
                    )
                else:
                    print(
                        f"{tokens:6d}  {stats['median_ms']:8.4f}  {gbps:10.2f}"
                        f"  {str(ours_mismatches):>14}"
                    )
                if sol_report is not None:
                    bounds = quant_sol_bounds(tokens, args.d, block_size, scale)
                    sol_report.add(
                        bounds,
                        stats["median_ms"],
                        label=f"ours b{block_size}/{scale} T={tokens}",
                    )
                    if tilelang_kernel is not None:
                        sol_report.add(
                            bounds,
                            tilelang_stats["median_ms"],
                            label=f"TL b{block_size}/{scale} T={tokens}",
                        )
        if sol_report is not None:
            sol_report.print()


if __name__ == "__main__":
    Fp8ActQuantBenchmark().run()
