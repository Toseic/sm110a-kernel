#!/usr/bin/env python3
"""Benchmark the warp-specialized TMA BF16 -> block FP4 (MXFP4-style) quantizer."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import BenchmarkBase, SolReport, benchmark_with_stats, sol_bounds
from sm110a_kernel import (
    bf16tofp4_quant_dequant_out,
    bf16tofp4_quant_out,
    compile_bf16tofp4_quant,
)


TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
QUANT_BLOCK = 32
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _pack_e2m1_rne(values):
    """Pack in-place-clamped FP32 values into E2M1 pairs, element 2i in the
    low nibble, with round-to-nearest-even ties (cvt.rn.satfinite.e2m1x2
    semantics)."""
    midpoints = torch.tensor(E2M1_MIDPOINTS, device=values.device,
                             dtype=torch.float32)
    magnitudes = values.abs().clamp_(max=6.0)
    code = torch.bucketize(magnitudes, midpoints, out_int32=True).to(torch.uint8)
    # bucketize places exact midpoints in the lower bin; RN-even keeps them
    # there only when the lower code's magnitude is even-valued.
    safe = code.long().clamp(max=len(E2M1_MIDPOINTS) - 1)
    ties = (magnitudes == midpoints[safe]) & ((code & 1) == 1)
    code = (code + ties.to(torch.uint8)) | (torch.signbit(values).to(torch.uint8) << 3)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def reference(input):
    tokens, d = input.shape
    blocks = input.float().view(tokens, d // QUANT_BLOCK, QUANT_BLOCK)
    amax = blocks.abs().amax(dim=-1)
    amax = torch.maximum(amax, torch.full_like(amax, 6.0 * 2.0 ** -126))
    # Build E8M0 bits explicitly with the kernel's IEEE bit tricks
    # (fast_log2_ceil / fast_pow2): PyTorch's generic dtype conversion has a
    # different rounding mode.
    bits = (amax * (1.0 / 6.0)).view(torch.int32)
    log2_ceil = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scales = (log2_ceil + 127).clamp(0, 255).to(torch.uint8).view(
        torch.float8_e8m0fnu)
    # Power-of-two scale: multiply by the exact reciprocal.
    reciprocal = torch.exp2(-log2_ceil.to(torch.float32))
    normalized = blocks * reciprocal.unsqueeze(-1)
    packed = _pack_e2m1_rne(normalized.view(tokens, d))
    return packed, scales


def dequant_reference(input):
    tokens, d = input.shape
    packed, scales = reference(input)
    codes = packed.view(torch.uint8)
    codes = torch.stack((codes & 0xF, codes >> 4), dim=-1).reshape(tokens, d)
    magnitudes = torch.tensor(
        E2M1_VALUES, device=input.device, dtype=torch.float32)[(codes & 0x7).long()]
    values = torch.where((codes & 0x8) != 0, -magnitudes, magnitudes)
    scale_values = torch.exp2(scales.view(torch.uint8).float() - 127.0)
    return (values.view(tokens, d // QUANT_BLOCK, QUANT_BLOCK)
            * scale_values.unsqueeze(-1)).reshape(tokens, d).to(torch.bfloat16)


def quant_sol_bounds(tokens, d, dequant):
    """Bandwidth-only SOL for one BF16->FP4 quantization launch."""
    # One BF16 read, a half byte of packed E2M1 (+ a full BF16 write when
    # fused with dequantization), and one E8M0 byte per 32 values.
    write_bytes = 2.5 if dequant else 0.5
    traffic_bytes = tokens * d * (2 + write_bytes + 1 / QUANT_BLOCK)
    return sol_bounds(
        flops=0,
        compulsory_bytes=traffic_bytes,
        a_and_b_bytes=traffic_bytes,
        tile_read_bytes=traffic_bytes,
    )


def compile_tilelang_baseline(d, inplace):
    """Compile the DeepSeek v4 TileLang block-FP4 quantizer once per D."""
    from models.deepseek_v4_flash.inference.kernel import fp4_quant_kernel

    return fp4_quant_kernel(d, QUANT_BLOCK, inplace=inplace)


class QuantBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--d", type=int, default=4096)
        parser.add_argument("--tokens", type=int, nargs="*", default=TOKENS)
        parser.add_argument(
            "--entry", choices=("quant", "dequant", "both"), default="both",
            help="which kernel entry to benchmark (default: both)")
        self.add_timing_args(parser, warmup=30, iters=100, samples=9)
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument(
            "--baseline", choices=("tilelang", "none"), default="tilelang",
            help="baseline implementation (default: models/deepseek_v4_flash TileLang kernel)",
        )
        self.add_sol_arg(parser)
        return parser

    def validate_args(self, args):
        if args.d <= 0 or args.d % QUANT_BLOCK:
            raise SystemExit("--d must be a positive multiple of 32")
        if not args.tokens or any(tokens <= 0 for tokens in args.tokens):
            raise SystemExit("--tokens values must be positive")
        self.validate_timing_args(args, allow_zero_warmup=True)

    def initialize(self):
        compile_bf16tofp4_quant()

    def run_with_args(self, args):
        entries = (("quant", "dequant") if args.entry == "both"
                   else (args.entry,))
        tilelang_kernels = {}
        if args.baseline == "tilelang":
            tilelang_kernels["quant"] = compile_tilelang_baseline(args.d, False)
            tilelang_kernels["dequant"] = compile_tilelang_baseline(args.d, True)
        for entry in entries:
            dequant = entry == "dequant"
            print(f"== {entry}: " +
                  ("fused quantize+dequantize, BF16 out" if dequant
                   else "packed E2M1 + E8M0 scales") +
                  (" ==" if args.baseline == "tilelang" else " (no baseline) =="))
            if args.baseline == "tilelang":
                print("tokens   ours(ms)  TL(ms)  TL/ours    ours GB/s"
                      "  TL GB/s  ours mismatch  TL mismatch")
            else:
                print("tokens   ours(ms)  ours GB/s  ours mismatch")
            sol_report = SolReport() if args.sol else None
            write_bytes = 2.5 if dequant else 0.5
            for tokens in args.tokens:
                input = torch.randn((tokens, args.d), device="cuda",
                                    dtype=torch.bfloat16)
                if dequant:
                    output = torch.empty_like(input)
                else:
                    output = torch.empty((tokens, args.d // 2), device="cuda",
                                         dtype=torch.float4_e2m1fn_x2)
                scales = torch.empty((tokens, args.d // QUANT_BLOCK),
                                     device="cuda", dtype=torch.float8_e8m0fnu)
                if dequant:
                    run_ours = lambda: bf16tofp4_quant_dequant_out(
                        input, output, scales)
                else:
                    run_ours = lambda: bf16tofp4_quant_out(
                        input, output, scales)
                stats = benchmark_with_stats(
                    run_ours, args.warmup, args.iters, args.samples,
                    cold_l2_flush=args._cold_l2_flush)
                tilelang_stats = None
                if entry in tilelang_kernels:
                    tilelang_output = torch.empty_like(output)
                    tilelang_scales = torch.empty_like(scales)
                    tilelang_stats = benchmark_with_stats(
                        lambda: tilelang_kernels[entry](
                            input, tilelang_output, tilelang_scales),
                        args.warmup, args.iters, args.samples,
                        cold_l2_flush=args._cold_l2_flush)
                if args.no_accuracy:
                    ours_mismatches = tilelang_mismatches = "-"
                else:
                    if dequant:
                        expected_output = dequant_reference(input)
                        ours_mismatches = (
                            int((output.view(torch.int16) !=
                                 expected_output.view(torch.int16)).sum().item()),
                            int((scales.view(torch.uint8) !=
                                 reference(input)[1].view(torch.uint8)).sum().item()),
                        )
                    else:
                        expected_output, expected_scales = reference(input)
                        ours_mismatches = (
                            int((output.view(torch.uint8) !=
                                 expected_output.view(torch.uint8)).sum().item()),
                            int((scales.view(torch.uint8) !=
                                 expected_scales.view(torch.uint8)).sum().item()),
                        )
                    if tilelang_stats is not None:
                        if dequant:
                            expected_scales = reference(input)[1]
                            tilelang_mismatches = (
                                int((tilelang_output.view(torch.int16) !=
                                     expected_output.view(torch.int16)).sum().item()),
                                int((tilelang_scales.view(torch.uint8) !=
                                     expected_scales.view(torch.uint8)).sum().item()),
                            )
                        else:
                            tilelang_mismatches = (
                                int((tilelang_output.view(torch.uint8) !=
                                     expected_output.view(torch.uint8)).sum().item()),
                                int((tilelang_scales.view(torch.uint8) !=
                                     expected_scales.view(torch.uint8)).sum().item()),
                            )
                bytes_moved = tokens * args.d * (2 + write_bytes + 1 / QUANT_BLOCK)
                gbps = bytes_moved / (stats["median_ms"] * 1e-3) / 1e9
                if tilelang_stats is not None:
                    tilelang_gbps = bytes_moved / (tilelang_stats["median_ms"] * 1e-3) / 1e9
                    print(f"{tokens:6d}  {stats['median_ms']:8.4f}  {tilelang_stats['median_ms']:6.4f}"
                          f"  {tilelang_stats['median_ms'] / stats['median_ms']:7.2f}"
                          f"  {gbps:10.2f}  {tilelang_gbps:7.2f}"
                          f"  {str(ours_mismatches):>14}  {str(tilelang_mismatches):>13}")
                else:
                    print(f"{tokens:6d}  {stats['median_ms']:8.4f}  {gbps:10.2f}"
                          f"  {str(ours_mismatches):>14}")
                if sol_report is not None:
                    bounds = quant_sol_bounds(tokens, args.d, dequant)
                    sol_report.add(bounds, stats["median_ms"], label=f"ours T={tokens}")
                    if tilelang_stats is not None:
                        sol_report.add(bounds, tilelang_stats["median_ms"],
                                       label=f"TileLang T={tokens}")
            if sol_report is not None:
                sol_report.print()


if __name__ == "__main__":
    QuantBenchmark().run()
