#!/usr/bin/env python3
"""Benchmark FA4-style MLA decoding, H=128, D_KV=512, D_rope=64.

Q_input=concat(Q_absorb,Q_rope), K=concat(cKV,K_rope), V=cKV.
K is stored as [B,1,S,576]; QK consumes its logical transpose.
The default suite uses B in {32,64,128}, S=8192, and one query token.
"""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    BenchmarkBase,
    SolReport,
    TablePrinter,
    benchmark_with_stats,
    sol_bounds,
)
from sm110a_kernel import Bf16Fa4MlaConfig, Bf16Fa4MlaRunner

H = 128
D_KV = 512
D_rope = 64


def make_inputs(batch, seq_len):
    def random(shape):
        return torch.empty(shape, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)

    q_absorb = random((batch, H, 1, D_KV))
    q_rope = random((batch, H, 1, D_rope))
    ckv = random((batch, 1, seq_len, D_KV))
    k_rope = random((batch, 1, seq_len, D_rope))
    return torch.cat((q_absorb, q_rope), -1), torch.cat((ckv, k_rope), -1), ckv


def reference_attention(q, k, v, scale):
    # Batch by batch to bound FP32 score storage at large B/S. Evaluate the
    # two terms independently to verify the absorbed-content/RoPE equation.
    out = torch.empty((q.size(0), H, 1, D_KV), device=q.device, dtype=q.dtype)
    for b in range(q.size(0)):
        qa = q[b, :, 0, :D_KV].float()
        qr = q[b, :, 0, D_KV:].float()
        ka = k[b, 0, :, :D_KV].float()
        kr = k[b, 0, :, D_KV:].float()
        scores = (qa @ ka.T + qr @ kr.T) * scale
        out[b, :, 0] = torch.softmax(scores, -1) @ v[b, 0].float()
    return out


def mla_sol_bounds(batch, kv_len, *, tile_kv=64):
    """Traffic-aware SOL for the MLA decode schedule.

    Compulsory traffic reads Q/K and writes O once. K stores
    concat(cKV, K_rope), so its cKV prefix is V. Each batch runs two
    independent CTAs, each producing all 512 output features for 64 heads.
    Q is partitioned over the CTAs; both CTAs read all of K.
    """
    element_bytes = 2  # BF16 Q, K, V, O
    q_bytes = element_bytes * batch * H * (D_KV + D_rope)
    k_bytes = element_bytes * batch * kv_len * (D_KV + D_rope)
    output_bytes = element_bytes * batch * H * D_KV

    # Per KV tile both CTAs stream the full K=[cKV,K_rope] tile once.
    kv_tile_count = math.ceil(kv_len / tile_kv)
    kv_tile_bytes = element_bytes * tile_kv * 2 * (D_KV + D_rope)
    tile_read_bytes = q_bytes + batch * kv_tile_count * kv_tile_bytes

    # Useful FLOPs (QK + PV); neither operation is duplicated over output D.
    flops = 2 * batch * H * kv_len * (D_KV + D_rope + D_KV)
    return sol_bounds(
        flops,
        q_bytes + k_bytes + output_bytes,
        q_bytes + k_bytes,
        tile_read_bytes,
    )


class Bf16Fa4MlaBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--single", action="store_true")
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--kv-len", "--s", type=int, default=8192)
        parser.add_argument("--scale", type=float, default=1 / math.sqrt(D_KV + D_rope))
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument(
            "--impl",
            choices=("1cta", "2cta"),
            default="1cta",
            help="MLA kernel implementation to benchmark (default: 1cta)",
        )
        parser.add_argument("--atol", type=float, default=2e-2)
        parser.add_argument("--rtol", type=float, default=2e-2)
        self.add_timing_args(parser, warmup=5, iters=20, samples=7)
        self.add_sol_arg(parser)
        parser.add_argument(
            "--debug", action="store_true", help="print benchmark records line by line"
        )
        return parser

    def validate_args(self, args):
        if args.batch <= 0 or args.kv_len <= 0:
            raise SystemExit("batch and KV length must be positive")
        if not math.isfinite(args.scale) or args.scale <= 0:
            raise SystemExit("scale must be finite and positive")
        self.validate_timing_args(args, allow_zero_warmup=True)
        return args

    def run_with_args(self, args):
        torch.manual_seed(0)
        print(
            f"[config] MLA decode: {args.impl}, QK M64N64, PV M64N128, D=512, 12 warps"
        )
        columns = (
            ("B", 6),
            ("S", 8),
            ("Kernel ms", 13),
            ("Kernel TF/s", 14),
            ("Max diff", 13),
        )
        table = TablePrinter(columns, debug=args.debug)
        table.print_header()
        sol_report = SolReport(debug=args.debug) if args.sol else None
        for batch in (args.batch,) if args.single else (32, 64, 128):
            q, k, v = make_inputs(batch, args.kv_len)
            out = torch.empty((batch, H, 1, D_KV), device=q.device, dtype=q.dtype)

            kernel = (
                Bf16Fa4MlaRunner(q, k, v, out, scale=args.scale)
                .prepare(Bf16Fa4MlaConfig(args.impl))
                .run
            )

            kernel()
            torch.cuda.synchronize()
            diff = "skipped"
            if not args.no_accuracy:
                ref = reference_attention(q, k, v, args.scale)
                torch.testing.assert_close(out, ref, atol=args.atol, rtol=args.rtol)
                diff = f"{(out.float() - ref.float()).abs().max().item():.6g}"
            stats = benchmark_with_stats(
                kernel,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=args._cold_l2_flush,
            )
            # Useful FLOPs; the two CTAs partition H and compute full D.
            flops = 2 * batch * H * args.kv_len * (D_KV + D_rope + D_KV)
            ms = stats["median_ms"]
            values = (
                batch,
                args.kv_len,
                f"{ms:.4f}",
                f"{flops / (ms * 1e9):.3f}",
                diff,
            )
            if sol_report:
                bounds = mla_sol_bounds(batch, args.kv_len)
            table.print_row(*values)
            if sol_report:
                sol_report.add(bounds, ms, label=f"MLA B={batch}, S={args.kv_len}")

        if sol_report:
            sol_report.print()


if __name__ == "__main__":
    Bf16Fa4MlaBenchmark().run()
