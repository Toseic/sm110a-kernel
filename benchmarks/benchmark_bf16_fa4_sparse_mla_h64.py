#!/usr/bin/env python3
"""Benchmark H=64 sparse MLA decode with indexed KV and attention sink."""

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
from sm110a_kernel import Bf16Fa4SparseMlaRunner

H = 64
D = 512


def make_inputs(batch, cache_len, selected_len, invalid):
    q = torch.empty((batch, H, 1, D), device="cuda", dtype=torch.bfloat16).uniform_(
        -1, 1
    )
    kv = torch.empty(
        (batch, cache_len, D), device="cuda", dtype=torch.bfloat16
    ).uniform_(-1, 1)
    sink = torch.randn(H, device="cuda", dtype=torch.float32)
    indices = torch.randint(
        cache_len, (batch, selected_len), device="cuda", dtype=torch.int32
    )
    window = min(128, selected_len, cache_len)
    indices[:, :window] = torch.arange(
        cache_len - window, cache_len, device="cuda", dtype=torch.int32
    )
    if invalid:
        indices[:, -invalid:] = -1
    return q, kv, indices, sink


def reference(q, kv, indices, sink, scale):
    output = torch.empty_like(q)
    for batch in range(q.size(0)):
        valid = indices[batch] >= 0
        selected = kv[batch].index_select(0, indices[batch, valid].long()).float()
        scores = (q[batch, :, 0].float() @ selected.T) * scale
        row_max = torch.maximum(scores.max(-1).values, sink)
        weights = torch.exp(scores - row_max[:, None])
        denom = weights.sum(-1) + torch.exp(sink - row_max)
        output[batch, :, 0] = (weights @ selected / denom[:, None]).to(q.dtype)
    return output


def sparse_mla_sol_bounds(batch, selected_len, valid_len):
    """Idealized SOL: one BF16 KV row read per valid selected index."""
    q_bytes = 2 * batch * H * D
    kv_bytes = 2 * batch * valid_len * D
    out_bytes = q_bytes
    index_bytes = 4 * batch * selected_len
    sink_bytes = 4 * H
    input_bytes = q_bytes + kv_bytes + index_bytes + sink_bytes
    tile_read_bytes = input_bytes + (batch - 1) * sink_bytes
    flops = 4 * batch * H * valid_len * D
    return sol_bounds(flops, input_bytes + out_bytes, input_bytes, tile_read_bytes)


class SparseMlaH64Benchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--kv-len", type=int, default=8192)
        parser.add_argument("--selected-len", type=int, default=640)
        parser.add_argument(
            "--invalid", type=int, default=0, help="number of trailing -1 indices"
        )
        parser.add_argument("--scale", type=float, default=D**-0.5)
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--atol", type=float, default=3e-2)
        parser.add_argument("--rtol", type=float, default=3e-2)
        self.add_timing_args(parser, warmup=5, iters=20, samples=7)
        self.add_sol_arg(parser)
        parser.add_argument("--debug", action="store_true")
        return parser

    def validate_args(self, args):
        if args.batch <= 0 or args.kv_len <= 0 or args.selected_len <= 0:
            raise SystemExit("batch, KV length, and selected length must be positive")
        if args.invalid < 0 or args.invalid >= args.selected_len:
            raise SystemExit("invalid must be in [0, selected length)")
        if not math.isfinite(args.scale) or args.scale <= 0:
            raise SystemExit("scale must be finite and positive")
        self.validate_timing_args(args, allow_zero_warmup=True)
        return args

    def run_with_args(self, args):
        torch.manual_seed(0)
        q, kv, indices, sink = make_inputs(
            args.batch, args.kv_len, args.selected_len, args.invalid
        )
        out = torch.empty_like(q)

        kernel = Bf16Fa4SparseMlaRunner(q, kv, indices, sink, out, scale=args.scale).run

        kernel()
        torch.cuda.synchronize()
        diff = "skipped"
        if not args.no_accuracy:
            expected = reference(q, kv, indices, sink, args.scale)
            torch.testing.assert_close(out, expected, atol=args.atol, rtol=args.rtol)
            diff = f"{(out.float() - expected.float()).abs().max().item():.6g}"
        stats = benchmark_with_stats(
            kernel,
            args.warmup,
            args.iters,
            args.samples,
            cold_l2_flush=args._cold_l2_flush,
        )
        ms = stats["median_ms"]
        valid = args.selected_len - args.invalid
        flops = 4 * args.batch * H * valid * D
        table = TablePrinter(
            (
                ("B", 5),
                ("KV", 8),
                ("Selected", 10),
                ("Kernel ms", 13),
                ("TF/s", 10),
                ("Max diff", 13),
            ),
            debug=args.debug,
        )
        table.print_header()
        table.print_row(
            args.batch,
            args.kv_len,
            valid,
            f"{ms:.4f}",
            f"{flops / (ms * 1e9):.3f}",
            diff,
        )
        if args.sol:
            report = SolReport(debug=args.debug)
            report.add(
                sparse_mla_sol_bounds(args.batch, args.selected_len, valid),
                ms,
                label=f"sparse MLA B={args.batch}, K={valid}",
            )
            report.print()


if __name__ == "__main__":
    SparseMlaH64Benchmark().run()
