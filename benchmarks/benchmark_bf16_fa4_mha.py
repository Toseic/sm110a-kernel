#!/usr/bin/env python3
"""Benchmark the BF16 FA4 MHA kernel on prefill and decode workloads.

The default suite uses 32 attention heads and head dimension 128:

* causal prefill: B=1, Q=KV in {1K, 2K, 4K, 8K}
* decode: Q=1, KV=8K, B in {32, 64, 128}

Decode is deliberately non-causal. The kernel's causal mask is upper-left,
so a one-token query would otherwise only attend KV position zero; non-causal
Q=1 models the final token attending its complete KV cache.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    BenchmarkBase,
    TablePrinter,
    benchmark_with_stats,
    sol_bounds,
    sol_table_columns,
    sol_table_values,
)
from sm110a_kernel import bf16_fa4_mha_out


PREFILL_LENGTHS = (1024, 2048, 4096, 8192)
DECODE_BATCH_SIZES = (32, 64, 128)
DEFAULT_HEADS = 32
DEFAULT_HEAD_DIM = 128


@dataclass(frozen=True)
class AttentionCase:
    name: str
    batch: int
    heads: int
    q_len: int
    kv_len: int
    head_dim: int
    causal: bool


def attention_tflops(case, elapsed_ms):
    # QK^T and PV each cost 2 FLOPs per multiply-accumulate.
    flops = 4 * case.batch * case.heads * case.q_len * case.kv_len * case.head_dim
    return flops / (elapsed_ms * 1e9)


def output_tokens_per_second(case, elapsed_ms):
    return case.batch * case.q_len * 1e3 / elapsed_ms


def attention_sol_bounds(case, *, tile_q=128, tile_kv=128):
    """Return a traffic-aware SOL model for this kernel's tiled dataflow.

    The compulsory model reads Q/K/V and writes O once.  The tiled model adds
    the K/V reloads caused by assigning one K/V mainloop to every pair of Q
    tiles.  For causal prefill it counts only the prefix visible to each pair.
    """
    element_bytes = 2  # BF16 Q, K, V, and O
    bh = case.batch * case.heads
    q_bytes = element_bytes * bh * case.q_len * case.head_dim
    kv_input_bytes = 2 * element_bytes * bh * case.kv_len * case.head_dim
    output_bytes = element_bytes * bh * case.q_len * case.head_dim
    q_pair_count = math.ceil(case.q_len / (2 * tile_q))
    kv_tile_count = math.ceil(case.kv_len / tile_kv)
    kv_tiles_per_head = 0
    for q_pair_idx in range(q_pair_count):
        if case.causal:
            q_pair_end = min((q_pair_idx + 1) * 2 * tile_q, case.q_len)
            kv_tiles_per_head += min(math.ceil(q_pair_end / tile_kv), kv_tile_count)
        else:
            kv_tiles_per_head += kv_tile_count

    # K and V are each TMA-loaded for every visited KV tile. Q is loaded once
    # per task, so it has no inter-Q-pair reload in this dual-Q-tile design.
    kv_tile_bytes = 2 * element_bytes * tile_kv * case.head_dim
    tile_read_bytes = q_bytes + bh * kv_tiles_per_head * kv_tile_bytes
    return sol_bounds(
        4 * bh * case.q_len * case.kv_len * case.head_dim,
        q_bytes + kv_input_bytes + output_bytes,
        q_bytes + kv_input_bytes,
        tile_read_bytes,
    )


def reference_attention(q, k, v, *, causal, scale):
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        mask = torch.ones((q.size(-2), k.size(-2)), device=q.device, dtype=torch.bool).triu(1)
        scores.masked_fill_(mask, float("-inf"))
    return torch.softmax(scores, dim=-1).matmul(v.float()).to(torch.bfloat16)


class Bf16Fa4MhaBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--single",
            action="store_true",
            help="benchmark one explicit shape instead of the default prefill/decode suite",
        )
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
        parser.add_argument("--q-len", type=int, default=1024)
        parser.add_argument("--kv-len", type=int, default=1024)
        parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
        parser.add_argument("--causal", action="store_true")
        parser.add_argument("--scale", type=float, default=None)
        self.add_timing_args(parser, warmup=5, iters=20, samples=7)
        self.add_sol_arg(parser)
        parser.add_argument("--debug", action="store_true", help="print benchmark records line by line")
        parser.add_argument(
            "--no-accuracy",
            action="store_true",
            help="skip the FP32 reference check in --single mode",
        )
        parser.add_argument("--atol", type=float, default=2e-2)
        parser.add_argument("--rtol", type=float, default=2e-2)
        return parser

    def validate_args(self, args):
        if min(args.batch, args.heads, args.q_len, args.kv_len, args.head_dim) <= 0:
            raise SystemExit("all dimensions must be positive")
        self.validate_timing_args(args, allow_zero_warmup=True)
        if args.scale is not None and (not math.isfinite(args.scale) or args.scale <= 0):
            raise SystemExit("--scale must be finite and positive")
        return args

    def cases(self, args):
        if args.single:
            yield AttentionCase(
                "single", args.batch, args.heads, args.q_len, args.kv_len,
                args.head_dim, args.causal,
            )
            return

        for seq_len in PREFILL_LENGTHS:
            yield AttentionCase(
                "prefill", 1, DEFAULT_HEADS, seq_len, seq_len,
                DEFAULT_HEAD_DIM, True,
            )
        for batch in DECODE_BATCH_SIZES:
            yield AttentionCase(
                "decode", batch, DEFAULT_HEADS, 1, 8192,
                DEFAULT_HEAD_DIM, False,
            )

    @staticmethod
    def make_tensors(case):
        q = torch.empty(
            (case.batch, case.heads, case.q_len, case.head_dim),
            device="cuda", dtype=torch.bfloat16,
        ).uniform_(-1, 1)
        k = torch.empty(
            (case.batch, case.heads, case.kv_len, case.head_dim),
            device="cuda", dtype=torch.bfloat16,
        ).uniform_(-1, 1)
        v = torch.empty_like(k).uniform_(-1, 1)
        return q, k, v, torch.empty_like(q)

    def run_case(self, case, args):
        q, k, v, out = self.make_tensors(case)
        scale = args.scale if args.scale is not None else 1.0 / math.sqrt(case.head_dim)

        def run():
            bf16_fa4_mha_out(q, k, v, out, causal=case.causal, scale=scale)

        run()  # Compile and validate the launch before timing.
        torch.cuda.synchronize()
        if args.single and not args.no_accuracy:
            reference = reference_attention(q, k, v, causal=case.causal, scale=scale)
            torch.testing.assert_close(out, reference, atol=args.atol, rtol=args.rtol)
            max_diff = (out.float() - reference.float()).abs().max().item()
            print(f"[accuracy] PASS (max_abs_diff={max_diff:.6g})")

        return benchmark_with_stats(
            run, args.warmup, args.iters, args.samples, cold_l2_flush=args._cold_l2_flush
        )

    def run_with_args(self, args):
        if args.single:
            print("[mode] single shape (use no arguments for the default workload suite)")
        else:
            print(
                "[mode] default suite: 32 heads, head_dim=128; "
                "prefill 1K-8K and decode KV=8K"
            )

        columns = (
            ("mode", 10), ("B", 6), ("H", 6), ("Q", 8), ("KV", 8),
            ("D", 6), ("causal", 9), ("median ms", 13), ("TF/s", 12),
            ("out tok/s", 16),
        ) + (sol_table_columns()[1:] if args.sol else ())
        table = TablePrinter(columns, debug=args.debug)
        table.print_header()
        for case in self.cases(args):
            stats = self.run_case(case, args)
            values = (
                case.name,
                case.batch,
                case.heads,
                case.q_len,
                case.kv_len,
                case.head_dim,
                str(case.causal),
                f"{stats['median_ms']:.4f}",
                f"{attention_tflops(case, stats['median_ms']):.3f}",
                f"{output_tokens_per_second(case, stats['median_ms']):.0f}",
            )
            if args.sol:
                values += sol_table_values(
                    attention_sol_bounds(case),
                    stats["median_ms"],
                )[1:]
            table.print_row(*values)


if __name__ == "__main__":
    Bf16Fa4MhaBenchmark().run()
