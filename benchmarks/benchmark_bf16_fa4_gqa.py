#!/usr/bin/env python3
"""Benchmark the BF16 FA4-style GQA kernel against official FlashAttention-4.

The kernel uses BHSD tensors: Q is ``[B, Hq, Q, D]`` and K/V are
``[B, Hkv, KV, D]``. It is specialized for ``Hq / Hkv == 4``. Two orthogonal
axes select the schedule:

* ``--tactic`` — 16-warp or 12-warp CTA schedule.
* ``--pack-gqa`` — fold the four query heads into the logical query-row
  dimension of one CTA instead of assigning one query head per CTA.

The default suite matches the MHA benchmark: causal prefill at B=1 with
Q=KV in {1K..16K} and non-causal decode at Q=1, KV=8K with B in {32, 64, 128}.
"""

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import BenchmarkBase, TablePrinter, benchmark_with_stats, sol_bounds
from sm110a_kernel import Bf16Fa4GqaTactic, bf16_fa4_gqa_out


PREFILL_LENGTHS = (1024, 2048, 4096, 8192, 16384)
DECODE_BATCH_SIZES = (32, 64, 128)
DEFAULT_Q_HEADS = 32
DEFAULT_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128


@dataclass(frozen=True)
class GqaCase:
    name: str
    batch: int
    q_heads: int
    kv_heads: int
    q_len: int
    kv_len: int
    head_dim: int
    causal: bool


def attention_valid_score_count(case):
    """Number of unmasked QK scores for one query head."""
    if not case.causal:
        return case.q_len * case.kv_len
    # The kernel follows FA's bottom-right causal alignment. Its causal path
    # requires KV >= Q, so row r sees KV-Q+r+1 keys.
    return case.q_len * (case.kv_len - case.q_len) + case.q_len * (case.q_len + 1) // 2


def attention_flops(case):
    # QK and PV each perform a multiply and an add for every visible score.
    return 4 * case.batch * case.q_heads * attention_valid_score_count(case) * case.head_dim


def attention_tflops(case, elapsed_ms):
    return attention_flops(case) / (elapsed_ms * 1e9)


def attention_sol_bounds(case, *, tile_q=128, tile_kv=128, packed_gqa=False):
    """Traffic-aware SOL for the selected GQA schedule.

    Compulsory traffic counts the physical GQA inputs once: there are Hq Q
    heads but only Hkv K/V heads. Packed schedules load each visited K/V tile
    once per K/V-head task; unpacked schedules re-read it once per query
    head (L2 serves the GQA-group reuse).
    """
    element_bytes = 2  # BF16 Q, K, V, O
    q_bytes = element_bytes * case.batch * case.q_heads * case.q_len * case.head_dim
    kv_input_bytes = 2 * element_bytes * case.batch * case.kv_heads * case.kv_len * case.head_dim
    output_bytes = element_bytes * case.batch * case.q_heads * case.q_len * case.head_dim

    packed_factor = case.q_heads // case.kv_heads if packed_gqa else 1
    logical_q_len = case.q_len * packed_factor
    q_pair_count = math.ceil(logical_q_len / (2 * tile_q))
    kv_tile_count = math.ceil(case.kv_len / tile_kv)
    kv_tiles_per_task = 0
    for q_pair_idx in range(q_pair_count):
        if case.causal:
            logical_q_pair_end = min((q_pair_idx + 1) * 2 * tile_q, logical_q_len)
            q_pair_end = math.ceil(logical_q_pair_end / packed_factor)
            kv_tiles_per_task += min(math.ceil(q_pair_end / tile_kv), kv_tile_count)
        else:
            kv_tiles_per_task += kv_tile_count

    # Both K and V are loaded once per scheduling task's visited tiles.
    task_count = case.batch * (case.kv_heads if packed_gqa else case.q_heads)
    kv_tile_bytes = 2 * element_bytes * tile_kv * case.head_dim
    tile_read_bytes = q_bytes + task_count * kv_tiles_per_task * kv_tile_bytes
    return sol_bounds(
        attention_flops(case),
        q_bytes + kv_input_bytes + output_bytes,
        q_bytes + kv_input_bytes,
        tile_read_bytes,
    )


def reference_attention(q, k, v, *, causal, scale):
    group_size = q.size(1) // k.size(1)
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        causal_offset = k.size(-2) - q.size(-2)
        scores.masked_fill_(
            torch.ones((q.size(-2), k.size(-2)), device=q.device, dtype=torch.bool).triu(
                1 + causal_offset
            ),
            float("-inf"),
        )
    return torch.softmax(scores, dim=-1).matmul(v.float()).to(torch.bfloat16)


def load_official_fa4():
    """Load FA4's forward entry point, which auto-enables packed GQA."""
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
    except ImportError as exc:
        raise SystemExit(
            'official FlashAttention-4 is required; install "flash-attn-4[cu13]"'
        ) from exc
    # FA4 currently emits this CuTe adapter warning while compiling despite
    # producing a valid forward kernel.
    warnings.filterwarnings(
        "ignore",
        message=r"Argument aux_data .*cannot be converted to a JitArgument.*",
        category=UserWarning,
    )
    return _flash_attn_fwd


class Bf16Fa4GqaBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--single", action="store_true")
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--q-heads", type=int, default=32)
        parser.add_argument("--kv-heads", type=int, default=8)
        parser.add_argument("--q-len", type=int, default=1024)
        parser.add_argument("--kv-len", type=int, default=1024)
        parser.add_argument("--head-dim", type=int, default=128)
        parser.add_argument("--causal", action="store_true")
        parser.add_argument("--scale", type=float, default=None)
        parser.add_argument(
            "--tactic",
            choices=("16warp", "12warp"),
            default="16warp",
        )
        parser.add_argument(
            "--pack-gqa",
            action="store_true",
            help="fold the four query heads into one packed-M CTA per K/V head",
        )
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--atol", type=float, default=2e-2)
        parser.add_argument("--rtol", type=float, default=2e-2)
        self.add_timing_args(parser, warmup=5, iters=20, samples=7)
        self.add_sol_arg(parser)
        return parser

    def validate_args(self, args):
        if min(args.batch, args.q_heads, args.kv_heads, args.q_len, args.kv_len, args.head_dim) <= 0:
            raise SystemExit("all dimensions must be positive")
        if args.q_heads != 4 * args.kv_heads:
            raise SystemExit("this kernel requires --q-heads == 4 * --kv-heads")
        if args.scale is not None and (not math.isfinite(args.scale) or args.scale <= 0):
            raise SystemExit("--scale must be finite and positive")
        self.validate_timing_args(args, allow_zero_warmup=True)
        return args

    def cases(self, args):
        if args.single:
            yield GqaCase("single", args.batch, args.q_heads, args.kv_heads, args.q_len,
                          args.kv_len, args.head_dim, args.causal)
            return
        for seq_len in PREFILL_LENGTHS:
            yield GqaCase("prefill", 1, DEFAULT_Q_HEADS, DEFAULT_KV_HEADS, seq_len,
                          seq_len, DEFAULT_HEAD_DIM, True)
        # Decode is deliberately non-causal, mirroring the MHA benchmark: a
        # one-token query under the bottom-right causal mask would only see
        # KV position zero, while non-causal Q=1 models the final token
        # attending its complete KV cache.
        for batch in DECODE_BATCH_SIZES:
            yield GqaCase("decode", batch, DEFAULT_Q_HEADS, DEFAULT_KV_HEADS, 1,
                          8192, DEFAULT_HEAD_DIM, False)

    def run_with_args(self, args):
        tactic = {
            "16warp": Bf16Fa4GqaTactic.WARP_16,
            "12warp": Bf16Fa4GqaTactic.WARP_12,
        }[args.tactic]
        schedule_label = f"{args.tactic}{'/packed' if args.pack_gqa else ''}"
        fa4_fwd = load_official_fa4()
        print(f"[tactic] {schedule_label}")
        columns = (("mode", 9), ("shape [B,Hq,Hkv,Q,KV,D]", 30), ("mask", 10),
                   ("Kernel ms", 13), ("Kernel TF/s", 14), ("FA4 ms", 13),
                   ("FA4/kernel", 14), ("Diff", 12))
        if args.sol:
            columns += (("SOL selected(ms)", 17), ("SOL attainment", 15))
        table = TablePrinter(columns)
        table.print_header()
        for case in self.cases(args):
            q = torch.empty((case.batch, case.q_heads, case.q_len, case.head_dim), device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
            k = torch.empty((case.batch, case.kv_heads, case.kv_len, case.head_dim), device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
            v = torch.empty_like(k).uniform_(-1, 1)
            out = torch.empty_like(q)
            # FA4 consumes BSHD and permits Hq != Hkv; these conversions are
            # outside the timed region, as in the MHA benchmark.
            q_fa4 = q.permute(0, 2, 1, 3).contiguous()
            k_fa4 = k.permute(0, 2, 1, 3).contiguous()
            v_fa4 = v.permute(0, 2, 1, 3).contiguous()
            out_fa4 = torch.empty_like(q_fa4)
            scale = args.scale or 1.0 / math.sqrt(case.head_dim)
            def kernel():
                bf16_fa4_gqa_out(q, k, v, out, causal=case.causal, scale=scale,
                                 tactic=tactic, pack_gqa=args.pack_gqa)
            def fa4():
                fa4_fwd(q_fa4, k_fa4, v_fa4, softmax_scale=scale,
                        causal=case.causal, out=out_fa4)
            kernel(); fa4(); torch.cuda.synchronize()
            out_fa4_bhsd = out_fa4.permute(0, 2, 1, 3)
            torch.testing.assert_close(out, out_fa4_bhsd, atol=args.atol, rtol=args.rtol)
            if args.single and not args.no_accuracy:
                torch.testing.assert_close(out, reference_attention(q, k, v, causal=case.causal, scale=scale), atol=args.atol, rtol=args.rtol)
            kernel_stats = benchmark_with_stats(kernel, args.warmup, args.iters, args.samples,
                                                cold_l2_flush=args._cold_l2_flush)
            fa4_stats = benchmark_with_stats(fa4, args.warmup, args.iters, args.samples,
                                             cold_l2_flush=args._cold_l2_flush)
            diff = (out.float() - out_fa4_bhsd.float()).abs().max().item()
            values = (case.name, f"{case.batch}x{case.q_heads}x{case.kv_heads}x{case.q_len}x{case.kv_len}x{case.head_dim}",
                      "causal" if case.causal else "noncausal", f"{kernel_stats['median_ms']:.4f}",
                      f"{attention_tflops(case, kernel_stats['median_ms']):.3f}",
                      f"{fa4_stats['median_ms']:.4f}", f"{fa4_stats['median_ms'] / kernel_stats['median_ms']:.3f}x", f"{diff:.6g}")
            if args.sol:
                bounds = attention_sol_bounds(case, packed_gqa=args.pack_gqa)
                values += (f"{bounds['sol_ms']:.4f}",
                           f"{100.0 * bounds['sol_ms'] / kernel_stats['median_ms']:.1f}%")
            table.print_row(*values)


if __name__ == "__main__":
    Bf16Fa4GqaBenchmark().run()
