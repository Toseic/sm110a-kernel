#!/usr/bin/env python3
"""Correctness and cold-L2 timing for H=64 sparse MLA prefill and append."""

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
    SolReport,
    TablePrinter,
    benchmark_with_stats,
    sol_bounds,
)
from sm110a_kernel import bf16_fa4_sparse_mla_prefill_out

H = 64
D = 512
PREFILL_Q_LENGTHS = (1024, 2048, 4096)
BATCHED_Q_LENGTHS = (1024,)
BATCHED_BATCH = 4
APPEND_Q_LENGTHS = (1, 16, 128)
APPEND_CACHE_LENGTH = 1024


@dataclass(frozen=True)
class PrefillCase:
    mode: str
    batch: int
    q_len: int
    kv_len: int
    selected_len: int


def sparse_mla_sol_bounds(case, indices):
    """Idealized traffic: one KV-row read per valid selection, with L2 reuse."""
    valid = (indices >= 0) & (indices < case.kv_len)
    valid_rows = int(valid.sum().item())
    unique_rows = sum(
        torch.unique(indices[b][valid[b]]).numel() for b in range(case.batch)
    )
    q_bytes = 2 * case.batch * H * case.q_len * D
    out_bytes = q_bytes
    kv_bytes = 2 * unique_rows * D
    kv_tile_read_bytes = 2 * valid_rows * D
    index_bytes = 4 * case.batch * case.q_len * case.selected_len
    sink_bytes = 4 * H
    compulsory_bytes = q_bytes + out_bytes + kv_bytes + index_bytes + sink_bytes
    input_bytes = compulsory_bytes - out_bytes
    tile_read_bytes = (
        q_bytes
        + kv_tile_read_bytes
        + index_bytes
        + case.batch * case.q_len * sink_bytes
    )
    flops = 4 * H * D * valid_rows
    return sol_bounds(flops, compulsory_bytes, input_bytes, tile_read_bytes)


def make_inputs(batch, q_len, cache_len, selected_len, invalid, out_of_range=0):
    assert cache_len >= q_len
    q = torch.empty((batch, H, q_len, D), device="cuda", dtype=torch.bfloat16).uniform_(
        -1, 1
    )
    kv = torch.empty(
        (batch, cache_len, D), device="cuda", dtype=torch.bfloat16
    ).uniform_(-1, 1)
    # Append queries are the last Q entries of a larger KV cache. Each query
    # sees the old cache plus its causal new prefix; ordinary prefill is the
    # special case with no old cache. Keep duplicate selections intentional.
    visible = (cache_len - q_len + torch.arange(1, q_len + 1, device="cuda")).view(
        1, q_len, 1
    )
    slots = torch.arange(selected_len, device="cuda").view(1, 1, selected_len)
    indices = (torch.rand((batch, q_len, selected_len), device="cuda") * visible).to(
        torch.int32
    )
    if cache_len != q_len:
        # Exercise KV offsets beyond Q length, not merely an oversized unused
        # allocation. Include both the newest visible row and an old row.
        indices[..., 0] = (visible - 1).squeeze(-1)
        if selected_len > 1:
            indices[..., 1] = 0
    indices.masked_fill_(slots >= visible, -1)
    if out_of_range:
        end = selected_len - invalid
        indices[..., end - out_of_range : end] = cache_len
    if invalid:
        indices[..., -invalid:] = -1
    sink = torch.randn(H, device="cuda", dtype=torch.float32)
    return q, kv, indices, sink


def reference(q, kv, indices, sink, scale):
    out = torch.empty_like(q)
    for b in range(q.size(0)):
        for t in range(q.size(2)):
            idx = indices[b, t]
            valid = (idx >= 0) & (idx < kv.size(1))
            selected = kv[b].index_select(0, idx[valid].long()).float()
            if selected.numel() == 0:
                out[b, :, t] = 0
                continue
            scores = (q[b, :, t].float() @ selected.T) * scale
            row_max = torch.maximum(scores.max(-1).values, sink)
            weights = torch.exp(scores - row_max[:, None])
            denom = weights.sum(-1) + torch.exp(sink - row_max)
            out[b, :, t] = (weights @ selected / denom[:, None]).to(q.dtype)
    return out


class SparseMlaPrefillBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--single", action="store_true", help="run one custom shape"
        )
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--q-len", type=int, default=16)
        parser.add_argument(
            "--kv-len",
            type=int,
            default=None,
            help="total KV length including appended queries; defaults to Q length",
        )
        parser.add_argument("--selected-len", type=int, default=640)
        parser.add_argument("--invalid", type=int, default=0)
        parser.add_argument(
            "--out-of-range",
            type=int,
            default=0,
            help="number of selections set to KV length (invalid upper bound)",
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
        if min(args.batch, args.q_len, args.selected_len) <= 0:
            raise SystemExit(
                "batch, Q length, KV length and selected length must be positive"
            )
        if args.kv_len is not None and args.kv_len <= 0:
            raise SystemExit("KV length must be positive")
        if args.single and args.kv_len is not None and args.kv_len < args.q_len:
            raise SystemExit("causal prefill/append requires --kv-len >= --q-len")
        if not args.single and args.kv_len is not None:
            raise SystemExit("--kv-len is only used with --single")
        if not 0 <= args.invalid <= args.selected_len:
            raise SystemExit("invalid must be in [0, selected length]")
        if not 0 <= args.out_of_range <= args.selected_len - args.invalid:
            raise SystemExit("invalid + out-of-range must not exceed selected length")
        if not math.isfinite(args.scale) or args.scale <= 0:
            raise SystemExit("scale must be finite and positive")
        self.validate_timing_args(args, allow_zero_warmup=True)
        return args

    def cases(self, args):
        if args.single:
            kv_len = args.kv_len if args.kv_len is not None else args.q_len
            yield PrefillCase(
                "append" if kv_len != args.q_len else "single",
                args.batch,
                args.q_len,
                kv_len,
                args.selected_len,
            )
            return
        for q_len in PREFILL_Q_LENGTHS:
            yield PrefillCase("prefill", 1, q_len, q_len, args.selected_len)
        for q_len in BATCHED_Q_LENGTHS:
            yield PrefillCase("batched", BATCHED_BATCH, q_len, q_len, args.selected_len)
        for q_len in APPEND_Q_LENGTHS:
            yield PrefillCase(
                "append", 1, q_len, APPEND_CACHE_LENGTH, args.selected_len
            )
        yield PrefillCase(
            "append", BATCHED_BATCH, 16, APPEND_CACHE_LENGTH, args.selected_len
        )

    def run_case(self, case, args):
        q, kv, indices, sink = make_inputs(
            case.batch,
            case.q_len,
            case.kv_len,
            case.selected_len,
            args.invalid,
            args.out_of_range,
        )
        out = torch.empty_like(q)

        def kernel():
            return bf16_fa4_sparse_mla_prefill_out(
                q, kv, indices, sink, out, scale=args.scale
            )

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
        valid_rows = int(((indices >= 0) & (indices < case.kv_len)).sum().item())
        flops = 4 * H * D * valid_rows
        bounds = sparse_mla_sol_bounds(case, indices) if args.sol else None
        return ms, flops, valid_rows, diff, bounds

    def run_with_args(self, args):
        torch.manual_seed(0)
        table = TablePrinter(
            (
                ("mode", 9),
                ("B", 5),
                ("Q", 5),
                ("KV", 8),
                ("K", 7),
                ("Valid/Q", 9),
                ("Kernel ms", 13),
                ("TF/s", 10),
                ("Max diff", 13),
            ),
            debug=args.debug,
        )
        table.print_header()
        sol_report = SolReport(debug=args.debug) if args.sol else None
        for case in self.cases(args):
            ms, flops, valid_rows, diff, bounds = self.run_case(case, args)
            table.print_row(
                case.mode,
                case.batch,
                case.q_len,
                case.kv_len,
                case.selected_len,
                f"{valid_rows / (case.batch * case.q_len):.1f}",
                f"{ms:.4f}",
                f"{flops / (ms * 1e9):.3f}",
                diff,
            )
            if sol_report is not None:
                sol_report.add(
                    bounds,
                    ms,
                    label=f"{case.mode} B={case.batch} Q={case.q_len} K={case.selected_len}",
                )
        if sol_report is not None:
            sol_report.print()


if __name__ == "__main__":
    SparseMlaPrefillBenchmark().run()
