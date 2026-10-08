#!/usr/bin/env python3
"""Cold-L2 prefill benchmark for the MXFP4 indexer score kernel."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    MMA_PEAK_TFLOPS,
    BenchmarkBase,
    SolReport,
    TablePrinter,
    benchmark_with_stats,
    sol_bounds,
)
from sm110a_kernel import (
    compile_indexer_score_prefill,
    from_blocked_mxfp4_scales,
    to_blocked_mxfp4_scales,
)
from sm110a_kernel.runner import TuneOptions

HEADS = 64
DIM = 128
SEQ_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768)
FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MXFP4_MMA_PEAK_TFLOPS = 4 * MMA_PEAK_TFLOPS


def indexer_score_work(qs, kvs, cluster_m=1, *, causal=True):
    q_tiles = (qs + 1) // 2
    kv_tiles = (kvs + 255) // 256
    kv_load_groups = (q_tiles + cluster_m - 1) // cluster_m
    active_tiles = [
        min(kv_tiles, ((group + 1) * cluster_m * 2 + 255) // 256)
        if causal
        else kv_tiles
        for group in range(kv_load_groups)
    ]
    flops = 2 * sum(active_tiles) * cluster_m * 2 * 256 * HEADS * DIM
    return q_tiles, kv_tiles, active_tiles, flops


def indexer_score_sol_bounds(qs, kvs, cluster_m=1, *, causal=True):
    """MMA and memory rooflines for the persistent Q-reuse schedule.

    Each query tile loads Q, Q scales, and head weights once, then traverses
    its visible 256-row KV tiles. Compulsory HBM includes FP32 output stores.
    It does not model epilogue instruction throughput or launch overhead.
    """
    q_tiles, _, active_tiles, flops = indexer_score_work(
        qs, kvs, cluster_m, causal=causal
    )
    q_bytes = qs * HEADS * DIM // 2
    k_bytes = kvs * DIM // 2
    q_scale_bytes = q_tiles * 1024
    k_scale_bytes = ((kvs + 127) // 128) * 1024
    weight_bytes = qs * HEADS * 2
    output_bytes = qs * kvs * 4
    input_bytes = q_bytes + k_bytes + q_scale_bytes + k_scale_bytes + weight_bytes
    compulsory_bytes = input_bytes + output_bytes
    q_tile_read_bytes = 128 * DIM // 2 + 1024 + 2 * HEADS * 2
    kv_tile_read_bytes = 256 * DIM // 2 + 2048
    tile_read_bytes = (
        q_tiles * q_tile_read_bytes + sum(active_tiles) * kv_tile_read_bytes
    )
    return sol_bounds(
        flops=flops,
        compulsory_bytes=compulsory_bytes,
        a_and_b_bytes=input_bytes,
        tile_read_bytes=tile_read_bytes,
        mma_peak_tflops=MXFP4_MMA_PEAK_TFLOPS,
    )


def make_packed_rows(rows):
    return torch.randint(
        0, 256, (rows, DIM // 2), device="cuda", dtype=torch.uint8
    ).view(torch.float4_e2m1fn_x2)


def make_blocked_scales(rows):
    padded_rows = (rows + 127) // 128 * 128
    logical = torch.randint(
        124, 127, (padded_rows, DIM // 32), device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    return to_blocked_mxfp4_scales(logical)


def decode_rows(packed_bytes, scale_bytes):
    """Decode selected packed rows and their logical E8M0 scales in FP32."""
    codes = torch.stack((packed_bytes & 15, packed_bytes >> 4), dim=-1).reshape(
        packed_bytes.size(0), DIM
    )
    magnitudes = torch.tensor(FP4_MAGNITUDES, device=packed_bytes.device)
    values = magnitudes[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -values, values)
    scales = (scale_bytes.to(torch.int32) << 23).view(torch.float32)
    return (values.view(-1, DIM // 32, 32) * scales.unsqueeze(-1)).reshape(-1, DIM)


def reference_rows(
    q, k, q_scales, k_scales, head_weight, query_indices, *, causal=True
):
    qs = q.size(0)
    kvs = k.size(0)
    q_row_ids = (
        query_indices[:, None] * HEADS + torch.arange(HEADS, device="cuda")[None, :]
    ).reshape(-1)
    q_scale_bytes = from_blocked_mxfp4_scales(q_scales, (qs + 1) // 2 * 128, DIM).view(
        torch.uint8
    )
    k_scale_bytes = from_blocked_mxfp4_scales(
        k_scales, (kvs + 127) // 128 * 128, DIM
    ).view(torch.uint8)
    q_dense = decode_rows(
        q.reshape(-1, DIM // 2).view(torch.uint8).index_select(0, q_row_ids),
        q_scale_bytes.index_select(0, q_row_ids),
    )
    k_dense = decode_rows(k.view(torch.uint8), k_scale_bytes[:kvs])
    dot = q_dense @ k_dense.T
    dot.clamp_min_(0)
    weights = head_weight.index_select(0, query_indices).float()
    scores = (dot.view(query_indices.numel(), HEADS, kvs) * weights[:, :, None]).sum(1)
    if causal:
        keys = torch.arange(kvs, device=scores.device)
        scores.masked_fill_(keys[None, :] > query_indices[:, None], -torch.inf)
    return scores


class IndexerScoreBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--seq-lens",
            type=int,
            nargs="+",
            default=SEQ_LENGTHS,
            help="prefill lengths; Qs and KVs both equal each length",
        )
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument(
            "--non-causal", action="store_true", help="disable the default causal mask"
        )
        parser.add_argument("--atol", type=float, default=5e-2)
        parser.add_argument("--rtol", type=float, default=1e-3)
        parser.add_argument(
            "--cluster-m",
            type=int,
            choices=(1, 2),
            help="run one CTA cluster size without autotuning",
        )
        parser.add_argument(
            "--sweep-cluster-m",
            action="store_true",
            help="remeasure cluster_m=1 and 2 with cold L2",
        )
        parser.add_argument(
            "--sol",
            action="store_true",
            help="report MXFP4 MMA, HBM, and L2 rooflines for each length",
        )
        self.add_timing_args(parser, warmup=3, iters=10, samples=5)
        return parser

    def validate_args(self, args):
        if not args.seq_lens or any(seq <= 0 for seq in args.seq_lens):
            raise SystemExit("--seq-lens must contain positive lengths")
        if args.atol < 0 or args.rtol < 0:
            raise SystemExit("--atol and --rtol must be non-negative")
        if args.cluster_m and args.sweep_cluster_m:
            raise SystemExit("--cluster-m and --sweep-cluster-m are mutually exclusive")
        self.validate_timing_args(args, allow_zero_warmup=True)
        return args

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        module = compile_indexer_score_prefill()
        table = TablePrinter(
            (
                ("Seq", 8),
                ("Cluster", 9),
                ("Candidates", 14),
                ("Kernel(ms)", 14),
                ("TF/s", 11),
                ("MaxDiff", 13),
            ),
        )
        table.print_header()
        sol_report = SolReport() if args.sol else None
        for seq in args.seq_lens:
            q = make_packed_rows(seq * HEADS).view(seq, HEADS, DIM // 2)
            k = make_packed_rows(seq)
            q_scales = make_blocked_scales(seq * HEADS)
            k_scales = make_blocked_scales(seq)
            head_weight = torch.randn((seq, HEADS), device="cuda", dtype=torch.float16)
            out = torch.empty((seq, seq), device="cuda", dtype=torch.float32)
            if args.sweep_cluster_m:
                cluster_m = module.select_cluster_m(
                    q,
                    k,
                    q_scales,
                    k_scales,
                    head_weight,
                    out,
                    causal=not args.non_causal,
                    options=TuneOptions(
                        warmup=args.warmup,
                        iters=args.iters,
                        samples=args.samples,
                        cold_l2_flush=args._cold_l2_flush,
                        force_sweep=True,
                        debug=True,
                    ),
                )
            elif args.cluster_m:
                cluster_m = args.cluster_m
            else:
                cluster_m = module.select_cluster_m(
                    q,
                    k,
                    q_scales,
                    k_scales,
                    head_weight,
                    out,
                    causal=not args.non_causal,
                )

            def kernel():
                return module.indexer_score_prefill_out(
                    q,
                    k,
                    q_scales,
                    k_scales,
                    head_weight,
                    out,
                    cluster_m,
                    causal=not args.non_causal,
                )

            kernel()
            torch.cuda.synchronize()
            max_diff = "skipped"
            if not args.no_accuracy:
                indices = torch.tensor(
                    sorted(
                        {
                            i
                            for i in (0, 1, 254, 255, 256, 257, seq // 2, seq - 1)
                            if i < seq
                        }
                    ),
                    device="cuda",
                    dtype=torch.long,
                )
                expected = reference_rows(
                    q,
                    k,
                    q_scales,
                    k_scales,
                    head_weight,
                    indices,
                    causal=not args.non_causal,
                )
                actual = out.index_select(0, indices)
                valid = torch.isfinite(expected)
                max_diff = f"{(actual[valid] - expected[valid]).abs().max().item():.6g}"
                torch.testing.assert_close(
                    actual, expected, atol=args.atol, rtol=args.rtol
                )

            stats = benchmark_with_stats(
                kernel,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=args._cold_l2_flush,
            )
            ms = stats["median_ms"]
            bounds = indexer_score_sol_bounds(
                seq, seq, cluster_m, causal=not args.non_causal
            )
            _, _, _, flops = indexer_score_work(
                seq, seq, cluster_m, causal=not args.non_causal
            )
            table.print_row(
                seq,
                cluster_m,
                seq * (seq + 1) // 2 if not args.non_causal else seq * seq,
                f"{ms:.4f}",
                f"{flops / (ms * 1e9):.2f}",
                max_diff,
            )
            if sol_report is not None:
                sol_report.add(
                    bounds,
                    ms,
                    label=f"seq={seq}",
                )
        if sol_report is not None:
            sol_report.print()


if __name__ == "__main__":
    IndexerScoreBenchmark().run()
