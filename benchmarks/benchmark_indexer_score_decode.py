#!/usr/bin/env python3
"""Cold-L2 benchmark for batched MXFP4 indexer-score decoding."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_indexer_score import (
    DIM,
    HEADS,
    MXFP4_MMA_PEAK_TFLOPS,
    make_blocked_scales,
    make_packed_rows,
    reference_rows,
)

from common import (
    BenchmarkBase,
    SolReport,
    TablePrinter,
    benchmark_with_stats,
    sol_bounds,
)
from sm110a_kernel import compile_indexer_score_decode

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
KV_LENGTHS = (16384,)
SPLIT_KVS = (1, 2, 4, 8)


def indexer_score_decode_sol_bounds(batch, kvs, split_kv):
    """Useful MMA work and approximate HBM/L2 traffic; excludes launch overhead."""
    kv_tiles = (kvs + 255) // 256
    q_bytes = batch * HEADS * DIM // 2
    k_bytes = batch * kvs * DIM // 2
    q_scale_bytes = batch * 1024
    k_scale_bytes = batch * ((kvs + 127) // 128) * 1024
    weight_bytes = batch * HEADS * 2
    output_bytes = batch * kvs * 4
    input_bytes = q_bytes + k_bytes + q_scale_bytes + k_scale_bytes + weight_bytes
    q_tile_read_bytes = HEADS * DIM // 2 + 1024 + HEADS * 2
    kv_tile_read_bytes = 256 * DIM // 2 + 2048
    tile_read_bytes = batch * (
        split_kv * q_tile_read_bytes + kv_tiles * kv_tile_read_bytes
    )
    return sol_bounds(
        flops=2 * batch * kvs * HEADS * DIM,
        compulsory_bytes=input_bytes + output_bytes,
        a_and_b_bytes=input_bytes,
        tile_read_bytes=tile_read_bytes,
        mma_peak_tflops=MXFP4_MMA_PEAK_TFLOPS,
    )


def sampled_reference(q, k, q_scales, k_scales, head_weight, indices):
    kvs = k.size(1)
    k_scale_stride = ((kvs + 127) // 128) * 1024
    query_index = torch.tensor([0], device="cuda", dtype=torch.long)
    return torch.stack(
        [
            reference_rows(
                q[b : b + 1],
                k[b],
                q_scales[b * 1024 : (b + 1) * 1024],
                k_scales[b * k_scale_stride : (b + 1) * k_scale_stride],
                head_weight[b : b + 1],
                query_index,
            )[0]
            for b in indices
        ]
    )


class IndexerScoreDecodeBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--batches", type=int, nargs="+", default=BATCHES)
        parser.add_argument("--kv-lens", type=int, nargs="+", default=KV_LENGTHS)
        parser.add_argument(
            "--split-kvs", type=int, nargs="+", choices=SPLIT_KVS, default=SPLIT_KVS
        )
        parser.add_argument("--seed", type=int, default=1234)
        auto_group = parser.add_mutually_exclusive_group()
        auto_group.add_argument(
            "--auto",
            dest="auto",
            action="store_true",
            default=True,
            help="also benchmark the autotuned split (default)",
        )
        auto_group.add_argument(
            "--no-auto",
            dest="auto",
            action="store_false",
            help="benchmark fixed splits only",
        )
        parser.add_argument(
            "--cache-path", default=None, help="autotune cache JSON path"
        )
        parser.add_argument(
            "--force-sweep", action="store_true", help="remeasure every valid split"
        )
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--atol", type=float, default=5e-2)
        parser.add_argument("--rtol", type=float, default=1e-3)
        parser.add_argument(
            "--sol",
            action="store_true",
            help="report approximate MMA, HBM, and L2 rooflines",
        )
        self.add_timing_args(parser, warmup=3, iters=10, samples=5)
        return parser

    def validate_args(self, args):
        if any(batch <= 0 for batch in args.batches):
            raise SystemExit("--batches must contain positive sizes")
        if any(kvs <= 0 for kvs in args.kv_lens):
            raise SystemExit("--kv-lens must contain positive lengths")
        if args.atol < 0 or args.rtol < 0:
            raise SystemExit("--atol and --rtol must be non-negative")
        if not args.auto and not any(
            split <= (kvs + 255) // 256
            for kvs in args.kv_lens
            for split in args.split_kvs
        ):
            raise SystemExit("no valid KV length/split_kv combination")
        self.validate_timing_args(args, allow_zero_warmup=True)
        return args

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        module = compile_indexer_score_decode()
        table = TablePrinter(
            (
                ("B", 7),
                ("KVs", 8),
                ("SplitKV", 9),
                ("Candidates", 14),
                ("Kernel(us)", 14),
                ("TF/s", 11),
                ("MaxDiff", 13),
            )
        )
        table.print_header()
        sol_report = SolReport() if args.sol else None
        skipped = []
        for batch in args.batches:
            for kvs in args.kv_lens:
                valid_splits = [s for s in args.split_kvs if s <= (kvs + 255) // 256]
                skipped.extend(
                    (batch, kvs, s) for s in args.split_kvs if s not in valid_splits
                )
                if not valid_splits and not args.auto:
                    continue

                q = make_packed_rows(batch * HEADS).view(batch, HEADS, DIM // 2)
                k = make_packed_rows(batch * kvs).view(batch, kvs, DIM // 2)
                q_scales = torch.cat([make_blocked_scales(HEADS) for _ in range(batch)])
                k_scales = torch.cat([make_blocked_scales(kvs) for _ in range(batch)])
                head_weight = torch.randn(
                    (batch, HEADS), device="cuda", dtype=torch.float16
                )
                out = torch.empty((batch, kvs), device="cuda", dtype=torch.float32)

                indices = sorted({0, batch // 2, batch - 1})
                expected = None
                if not args.no_accuracy:
                    expected = sampled_reference(
                        q, k, q_scales, k_scales, head_weight, indices
                    )

                cases = [(str(split), split) for split in valid_splits]
                if args.auto:
                    selected = module.select_split_kv(
                        q,
                        k,
                        q_scales,
                        k_scales,
                        head_weight,
                        out,
                        options=self.make_tune_options(args),
                    )
                    cases.append((f"auto({selected})", selected))

                for split_label, split_kv in cases:

                    def kernel():
                        return module.indexer_score_decode_out(
                            q, k, q_scales, k_scales, head_weight, out, split_kv
                        )

                    max_diff = "skipped"
                    if expected is not None:
                        out.fill_(float("nan"))
                    kernel()
                    torch.cuda.synchronize()
                    if expected is not None:
                        actual = out[indices]
                        max_diff = f"{(actual - expected).abs().max().item():.6g}"
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
                    flops = 2 * batch * kvs * HEADS * DIM
                    table.print_row(
                        batch,
                        kvs,
                        split_label,
                        batch * kvs,
                        f"{ms * 1000:.2f}",
                        f"{flops / (ms * 1e9):.2f}",
                        max_diff,
                    )
                    if sol_report is not None:
                        sol_report.add(
                            indexer_score_decode_sol_bounds(batch, kvs, split_kv),
                            ms,
                            label=f"B={batch},KVs={kvs},split={split_kv}",
                        )
        if skipped:
            print("Skipped split_kv > ceil(KVs/256): " + ", ".join(map(str, skipped)))
        if sol_report is not None:
            sol_report.print()


if __name__ == "__main__":
    IndexerScoreDecodeBenchmark().run()
