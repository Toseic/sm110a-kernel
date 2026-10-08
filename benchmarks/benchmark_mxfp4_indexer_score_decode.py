#!/usr/bin/env python3
"""Cold-L2 benchmark for batched MXFP4 indexer-score decoding."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_mxfp4_indexer_score_prefill import (
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


def random_kv_lengths(batch, kvs, generator):
    """Shuffle balanced long/short pairs while keeping sum(lengths)=B*KVs."""
    lengths = [kvs] * batch
    if batch < 2 or kvs < 2:
        return lengths
    order = torch.randperm(batch, generator=generator).tolist()
    for pair in range(batch // 2):
        delta = torch.randint(1, kvs // 2 + 1, (1,), generator=generator).item()
        lengths[order[2 * pair]] += delta
        lengths[order[2 * pair + 1]] -= delta
    return lengths


def indexer_score_decode_sol_bounds(lengths, kv_capacity, split_kv):
    """Executed MMA tiles and approximate cold-L2 HBM/L2 traffic."""
    batch = len(lengths)
    total_tiles = (kv_capacity + 255) // 256
    active_tiles = sum((length + 255) // 256 for length in lengths)
    active_splits = sum(
        (length + 255) // 256 > total_tiles * split // split_kv
        for length in lengths
        for split in range(split_kv)
    )
    q_bytes = batch * HEADS * DIM // 2
    k_bytes = sum(lengths) * DIM // 2
    q_scale_bytes = batch * 1024
    k_scale_bytes = sum((length + 127) // 128 for length in lengths) * 1024
    weight_bytes = batch * HEADS * 2
    output_bytes = batch * kv_capacity * 4
    input_bytes = q_bytes + k_bytes + q_scale_bytes + k_scale_bytes + weight_bytes
    q_tile_read_bytes = HEADS * DIM // 2 + 1024 + HEADS * 2
    kv_tile_read_bytes = 256 * DIM // 2 + 2048
    tile_read_bytes = (
        active_splits * q_tile_read_bytes + active_tiles * kv_tile_read_bytes
    )
    return sol_bounds(
        flops=2 * active_tiles * 256 * HEADS * DIM,
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
                causal=False,
            )[0]
            for b in indices
        ]
    )


def make_paged_k(k, k_scales, page_size):
    batch, kvs, packed_dim = k.shape
    pages_per_batch = (kvs + page_size - 1) // page_size
    num_pages = batch * pages_per_batch
    permutation = torch.randperm(num_pages, device="cuda", dtype=torch.int32)
    block_table = permutation.view(batch, pages_per_batch).contiguous()
    paged_k = make_packed_rows(num_pages * page_size).view(
        num_pages, page_size, packed_dim
    )
    paged_scales = make_blocked_scales(num_pages * page_size)
    src_k = k.view(torch.uint8)
    dst_k = paged_k.view(torch.uint8)
    src_scales = k_scales.view(torch.uint8).view(batch, (kvs + 127) // 128, 1024)
    dst_scales = paged_scales.view(torch.uint8).view(num_pages, page_size // 128, 1024)
    for b in range(batch):
        for logical_page in range(pages_per_batch):
            physical_page = block_table[b, logical_page].item()
            start = logical_page * page_size
            count = min(page_size, kvs - start)
            dst_k[physical_page, :count].copy_(src_k[b, start : start + count])
            scale_start = start // 128
            scale_count = (count + 127) // 128
            dst_scales[physical_page, :scale_count].copy_(
                src_scales[b, scale_start : scale_start + scale_count]
            )
    return paged_k, paged_scales, block_table


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
        parser.add_argument("--paged-page-size", type=int, choices=(128, 256))
        ragged_group = parser.add_mutually_exclusive_group()
        ragged_group.add_argument(
            "--ragged",
            dest="ragged",
            action="store_true",
            default=True,
            help="random balanced KV lengths with fixed total tokens (default)",
        )
        ragged_group.add_argument(
            "--uniform",
            dest="ragged",
            action="store_false",
            help="use KVs for every batch without extra KV capacity",
        )
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
            split <= (kvs + (kvs // 2 if args.ragged and batch > 1 else 0) + 255) // 256
            for batch in args.batches
            for kvs in args.kv_lens
            for split in args.split_kvs
        ):
            raise SystemExit("no valid KV length/split_kv combination")
        self.validate_timing_args(args, allow_zero_warmup=True)
        return args

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        length_generator = torch.Generator().manual_seed(args.seed)
        module = compile_indexer_score_decode()
        table = TablePrinter(
            (
                ("B", 7),
                ("KVs", 8),
                ("Kcap", 8),
                ("KV range", 13),
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
                kv_capacity = kvs + kvs // 2 if args.ragged and batch > 1 else kvs
                valid_splits = [
                    s for s in args.split_kvs if s <= (kv_capacity + 255) // 256
                ]
                skipped.extend(
                    (batch, kvs, s) for s in args.split_kvs if s not in valid_splits
                )
                if not valid_splits and not args.auto:
                    continue

                q = make_packed_rows(batch * HEADS).view(batch, HEADS, DIM // 2)
                k = make_packed_rows(batch * kv_capacity).view(
                    batch, kv_capacity, DIM // 2
                )
                q_scales = torch.cat([make_blocked_scales(HEADS) for _ in range(batch)])
                k_scales = torch.cat(
                    [make_blocked_scales(kv_capacity) for _ in range(batch)]
                )
                block_table = None
                kv_seqlens = None
                lengths = (
                    random_kv_lengths(batch, kvs, length_generator)
                    if args.ragged
                    else [kvs] * batch
                )
                assert sum(lengths) == batch * kvs
                if args.ragged or args.paged_page_size:
                    kv_seqlens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
                dense_k, dense_k_scales = k, k_scales
                if args.paged_page_size:
                    k, k_scales, block_table = make_paged_k(
                        dense_k, dense_k_scales, args.paged_page_size
                    )
                head_weight = torch.randn(
                    (batch, HEADS), device="cuda", dtype=torch.float16
                )
                out = torch.empty(
                    (batch, kv_capacity), device="cuda", dtype=torch.float32
                )

                indices = sorted({0, batch // 2, batch - 1})
                expected = None
                if not args.no_accuracy:
                    expected = sampled_reference(
                        q, dense_k, q_scales, dense_k_scales, head_weight, indices
                    )
                    if kv_seqlens is not None:
                        for row, b in enumerate(indices):
                            expected[row, lengths[b] :] = float("-inf")

                cases = [(str(split), split) for split in valid_splits]
                if args.auto:
                    selected = module.select_split_kv(
                        q,
                        k,
                        q_scales,
                        k_scales,
                        head_weight,
                        out,
                        block_table=block_table,
                        kv_seqlens=kv_seqlens,
                        max_kv_len=kv_capacity,
                        options=self.make_tune_options(args),
                    )
                    cases.append((f"auto({selected})", selected))

                for split_label, split_kv in cases:

                    def kernel():
                        return module.indexer_score_decode_out(
                            q,
                            k,
                            q_scales,
                            k_scales,
                            head_weight,
                            out,
                            split_kv,
                            block_table=block_table,
                            kv_seqlens=kv_seqlens,
                            max_kv_len=kv_capacity,
                        )

                    max_diff = "skipped"
                    if expected is not None:
                        out.fill_(float("nan"))
                    kernel()
                    torch.cuda.synchronize()
                    if expected is not None:
                        actual = out[indices]
                        finite = torch.isfinite(expected)
                        max_diff = f"{(actual[finite] - expected[finite]).abs().max().item():.6g}"
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
                    flops = 2 * sum(lengths) * HEADS * DIM
                    table.print_row(
                        batch,
                        kvs,
                        kv_capacity,
                        f"{min(lengths)}-{max(lengths)}",
                        split_label,
                        sum(lengths),
                        f"{ms * 1000:.2f}",
                        f"{flops / (ms * 1e9):.2f}",
                        max_diff,
                    )
                    if sol_report is not None:
                        sol_report.add(
                            indexer_score_decode_sol_bounds(
                                lengths, kv_capacity, split_kv
                            ),
                            ms,
                            label=f"B={batch},KVs={kvs},split={split_label}",
                        )
        if skipped:
            print("Skipped split_kv > ceil(Kcap/256): " + ", ".join(map(str, skipped)))
        if sol_report is not None:
            sol_report.print()


if __name__ == "__main__":
    IndexerScoreDecodeBenchmark().run()
