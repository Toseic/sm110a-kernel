#!/usr/bin/env python3
"""Cold-L2 batched/paged prefill benchmark for the MXFP4 indexer score kernel."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepgemm_baseline import (
    mqa_fused_pages,
    mqa_logits_dense,
    mqa_logits_paged,
    mqa_scale_words,
    mqa_weights,
)

from common import (
    MMA_PEAK_TFLOPS,
    BenchmarkBase,
    SolReport,
    TablePrinter,
    benchmark_with_stats,
    sol_bounds,
)
from sm110a_kernel import (
    IndexerScorePrefillRunner,
    from_blocked_mxfp4_scales,
    pack_indexer_score_prefill_scales,
    to_blocked_mxfp4_scales,
)

HEADS = 64
DIM = 128
TILE_KV = 384
SEQ_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768)
EXTRA_BATCH_SEQ = 8192
EXTRA_BATCHES = (2, 4, 8, 16)
FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MXFP4_MMA_PEAK_TFLOPS = 4 * MMA_PEAK_TFLOPS


def indexer_score_work(qs, kvs, cluster_m=1, *, causal=True):
    q_tiles = (qs + 1) // 2
    kv_tiles = (kvs + TILE_KV - 1) // TILE_KV
    kv_load_groups = (q_tiles + cluster_m - 1) // cluster_m
    active_tiles = [
        min(kv_tiles, ((group + 1) * cluster_m * 2 + TILE_KV - 1) // TILE_KV)
        if causal
        else kv_tiles
        for group in range(kv_load_groups)
    ]
    flops = 2 * sum(active_tiles) * cluster_m * 2 * TILE_KV * HEADS * DIM
    return q_tiles, kv_tiles, active_tiles, flops


def indexer_score_sol_bounds(qs, kvs, cluster_m=1, *, causal=True):
    """MMA and memory rooflines for the persistent Q-reuse schedule.

    Each query tile loads Q, Q scales, and head weights once, then traverses
    its visible 128-row KV tiles. Compulsory HBM includes BF16 output stores.
    It does not model epilogue instruction throughput or launch overhead.
    """
    q_tiles, _, active_tiles, flops = indexer_score_work(
        qs, kvs, cluster_m, causal=causal
    )
    q_bytes = qs * HEADS * DIM // 2
    k_bytes = kvs * DIM // 2
    q_scale_bytes = q_tiles * 512
    k_scale_bytes = ((kvs + 127) // 128) * 512
    weight_bytes = qs * HEADS * 2
    output_bytes = (sum(min(kvs, q + 1) for q in range(qs)) if causal else qs * kvs) * 2
    input_bytes = q_bytes + k_bytes + q_scale_bytes + k_scale_bytes + weight_bytes
    compulsory_bytes = input_bytes + output_bytes
    q_tile_read_bytes = 128 * DIM // 2 + 512 + 2 * HEADS * 2
    kv_tile_read_bytes = TILE_KV * DIM // 2 + (TILE_KV // 128) * 512
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


def prepare_deepgemm_mqa(
    q,
    k,
    q_scales,
    k_scales,
    head_weight,
    q_lengths,
    kv_lengths,
    *,
    block_table=None,
    page_size=None,
    causal=True,
):
    """Adapt the same MXFP4 operands to DeepGEMM before timing."""
    batch = len(q_lengths)
    q_scale_stride = q_scales.numel() // batch
    q_words = torch.cat(
        [
            mqa_scale_words(
                q_scales[b * q_scale_stride : (b + 1) * q_scale_stride],
                q_scale_stride // 1024 * 128,
                q_lengths[b] * HEADS,
                from_blocked_scales=from_blocked_mxfp4_scales,
            ).view(q_lengths[b], HEADS)
            for b in range(batch)
        ]
    )
    q_flat = q.reshape(-1, HEADS, DIM // 2)
    if q.ndim == 4:
        q_flat = torch.cat([q[b, : q_lengths[b]] for b in range(batch)])
        weights = torch.cat([head_weight[b, : q_lengths[b]] for b in range(batch)])
    else:
        weights = head_weight
    weights = mqa_weights(weights)
    if block_table is None:
        kv_capacity = k.size(1)
        k_scale_stride = k_scales.numel() // batch
        k_words = torch.cat(
            [
                mqa_scale_words(
                    k_scales[b * k_scale_stride : (b + 1) * k_scale_stride],
                    k_scale_stride // 1024 * 128,
                    kv_capacity,
                    from_blocked_scales=from_blocked_mxfp4_scales,
                )
                for b in range(batch)
            ]
        )
        starts = torch.cat(
            [
                torch.full((length,), b * kv_capacity, device="cuda", dtype=torch.int32)
                for b, length in enumerate(q_lengths)
            ]
        )
        ends = torch.cat(
            [
                starts[sum(q_lengths[:b]) : sum(q_lengths[: b + 1])]
                + (torch.arange(length, device="cuda", dtype=torch.int32) + 1).clamp(
                    max=kv_lengths[b]
                )
                if causal
                else starts[sum(q_lengths[:b]) : sum(q_lengths[: b + 1])]
                + kv_lengths[b]
                for b, length in enumerate(q_lengths)
            ]
        )
        return mqa_logits_dense(
            q_flat.contiguous(),
            q_words.contiguous(),
            k.reshape(-1, DIM // 2).contiguous(),
            k_words,
            weights,
            starts,
            ends,
            max(kv_lengths),
        )
    if page_size not in (128,):
        raise ValueError("DeepGEMM paged MQA supports --page-size 128 only")
    k_words = mqa_scale_words(
        k_scales,
        k.size(0) * page_size,
        k.size(0) * page_size,
        from_blocked_scales=from_blocked_mxfp4_scales,
    ).view(k.size(0), page_size)
    fused_k = mqa_fused_pages(k, k_words)
    context = torch.cat(
        [
            (torch.arange(length, device="cuda", dtype=torch.int32) + 1).clamp(
                max=kv_lengths[b]
            )
            if causal
            else torch.full((length,), kv_lengths[b], device="cuda", dtype=torch.int32)
            for b, length in enumerate(q_lengths)
        ]
    ).view(-1, 1)
    indices = torch.repeat_interleave(
        torch.arange(batch, device="cuda", dtype=torch.int32),
        torch.tensor(q_lengths, device="cuda", dtype=torch.int32),
    )
    expanded_table = block_table.repeat_interleave(
        torch.tensor(q_lengths, device="cuda"), dim=0
    ).contiguous()
    return mqa_logits_paged(
        q_flat.contiguous().unsqueeze(1),
        q_words.contiguous().unsqueeze(1),
        fused_k,
        weights,
        context,
        expanded_table,
        indices,
        page_size * block_table.size(1),
    )


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


def bf16_head_reduce(dot, weights):
    """Mirror the kernel's BF16 FMA order and two final BF16 additions."""
    values = dot.view(-1, HEADS, dot.size(-1)).clamp_min(0).bfloat16()
    weights = weights.bfloat16()
    partial = [torch.zeros_like(values[:, 0]) for _ in range(4)]
    for head in range(HEADS):
        slot = head % 4
        partial[slot] = torch.addcmul(
            partial[slot].float(),
            values[:, head].float(),
            weights[:, head, None].float(),
        ).bfloat16()
    even = (partial[0].float() + partial[2].float()).bfloat16()
    odd = (partial[1].float() + partial[3].float()).bfloat16()
    return (even.float() + odd.float()).bfloat16().float()


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
    weights = head_weight.index_select(0, query_indices).float()
    scores = bf16_head_reduce(dot, weights)
    if causal:
        keys = torch.arange(kvs, device=scores.device)
        scores.masked_fill_(keys[None, :] > query_indices[:, None], -torch.inf)
    return scores


def reference_batched_rows(
    q,
    k,
    q_scales,
    k_scales,
    head_weight,
    batch,
    query_indices,
    q_lengths,
    kv_lengths,
    output_kv,
    *,
    block_table=None,
    cu_seqlens_q=None,
    causal=True,
):
    """Sample logical batch rows, including page lookup and packed-Q offsets."""
    kv_len = kv_lengths[batch]
    q_scale_tile_bytes = q_scales.numel() // len(q_lengths)
    batch_q_scales = q_scales[
        batch * q_scale_tile_bytes : (batch + 1) * q_scale_tile_bytes
    ]
    q_scale_bytes = from_blocked_mxfp4_scales(
        batch_q_scales, q_scale_tile_bytes // 1024 * 128, DIM
    ).view(torch.uint8)
    if cu_seqlens_q is not None:
        q_offset = sum(q_lengths[:batch])
        q_flat = q.reshape(-1, DIM // 2).view(torch.uint8)
    else:
        q_offset = 0
        q_flat = q[batch].reshape(-1, DIM // 2).view(torch.uint8)
    q_code_row_ids = (
        (q_offset + query_indices)[:, None] * HEADS
        + torch.arange(HEADS, device="cuda")[None, :]
    ).reshape(-1)
    q_scale_row_ids = (
        query_indices[:, None] * HEADS + torch.arange(HEADS, device="cuda")[None, :]
    ).reshape(-1)
    q_dense = decode_rows(
        q_flat.index_select(0, q_code_row_ids),
        q_scale_bytes.index_select(0, q_scale_row_ids),
    )
    if block_table is not None:
        page_size = k.size(1)
        page_ids = block_table[batch, : (kv_len + page_size - 1) // page_size]
        physical_rows = (
            (
                page_ids[:, None] * page_size
                + torch.arange(page_size, device="cuda")[None, :]
            )
            .reshape(-1)[:kv_len]
            .long()
        )
        k_flat = k.reshape(-1, DIM // 2).view(torch.uint8)
        k_scale_bytes = from_blocked_mxfp4_scales(
            k_scales, k.size(0) * page_size, DIM
        ).view(torch.uint8)
        k_codes = k_flat.index_select(0, physical_rows)
        k_scale = k_scale_bytes.index_select(0, physical_rows)
    else:
        k_codes = k[batch, :kv_len].view(torch.uint8)
        scale_bytes_per_batch = ((k.size(1) + 127) // 128) * 1024
        k_scale_bytes = from_blocked_mxfp4_scales(
            k_scales[
                batch * scale_bytes_per_batch : (batch + 1) * scale_bytes_per_batch
            ],
            ((k.size(1) + 127) // 128) * 128,
            DIM,
        ).view(torch.uint8)
        k_scale = k_scale_bytes[:kv_len]
    k_dense = decode_rows(k_codes, k_scale)
    dot = q_dense @ k_dense.T
    weights = (
        head_weight.index_select(0, q_offset + query_indices)
        if cu_seqlens_q is not None
        else head_weight[batch].index_select(0, query_indices)
    )
    scores = bf16_head_reduce(dot, weights.float())
    if causal:
        scores.masked_fill_(
            torch.arange(kv_len, device="cuda")[None, :] > query_indices[:, None],
            -torch.inf,
        )
    expected = torch.full((query_indices.numel(), output_kv), -torch.inf, device="cuda")
    expected[:, :kv_len] = scores
    return expected


class IndexerScoreBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--seq-lens",
            type=int,
            nargs="+",
            help="maximum Qs and KVs per batch; default sweep keeps the original lengths",
        )
        batch_group = parser.add_mutually_exclusive_group()
        batch_group.add_argument(
            "--batch", type=int, help="one batch size for every selected length"
        )
        batch_group.add_argument(
            "--batches",
            type=int,
            nargs="+",
            help="batch sizes to cross with --seq-lens",
        )
        parser.add_argument(
            "--paged-kv", action="store_true", help="use paged K and block_table"
        )
        parser.add_argument("--page-size", type=int, choices=(128, 256), default=128)
        parser.add_argument(
            "--packed-q",
            action="store_true",
            help="pack variable Q with cu_seqlens_q (requires --paged-kv)",
        )
        parser.add_argument(
            "--q-lengths",
            type=int,
            nargs="+",
            help="per-batch Q lengths; requires --packed-q and one --seq-lens value",
        )
        parser.add_argument(
            "--kv-lengths",
            type=int,
            nargs="+",
            help="per-batch KV lengths; requires one --seq-lens value",
        )
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument("--baseline", choices=("none", "deepgemm"), default="none")
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument(
            "--non-causal", action="store_true", help="disable the default causal mask"
        )
        parser.add_argument("--atol", type=float, default=1e-1)
        parser.add_argument("--rtol", type=float, default=1e-2)
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
        default_sweep = (
            args.seq_lens is None and args.batch is None and args.batches is None
        )
        if args.seq_lens is None:
            args.seq_lens = SEQ_LENGTHS
        if not args.seq_lens or any(seq <= 0 for seq in args.seq_lens):
            raise SystemExit("--seq-lens must contain positive lengths")
        batches = args.batches or [args.batch or 1]
        if any(batch <= 0 or batch > 65535 for batch in batches):
            raise SystemExit("batch sizes must be in [1, 65535]")
        if (args.q_lengths or args.kv_lengths) and args.batches:
            raise SystemExit("explicit per-batch lengths require one --batch value")
        if (args.q_lengths or args.kv_lengths) and args.batch is None:
            raise SystemExit("explicit per-batch lengths require --batch")
        if args.packed_q and not args.paged_kv:
            raise SystemExit("--packed-q requires --paged-kv")
        if args.q_lengths and not args.packed_q:
            raise SystemExit("--q-lengths requires --packed-q")
        if (args.q_lengths or args.kv_lengths) and len(args.seq_lens) != 1:
            raise SystemExit("explicit batch lengths require one --seq-lens value")
        for name in ("q_lengths", "kv_lengths"):
            lengths = getattr(args, name)
            if lengths is not None and (
                len(lengths) != args.batch
                or any(n <= 0 or n > args.seq_lens[0] for n in lengths)
            ):
                raise SystemExit(
                    f"--{name.replace('_', '-')} needs {args.batch} lengths in [1, seq]"
                )
        if args.atol < 0 or args.rtol < 0:
            raise SystemExit("--atol and --rtol must be non-negative")
        if args.cluster_m and args.sweep_cluster_m:
            raise SystemExit("--cluster-m and --sweep-cluster-m are mutually exclusive")
        if args.baseline == "deepgemm" and args.paged_kv and args.page_size != 128:
            raise SystemExit("DeepGEMM paged MQA supports --page-size 128 only")
        args.cases = (
            [(seq, 1) for seq in SEQ_LENGTHS]
            + [(EXTRA_BATCH_SEQ, batch) for batch in EXTRA_BATCHES]
            if default_sweep and not (args.q_lengths or args.kv_lengths)
            else [(seq, batch) for seq in args.seq_lens for batch in batches]
        )
        self.validate_timing_args(args, allow_zero_warmup=True)
        return args

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        columns = (
            ("Seq", 8),
            ("Batch", 8),
            ("KV", 8),
            ("Q", 8),
            ("Cluster", 9),
            ("Candidates", 14),
            ("Kernel(ms)", 14),
            ("TF/s", 11),
        )
        if args.baseline == "deepgemm":
            columns += (("Baseline(ms)", 14), ("Speedup", 10))
        columns += (("MaxDiff", 13),)
        if args.baseline == "deepgemm":
            columns += (("BaselineDiff", 14),)
        table = TablePrinter(columns)
        table.print_header()
        sol_report = SolReport() if args.sol else None
        for seq, batch_size in args.cases:
            q_lengths = args.q_lengths or [seq] * batch_size
            kv_lengths = args.kv_lengths or [seq] * batch_size
            cu_seqlens_q = None
            if args.packed_q:
                total_q = sum(q_lengths)
                q = make_packed_rows(total_q * HEADS).view(total_q, HEADS, DIM // 2)
                q_scales = torch.cat(
                    [make_blocked_scales(seq * HEADS) for _ in range(batch_size)]
                )
                head_weight = torch.randn(
                    (total_q, HEADS), device="cuda", dtype=torch.bfloat16
                )
                cu_seqlens_q = torch.tensor(
                    [0] + list(torch.tensor(q_lengths).cumsum(0).tolist()),
                    device="cuda",
                    dtype=torch.int32,
                )
                out = torch.empty((total_q, seq), device="cuda", dtype=torch.bfloat16)
            else:
                q = torch.stack(
                    [
                        make_packed_rows(seq * HEADS).view(seq, HEADS, DIM // 2)
                        for _ in range(batch_size)
                    ]
                )
                q_scales = torch.cat(
                    [make_blocked_scales(seq * HEADS) for _ in range(batch_size)]
                )
                head_weight = torch.randn(
                    (batch_size, seq, HEADS), device="cuda", dtype=torch.bfloat16
                )
                out = torch.empty(
                    (batch_size, seq, seq), device="cuda", dtype=torch.bfloat16
                )
            block_table = None
            if args.paged_kv:
                pages_per_batch = (seq + args.page_size - 1) // args.page_size
                num_pages = batch_size * pages_per_batch
                k = make_packed_rows(num_pages * args.page_size).view(
                    num_pages, args.page_size, DIM // 2
                )
                k_scales = make_blocked_scales(num_pages * args.page_size)
                # A permutation ensures the benchmark exercises page lookup,
                # rather than an accidentally contiguous physical order.
                block_table = torch.randperm(
                    num_pages, device="cuda", dtype=torch.int32
                ).view(batch_size, pages_per_batch)
            else:
                k = torch.stack([make_packed_rows(seq) for _ in range(batch_size)])
                k_scales = torch.cat(
                    [make_blocked_scales(seq) for _ in range(batch_size)]
                )
            kv_seqlens = torch.tensor(kv_lengths, device="cuda", dtype=torch.int32)
            extra = dict(
                block_table=block_table,
                cu_seqlens_q=cu_seqlens_q,
                kv_seqlens=kv_seqlens,
                max_q_len=seq if args.packed_q else None,
                max_kv_len=seq if args.paged_kv else None,
            )
            # Match DeepGEMM: adapt scale layout once, before tuning and timing.
            compact_q_scales = pack_indexer_score_prefill_scales(q_scales)
            compact_k_scales = pack_indexer_score_prefill_scales(k_scales)
            runner = IndexerScorePrefillRunner(
                q,
                k,
                compact_q_scales,
                compact_k_scales,
                head_weight,
                out,
                causal=not args.non_causal,
                **extra,
            )
            if args.cluster_m:
                from sm110a_kernel.kernels.mxfp4_indexer_score import (
                    IndexerScorePrefillConfig,
                )

                prepared = runner.prepare(IndexerScorePrefillConfig(args.cluster_m))
            else:
                options = self.make_tune_options(
                    args,
                    force_sweep=args.sweep_cluster_m,
                    debug=args.sweep_cluster_m,
                )
                prepared = runner.tune(options).prepared
            cluster_m = prepared.candidate.cluster_m
            kernel = prepared.run

            baseline = None
            baseline_out = None
            if args.baseline == "deepgemm":
                baseline = prepare_deepgemm_mqa(
                    q,
                    k,
                    q_scales,
                    k_scales,
                    head_weight,
                    q_lengths,
                    kv_lengths,
                    block_table=block_table,
                    page_size=args.page_size if args.paged_kv else None,
                    causal=not args.non_causal,
                )
                baseline_out = baseline()
                torch.cuda.synchronize()

            # A prefilled output makes masked writes observable during the
            # accuracy check. Filling is outside the timed kernel call.
            masked_sentinel = 7.25
            if not args.no_accuracy:
                out.fill_(masked_sentinel)
            kernel()
            torch.cuda.synchronize()
            max_diff = "skipped"
            baseline_diff = "skipped"
            if not args.no_accuracy:
                largest_diff = 0.0
                largest_baseline_diff = 0.0
                q_offset = 0
                for batch in range(batch_size):
                    local_q = q_lengths[batch]
                    indices = torch.tensor(
                        sorted(
                            {
                                i
                                for i in (
                                    0,
                                    1,
                                    254,
                                    255,
                                    256,
                                    257,
                                    local_q // 2,
                                    local_q - 1,
                                )
                                if i < local_q
                            }
                        ),
                        device="cuda",
                        dtype=torch.long,
                    )
                    expected = reference_batched_rows(
                        q,
                        k,
                        q_scales,
                        k_scales,
                        head_weight,
                        batch,
                        indices,
                        q_lengths,
                        kv_lengths,
                        seq,
                        block_table=block_table,
                        cu_seqlens_q=cu_seqlens_q,
                        causal=not args.non_causal,
                    )
                    actual = (
                        out.index_select(0, q_offset + indices)
                        if args.packed_q
                        else out[batch].index_select(0, indices)
                    )
                    actual = actual.float()
                    valid = torch.isfinite(expected)
                    if valid.any():
                        largest_diff = max(
                            largest_diff,
                            (actual[valid] - expected[valid]).abs().max().item(),
                        )
                        if baseline_out is not None:
                            baseline_actual = baseline_out.index_select(
                                0, q_offset + indices
                            )[:, :seq].float()
                            baseline_valid = valid[:, : baseline_actual.size(1)]
                            baseline_expected = expected[:, : baseline_actual.size(1)]
                            largest_baseline_diff = max(
                                largest_baseline_diff,
                                (
                                    baseline_actual[baseline_valid]
                                    - baseline_expected[baseline_valid]
                                )
                                .abs()
                                .max()
                                .item(),
                            )
                            torch.testing.assert_close(
                                baseline_actual[baseline_valid],
                                baseline_expected[baseline_valid],
                                atol=args.atol,
                                rtol=args.rtol,
                            )
                    torch.testing.assert_close(
                        actual[~valid],
                        torch.full_like(actual[~valid], masked_sentinel),
                        atol=0,
                        rtol=0,
                    )
                    expected.masked_fill_(~valid, masked_sentinel)
                    torch.testing.assert_close(
                        actual, expected, atol=args.atol, rtol=args.rtol
                    )
                    q_offset += local_q
                max_diff = f"{largest_diff:.6g}"
                baseline_diff = f"{largest_baseline_diff:.6g}"

            stats = benchmark_with_stats(
                kernel,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=args._cold_l2_flush,
            )
            ms = stats["median_ms"]
            baseline_ms = (
                benchmark_with_stats(
                    baseline,
                    args.warmup,
                    args.iters,
                    args.samples,
                    cold_l2_flush=args._cold_l2_flush,
                )["median_ms"]
                if baseline is not None
                else None
            )
            per_batch_bounds = [
                indexer_score_sol_bounds(qs, kvs, cluster_m, causal=not args.non_causal)
                for qs, kvs in zip(q_lengths, kv_lengths)
            ]
            bounds = {
                key: sum(item[key] for item in per_batch_bounds)
                for key in per_batch_bounds[0]
            }
            bounds["sol_ms"] = max(
                bounds["mma_ms"], bounds["hbm_ms"], bounds["pure_l2_ms"]
            )
            flops = sum(
                indexer_score_work(qs, kvs, cluster_m, causal=not args.non_causal)[3]
                for qs, kvs in zip(q_lengths, kv_lengths)
            )
            row = (
                seq,
                batch_size,
                "paged" if args.paged_kv else "dense",
                "packed" if args.packed_q else "fixed",
                cluster_m,
                sum(
                    min(kv_len, q_idx + 1) if not args.non_causal else kv_len
                    for q_len, kv_len in zip(q_lengths, kv_lengths)
                    for q_idx in range(q_len)
                ),
                f"{ms:.4f}",
                f"{flops / (ms * 1e9):.2f}",
            )
            if baseline_ms is not None:
                row += (f"{baseline_ms:.4f}", f"{baseline_ms / ms:.2f}x")
            row += (max_diff,)
            if baseline_ms is not None:
                row += (baseline_diff,)
            table.print_row(*row)
            if sol_report is not None:
                sol_report.add(
                    bounds,
                    ms,
                    label=f"seq={seq},B={batch_size},{'paged' if args.paged_kv else 'dense'}",
                )
        if sol_report is not None:
            sol_report.print()


if __name__ == "__main__":
    IndexerScoreBenchmark().run()
