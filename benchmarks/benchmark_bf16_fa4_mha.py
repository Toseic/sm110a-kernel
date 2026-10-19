#!/usr/bin/env python3
"""Accuracy and latency of paged BF16 FA4 MHA against attention baselines.

K/V are stored in shuffled pages (128 tokens by default). Prefill may pack variable
length Q sequences with --batched-q. All metadata is built outside timing.
"""

import argparse
import gc
import math
import random
import sys
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn.functional as F

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
from sm110a_kernel import (
    Bf16Fa4MhaConfig,
    Bf16Fa4MhaTactic,
)
from sm110a_kernel.kernels.bf16_fa4_mha import Bf16Fa4MhaRunner

DEFAULT_PAGE_SIZE = 128
PREFILL_LENGTHS = (1024, 2048, 4096, 8192, 16384)
DECODE_BATCH_SIZES = (32, 64, 128)
BATCHED_PREFILL_LENGTHS = PREFILL_LENGTHS
BATCHED_PREFILL_BATCH = 4
TRTLLM_GEN_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))


def load_official_fa4():
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
    except ImportError as exc:
        raise SystemExit(
            'install the FA4 baseline with: pip install "flash-attn-4[cu13]"'
        ) from exc
    warnings.filterwarnings(
        "ignore",
        message=r"Argument aux_data .*cannot be converted to a JitArgument.*",
        category=UserWarning,
    )
    try:
        label = version("flash-attn-4")
    except PackageNotFoundError:
        label = "unknown version"
    return _flash_attn_fwd, label


def load_trtllm_gen():
    capability = torch.cuda.get_device_capability()
    if capability not in TRTLLM_GEN_SUPPORTED_CAPABILITIES:
        raise SystemExit(
            f"TRTLLM-Gen FMHA does not support SM{capability[0]}{capability[1]}"
        )
    try:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek
    except ImportError as exc:
        raise SystemExit(
            'install the TRTLLM-Gen baseline with: pip install "flashinfer-python[cu13]>=0.6.18"'
        ) from exc
    try:
        label = version("flashinfer-python")
    except PackageNotFoundError:
        label = "unknown version"
    return trtllm_ragged_attention_deepseek, label


@dataclass(frozen=True)
class AttentionCase:
    name: str
    batch: int
    heads: int
    q_len: int
    kv_len: int
    head_dim: int
    causal: bool
    q_lengths: tuple[int, ...] | None = None
    kv_lengths: tuple[int, ...] | None = None
    packed_q: bool = False


def valid_scores(q_len, kv_len, causal):
    if not causal:
        return q_len * kv_len
    rows = min(q_len, kv_len)
    return rows * (rows + 1) // 2 + (q_len - rows) * kv_len


def attention_sol_bounds(case, q_lengths, kv_lengths, packed_q, page_size):
    """Model the actual ragged work and paged K/V reads of this kernel."""
    element_bytes = 2
    q_rows = sum(q_lengths) if packed_q else case.batch * case.q_len
    output_rows = sum(q_lengths)
    kv_rows = sum(math.ceil(length / page_size) * page_size for length in kv_lengths)
    q_bytes = q_rows * case.heads * case.head_dim * element_bytes
    kv_bytes = 2 * kv_rows * case.heads * case.head_dim * element_bytes
    output_bytes = output_rows * case.heads * case.head_dim * element_bytes
    tile_read_bytes = q_bytes
    for q_len, kv_len in zip(q_lengths, kv_lengths):
        for q_start in range(0, q_len, 256):
            visible_kv = min(q_start + 256, kv_len) if case.causal else kv_len
            kv_tiles = math.ceil(visible_kv / 128)
            tile_read_bytes += (
                2 * kv_tiles * 128 * case.heads * case.head_dim * element_bytes
            )
    flops = (
        4
        * case.heads
        * case.head_dim
        * sum(
            valid_scores(q_len, kv_len, case.causal)
            for q_len, kv_len in zip(q_lengths, kv_lengths)
        )
    )
    return sol_bounds(
        flops, q_bytes + kv_bytes + output_bytes, q_bytes + kv_bytes, tile_read_bytes
    )


def reference_attention(q, k, v, *, causal, scale):
    """Chunk Q so a 16K prefill does not materialize the full score matrix."""
    kf, vf = k.float(), v.float()
    pieces = []
    for start in range(0, q.size(1), 64):
        end = min(start + 64, q.size(1))
        scores = (
            torch.matmul(
                q[:, start:end].float().transpose(1, 2),
                kf.transpose(1, 2).transpose(-2, -1),
            )
            * scale
        )
        if causal:
            rows = torch.arange(start, end, device=q.device)
            cols = torch.arange(k.size(1), device=q.device)
            scores.masked_fill_(cols[None, :] > rows[:, None], float("-inf"))
        piece = torch.softmax(scores, dim=-1).matmul(vf.transpose(1, 2))
        pieces.append(piece.transpose(1, 2).to(torch.bfloat16))
    return torch.cat(pieces, dim=1)


class Bf16Fa4MhaBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--single", action="store_true")
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--heads", type=int, default=32)
        parser.add_argument("--q-len", type=int, default=1024)
        parser.add_argument("--kv-len", type=int, default=1024)
        parser.add_argument("--head-dim", type=int, default=128)
        parser.add_argument(
            "--page-size",
            type=int,
            default=DEFAULT_PAGE_SIZE,
            help="KV page size; must divide 128 (default: 128)",
        )
        parser.add_argument("--causal", action="store_true")
        parser.add_argument(
            "--batched-q",
            action="store_true",
            help="pack prefill Q as [sum_q, heads, dim]",
        )
        parser.add_argument(
            "--q-lengths",
            type=str,
            default=None,
            help="comma-separated per-batch Q lengths for --single --batched-q",
        )
        parser.add_argument(
            "--kv-lengths",
            type=str,
            default=None,
            help="comma-separated per-batch KV lengths for --single",
        )
        parser.add_argument("--scale", type=float, default=None)
        parser.add_argument("--baseline", choices=("fa4", "trtllm-gen"), default="fa4")
        parser.add_argument(
            "--tactic",
            choices=("auto", "16warp", "12warp"),
            default="auto",
            help="CTA schedule; auto benchmarks and caches compatible schedules",
        )
        parser.add_argument(
            "--no-accuracy",
            action="store_true",
            help="skip the additional FP32 check; baseline comparison remains enabled",
        )
        parser.add_argument("--atol", type=float, default=2e-2)
        parser.add_argument("--rtol", type=float, default=2e-2)
        parser.add_argument("--debug", action="store_true")
        self.add_timing_args(parser, warmup=5, iters=20, samples=7)
        self.add_cache_args(parser)
        self.add_sol_arg(parser)
        return parser

    def validate_args(self, args):
        if (
            min(args.batch, args.heads, args.q_len, args.kv_len, args.page_size) <= 0
            or args.head_dim != 128
            or args.page_size > 128
            or 128 % args.page_size
        ):
            raise SystemExit(
                "dimensions must be positive, head_dim must be 128, and page_size must divide 128"
            )
        if args.scale is not None and (
            not math.isfinite(args.scale) or args.scale <= 0
        ):
            raise SystemExit("--scale must be finite and positive")
        self.validate_timing_args(args, allow_zero_warmup=True)
        for name, maximum in (("q_lengths", args.q_len), ("kv_lengths", args.kv_len)):
            raw = getattr(args, name)
            if raw is None:
                continue
            if not args.single or (name == "q_lengths" and not args.batched_q):
                raise SystemExit(
                    f"--{name.replace('_', '-')} requires --single"
                    + (" --batched-q" if name == "q_lengths" else "")
                )
            try:
                lengths = tuple(int(value) for value in raw.split(","))
            except ValueError as exc:
                raise SystemExit(
                    f"--{name.replace('_', '-')} must contain integers"
                ) from exc
            if len(lengths) != args.batch or any(
                length < 1 or length > maximum for length in lengths
            ):
                raise SystemExit(
                    f"--{name.replace('_', '-')} needs {args.batch} values in [1, {maximum}]"
                )
            setattr(args, name, lengths)
        if args.batched_q and args.single and args.q_len == 1:
            raise SystemExit("--batched-q is intended for prefill (q_len > 1)")
        return args

    def cases(self, args):
        if args.single:
            yield AttentionCase(
                "single",
                args.batch,
                args.heads,
                args.q_len,
                args.kv_len,
                args.head_dim,
                args.causal,
            )
            return
        for length in PREFILL_LENGTHS:
            yield AttentionCase("prefill", 1, 32, length, length, 128, True)
        for length in BATCHED_PREFILL_LENGTHS:
            rng = random.Random(self.seed + length)
            lengths = [length] + [
                rng.randint(length // 2, length - 1)
                for _ in range(BATCHED_PREFILL_BATCH - 1)
            ]
            rng.shuffle(lengths)
            yield AttentionCase(
                "batched",
                BATCHED_PREFILL_BATCH,
                32,
                length,
                length,
                128,
                True,
                tuple(lengths),
                tuple(lengths),
                True,
            )
        if args.batched_q:
            return
        for batch in DECODE_BATCH_SIZES:
            yield AttentionCase("decode", batch, 32, 1, 8192, 128, False)

    def run_case(self, case, args):
        q_lengths = case.q_lengths or args.q_lengths or (case.q_len,) * case.batch
        kv_lengths = case.kv_lengths or args.kv_lengths or (case.kv_len,) * case.batch
        packed_q = case.packed_q or args.batched_q
        if case.causal and any(
            q_len != kv_len for q_len, kv_len in zip(q_lengths, kv_lengths)
        ):
            raise SystemExit(
                "causal baseline alignment differs when Q and KV lengths differ; "
                "use matching per-batch lengths or non-causal attention"
            )
        scale = args.scale if args.scale is not None else case.head_dim**-0.5
        shape_q = (case.batch, case.q_len, case.heads, case.head_dim)
        shape_kv = (case.batch, case.kv_len, case.heads, case.head_dim)
        q = torch.empty(shape_q, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
        k = torch.empty(shape_kv, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
        v = torch.empty_like(k).uniform_(-1, 1)

        # Logical [B, pages, H, 16, D] -> randomly numbered physical pages.
        page_size = args.page_size
        pages_per_batch = (case.kv_len + page_size - 1) // page_size
        padded = pages_per_batch * page_size - case.kv_len
        permutation = torch.randperm(case.batch * pages_per_batch, device="cuda")
        block_table = permutation.reshape(case.batch, pages_per_batch).to(torch.int32)

        def make_pages(tensor, *, fa4_layout=False):
            logical = F.pad(tensor, (0, 0, 0, 0, 0, padded)).view(
                case.batch, pages_per_batch, page_size, case.heads, case.head_dim
            )
            if fa4_layout:
                logical = logical.reshape(-1, page_size, case.heads, case.head_dim)
            else:
                logical = (
                    logical.permute(0, 1, 3, 2, 4)
                    .contiguous()
                    .view(-1, case.heads, page_size, case.head_dim)
                )
            physical = torch.empty_like(logical)
            physical.index_copy_(0, permutation, logical)
            return physical

        k_pages, v_pages = make_pages(k), make_pages(v)
        kv_seqlens = torch.tensor(kv_lengths, device="cuda", dtype=torch.int32)

        if packed_q:
            offsets = [0]
            for length in q_lengths:
                offsets.append(offsets[-1] + length)
            cu_seqlens_q = torch.tensor(offsets, device="cuda", dtype=torch.int32)
            q_input = torch.cat(
                [q[b, :length] for b, length in enumerate(q_lengths)], dim=0
            )
        else:
            cu_seqlens_q = None
            q_input = q.permute(0, 2, 1, 3).contiguous()
        out = torch.empty_like(q_input)

        kernel_runner = Bf16Fa4MhaRunner(
            q_input,
            k_pages,
            v_pages,
            out,
            causal=case.causal,
            scale=scale,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            kv_seqlens=kv_seqlens,
            max_q_len=case.q_len,
            max_kv_len=case.kv_len,
        )

        def make_kernel_run(tactic):
            return kernel_runner.prepare(Bf16Fa4MhaConfig(tactic=tactic)).run

        if args.baseline == "fa4":
            fa4_k_pages, fa4_v_pages = (
                make_pages(k, fa4_layout=True),
                make_pages(v, fa4_layout=True),
            )
            fa4_q = q_input if packed_q else q
            baseline_out = torch.empty_like(fa4_q)

            def run_baseline():
                self._baseline(
                    fa4_q,
                    fa4_k_pages,
                    fa4_v_pages,
                    page_table=block_table,
                    cu_seqlens_q=cu_seqlens_q,
                    seqused_k=kv_seqlens,
                    max_seqlen_q=case.q_len,
                    max_seqlen_k=case.kv_len,
                    softmax_scale=scale,
                    causal=case.causal,
                    out=baseline_out,
                )

            def baseline_batch(b, q_len):
                return (
                    baseline_out[offsets[b] : offsets[b + 1]]
                    if packed_q
                    else baseline_out[b, :q_len]
                )
        else:
            q_base = torch.cat(
                [q[b, :length] for b, length in enumerate(q_lengths)], dim=0
            )
            k_base = torch.cat(
                [k[b, :length] for b, length in enumerate(kv_lengths)], dim=0
            )
            v_base = torch.cat(
                [v[b, :length] for b, length in enumerate(kv_lengths)], dim=0
            )
            baseline_out = torch.empty_like(q_base)
            q_offsets = [0]
            kv_offsets = [0]
            for q_len, kv_len in zip(q_lengths, kv_lengths):
                q_offsets.append(q_offsets[-1] + q_len)
                kv_offsets.append(kv_offsets[-1] + kv_len)
            q_cu = torch.tensor(q_offsets, device="cuda", dtype=torch.int32)
            kv_cu = torch.tensor(kv_offsets, device="cuda", dtype=torch.int32)
            workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)

            def run_baseline():
                self._baseline(
                    q_base,
                    k_base,
                    v_base,
                    workspace_buffer=workspace,
                    seq_lens=kv_seqlens,
                    max_q_len=case.q_len,
                    max_kv_len=case.kv_len,
                    bmm1_scale=scale,
                    bmm2_scale=1.0,
                    o_sf_scale=1.0,
                    batch_size=case.batch,
                    window_left=-1,
                    cum_seq_lens_q=q_cu,
                    cum_seq_lens_kv=kv_cu,
                    enable_pdl=False,
                    is_causal=case.causal,
                    return_lse=False,
                    out=baseline_out,
                    backend="trtllm-gen",
                    skip_all_rows_active_check=True,
                )

            def baseline_batch(b, q_len):
                return baseline_out[q_offsets[b] : q_offsets[b + 1]]

        if args.tactic == "auto":
            # The 16-warp epilogue cannot directly store packed Q tails and
            # dispatches to 12-warp internally, so sweep only the real tactic.
            candidates = (
                (Bf16Fa4MhaConfig(tactic=Bf16Fa4MhaTactic.WARP_12),)
                if packed_q
                else (
                    Bf16Fa4MhaConfig(tactic=Bf16Fa4MhaTactic.WARP_12),
                    Bf16Fa4MhaConfig(tactic=Bf16Fa4MhaTactic.WARP_16),
                )
            )
            selection = kernel_runner.tune(
                self.make_tune_options(args, role="ours"),
                candidates=candidates,
                benchmark=lambda prepared: benchmark_with_stats(
                    prepared.run,
                    args.warmup,
                    args.iters,
                    args.samples,
                    cold_l2_flush=args._cold_l2_flush,
                ),
                probe=None,
            )
            run_kernel, selected_tactic = (
                selection.prepared.run,
                selection.candidate.tactic,
            )
        else:
            selected_tactic = (
                Bf16Fa4MhaTactic.WARP_12
                if args.tactic == "12warp"
                else Bf16Fa4MhaTactic.WARP_16
            )
            run_kernel = make_kernel_run(selected_tactic)

        run_kernel()
        run_baseline()
        torch.cuda.synchronize()
        max_diff = 0.0
        for b, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths)):
            actual = (
                out[int(cu_seqlens_q[b]) : int(cu_seqlens_q[b + 1])]
                if packed_q
                else out[b, :, :q_len].transpose(0, 1)
            )
            expected = baseline_batch(b, q_len)
            torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
            max_diff = max(
                max_diff, (actual.float() - expected.float()).abs().max().item()
            )
            if args.single and not args.no_accuracy:
                reference = reference_attention(
                    q[b : b + 1, :q_len],
                    k[b : b + 1, :kv_len],
                    v[b : b + 1, :kv_len],
                    causal=case.causal,
                    scale=scale,
                )[0]
                torch.testing.assert_close(
                    actual, reference, atol=args.atol, rtol=args.rtol
                )
                torch.testing.assert_close(
                    expected, reference, atol=args.atol, rtol=args.rtol
                )
        flops = (
            4
            * case.heads
            * case.head_dim
            * sum(
                valid_scores(q_len, kv_len, case.causal)
                for q_len, kv_len in zip(q_lengths, kv_lengths)
            )
        )
        kernel_stats = benchmark_with_stats(
            run_kernel,
            args.warmup,
            args.iters,
            args.samples,
            cold_l2_flush=args._cold_l2_flush,
        )
        baseline_stats = benchmark_with_stats(
            run_baseline,
            args.warmup,
            args.iters,
            args.samples,
            cold_l2_flush=args._cold_l2_flush,
        )
        bounds = (
            attention_sol_bounds(case, q_lengths, kv_lengths, packed_q, args.page_size)
            if args.sol
            else None
        )
        return (
            kernel_stats["median_ms"],
            baseline_stats["median_ms"],
            max_diff,
            flops,
            bounds,
            selected_tactic,
        )

    def run_with_args(self, args):
        if args.baseline == "fa4":
            self._baseline, label = load_official_fa4()
            baseline_name = "FA4"
        else:
            self._baseline, label = load_trtllm_gen()
            baseline_name = "TRTLLM-Gen"
        print(f"[baseline] {baseline_name} {label}")
        print(
            f"[layout] shuffled {args.page_size}-token KV pages"
            + ("; packed Q (12warp epilogue)" if args.batched_q else "")
        )
        columns = (
            ("mode", 9),
            ("shape [B,H,Q,KV,D]", 24),
            ("mask", 10),
            ("tactic", 10),
            ("Kernel ms", 13),
            ("Kernel TF/s", 14),
            (f"{baseline_name} ms", 16),
            (f"vs {baseline_name}", 16),
            ("Diff", 12),
        )
        table = TablePrinter(columns, debug=args.debug)
        table.print_header()
        sol_report = SolReport(debug=args.debug) if args.sol else None
        for case in self.cases(args):
            # Prepared callables can keep their runner alive until cyclic GC.
            # Reclaim the preceding shape before allocating another paged KV copy.
            gc.collect()
            torch.cuda.empty_cache()
            kernel_ms, baseline_ms, diff, flops, bounds, selected_tactic = (
                self.run_case(case, args)
            )
            table.print_row(
                case.name,
                f"{case.batch}x{case.heads}x{case.q_len}x{case.kv_len}x{case.head_dim}",
                "causal" if case.causal else "noncausal",
                "12warp" if selected_tactic == Bf16Fa4MhaTactic.WARP_12 else "16warp",
                f"{kernel_ms:.4f}",
                f"{flops / (kernel_ms * 1e9):.3f}",
                f"{baseline_ms:.4f}",
                f"{baseline_ms / kernel_ms:.3f}x",
                f"{diff:.6g}",
            )
            if sol_report:
                q_lengths = (
                    case.q_lengths or args.q_lengths or (case.q_len,) * case.batch
                )
                kv_lengths = (
                    case.kv_lengths or args.kv_lengths or (case.kv_len,) * case.batch
                )
                q_symbol = (
                    "≤" if any(length < case.q_len for length in q_lengths) else "="
                )
                kv_symbol = (
                    "≤" if any(length < case.kv_len for length in kv_lengths) else "="
                )
                sol_report.add(
                    bounds,
                    kernel_ms,
                    label=f"MHA {case.name} "
                    f"B={case.batch} H={case.heads} "
                    f"Q{q_symbol}{case.q_len} KV{kv_symbol}{case.kv_len}",
                )
        if sol_report:
            sol_report.print()


if __name__ == "__main__":
    Bf16Fa4MhaBenchmark().run()
