#!/usr/bin/env python3
"""Runner-backed config and token sweep for fused-permute NVFP4 FC1."""

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
    TablePrinter,
    benchmark_with_stats,
    print_sol_report,
    quick_probe,
    sol_bounds,
    tflops_from_ms,
)
from sm110a_kernel import (
    Nvfp4Fc1Config,
    compile_nvfp4_fc1,
    nvfp4_fc1_out,
    prepare_nvfp4_fc1_metadata,
    quantize_nvfp4_rows,
)
from sm110a_kernel.runner import AutotuneRunner, PreparedKernel, TuneOptions


DEFAULT_EXPERTS = 256
DEFAULT_N = 2048
DEFAULT_K = 6144
DEFAULT_TOKEN_VALUES = (64 * 32, 128 * 32, 256 * 32, 512 * 32, 1024 * 32)
DEFAULT_TOPK = 8
NVFP4_MMA_PEAK_TFLOPS = 4.0 * 250.0
_MODE_LABELS = {0: "np", 1: "pe", 2: "clc"}

NVFP4_FC1_CONFIGS = tuple(
    Nvfp4Fc1Config(
        tile_n=tile_n,
        tile_k=tile_k,
        epilogue_tile_n=epilogue_tile_n,
        mode=mode,
        direct_gmem=direct_gmem,
    )
    for tile_n in (64, 128)
    for tile_k in (64, 128, 256)
    for epilogue_tile_n in (32, 64, 128)
    for mode in (0, 1, 2)
    for direct_gmem in (False, True)
    if tile_n % epilogue_tile_n == 0
)


def format_candidate(config):
    epilogue = "dg" if config.direct_gmem else "tma"
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/e{config.epilogue_tile_n}/{_MODE_LABELS[config.mode]}/{epilogue}"
    )


def compatible_candidates(n, k):
    return tuple(
        config for config in NVFP4_FC1_CONFIGS
        if n % config.tile_n == 0 and n % 128 == 0 and k % config.tile_k == 0
    )


def selected_candidates(n, k, args):
    candidates = compatible_candidates(n, k)
    if args.tile_n is not None:
        candidates = tuple(c for c in candidates if c.tile_n == args.tile_n)
    if args.tile_k is not None:
        candidates = tuple(c for c in candidates if c.tile_k == args.tile_k)
    if args.epilogue_tile_n is not None:
        candidates = tuple(
            c for c in candidates if c.epilogue_tile_n == args.epilogue_tile_n
        )
    if args.mode is not None:
        candidates = tuple(c for c in candidates if c.mode == args.mode)
    if args.direct_gmem:
        candidates = tuple(c for c in candidates if c.direct_gmem)
    return candidates


def nvfp4_fc1_sol_bounds(router_scores, topk, n, k, config):
    """Model main-kernel traffic; routing and task construction are excluded."""
    experts = router_scores.size(1)
    selected = torch.argsort(
        torch.nan_to_num(router_scores.float(), nan=-float("inf")),
        dim=1,
        descending=True,
        stable=True,
    )[:, :topk]
    expert_rows = torch.bincount(selected.reshape(-1), minlength=experts)
    active_experts = int((expert_rows > 0).sum().item())
    m = selected.numel()
    m_tasks = sum(
        math.ceil(rows / config.tile_m) for rows in expert_rows.tolist() if rows
    )
    task_count = m_tasks * (n // config.tile_n)

    flops = 2 * m * n * k
    routed_a_bytes = m * (k / 2 + k / 16)
    b_bytes = active_experts * n * (k / 2 + k / 16)
    c_bytes = 2 * m * n
    staged_sfb_n = max(config.tile_n, 128)
    tile_read_bytes = task_count * (
        config.tile_m * (k / 2 + k / 16)
        + config.tile_n * k / 2
        + staged_sfb_n * k / 16
    )
    return sol_bounds(
        flops,
        routed_a_bytes + b_bytes + c_bytes,
        routed_a_bytes + b_bytes,
        tile_read_bytes,
        mma_peak_tflops=NVFP4_MMA_PEAK_TFLOPS,
    )


def build_reference(aq, bq, sfa, sfb, metadata, experts, n, k, out):
    """Build the routed result without materializing [experts,tokens,N]."""
    rows = metadata.source_rows.numel()
    grouped_layout = metadata.routing[2 * experts:2 * experts + rows]
    counts = metadata.routing[:experts].cpu().tolist()
    row_start = 0
    for expert, row_count in enumerate(counts):
        if not row_count:
            continue
        full_expert = torch._scaled_mm(
            aq,
            bq[expert].t(),
            sfa,
            sfb[expert * n * (k // 16):(expert + 1) * n * (k // 16)],
            out_dtype=torch.bfloat16,
        )
        row_end = row_start + row_count
        if not torch.all(grouped_layout[row_start:row_end] == expert):
            raise AssertionError("metadata rows are not grouped by expert")
        out[row_start:row_end].copy_(
            full_expert.index_select(
                0, metadata.source_rows[row_start:row_end].long()
            )
        )
        row_start = row_end


class Nvfp4Fc1Benchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
        parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
        parser.add_argument("--n", type=int, default=DEFAULT_N)
        parser.add_argument("--k", type=int, default=DEFAULT_K)
        parser.add_argument(
            "--token", "--tokens", dest="token", type=int, default=0,
            help="run one token count; default runs the standard token sweep",
        )
        self.add_timing_args(parser, warmup=2, iters=5, samples=3)
        self.add_cache_args(parser)
        self.add_pre_parallel_jit_arg(parser)
        self.add_sol_arg(parser)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--seed", type=int, default=self.seed)
        parser.add_argument("--tile-n", type=int, choices=(64, 128))
        parser.add_argument("--tile-k", type=int, choices=(64, 128, 256))
        parser.add_argument("--epilogue-tile-n", type=int, choices=(32, 64, 128))
        parser.add_argument("--mode", type=int, choices=(0, 1, 2))
        parser.add_argument(
            "--direct-gmem",
            action="store_true",
            help="only tune direct-GMEM epilogue candidates",
        )
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if args.experts <= 0 or args.topk <= 0 or args.topk > args.experts:
            raise SystemExit("experts must be positive and topk must be in [1, experts]")
        if args.n <= 0 or args.n % 128 or args.k <= 0 or args.k % 64:
            raise SystemExit("N must be a positive multiple of 128 and K of 64")
        if args.token < 0:
            raise SystemExit("token must be non-negative")
        token_values = (args.token,) if args.token else DEFAULT_TOKEN_VALUES
        if any(tokens <= 0 or tokens % 128 for tokens in token_values):
            raise SystemExit("every token count must be a positive multiple of 128")
        if not selected_candidates(args.n, args.k, args):
            raise SystemExit("no compatible NVFP4 FC1 configs match the filters")
        args.token_values = token_values
        return args

    def resolve_kernel(self, aq, bq, sfa, sfb, router, out, args):
        candidates = selected_candidates(args.n, args.k, args)

        def prepare(config):
            metadata = prepare_nvfp4_fc1_metadata(
                router, args.topk, args.n, config=config
            )
            run = lambda: nvfp4_fc1_out(
                aq, bq, sfa, sfb, metadata, out, config=config
            )
            run()
            torch.cuda.synchronize()
            return PreparedKernel(config, run), metadata

        prepared_metadata = {}

        def prepare_for_runner(config):
            prepared, metadata = prepare(config)
            prepared_metadata[config] = metadata
            return prepared

        options = TuneOptions(
            warmup=args.warmup,
            iters=args.iters,
            samples=args.samples,
            cold_l2_flush=args._cold_l2_flush,
            cache_path=args.cache_path,
            force_sweep=args.force_sweep,
            debug=args.debug,
            pre_parallel_jit=args.pre_parallel_jit,
        )
        routed_m = aq.size(0) * args.topk
        selection = AutotuneRunner(options).resolve(
            candidates,
            namespace="nvfp4_fc1",
            key=(aq.size(0), args.experts, args.topk, args.n, args.k),
            prepare=prepare_for_runner,
            benchmark=lambda prepared: benchmark_with_stats(
                prepared.run,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=args._cold_l2_flush,
            ),
            probe=lambda prepared: quick_probe(
                prepared.run, args.iters, cold_l2_flush=args._cold_l2_flush
            ),
            format_candidate=format_candidate,
            metric_formatter=lambda _config, stats: (
                f"{tflops_from_ms(routed_m, args.n, args.k, stats['median_ms']):.1f} TF/s"
            ),
            pre_jit=compile_nvfp4_fc1,
        )
        return selection, prepared_metadata[selection.candidate]

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        b = torch.randn(
            (args.experts, args.n, args.k), device="cuda", dtype=torch.bfloat16
        )
        bq_flat, sfb = quantize_nvfp4_rows(
            b.reshape(args.experts * args.n, args.k)
        )
        del b
        bq = bq_flat.reshape(args.experts, args.n, args.k // 2)

        print(
            f"[config] experts={args.experts} topk={args.topk} "
            f"n={args.n} k={args.k} tokens={args.token_values}"
        )
        print("[runner] config selection cached by tokens/experts/topk/N/K")
        table = TablePrinter((
            ("Tokens", 9), ("M", 10), ("TopK", 7), ("E", 6),
            ("N", 8), ("K", 8), ("Kernel(ms)", 14), ("TF/s", 12),
            ("MaxDiff", 12), ("Cache", 8), ("BestConfig", 34),
        ))
        table.print_header()

        for tokens in args.token_values:
            a = torch.randn((tokens, args.k), device="cuda", dtype=torch.bfloat16)
            router = torch.randn(
                (tokens, args.experts), device="cuda", dtype=torch.bfloat16
            )
            aq, sfa = quantize_nvfp4_rows(a)
            del a
            out = torch.empty(
                (tokens * args.topk, args.n), device="cuda", dtype=torch.bfloat16
            )
            selection, metadata = self.resolve_kernel(
                aq, bq, sfa, sfb, router, out, args
            )

            max_diff = 0.0
            if not args.no_accuracy:
                reference = torch.empty_like(out)
                build_reference(
                    aq, bq, sfa, sfb, metadata,
                    args.experts, args.n, args.k, reference,
                )
                selection.prepared.run()
                torch.cuda.synchronize()
                max_diff = (out.float() - reference.float()).abs().max().item()
                if not torch.equal(out, reference):
                    raise AssertionError(f"NVFP4 FC1 mismatch: max diff {max_diff}")

            elapsed = selection.stats["median_ms"]
            routed_m = tokens * args.topk
            table.print_row(
                tokens, routed_m, args.topk, args.experts, args.n, args.k,
                f"{elapsed:.4f}",
                f"{tflops_from_ms(routed_m, args.n, args.k, elapsed):.2f}",
                f"{max_diff:.6f}",
                "hit" if selection.cache_hit else "miss",
                format_candidate(selection.candidate),
            )
            if args.sol:
                print_sol_report(
                    nvfp4_fc1_sol_bounds(
                        router, args.topk, args.n, args.k, selection.candidate
                    ),
                    elapsed,
                    label="NVFP4 FC1 main kernel (routing excluded)",
                )


if __name__ == "__main__":
    Nvfp4Fc1Benchmark().run()
