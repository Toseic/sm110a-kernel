#!/usr/bin/env python3
"""Autotune and benchmark fused routed FC2 plus token finalize."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    BenchmarkBase,
    benchmark_with_stats,
    print_sol_report,
    quick_probe,
    sol_bounds,
)
from sm110a_kernel import (
    Fc2FinalizeConfig,
    GroupedMConfig,
    fc2_finalize_with_routing_out,
    pack_routing_metadata,
    prepare_grouped_m_metadata,
)
from sm110a_kernel.runner import AutotuneRunner, PreparedKernel, TuneOptions


DEFAULT_EXPERTS = 256
DEFAULT_H = 6144
DEFAULT_I = 2048
DEFAULT_TOKEN_VALUES = (64 * 32, 128 * 32, 256 * 32, 512 * 32, 1024 * 32)
DEFAULT_TOPK = 8
FC2_CONFIGS = tuple(
    Fc2FinalizeConfig(
        tile_n=tile_n,
        tile_k=tile_k,
        cluster_m=cluster_m,
    )
    for tile_n in (64, 128, 256)
    for tile_k in (64, 128)
    for cluster_m in (1, 2)
)


def fc2_tflops(tokens, topk, h, i, elapsed_ms):
    return 2.0 * tokens * topk * h * i / (elapsed_ms * 1e-3) / 1e12


def fc2_sol_bounds(router_scores, topk, h, i, config):
    """SOL model for the selected FC2 grouped-GEMM configuration.

    Every expert/M-cluster/N-tile task TMA-loads a full cluster-M A tile and
    an N tile of that expert's B matrix.  Count those tile loads rather than
    treating FC1 and the active expert weights as single-use tensors.
    """
    tokens, experts = router_scores.shape
    selected = torch.topk(router_scores.float(), topk, dim=1).indices
    expert_rows = torch.bincount(selected.reshape(-1), minlength=experts)
    active_experts = int((expert_rows > 0).sum().item())
    m = tokens * topk
    flops = 2 * m * h * i
    a_bytes = 2 * m * i
    b_bytes = 2 * active_experts * h * i
    out_bytes = 2 * tokens * h
    m_tiles = torch.div(
        expert_rows + config.cluster_m * config.tile_m - 1,
        config.cluster_m * config.tile_m,
        rounding_mode="floor",
    ).sum().item()
    task_count = m_tiles * (h // config.tile_n)
    tile_read_bytes = 2 * task_count * (
        config.cluster_m * config.tile_m + config.tile_n
    ) * i
    # Fused FC2 reads its BF16 FC1 rows and directly reduces into the finalized output.
    return sol_bounds(flops, a_bytes + b_bytes + out_bytes,
                      a_bytes + b_bytes, tile_read_bytes)


def format_candidate(config):
    return f"128x{config.tile_n}x{config.tile_k}/c{config.cluster_m}/persistent/fusion"


def compatible_candidates(h, i, args):
    candidates = tuple(config for config in FC2_CONFIGS if h % config.tile_n == 0 and i % config.tile_k == 0)
    if args.tile_n is not None:
        candidates = tuple(config for config in candidates if config.tile_n == args.tile_n)
    if args.tile_k is not None:
        candidates = tuple(config for config in candidates if config.tile_k == args.tile_k)
    if args.cluster_m is not None:
        candidates = tuple(config for config in candidates if config.cluster_m == args.cluster_m)
    return candidates


def make_fc1_layout(tokens, experts, topk, i):
    router_scores = torch.empty((tokens, experts), device="cuda", dtype=torch.bfloat16).uniform_(-1.0, 1.0)
    topk_result = torch.topk(router_scores.float(), topk, dim=1)
    selected = topk_result.indices.reshape(-1)
    # Select the top-k experts and normalize their logits, so each token's
    # selected-expert routing weights sum to one.
    routing_weights = torch.softmax(topk_result.values, dim=1).to(torch.bfloat16).contiguous()
    grouped_order = torch.argsort(selected, stable=True)
    fc1_unordered = torch.empty((tokens * topk, i), device="cuda", dtype=torch.bfloat16).uniform_(-1.0, 1.0)
    routing = pack_routing_metadata(selected, experts)
    return (fc1_unordered.index_select(0, grouped_order).contiguous(), routing, routing_weights,
            router_scores, selected, grouped_order)


def prepare_fc2_metadata(routing, weights, fc1, config):
    experts, h, _ = weights.shape
    grouped_layout = routing.narrow(0, 2 * experts, fc1.size(0))
    grouped_config = GroupedMConfig(
        tile_m=config.tile_m, tile_n=config.tile_n, tile_k=config.tile_k,
        epilogue_tile_n=config.epilogue_tile_n, cluster_m=config.cluster_m,
        mode=1)
    return prepare_grouped_m_metadata(grouped_layout, experts, h, grouped_config)


def run_fc2(fc1, weights, routing_weights, routing, grouped_metadata, topk, out, config):
    fc2_finalize_with_routing_out(
        fc1, weights, topk, routing_weights, routing, grouped_metadata, out, config=config)


def benchmark_fc2(fc1, weights, routing_weights, routing, topk, out, config, args):
    grouped_metadata = prepare_fc2_metadata(routing, weights, fc1, config)
    run_fc2(fc1, weights, routing_weights, routing, grouped_metadata, topk, out, config)
    torch.cuda.synchronize()
    return benchmark_with_stats(
        lambda: run_fc2(fc1, weights, routing_weights, routing, grouped_metadata, topk, out, config),
        args.warmup, args.iters, args.samples, cold_l2_flush=args._cold_l2_flush,
    )


def probe_fc2(fc1, weights, routing_weights, routing, topk, out, config, args):
    grouped_metadata = prepare_fc2_metadata(routing, weights, fc1, config)
    return quick_probe(
        lambda: run_fc2(fc1, weights, routing_weights, routing, grouped_metadata, topk, out, config),
        args.iters,
        cold_l2_flush=args._cold_l2_flush,
    )


def reference_fc2(fc1, weights, routing_weights, selected, grouped_order, tokens, topk, h):
    grouped_experts = selected.index_select(0, grouped_order)
    grouped = torch.empty((fc1.size(0), h), device="cuda", dtype=torch.bfloat16)
    for expert in range(weights.size(0)):
        rows = (grouped_experts == expert).nonzero().flatten()
        if rows.numel():
            grouped[rows] = (fc1[rows].float() @ weights[expert].float().T).bfloat16()
    inverse = torch.empty_like(grouped_order)
    inverse[grouped_order] = torch.arange(grouped_order.numel(), device="cuda")
    token_major = grouped.float().index_select(0, inverse).view(tokens, topk, h)
    return (token_major * routing_weights.float().unsqueeze(-1)).sum(1).bfloat16()


class Fc2FinalizeBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = super().build_parser()
        self.add_timing_args(parser, warmup=2, iters=3, samples=3)
        self.add_cache_args(parser)
        self.add_sol_arg(parser)
        parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
        parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
        parser.add_argument("--h", type=int, default=DEFAULT_H)
        parser.add_argument("--i", type=int, default=DEFAULT_I)
        parser.add_argument("--token", type=int, default=0)
        parser.add_argument("--seed", type=int, default=self.seed)
        parser.add_argument("--atol", type=float, default=2e-2)
        parser.add_argument("--rtol", type=float, default=2e-2)
        parser.add_argument("--check", action="store_true")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--tile-n", type=int, choices=(64, 128, 256))
        parser.add_argument("--tile-k", type=int, choices=(64, 128))
        parser.add_argument("--cluster-m", type=int, choices=(1, 2))
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if args.experts <= 0 or args.h <= 0 or args.i <= 0:
            raise SystemExit("--experts, --h, and --i must be positive")
        if args.topk <= 0 or args.topk > args.experts:
            raise SystemExit("--topk must be positive and no larger than --experts")
        if args.token < 0:
            raise SystemExit("--token must be non-negative")
        return args

    def make_result_table(self):
        return self.make_table_printer((
            ("Tokens", 8), ("M", 10), ("TopK", 8), ("E", 8),
            ("H", 8), ("I", 8), ("KernelMed(ms)", 16),
            ("E2E TF/s", 16), ("MaxDiff", 12), ("BestConfig", 42),
        ))

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        tokens_values = (args.token,) if args.token else DEFAULT_TOKEN_VALUES
        candidates = compatible_candidates(args.h, args.i, args)
        if not candidates:
            raise SystemExit("no compatible FC2 configs match the requested filters")
        print(f"[config] experts={args.experts} h={args.h} i={args.i} topk={args.topk} tokens={tokens_values}")
        print("[runner] resolving and caching the best compatible config")
        table = self.make_result_table()
        table.print_header()
        weights = torch.empty((args.experts, args.h, args.i), device="cuda", dtype=torch.bfloat16).uniform_(-1.0, 1.0)
        for tokens in tokens_values:
            fc1, routing, routing_weights, router_scores, selected, grouped_order = make_fc1_layout(
                tokens, args.experts, args.topk, args.i
            )
            out = torch.empty((tokens, args.h), device="cuda", dtype=torch.bfloat16)
            def prepare(config):
                metadata = prepare_fc2_metadata(routing, weights, fc1, config)
                run = lambda: run_fc2(fc1, weights, routing_weights, routing, metadata, args.topk, out, config)
                run()
                torch.cuda.synchronize()
                return PreparedKernel(config, run)

            options = TuneOptions(
                warmup=args.warmup, iters=args.iters, samples=args.samples,
                cold_l2_flush=args._cold_l2_flush, cache_path=args.cache_path,
                force_sweep=args.force_sweep, debug=args.debug,
            )
            selection = AutotuneRunner(options).resolve(
                candidates,
                prepare=prepare,
                benchmark=lambda prepared: benchmark_with_stats(
                    prepared.run, args.warmup, args.iters, args.samples,
                    cold_l2_flush=args._cold_l2_flush,
                ),
                probe=lambda prepared: quick_probe(
                    prepared.run, args.iters, cold_l2_flush=args._cold_l2_flush,
                ),
                format_candidate=format_candidate,
                namespace="fc2_finalize",
                key=(tokens, args.experts, args.topk, args.h, args.i),
                metric_formatter=lambda _config, result: f"{fc2_tflops(tokens, args.topk, args.h, args.i, result['median_ms']):.1f} TF/s",
            )
            best_config, stats = selection.candidate, selection.stats
            max_abs_diff = 0.0
            if args.check:
                selection.prepared.run()
                reference = reference_fc2(fc1, weights, routing_weights, selected, grouped_order, tokens, args.topk, args.h)
                torch.cuda.synchronize()
                max_abs_diff = (out.float() - reference.float()).abs().max().item()
                if not torch.allclose(out.float(), reference.float(), atol=args.atol, rtol=args.rtol):
                    raise SystemExit(f"[check] FAIL max_abs_diff={max_abs_diff:.6f}")
                print(f"[check tokens={tokens}] PASS max_abs_diff={max_abs_diff:.6f}")
            table.print_row(tokens, tokens * args.topk, args.topk, args.experts, args.h, args.i,
                            f"{stats['median_ms']:.4f}",
                            f"{fc2_tflops(tokens, args.topk, args.h, args.i, stats['median_ms']):.3f}",
                            f"{max_abs_diff:.6f}", format_candidate(best_config))
            if args.sol:
                print_sol_report(fc2_sol_bounds(router_scores, args.topk, args.h, args.i, best_config),
                                 stats["median_ms"], label="FC2 grouped GEMM + finalize")


if __name__ == "__main__":
    Fc2FinalizeBenchmark().run()
