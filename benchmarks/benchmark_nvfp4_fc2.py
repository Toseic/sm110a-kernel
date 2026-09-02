#!/usr/bin/env python3
"""Runner-backed sweep and benchmark for routed NVFP4 FC2."""

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    BenchmarkBase,
    TablePrinter,
    benchmark_with_stats,
    print_sol_report,
    quick_probe,
    sol_bounds,
)
from sm110a_kernel import (  # noqa: E402
    CuteDslNvfp4Fc2Runner,
    Nvfp4Fc2Config,
    compile_nvfp4_fc2,
    from_blocked_nvfp4_scales,
    nvfp4_fc2_out,
    pack_routing_metadata,
    prepare_nvfp4_fc2_metadata,
    quantize_nvfp4_rows,
)
from sm110a_kernel.runner import AutotuneRunner, PreparedKernel  # noqa: E402


DEFAULT_EXPERTS = 256
DEFAULT_N = 6144
DEFAULT_K = 2048
# Keep the FC2 sweep aligned with benchmark_nvfp4_fc1.py so the two MoE
# projections are compared at the same routed-token counts.
DEFAULT_TOKEN_VALUES = (64 * 32, 128 * 32, 256 * 32, 512 * 32, 1024 * 32)
DEFAULT_TOPK = 8
NVFP4_MMA_PEAK_TFLOPS = 4.0 * 250.0
_MODE_LABELS = {0: "np", 1: "pe", 2: "clc"}


NVFP4_FC2_CONFIGS = tuple(
    Nvfp4Fc2Config(
        tile_n=tile_n,
        tile_k=tile_k,
        epilogue_tile_n=epilogue_tile_n,
        cluster_m=cluster_m,
        mode=mode,
    )
    for tile_n in (64, 128, 256)
    for tile_k in (64, 128, 256)
    for epilogue_tile_n in (32, 64, 128)
    for cluster_m in (1, 2)
    for mode in (0, 1, 2)
    if tile_n % epilogue_tile_n == 0
)


def fc2_tflops(tokens, topk, n, k, elapsed_ms):
    return 2.0 * tokens * topk * n * k / (elapsed_ms * 1e-3) / 1e12


def format_candidate(config):
    order = "n" if config.macro_swizzle_n_major else "m"
    if config.macro_swizzle_serpentine:
        order += "s"
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/e{config.epilogue_tile_n}/c{config.cluster_m}/{_MODE_LABELS[config.mode]}"
        f"/r{config.reduce_tile_n}/{order}{config.macro_swizzle_m}x{config.macro_swizzle_n}"
    )


def compatible_candidates(n, k, args):
    candidates = tuple(
        config for config in NVFP4_FC2_CONFIGS
        if n % config.tile_n == 0 and n % 128 == 0 and k % config.tile_k == 0
    )
    if args.tile_n is not None:
        candidates = tuple(c for c in candidates if c.tile_n == args.tile_n)
    if args.tile_k is not None:
        candidates = tuple(c for c in candidates if c.tile_k == args.tile_k)
    if args.epilogue_tile_n is not None:
        candidates = tuple(
            c for c in candidates if c.epilogue_tile_n == args.epilogue_tile_n
        )
    if args.cluster_m is not None:
        candidates = tuple(c for c in candidates if c.cluster_m == args.cluster_m)
    if args.mode is not None:
        candidates = tuple(c for c in candidates if c.mode == args.mode)
    if args.reduce_tile_n is not None:
        candidates = tuple(c for c in candidates if c.tile_n % args.reduce_tile_n == 0)
    n_major = args.macro_order.startswith("n")
    serpentine = args.macro_order.endswith("serp")
    candidates = tuple(
        replace(
            c,
            reduce_tile_n=args.reduce_tile_n or c.reduce_tile_n,
            macro_swizzle_n_major=n_major,
            macro_swizzle_serpentine=serpentine,
            macro_swizzle_m=args.macro_m,
            macro_swizzle_n=args.macro_n,
        )
        for c in candidates
    )
    return candidates


def nvfp4_fc2_sol_bounds(router_scores, topk, n, k, config):
    """Model NVFP4 FC2 traffic, including the grouped workspace reduction."""
    tokens, experts = router_scores.shape
    selected = torch.topk(router_scores.float(), topk, dim=1).indices
    expert_rows = torch.bincount(selected.reshape(-1), minlength=experts)
    active_experts = int((expert_rows > 0).sum().item())
    m = tokens * topk

    flops = 2 * m * n * k
    a_bytes = m * (k / 2 + k / 16)
    b_bytes = active_experts * n * (k / 2 + k / 16)
    workspace_bytes = 2 * m * n
    output_bytes = 2 * tokens * n
    compulsory_bytes = a_bytes + b_bytes + workspace_bytes + output_bytes

    cluster_tile_m = config.cluster_m * config.tile_m
    task_count = sum(
        math.ceil(int(rows) / cluster_tile_m) * (n // config.tile_n)
        for rows in expert_rows.tolist() if rows
    )
    staged_sfb_n = max(config.tile_n, 128)
    tile_read_bytes = task_count * (
        cluster_tile_m * (k / 2 + k / 16)
        + config.tile_n * k / 2
        + staged_sfb_n * k / 16
    )
    return sol_bounds(
        flops,
        compulsory_bytes,
        a_bytes + b_bytes,
        tile_read_bytes,
        mma_peak_tflops=NVFP4_MMA_PEAK_TFLOPS,
    )


def unpack_e2m1(q, rows, k):
    raw = q.view(torch.uint8)
    codes = torch.stack((raw & 0xF, raw >> 4), dim=-1).reshape(rows, k)
    magnitudes = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device=q.device, dtype=torch.float32,
    )
    values = magnitudes[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -values, values)


def dequantize(q, scales, rows, k):
    logical_scales = from_blocked_nvfp4_scales(scales, rows, k).float()
    return unpack_e2m1(q, rows, k) * logical_scales.repeat_interleave(16, dim=1)


def make_case(tokens, experts, topk, n, k):
    router_scores = torch.randn(
        (tokens, experts), device="cuda", dtype=torch.bfloat16
    )
    topk_result = torch.topk(router_scores.float(), topk, dim=1)
    selected = topk_result.indices.to(torch.int32)
    selected_flat = selected.reshape(-1)
    grouped_order = torch.argsort(selected_flat, stable=True)
    routing = pack_routing_metadata(selected_flat, experts)
    grouped_layout = routing.narrow(0, 2 * experts, selected_flat.numel())
    routing_weights = torch.softmax(topk_result.values, dim=1).to(torch.bfloat16)

    # FC2 consumes expert-grouped FC1 rows. Keep only quantized activations and
    # weights, matching the memory behavior of the operator under benchmark.
    fc1 = torch.randn(
        (tokens * topk, k), device="cuda", dtype=torch.bfloat16
    ).index_select(0, grouped_order)
    aq, sfa = quantize_nvfp4_rows(fc1)
    del fc1

    bq_parts, sfb_parts = [], []
    for _ in range(experts):
        b_expert = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        bq_expert, sfb_expert = quantize_nvfp4_rows(b_expert)
        bq_parts.append(bq_expert)
        sfb_parts.append(sfb_expert)
        del b_expert
    bq = torch.stack(bq_parts)
    sfb = torch.cat(sfb_parts)
    return aq, bq, sfa, sfb, routing, grouped_layout, routing_weights, router_scores


def reference(case, tokens, experts, topk, n, k):
    aq, bq, sfa, sfb, routing, grouped_layout, routing_weights, _ = case
    m = tokens * topk
    a = dequantize(aq, sfa, m, k)
    grouped = torch.empty((m, n), device="cuda", dtype=torch.float32)
    scale_elems = n * (k // 16)
    for expert in range(experts):
        rows = (grouped_layout == expert).nonzero().flatten()
        if rows.numel():
            b = dequantize(
                bq[expert], sfb[expert * scale_elems:(expert + 1) * scale_elems],
                n, k,
            )
            grouped[rows] = a[rows] @ b.t()

    assignment_to_grouped = routing[2 * experts + 2 * m:]
    assignments = torch.arange(m, device="cuda").view(tokens, topk)
    grouped_rows = assignment_to_grouped[assignments]
    return (
        grouped[grouped_rows].view(tokens, topk, n)
        * routing_weights.float().unsqueeze(-1)
    ).sum(1).bfloat16()


class Nvfp4Fc2Benchmark(BenchmarkBase):
    def build_parser(self):
        parser = super().build_parser()
        self.add_timing_args(parser, warmup=2, iters=3, samples=3)
        self.add_cache_args(parser)
        self.add_pre_parallel_jit_arg(parser)
        self.add_sol_arg(parser)
        parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
        parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
        parser.add_argument("--n", type=int, default=DEFAULT_N)
        parser.add_argument("--k", type=int, default=DEFAULT_K)
        parser.add_argument("--token", "--tokens", dest="token", type=int, default=0)
        parser.add_argument("--seed", type=int, default=self.seed)
        parser.add_argument("--atol", type=float, default=0.25)
        parser.add_argument("--rtol", type=float, default=2e-2)
        parser.add_argument("--check", action="store_true")
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument(
            "--baseline", choices=("cutedsl",), default="cutedsl",
            help="baseline implementation to benchmark against (default: cutedsl)",
        )
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--tile-n", type=int, choices=(64, 128, 256))
        parser.add_argument("--tile-k", type=int, choices=(64, 128, 256))
        parser.add_argument("--epilogue-tile-n", type=int, choices=(32, 64, 128))
        parser.add_argument("--cluster-m", type=int, choices=(1, 2))
        parser.add_argument("--mode", type=int, choices=(0, 1, 2))
        parser.add_argument("--reduce-tile-n", type=int, choices=(32, 64, 128, 256))
        parser.add_argument(
            "--macro-order", choices=("n", "n-serp", "m", "m-serp"), default="n"
        )
        parser.add_argument("--macro-m", type=int, choices=(1, 2, 4, 8, 16), default=8)
        parser.add_argument("--macro-n", type=int, choices=(1, 2, 4, 8, 16), default=8)
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if args.experts <= 0 or args.n <= 0 or args.k <= 0:
            raise SystemExit("--experts, --n, and --k must be positive")
        if args.topk <= 0 or args.topk > args.experts:
            raise SystemExit("--topk must be positive and no larger than --experts")
        if args.n % 128 or args.k % 64:
            raise SystemExit("N must be divisible by 128 and K by 64")
        if args.token < 0:
            raise SystemExit("--token must be non-negative")
        args.token_values = (args.token,) if args.token else DEFAULT_TOKEN_VALUES
        if any(tokens <= 0 or tokens % 128 for tokens in args.token_values):
            raise SystemExit("every token count must be a positive multiple of 128")
        if args.check and args.no_accuracy:
            raise SystemExit("--check and --no-accuracy are mutually exclusive")
        if not compatible_candidates(args.n, args.k, args):
            raise SystemExit("no compatible NVFP4 FC2 configs match the filters")
        return args

    def make_result_table(self):
        return TablePrinter((
            ("Tokens", 9), ("M", 10), ("TopK", 7), ("E", 6),
            ("N", 8), ("K", 8), ("Kernel(ms)", 14), ("Kernel TF/s", 15),
            ("CuTE DSL(ms)", 14), ("CuTE DSL TF/s", 15),
            ("Speedup", 10), ("MaxDiff", 12), ("Cache", 8),
            ("BestConfig", 42),
        ), debug=getattr(self, "_debug_enabled", False))

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        candidates = compatible_candidates(args.n, args.k, args)
        print(
            f"[config] experts={args.experts} n={args.n} k={args.k} "
            f"topk={args.topk} tokens={args.token_values}"
        )
        print("[baseline] TRT-LLM CuTE DSL fused-finalize NVFP4 FC2 (autotuned)")
        print("[runner] resolving and caching the best compatible config")
        table = self.make_result_table()
        table.print_header()

        for tokens in args.token_values:
            case = make_case(tokens, args.experts, args.topk, args.n, args.k)
            aq, bq, sfa, sfb, routing, grouped_layout, routing_weights, router_scores = case
            out = torch.empty((tokens, args.n), device="cuda", dtype=torch.bfloat16)
            baseline_out = torch.empty_like(out)
            prepared_metadata = {}

            # Sweep the CuTE DSL baseline before the native FC2 candidates, as
            # in benchmark_nvfp4_fc1.py.  The native sweep prints many more
            # configurations, so running it first obscures baseline results.
            baseline_runner = CuteDslNvfp4Fc2Runner(
                aq, bq, sfa, sfb, routing, routing_weights,
                baseline_out, args.topk,
            )
            baseline_runner.autotune(self.make_tune_options(
                args, role="baseline", pre_parallel_jit=args.pre_parallel_jit,
            ))
            baseline_fn = baseline_runner.run
            baseline_stats = benchmark_with_stats(
                baseline_fn, args.warmup, args.iters, args.samples,
                cold_l2_flush=args._cold_l2_flush,
            )

            def prepare(config):
                metadata = prepare_nvfp4_fc2_metadata(
                    grouped_layout, args.experts, args.n, config=config
                )
                run = lambda: nvfp4_fc2_out(
                    aq, bq, sfa, sfb, routing, routing_weights,
                    metadata, args.topk, out, config=config,
                )
                # The fused epilogue scatter-adds into ``out``.  Start each
                # tactic from zero without putting the memset in the timed path.
                out.zero_()
                run()
                torch.cuda.synchronize()
                prepared_metadata[config] = metadata
                return PreparedKernel(config, run)

            options = self.make_tune_options(
                args, role="ours", pre_parallel_jit=args.pre_parallel_jit
            )
            selection = AutotuneRunner(options).resolve(
                candidates,
                prepare=prepare,
                benchmark=lambda prepared: benchmark_with_stats(
                    prepared.run, args.warmup, args.iters, args.samples,
                    cold_l2_flush=args._cold_l2_flush,
                ),
                probe=lambda prepared: quick_probe(
                    prepared.run, args.iters, cold_l2_flush=args._cold_l2_flush
                ),
                format_candidate=format_candidate,
                namespace="nvfp4_fc2",
                # v2 retunes after switching the API default to the
                # persistent 128x256x256 configuration.
                key=("v2", tokens, args.experts, args.topk, args.n, args.k),
                pre_jit=compile_nvfp4_fc2,
                metric_formatter=lambda config, stats: (
                    f"{fc2_tflops(tokens, args.topk, args.n, args.k, stats['median_ms']):.1f} TF/s"
                ),
            )

            max_abs_diff = 0.0
            if args.check and not args.no_accuracy:
                # Both finalize kernels accumulate.  Zero both destinations
                # immediately before the validation launches so the comparison
                # sees a single-kernel result, not the timed accumulation.
                # BF16 scatter-add reduction order differs between the two
                # kernels, so this comparison is opt-in via --check.
                out.zero_()
                selection.prepared.run()
                baseline_out.zero_()
                baseline_fn()
                torch.cuda.synchronize()
                max_abs_diff = (out.float() - baseline_out.float()).abs().max().item()
                if not torch.allclose(
                    out.float(), baseline_out.float(), atol=args.atol, rtol=args.rtol
                ):
                    raise SystemExit(
                        f"[check] FAIL tokens={tokens} "
                        f"max_abs_diff={max_abs_diff:.6f}"
                    )
                print(
                    f"[check tokens={tokens}] PASS vs CuTE DSL "
                    f"max_abs_diff={max_abs_diff:.6f}"
                )

            if args.check and not args.no_accuracy:
                # Optional independent validation against the torch reference.
                out.zero_()
                selection.prepared.run()
                ref = reference(
                    case, tokens, args.experts, args.topk, args.n, args.k
                )
                torch.cuda.synchronize()
                ref_diff = (out.float() - ref.float()).abs().max().item()
                if not torch.allclose(
                    out.float(), ref.float(), atol=args.atol, rtol=args.rtol
                ):
                    raise SystemExit(
                        f"[check] FAIL vs reference tokens={tokens} "
                        f"max_abs_diff={ref_diff:.6f}"
                    )
                print(
                    f"[check tokens={tokens}] PASS vs reference "
                    f"max_abs_diff={ref_diff:.6f}"
                )

            stats = selection.stats
            kernel_ms = stats["median_ms"]
            baseline_ms = baseline_stats["median_ms"]
            table.print_row(
                tokens, tokens * args.topk, args.topk, args.experts,
                args.n, args.k, f"{kernel_ms:.4f}",
                f"{fc2_tflops(tokens, args.topk, args.n, args.k, kernel_ms):.2f}",
                f"{baseline_ms:.4f}",
                f"{fc2_tflops(tokens, args.topk, args.n, args.k, baseline_ms):.2f}",
                f"{baseline_ms / kernel_ms:.2f}x" if kernel_ms > 0 else "N/A",
                f"{max_abs_diff:.6f}",
                "hit" if selection.cache_hit else "miss",
                format_candidate(selection.candidate),
            )
            if args.sol:
                print_sol_report(
                    nvfp4_fc2_sol_bounds(
                        router_scores, args.topk, args.n, args.k, selection.candidate
                    ),
                    kernel_ms,
                    label="NVFP4 FC2 grouped GEMM + reduction",
                    debug=args.debug,
                )


if __name__ == "__main__":
    Nvfp4Fc2Benchmark().run()
