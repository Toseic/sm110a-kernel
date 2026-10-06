#!/usr/bin/env python3
"""Sweep and benchmark for routed MXFP4 (W4A8) FC2 with NVFP4 parity timing.

Times the MXFP4 W4A8 kernel against the NVFP4 W4A4 kernel on the same routed
shapes (the performance target is parity with NVFP4), and validates the MXFP4
result against an in-file PyTorch reference: dequantize the FP8 activations
and MXFP4 weights, run the grouped GEMM per expert, and apply the
routing-weight reduction. Both sides consume identical quantized operands, so
only FP32 accumulation order differs and a plain tolerance applies.
"""

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
    SolReport,
    TablePrinter,
    benchmark_with_stats,
    quick_probe,
    sol_bounds,
)
from sm110a_kernel import (  # noqa: E402
    Mxfp4Fc2Config,
    Nvfp4Fc2Config,
    bf16tofp8_quant,
    compile_mxfp4_fc2,
    compile_nvfp4_fc2,
    mxfp4_fc2_out,
    nvfp4_fc2_out,
    pack_routing_metadata,
    prepare_mxfp4_fc2_metadata,
    prepare_nvfp4_fc2_metadata,
    quantize_fp8mxfp4_weights,
    quantize_nvfp4_rows,
    to_blocked_mxfp4_sfa,
)
from sm110a_kernel.runner import AutotuneRunner, PreparedKernel  # noqa: E402

DEFAULT_EXPERTS = 256
DEFAULT_N = 6144
DEFAULT_K = 2048
# Keep the FC2 sweep aligned with benchmark_nvfp4_fc1.py so the two MoE
# projections are compared at the same routed-token counts.
DEFAULT_TOKEN_VALUES = (64 * 32, 128 * 32, 256 * 32, 512 * 32, 1024 * 32)
# DeepSeek-V4-Flash routes each token to its top-6 experts.
DEFAULT_TOPK = 6
ROUTER_POWERLAW_ALPHA = 0.8
NVFP4_MMA_PEAK_TFLOPS = 4.0 * 250.0
MXFP4_W4A8_MMA_PEAK_TFLOPS = 2.0 * 250.0

# Task metadata carries the cluster-M tile geometry and is prepared for each
# candidate below.  Keep both cluster shapes in the normal sweep.
MXFP4_FC2_CONFIGS = tuple(
    Mxfp4Fc2Config(
        tile_k=128,
        epilogue_tile_n=epilogue_tile_n,
        cluster_m=cluster_m,
    )
    for epilogue_tile_n in (32, 64)
    for cluster_m in (1, 2)
)

# NVFP4 parity reference candidates (tile_n=256 persistent family).
NVFP4_FC2_PARITY_CONFIGS = tuple(
    Nvfp4Fc2Config(
        tile_n=256,
        tile_k=tile_k,
        epilogue_tile_n=32,
        cluster_m=cluster_m,
    )
    for tile_k in (256, 128)
    for cluster_m in (1, 2)
)


def fc2_tflops(tokens, topk, n, k, elapsed_ms):
    return 2.0 * tokens * topk * n * k / (elapsed_ms * 1e-3) / 1e12


def format_candidate(config):
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/e{config.epilogue_tile_n}/c{config.cluster_m}/persistent"
        f"/r{config.reduce_tile_n}"
    )


def mxfp4_candidates(args):
    candidates = MXFP4_FC2_CONFIGS
    if args.tile_k is not None:
        candidates = tuple(c for c in candidates if c.tile_k == args.tile_k)
    if args.cluster_m is not None:
        candidates = tuple(replace(c, cluster_m=args.cluster_m) for c in candidates)
    if args.reduce_tile_n is not None:
        candidates = tuple(
            replace(c, reduce_tile_n=args.reduce_tile_n) for c in candidates
        )
    return candidates


def mxfp4_fc2_sol_bounds(router_scores, topk, n, k, config):
    """Model W4A8 FC2 traffic, including the fused routing reduction."""
    tokens, experts = router_scores.shape
    selected = torch.topk(router_scores.float(), topk, dim=1).indices
    expert_rows = torch.bincount(selected.reshape(-1), minlength=experts)
    active_experts = int((expert_rows > 0).sum().item())
    m = tokens * topk

    flops = 2 * m * n * k
    a_bytes = m * (k + k / 8)
    b_bytes = active_experts * n * (k / 2 + k / 8)
    output_bytes = 2 * tokens * n
    compulsory_bytes = a_bytes + b_bytes + output_bytes

    cluster_tile_m = config.cluster_m * config.tile_m
    task_count = sum(
        math.ceil(int(rows) / cluster_tile_m) * (n // config.tile_n)
        for rows in expert_rows.tolist()
        if rows
    )
    staged_sfb_n = max(config.tile_n, 128)
    tile_read_bytes = task_count * (
        cluster_tile_m * (k + k / 8) + config.tile_n * k / 2 + staged_sfb_n * k / 8
    )
    return sol_bounds(
        flops,
        compulsory_bytes,
        a_bytes + b_bytes,
        tile_read_bytes,
        mma_peak_tflops=MXFP4_W4A8_MMA_PEAK_TFLOPS,
    )


def unpack_e2m1(q, rows, k):
    raw = q.view(torch.uint8)
    codes = torch.stack((raw & 0xF, raw >> 4), dim=-1).reshape(rows, k)
    magnitudes = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device=q.device,
        dtype=torch.float32,
    )
    values = magnitudes[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -values, values)


def blocked_to_logical_scales(scales, rows, k):
    tiles = scales.view(rows // 128, k // 32, 32, 4, 4).view(torch.uint8)
    blocks = tiles.transpose(2, 3).reshape(rows // 128, k // 32, 128, 4)
    logical = blocks.permute(0, 2, 1, 3)[..., 0].reshape(rows, k // 32)
    return logical.view(torch.float8_e8m0fnu)


def make_case(tokens, experts, topk, n, k):
    router_scores = (
        lambda logits: (
            torch.randn((tokens, experts), device="cuda") * logits.unsqueeze(0)
        ).to(torch.bfloat16)
    )(
        torch.arange(1, experts + 1, device="cuda", dtype=torch.float32).pow(
            -ROUTER_POWERLAW_ALPHA
        )
    )
    topk_result = torch.topk(router_scores.float(), topk, dim=1)
    selected = topk_result.indices.to(torch.int32)
    selected_flat = selected.reshape(-1)
    grouped_order = torch.argsort(selected_flat, stable=True)
    routing = pack_routing_metadata(selected_flat, experts)
    grouped_layout = routing.narrow(0, 2 * experts, selected_flat.numel())
    routing_weights = torch.softmax(topk_result.values, dim=1).to(torch.bfloat16)

    # FC2 consumes expert-grouped FC1 rows in W4A8 form: FP8 E4M3 with one
    # E8M0 scale per 128 K elements (the bf16tofp8_quant output), expanded to
    # the blocked per-32 layout the MMA consumes.
    fc1 = torch.randn(
        (tokens * topk, k), device="cuda", dtype=torch.bfloat16
    ).index_select(0, grouped_order)
    aq, sfa128 = bf16tofp8_quant(fc1)
    del fc1
    sfa = to_blocked_mxfp4_sfa(sfa128)
    nv_aq, nv_sfa = quantize_nvfp4_rows(
        torch.randn((tokens * topk, k), device="cuda", dtype=torch.bfloat16)
    )

    bq_parts, sfb_parts, nv_bq_parts, nv_sfb_parts = [], [], [], []
    for _ in range(experts):
        b_expert = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        bq_expert, sfb_expert = quantize_fp8mxfp4_weights(b_expert)
        bq_parts.append(bq_expert.view(torch.float4_e2m1fn_x2))
        sfb_parts.append(sfb_expert.reshape(-1))
        nv_bq_expert, nv_sfb_expert = quantize_nvfp4_rows(b_expert)
        nv_bq_parts.append(nv_bq_expert)
        nv_sfb_parts.append(nv_sfb_expert)
        del b_expert
    bq = torch.stack(bq_parts)
    sfb = torch.cat(sfb_parts)
    nv_bq = torch.stack(nv_bq_parts)
    nv_sfb = torch.cat(nv_sfb_parts)
    return (
        aq,
        sfa128,
        bq,
        sfb,
        sfa,
        nv_aq,
        nv_sfa,
        nv_bq,
        nv_sfb,
        routing,
        grouped_layout,
        routing_weights,
        router_scores,
    )


def mxfp4_fc2_reference(
    aq,
    sfa128,
    bq,
    sfb,
    routing,
    grouped_layout,
    routing_weights,
    tokens,
    experts,
    topk,
    n,
    k,
):
    m = tokens * topk
    a = aq.float() * sfa128.float().repeat_interleave(128, dim=1)
    grouped = torch.empty((m, n), device="cuda", dtype=torch.float32)
    scale_elems = n * (k // 32) * 4
    for expert in range(experts):
        rows = (grouped_layout == expert).nonzero().flatten()
        if rows.numel():
            sfb_e = blocked_to_logical_scales(
                sfb[expert * scale_elems : (expert + 1) * scale_elems], n, k
            )
            b = unpack_e2m1(bq[expert], n, k) * sfb_e.float().repeat_interleave(
                32, dim=1
            )
            grouped[rows] = a[rows] @ b.t()

    assignment_to_grouped = routing[2 * experts + 2 * m :]
    assignments = torch.arange(m, device="cuda").view(tokens, topk)
    grouped_rows = assignment_to_grouped[assignments]
    return (
        (
            grouped[grouped_rows].view(tokens, topk, n)
            * routing_weights.float().unsqueeze(-1)
        )
        .sum(1)
        .bfloat16()
    )


class Mxfp4Fc2Benchmark(BenchmarkBase):
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
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--tile-k", type=int, choices=(128,))
        parser.add_argument("--cluster-m", type=int, choices=(1, 2))
        parser.add_argument("--reduce-tile-n", type=int, choices=(32, 64, 128, 256))
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if args.experts <= 0 or args.n <= 0 or args.k <= 0:
            raise SystemExit("--experts, --n, and --k must be positive")
        if args.topk <= 0 or args.topk > args.experts:
            raise SystemExit("--topk must be positive and no larger than --experts")
        if args.n % 256 or args.k % 128:
            raise SystemExit("N must be divisible by 256 and K by 128")
        if args.token < 0:
            raise SystemExit("--token must be non-negative")
        args.token_values = (args.token,) if args.token else DEFAULT_TOKEN_VALUES
        if any(tokens <= 0 or tokens % 128 for tokens in args.token_values):
            raise SystemExit("every token count must be a positive multiple of 128")
        if not mxfp4_candidates(args):
            raise SystemExit("no compatible MXFP4 FC2 configs match the filters")
        return args

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        print(
            f"[config] experts={args.experts} n={args.n} k={args.k} "
            f"topk={args.topk} tokens={args.token_values} "
            f"routing=powerlaw(alpha={ROUTER_POWERLAW_ALPHA})"
        )
        print("[parity] NVFP4 W4A4 kernel on the same shapes (tile_n=256 family)")
        table = TablePrinter(
            (
                ("Tokens", 9),
                ("M", 10),
                ("TopK", 7),
                ("E", 6),
                ("N", 8),
                ("K", 8),
                ("MXFP4(ms)", 12),
                ("MXFP4 TF/s", 13),
                ("NVFP4(ms)", 12),
                ("NVFP4 TF/s", 13),
                ("NVFP4/MXFP4", 12),
                ("MaxDiff", 10),
                ("Cache", 8),
                ("MXFP4Config", 32),
            ),
            debug=args.debug,
        )
        table.print_header()
        sol_report = SolReport(debug=args.debug) if args.sol else None

        for tokens in args.token_values:
            case = make_case(tokens, args.experts, args.topk, args.n, args.k)
            (
                aq,
                sfa128,
                bq,
                sfb,
                sfa,
                nv_aq,
                nv_sfa,
                nv_bq,
                nv_sfb,
                routing,
                grouped_layout,
                routing_weights,
                router_scores,
            ) = case
            out = torch.empty((tokens, args.n), device="cuda", dtype=torch.bfloat16)
            nv_out = torch.empty_like(out)
            m = tokens * args.topk

            # FC2 task packets are parameterized by cluster_m as well.  Build
            # them with the candidate rather than using default cluster-1
            # metadata for the cluster-2 NVFP4 parity candidates.
            def prepare_mxfp4(config):
                metadata = prepare_mxfp4_fc2_metadata(
                    grouped_layout, args.experts, args.n, config=config
                )
                run = lambda: mxfp4_fc2_out(
                    aq,
                    bq,
                    sfa,
                    sfb,
                    routing,
                    routing_weights,
                    metadata,
                    args.topk,
                    out,  # noqa: F821
                    config=config,
                )
                # The fused epilogue scatter-adds into the destination.
                # Start each tactic from zero without timing the memset.
                out.zero_()  # noqa: F821
                run()
                torch.cuda.synchronize()
                return PreparedKernel(config, run)

            def prepare_nvfp4(config):
                metadata = prepare_nvfp4_fc2_metadata(
                    grouped_layout, args.experts, args.n, config=config
                )
                run = lambda: nvfp4_fc2_out(
                    nv_aq,
                    nv_bq,
                    nv_sfa,
                    nv_sfb,
                    routing,
                    routing_weights,
                    metadata,
                    args.topk,
                    nv_out,  # noqa: F821
                    config=config,
                )
                nv_out.zero_()  # noqa: F821
                run()
                torch.cuda.synchronize()
                return PreparedKernel(config, run)

            options = self.make_tune_options(
                args, role="ours", pre_parallel_jit=args.pre_parallel_jit
            )
            nv_options = self.make_tune_options(
                args, role="baseline", pre_parallel_jit=args.pre_parallel_jit
            )
            ours_selection = AutotuneRunner(options).resolve(
                mxfp4_candidates(args),
                namespace="mxfp4_fc2",
                key=("v1-w4a8", tokens, args.experts, args.topk, args.n, args.k),
                prepare=prepare_mxfp4,
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
                metric_formatter=lambda _c, stats: (
                    f"{fc2_tflops(tokens, args.topk, args.n, args.k, stats['median_ms']):.1f} TF/s"
                ),
                pre_jit=compile_mxfp4_fc2,
            )
            nv_selection = AutotuneRunner(nv_options).resolve(
                NVFP4_FC2_PARITY_CONFIGS,
                namespace="nvfp4_fc2",
                key=("mxfp4-parity", tokens, args.experts, args.topk, args.n, args.k),
                prepare=prepare_nvfp4,
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
                metric_formatter=lambda _c, stats: (
                    f"{fc2_tflops(tokens, args.topk, args.n, args.k, stats['median_ms']):.1f} TF/s"
                ),
                pre_jit=compile_nvfp4_fc2,
            )

            max_abs_diff = float("nan")
            if not args.no_accuracy:
                out.zero_()
                ours_selection.prepared.run()
                torch.cuda.synchronize()
                ref = mxfp4_fc2_reference(
                    aq,
                    sfa128,
                    bq,
                    sfb,
                    routing,
                    grouped_layout,
                    routing_weights,
                    tokens,
                    args.experts,
                    args.topk,
                    args.n,
                    args.k,
                )
                max_abs_diff = (out.float() - ref.float()).abs().max().item()
                # The fused epilogue accumulates the top-k contributions with
                # BF16 cp.reduce.async.bulk adds, so the error scale is a few
                # BF16 ULPs of the largest partial sum, not of each output
                # element (cancellation makes small outputs carry large-sum
                # rounding).  Scale atol accordingly; the fp32-summed
                # reference itself is exact to accumulation order.
                scaled_atol = max(args.atol, 0.03 * ref.float().abs().max().item())
                if not torch.allclose(
                    out.float(), ref.float(), atol=scaled_atol, rtol=args.rtol
                ):
                    raise AssertionError(
                        f"MXFP4 FC2 mismatch tokens={tokens}: "
                        f"max_abs_diff={max_abs_diff:.6f} "
                        f"scaled_atol={scaled_atol:.3f}"
                    )
                del ref

            stats = ours_selection.stats
            kernel_ms = stats["median_ms"]
            nv_ms = nv_selection.stats["median_ms"]
            table.print_row(
                tokens,
                m,
                args.topk,
                args.experts,
                args.n,
                args.k,
                f"{kernel_ms:.4f}",
                f"{fc2_tflops(tokens, args.topk, args.n, args.k, kernel_ms):.2f}",
                f"{nv_ms:.4f}",
                f"{fc2_tflops(tokens, args.topk, args.n, args.k, nv_ms):.2f}",
                f"{kernel_ms / nv_ms:.2f}x" if nv_ms > 0 else "N/A",
                f"{max_abs_diff:.4f}",
                "hit" if ours_selection.cache_hit else "miss",
                format_candidate(ours_selection.candidate),
            )
            if args.sol:
                sol_report.add(
                    mxfp4_fc2_sol_bounds(
                        router_scores,
                        args.topk,
                        args.n,
                        args.k,
                        ours_selection.candidate,
                    ),
                    kernel_ms,
                    label=f"MXFP4 FC2 kernel (tokens={tokens})",
                )
            del case, out, nv_out
            torch.cuda.empty_cache()
        if sol_report:
            sol_report.print()


if __name__ == "__main__":
    Mxfp4Fc2Benchmark().run()
