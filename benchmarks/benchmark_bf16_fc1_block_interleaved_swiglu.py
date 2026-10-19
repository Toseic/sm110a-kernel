#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sm110a_kernel as ext
from common import (
    BenchmarkBase,
    SolReport,
    benchmark_with_stats,
    quick_probe,
    sol_bounds,
)
from sm110a_kernel import (
    GroupedMConfig,
    bf16_grouped_m_gemm_out,
    prepare_grouped_m_metadata,
)
from sm110a_kernel.kernels.fc1_block_interleaved_swiglu import (
    Fc1BlockInterleavedSwiGLURunner,
    compatible_candidates,
    format_candidate,
)

DEFAULT_EXPERTS = 256
DEFAULT_N = 2048
DEFAULT_K = 6144
DEFAULT_TOKEN_VALUES = (64 * 32, 128 * 32, 256 * 32, 512 * 32, 1024 * 32)
DEFAULT_TOPK = 8
INTERLEAVE_BLOCK = 32


def routed_fc1_swiglu_tflops_from_ms(tokens, topk, n, k, elapsed_ms):
    """Return end-to-end GEMM-equivalent TFLOP/s for routed FC1 SwiGLU."""
    m = tokens * topk
    flops = 4.0 * m * n * k  # gate and up projections, each 2 * M * N * K
    return flops / (elapsed_ms * 1.0e-3) / 1.0e12


def fc1_main_kernel_sol_bounds(router_scores, topk, n, k, config):
    """SOL model for FC1's main kernel; routing/scheduling are excluded."""
    experts = router_scores.size(1)
    selected = torch.topk(
        router_scores.float(), topk, dim=1, largest=True, sorted=True
    ).indices
    expert_rows = torch.bincount(selected.reshape(-1), minlength=experts)
    m = selected.numel()
    m_task_count = (
        torch.div(
            expert_rows + config.cluster_m * config.tile_m - 1,
            config.cluster_m * config.tile_m,
            rounding_mode="floor",
        )
        .sum()
        .item()
    )
    task_count = m_task_count * (n // ((config.tile_n // 2) * config.cluster_n))
    flops = 4 * m * n * k
    a_bytes = 2 * m * k
    active_experts = int((expert_rows > 0).sum().item())
    b_bytes = 4 * active_experts * n * k
    c_bytes = 2 * m * n
    tile_read_bytes = (
        2
        * task_count
        * (config.cluster_m * config.tile_m + config.cluster_n * config.tile_n)
        * k
    )
    return sol_bounds(
        flops, a_bytes + b_bytes + c_bytes, a_bytes + b_bytes, tile_read_bytes
    )


def interleave_gate_up_weights(gate_w, up_w, block_size=INTERLEAVE_BLOCK):
    if gate_w.shape != up_w.shape:
        raise ValueError("gate_w and up_w must have the same shape")
    if gate_w.ndim != 3:
        raise ValueError("weights must have shape [experts, n, k]")
    experts, n, k = gate_w.shape
    if n % block_size != 0:
        raise ValueError(f"n must be divisible by interleave block size {block_size}")

    gate_blocks = gate_w.view(experts, n // block_size, block_size, k)
    up_blocks = up_w.view(experts, n // block_size, block_size, k)
    interleaved = torch.stack((gate_blocks, up_blocks), dim=2)
    return interleaved.reshape(experts, 2 * n, k).contiguous()


def swiglu(gate, up):
    return torch.nn.functional.silu(gate.float()) * up.float()


def run_baseline(a, gate_w, up_w, layout, out, config):
    gate_acc = torch.empty_like(out)
    up_acc = torch.empty_like(out)
    metadata = prepare_grouped_m_metadata(
        layout, gate_w.size(0), gate_w.size(1), config=config
    )
    bf16_grouped_m_gemm_out(a, gate_w, metadata, gate_acc, config=config, swap_ab=False)
    bf16_grouped_m_gemm_out(a, up_w, metadata, up_acc, config=config, swap_ab=False)
    out.copy_(swiglu(gate_acc, up_acc).to(torch.bfloat16))


def build_routed_reference(a, router_scores, topk):
    selected = torch.topk(
        router_scores.float(), topk, dim=1, largest=True, sorted=True
    ).indices
    selected_flat = selected.reshape(-1)
    source_tokens = (
        torch.arange(a.size(0), device=a.device, dtype=torch.int64)
        .view(-1, 1)
        .expand(-1, topk)
        .reshape(-1)
    )
    grouped_order = torch.argsort(selected_flat, stable=True)
    grouped_a = a.index_select(
        0, source_tokens.index_select(0, grouped_order)
    ).contiguous()
    grouped_layout = (
        selected_flat.index_select(0, grouped_order).to(torch.int32).contiguous()
    )
    return grouped_a, grouped_layout


def resolve_fused_impl():
    func = getattr(ext, "fc1_block_interleaved_swiglu_out", None)
    if callable(func):
        return "fc1_block_interleaved_swiglu_out", func
    return None, None


def selected_candidates(n, k, args):
    candidates = compatible_candidates(n, k)
    if args.tile_n is not None:
        candidates = tuple(c for c in candidates if c.tile_n == args.tile_n)
    if args.tile_k is not None:
        candidates = tuple(c for c in candidates if c.tile_k == args.tile_k)
    if args.cluster_m is not None:
        candidates = tuple(c for c in candidates if c.cluster_m == args.cluster_m)
    if args.cluster_n is not None:
        candidates = tuple(c for c in candidates if c.cluster_n == args.cluster_n)
    if args.direct_gmem:
        candidates = tuple(c for c in candidates if c.direct_gmem)
    return candidates


class Fc1BlockInterleavedSwiGLUBenchmark(BenchmarkBase):
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
        parser.add_argument("--seed", type=int, default=self.seed)
        parser.add_argument("--atol", type=float, default=2e-2)
        parser.add_argument("--rtol", type=float, default=2e-2)
        parser.add_argument("--check", action="store_true")
        parser.add_argument(
            "--debug",
            action="store_true",
            help="print per-config performance after the benchmark",
        )
        parser.add_argument(
            "--baseline-tile-n",
            type=int,
            default=128,
            help="grouped GEMM baseline tile_n",
        )
        parser.add_argument(
            "--baseline-tile-k",
            type=int,
            default=128,
            help="grouped GEMM baseline tile_k",
        )
        parser.add_argument(
            "--baseline-cluster-m",
            type=int,
            choices=(1, 2),
            default=1,
            help="grouped GEMM baseline cluster_m",
        )
        parser.add_argument(
            "--baseline-mode",
            type=int,
            choices=(0, 1, 2),
            default=2,
            help="grouped GEMM baseline mode",
        )
        parser.add_argument(
            "--token",
            type=int,
            default=0,
            help="run a single sweep token instead of the default token sweep",
        )
        parser.add_argument("--tile-n", type=int, choices=(64, 128, 256))
        parser.add_argument("--tile-k", type=int, choices=(64, 128))
        parser.add_argument("--cluster-m", type=int, choices=(1, 2))
        parser.add_argument("--cluster-n", type=int, choices=(1, 2))
        parser.add_argument(
            "--direct-gmem",
            action="store_true",
            help="profile only the direct-GMEM epilogue (when selecting a config)",
        )
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if args.experts <= 0:
            raise SystemExit("--experts must be positive")
        if args.token < 0:
            raise SystemExit("--token must be non-negative")
        if args.topk <= 0 or args.topk > args.experts:
            raise SystemExit("--topk must be positive and no larger than --experts")
        if args.n <= 0 or args.k <= 0:
            raise SystemExit("--n and --k must be positive")
        if args.n % INTERLEAVE_BLOCK != 0:
            raise SystemExit(
                f"--n must be divisible by interleave block size {INTERLEAVE_BLOCK}"
            )
        if args.n % args.baseline_tile_n != 0:
            raise SystemExit("--n must be divisible by --baseline-tile-n")
        if (2 * args.n) % args.baseline_tile_n != 0:
            raise SystemExit("2 * --n must be divisible by --baseline-tile-n")
        if args.k % args.baseline_tile_k != 0:
            raise SystemExit("--k must be divisible by --baseline-tile-k")
        return args

    def make_result_table(self):
        return self.make_table_printer(
            (
                ("Tokens", 8),
                ("M", 10),
                ("TopK", 8),
                ("E", 8),
                ("N", 8),
                ("K", 8),
                ("KernelMed(ms)", 16),
                ("E2E TF/s", 16),
                ("MaxDiff", 12),
                ("BestConfig", 36),
            )
        )

    def resolve_fused_kernel(self, a, interleaved_w, router_scores, topk, out, args):
        kernel_runner = Fc1BlockInterleavedSwiGLURunner(
            a, interleaved_w, router_scores, topk, out, dynamic_metadata=False
        )
        candidates = selected_candidates(interleaved_w.size(1) // 2, a.size(1), args)

        options = self.make_tune_options(
            args,
            role="ours",
            pre_parallel_jit=args.pre_parallel_jit,
        )
        return kernel_runner.tune(
            options,
            candidates=candidates,
            benchmark=lambda prepared: benchmark_with_stats(
                prepared.run,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            ),
            probe=lambda prepared: quick_probe(
                prepared.run,
                args.iters,
                cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            ),
            metric_formatter=lambda _config, stats: (
                f"{routed_fc1_swiglu_tflops_from_ms(a.size(0), topk, interleaved_w.size(1) // 2, a.size(1), stats['median_ms']):.1f} TF/s"
            ),
        )

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        token_values = (args.token,) if args.token > 0 else DEFAULT_TOKEN_VALUES

        # The fused kernel consumes only this physical `[gate, up]` layout.
        # Initialize it in place rather than retaining gate, up, and a third
        # interleaved copy of the complete expert-weight set.
        interleaved_w = torch.empty(
            (args.experts, 2 * args.n, args.k),
            device="cuda",
            dtype=torch.bfloat16,
        ).uniform_(-1.0, 1.0)

        baseline_config = GroupedMConfig(
            tile_m=128,
            tile_n=args.baseline_tile_n,
            tile_k=args.baseline_tile_k,
            epilogue_tile_n=32,
            cluster_m=args.baseline_cluster_m,
            mode=args.baseline_mode,
            direct_gmem=False,
        )
        candidates = selected_candidates(args.n, args.k, args)
        if not candidates:
            raise SystemExit("no compatible FC1 configs match the requested filters")
        print(
            f"[config] experts={args.experts} n={args.n} k={args.k} "
            f"topk={args.topk} tokens={token_values}"
        )
        print(f"[config] interleave_block={INTERLEAVE_BLOCK}")
        print(
            f"[baseline] grouped-gemm tile_m=128 tile_n={args.baseline_tile_n} "
            f"tile_k={args.baseline_tile_k} cluster_m={args.baseline_cluster_m} mode={args.baseline_mode}"
        )
        self.make_result_table().print_header()
        sol_report = SolReport(debug=args.debug) if args.sol else None
        for tokens in token_values:
            m = tokens * args.topk
            a = torch.empty(
                (tokens, args.k), device="cuda", dtype=torch.bfloat16
            ).uniform_(-1.0, 1.0)
            router_scores = torch.empty(
                (tokens, args.experts), device="cuda", dtype=torch.bfloat16
            ).uniform_(-1.0, 1.0)
            baseline_out = torch.empty((m, args.n), device="cuda", dtype=torch.bfloat16)
            test_out = torch.empty_like(baseline_out)
            selection = self.resolve_fused_kernel(
                a, interleaved_w, router_scores, args.topk, test_out, args
            )
            kernel_stats, best_config = selection.stats, selection.candidate

            max_abs_diff = 0.0
            if args.check:
                # Materialize the separate baseline operands only for an
                # explicit correctness check, after tuning has completed.
                weight_blocks = interleaved_w.view(
                    args.experts,
                    args.n // INTERLEAVE_BLOCK,
                    2,
                    INTERLEAVE_BLOCK,
                    args.k,
                )
                gate_w = weight_blocks[:, :, 0].reshape(args.experts, args.n, args.k)
                up_w = weight_blocks[:, :, 1].reshape(args.experts, args.n, args.k)
                grouped_a, grouped_layout = build_routed_reference(
                    a, router_scores, args.topk
                )
                run_baseline(
                    grouped_a,
                    gate_w,
                    up_w,
                    grouped_layout,
                    baseline_out,
                    baseline_config,
                )
                del gate_w, up_w, weight_blocks
                selection.prepared.run()
                torch.cuda.synchronize()
                max_abs_diff = (
                    (baseline_out.float() - test_out.float()).abs().max().item()
                )
                passed = torch.allclose(
                    baseline_out.float(),
                    test_out.float(),
                    atol=args.atol,
                    rtol=args.rtol,
                )
                print(
                    f"[check tokens={tokens}] max_abs_diff={max_abs_diff:.6f} "
                    f"atol={args.atol} rtol={args.rtol}"
                )
                print("[check] PASS" if passed else "[check] FAIL")
                if not passed:
                    raise SystemExit(1)

            self.make_result_table().print_row(
                tokens,
                m,
                args.topk,
                args.experts,
                args.n,
                args.k,
                f"{kernel_stats['median_ms']:.4f}",
                f"{routed_fc1_swiglu_tflops_from_ms(tokens, args.topk, args.n, args.k, kernel_stats['median_ms']):.3f}",
                f"{max_abs_diff:.6f}",
                format_candidate(best_config),
            )
            if args.sol:
                sol_report.add(
                    fc1_main_kernel_sol_bounds(
                        router_scores, args.topk, args.n, args.k, best_config
                    ),
                    kernel_stats["median_ms"],
                    label=f"FC1 main kernel (tokens={tokens}, routing excluded)",
                )
        if sol_report:
            sol_report.print()


if __name__ == "__main__":
    Fc1BlockInterleavedSwiGLUBenchmark().run()
