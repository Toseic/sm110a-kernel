#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import BenchmarkBase, benchmark_with_stats, quick_probe
import sm110a_kernel as ext
from sm110a_kernel import (
    Fc1BlockInterleavedSwiGLUConfig,
    GroupedMConfig,
    bf16_grouped_m_gemm_out,
    compile_fc1_block_interleaved_swiglu_configs,
)


DEFAULT_EXPERTS = 256
DEFAULT_N = 2048
DEFAULT_K = 6144
DEFAULT_TOKENS_PER_EXPERT_VALUES = (64, 128, 256, 512, 1024)
INTERLEAVE_BLOCK = 32
FC1_CONFIGS = tuple(
    Fc1BlockInterleavedSwiGLUConfig(
        tile_m=128,
        tile_n=tile_n,
        tile_k=tile_k,
        cluster_m=cluster_m,
        direct_gmem=direct_gmem,
    )
    for tile_n in (64, 128, 256)
    for tile_k in (64, 128)
    for cluster_m in (1, 2)
    for direct_gmem in (False, True)
    if tile_n % (8 * cluster_m) == 0
    and tile_n % (2 * INTERLEAVE_BLOCK) == 0
)


def fc1_swiglu_tflops_from_ms(m, n, k, elapsed_ms):
    """Return GEMM-equivalent TFLOP/s for both FC1 SwiGLU projections."""
    flops = 4.0 * m * n * k  # gate and up projections, each 2 * M * N * K
    return flops / (elapsed_ms * 1.0e-3) / 1.0e12


def build_grouped_layout(tokens_per_expert, experts):
    layout = torch.arange(experts, device="cuda", dtype=torch.int32)
    return layout.repeat_interleave(tokens_per_expert)


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


def deinterleave_gate_up_output(interleaved_out, block_size=INTERLEAVE_BLOCK):
    if interleaved_out.ndim != 2:
        raise ValueError("interleaved output must have shape [m, 2n]")
    m, two_n = interleaved_out.shape
    if two_n % (2 * block_size) != 0:
        raise ValueError(
            f"interleaved width must be divisible by {2 * block_size}"
        )

    logical_n = two_n // 2
    view = interleaved_out.view(m, logical_n // block_size, 2, block_size)
    gate = view[:, :, 0, :].reshape(m, logical_n)
    up = view[:, :, 1, :].reshape(m, logical_n)
    return gate, up


def swiglu(gate, up):
    return torch.nn.functional.silu(gate.float()) * up.float()


def run_baseline(a, gate_w, up_w, layout, out, config):
    gate_acc = torch.empty_like(out)
    up_acc = torch.empty_like(out)
    bf16_grouped_m_gemm_out(a, gate_w, layout, gate_acc, config=config, swap_ab=False)
    bf16_grouped_m_gemm_out(a, up_w, layout, up_acc, config=config, swap_ab=False)
    out.copy_(swiglu(gate_acc, up_acc).to(torch.bfloat16))


def run_interleaved_reference(a, interleaved_w, layout, out, config):
    interleaved_out = torch.empty(
        (a.size(0), interleaved_w.size(1)), device="cuda", dtype=torch.bfloat16
    )
    bf16_grouped_m_gemm_out(
        a, interleaved_w, layout, interleaved_out, config=config, swap_ab=False
    )
    gate_acc, up_acc = deinterleave_gate_up_output(interleaved_out)
    out.copy_(swiglu(gate_acc, up_acc).to(torch.bfloat16))


def resolve_fused_impl():
    func = getattr(ext, "fc1_block_interleaved_swiglu_out", None)
    if callable(func):
        return "fc1_block_interleaved_swiglu_out", func
    return None, None


def run_fused(a, interleaved_w, layout, out, config):
    impl_name, impl = resolve_fused_impl()
    if impl is None:
        raise RuntimeError("fused fc1_block_interleaved_swiglu_out binding is unavailable")
    impl(a, interleaved_w, layout, out, config=config)
    return impl_name


def format_candidate(config):
    epilogue = "direct-gmem" if config.direct_gmem else "tmem-smem-tma"
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}/c{config.cluster_m}/"
        f"{epilogue}"
    )


def compatible_candidates(n, k):
    return tuple(
        config
        for config in FC1_CONFIGS
        if n % (config.tile_n // 2) == 0 and k % config.tile_k == 0
    )


def compile_fc1_configs(configs, debug):
    if debug:
        configs = list(configs)
        print(f"[compile] compiling {len(configs)} configs...", flush=True)
        compile_fc1_block_interleaved_swiglu_configs(configs)
    else:
        compile_fc1_block_interleaved_swiglu_configs(configs)


def benchmark_fused_kernel(a, interleaved_w, layout, out, config, args):
    run_fused(a, interleaved_w, layout, out, config)
    torch.cuda.synchronize()
    return benchmark_with_stats(
        lambda: run_fused(a, interleaved_w, layout, out, config),
        args.warmup,
        args.iters,
        args.samples,
    )


def probe_fused_kernel(a, interleaved_w, layout, out, config):
    run_fused(a, interleaved_w, layout, out, config)
    torch.cuda.synchronize()
    return quick_probe(lambda: run_fused(a, interleaved_w, layout, out, config))


class Fc1BlockInterleavedSwiGLUBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = super().build_parser()
        self.add_timing_args(parser, warmup=2, iters=3, samples=3)
        parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
        parser.add_argument("--tokens-per-expert", type=int, default=0)
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
            default=1,
            help="grouped GEMM baseline mode",
        )
        parser.add_argument(
            "--token",
            type=int,
            default=0,
            help="run a single sweep token instead of the default token sweep",
        )
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if args.experts <= 0:
            raise SystemExit("--experts must be positive")
        if args.tokens_per_expert < 0:
            raise SystemExit("--tokens-per-expert must be non-negative")
        if args.token < 0:
            raise SystemExit("--token must be non-negative")
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
                ("Tok/E", 8),
                ("M", 10),
                ("E", 8),
                ("N", 8),
                ("K", 8),
                ("KernelMed(ms)", 16),
                ("Kernel TF/s", 16),
                ("MaxDiff", 12),
                ("BestConfig", 36),
                ("Path", 30),
            )
        )

    def benchmark_best_fused_kernel(self, a, interleaved_w, layout, out, args):
        candidates = compatible_candidates(interleaved_w.size(1) // 2, a.size(1))
        if not candidates:
            raise RuntimeError(
                f"no fc1 config supports N={interleaved_w.size(1) // 2}, K={a.size(1)}"
            )

        if args.debug:
            print("  [debug] per-config performance", flush=True)

        best_config, best_stats, _raw_results = self.autotune_candidates(
            candidates,
            args=args,
            benchmark_candidate=lambda config: benchmark_fused_kernel(
                a, interleaved_w, layout, out, config, args
            ),
            format_candidate=format_candidate,
            probe_candidate=lambda config: probe_fused_kernel(
                a, interleaved_w, layout, out, config
            ),
            metric_formatter=lambda _config, stats: (
                f"{fc1_swiglu_tflops_from_ms(a.size(0), interleaved_w.size(1) // 2, a.size(1), stats['median_ms']):.1f} TF/s"
            ),
            candidate_width=36,
            flush=True,
        )
        return best_stats, best_config

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        tokens_per_expert_values = (
            (args.token,)
            if args.token > 0
            else ((args.tokens_per_expert,) if args.tokens_per_expert > 0 else DEFAULT_TOKENS_PER_EXPERT_VALUES)
        )

        gate_w = torch.empty(
            (args.experts, args.n, args.k), device="cuda", dtype=torch.bfloat16
        ).uniform_(-1.0, 1.0)
        up_w = torch.empty(
            (args.experts, args.n, args.k), device="cuda", dtype=torch.bfloat16
        ).uniform_(-1.0, 1.0)
        interleaved_w = interleave_gate_up_weights(gate_w, up_w)

        baseline_config = GroupedMConfig(
            tile_m=128,
            tile_n=args.baseline_tile_n,
            tile_k=args.baseline_tile_k,
            epilogue_tile_n=32,
            cluster_m=args.baseline_cluster_m,
            mode=args.baseline_mode,
            direct_gmem=False,
        )
        candidates = compatible_candidates(args.n, args.k)
        print(
            f"[config] experts={args.experts} n={args.n} k={args.k} "
            f"tokens_per_expert={tokens_per_expert_values}"
        )
        print(f"[config] interleave_block={INTERLEAVE_BLOCK}")
        print(
            f"[baseline] grouped-gemm tile_m=128 tile_n={args.baseline_tile_n} "
            f"tile_k={args.baseline_tile_k} cluster_m={args.baseline_cluster_m} mode={args.baseline_mode}"
        )
        print(
            "[autotune] configs="
            + ", ".join(format_candidate(config) for config in candidates)
        )
        compile_fc1_configs(dict.fromkeys(candidates), args.debug)
        self.make_result_table().print_header()
        for tokens_per_expert in tokens_per_expert_values:
            m = args.experts * tokens_per_expert
            layout = build_grouped_layout(tokens_per_expert, args.experts)
            a = torch.empty((m, args.k), device="cuda", dtype=torch.bfloat16).uniform_(-1.0, 1.0)
            baseline_out = torch.empty((m, args.n), device="cuda", dtype=torch.bfloat16)
            test_out = torch.empty_like(baseline_out)
            kernel_stats, best_config = self.benchmark_best_fused_kernel(
                a, interleaved_w, layout, test_out, args
            )

            max_abs_diff = 0.0
            path_name = "fused"
            if args.check:
                run_baseline(a, gate_w, up_w, layout, baseline_out, baseline_config)
                path_name = run_fused(a, interleaved_w, layout, test_out, best_config)
                torch.cuda.synchronize()
                max_abs_diff = (baseline_out.float() - test_out.float()).abs().max().item()
                passed = torch.allclose(
                    baseline_out.float(),
                    test_out.float(),
                    atol=args.atol,
                    rtol=args.rtol,
                )
                print(
                    f"[check tok/e={tokens_per_expert}] max_abs_diff={max_abs_diff:.6f} "
                    f"atol={args.atol} rtol={args.rtol}"
                )
                print("[check] PASS" if passed else "[check] FAIL")
                if not passed:
                    raise SystemExit(1)

            self.make_result_table().print_row(
                tokens_per_expert,
                m,
                args.experts,
                args.n,
                args.k,
                f"{kernel_stats['median_ms']:.4f}",
                f"{fc1_swiglu_tflops_from_ms(m, args.n, args.k, kernel_stats['median_ms']):.3f}",
                f"{max_abs_diff:.6f}",
                format_candidate(best_config),
                path_name,
            )


if __name__ == "__main__":
    Fc1BlockInterleavedSwiGLUBenchmark().run()
