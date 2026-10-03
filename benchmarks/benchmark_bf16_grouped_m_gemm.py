#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    BenchmarkBase,
    SolReport,
    benchmark_with_stats,
    quick_probe,
    tflops_from_ms,
    sol_bounds,
)
from sm110a_kernel import (
    GroupedMConfig,
    compile_bf16_grouped_m_gemm,
    bf16_grouped_m_gemm_out,
    prepare_grouped_m_metadata,
)
from sm110a_kernel.runner import AutotuneRunner, PreparedKernel
from deepgemm_baseline import bf16_grouped_nt as deepgemm_bf16_grouped_nt

'''
roofline analysis:
Thor: 
250Tflops/s & 0.273TB/s
peak AI: 250 / 0.273 = 915 flops/Byte

grouped gemm: [m, k]@[e, k, n]
flops: 2 * m * n * k flops
memory: 2 * m * k + 2 * e * k * n + 2 * n * k -> 2 * e * k * n Byte
AI: 2 * m * n * k / 2 * e * k * n = m / e flops/Byte

e = 256
peak_m / e = 915
peak_m = 915 * e

'''




EXPERTS = 256
N = 4096
K = 6144
DEFAULT_TOKENS_PER_EXPERT_VALUES = (64, 128, 256, 512, 1024)
GROUPED_M_CONFIGS = tuple(
    GroupedMConfig(
        tile_m=128,
        tile_n=tile_n,
        tile_k=tile_k,
        epilogue_tile_n=epilogue_tile_n,
        cluster_m=cluster_m,
        mode=mode,
        direct_gmem=direct_gmem,
    )
    for tile_n in (128, 256)
    for tile_k in (64, 128)
    for epilogue_tile_n in (32, 64)
    for cluster_m in (1, 2)
    for mode in (0, 1, 2)
    for direct_gmem in (False,)
    if tile_n % epilogue_tile_n == 0
)
def build_layout(tokens_per_expert, experts):
    layout = torch.arange(experts, device="cuda", dtype=torch.int32)
    return layout.repeat_interleave(tokens_per_expert)


def make_workload(tokens_per_expert, b):
    experts, n, k = b.shape
    layout = build_layout(tokens_per_expert, experts)
    valid_m = experts * tokens_per_expert
    a = torch.empty((valid_m, k), device="cuda", dtype=torch.bfloat16)
    a.uniform_(-1.0, 1.0)
    out_kernel = torch.empty((valid_m, n), device="cuda", dtype=torch.bfloat16)
    out_baseline = torch.empty((valid_m, n), device="cuda", dtype=torch.bfloat16)
    return a, layout, out_kernel, out_baseline


def make_deepgemm_workload(a, layout, b, tokens_per_expert):
    """Pad each contiguous expert segment for DeepGEMM's grouped contract.

    DeepGEMM requires its per-expert rows to be 128-aligned; the native
    benchmark deliberately stores the same segments compactly.  This packing
    is setup work, outside the timed region, just as the native metadata is.
    """
    experts, n, k = b.shape
    padded_tokens = math.ceil(tokens_per_expert / 128) * 128
    padded_a = torch.zeros(
        (experts * padded_tokens, k), device=a.device, dtype=a.dtype
    )
    padded_layout = torch.full(
        (experts * padded_tokens,), -1, device=a.device, dtype=torch.int32
    )
    padded_a = padded_a.view(experts, padded_tokens, k)
    padded_a[:, :tokens_per_expert].copy_(a.view(experts, tokens_per_expert, k))
    padded_layout = padded_layout.view(experts, padded_tokens)
    padded_layout[:, :tokens_per_expert] = torch.arange(
        experts, device=a.device, dtype=torch.int32
    ).unsqueeze(1)
    padded_out = torch.empty(
        (experts * padded_tokens, n), device=a.device, dtype=a.dtype
    )

    def valid_output():
        return padded_out.view(experts, padded_tokens, n)[:, :tokens_per_expert].reshape(-1, n)

    return padded_a.reshape(-1, k), padded_layout.reshape(-1), padded_out, valid_output


def run_grouped_baseline(a, b, tokens_per_expert, out):
    for expert in range(b.size(0)):
        offset = expert * tokens_per_expert
        a_slice = a[offset : offset + tokens_per_expert]
        out_slice = out[offset : offset + tokens_per_expert]
        torch.mm(a_slice, b[expert].t(), out=out_slice)


def benchmark_grouped_baseline(a, b, tokens_per_expert, out, args):
    return benchmark_with_stats(
        lambda: run_grouped_baseline(a, b, tokens_per_expert, out),
        args.warmup,
        args.iters,
        args.samples,
        cold_l2_flush=getattr(args, "_cold_l2_flush", None),
    )


def benchmark_grouped_kernel(
    a, b, layout, out, config, swap_ab, args
):
    metadata = prepare_grouped_m_metadata(
        layout, b.size(0), b.size(1), config=config, swap_ab=swap_ab)
    # Compile and launch once before recording CUDA events. This also rejects a
    # configuration that cannot launch on the current device.
    bf16_grouped_m_gemm_out(
        a, b, metadata, out, config=config, swap_ab=swap_ab
    )
    torch.cuda.synchronize()
    return benchmark_with_stats(
        lambda: bf16_grouped_m_gemm_out(
            a, b, metadata, out, config=config, swap_ab=swap_ab
        ),
        args.warmup,
        args.iters,
        args.samples,
        cold_l2_flush=getattr(args, "_cold_l2_flush", None),
    )


def probe_grouped_kernel(a, b, layout, out, config, swap_ab, args):
    metadata = prepare_grouped_m_metadata(
        layout, b.size(0), b.size(1), config=config, swap_ab=swap_ab)
    bf16_grouped_m_gemm_out(
        a, b, metadata, out, config=config, swap_ab=swap_ab
    )
    torch.cuda.synchronize()
    return quick_probe(
        lambda: bf16_grouped_m_gemm_out(
            a, b, metadata, out, config=config, swap_ab=swap_ab
        ),
        args.iters,
        cold_l2_flush=getattr(args, "_cold_l2_flush", None),
    )


def compatible_candidates(n, k):
    candidates = []
    for config in GROUPED_M_CONFIGS:
        if k % config.tile_k:
            continue
        if n % config.tile_n == 0:
            candidates.append((config, False))
        # if config.cluster_m == 1 and n % (config.cluster_m * config.tile_m) == 0:
        #     candidates.append((config, True)) # disable swapab [tmp]
    return tuple(candidates)


_MODE_LABELS = {0: "np", 1: "pe", 2: "clc"}


def format_config(config):
    base = (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/e{config.epilogue_tile_n}/c{config.cluster_m}"
        f"/{_MODE_LABELS[config.mode]}"
    )
    return base + "+dg" if getattr(config, "direct_gmem", False) else base


def format_candidate(config, swap_ab):
    return f"{format_config(config)}/{'swap' if swap_ab else 'noswap'}"


def max_valid_abs_diff(ref, test):
    return (ref.float() - test.float()).abs().max().item()


def print_result(
    tokens_per_expert,
    padded_tokens,
    experts,
    n,
    k,
    kernel_stats,
    baseline_stats,
    diff,
    best_config,
    best_swap_ab,
):
    valid_m = experts * tokens_per_expert
    return (
        tokens_per_expert,
        padded_tokens,
        valid_m,
        experts,
        n,
        k,
        f"{kernel_stats['median_ms']:.4f}",
        f"{tflops_from_ms(valid_m, n, k, kernel_stats['median_ms']):.3f}",
        f"{baseline_stats['median_ms']:.4f}",
        f"{tflops_from_ms(valid_m, n, k, baseline_stats['median_ms']):.3f}",
        f"{baseline_stats['median_ms'] / kernel_stats['median_ms']:.2f}",
        f"{diff:.6f}",
        format_candidate(best_config, best_swap_ab),
    )


def grouped_m_gemm_sol_bounds(tokens_per_expert, experts, n, k, config):
    """SOL model for this benchmark's uniform expert-contiguous workload."""
    m = tokens_per_expert * experts
    task_count = (
        experts
        * math.ceil(tokens_per_expert / (config.cluster_m * config.tile_m))
        * math.ceil(n / config.tile_n)
    )
    flops = 2 * m * n * k
    a_bytes = 2 * m * k
    b_bytes = 2 * experts * n * k
    c_bytes = 2 * m * n
    tile_read_bytes = (
        2
        * task_count
        * (config.cluster_m * config.tile_m + config.tile_n)
        * k
    )
    return sol_bounds(
        flops, a_bytes + b_bytes + c_bytes, a_bytes + b_bytes, tile_read_bytes
    )


def print_debug_results(results, experts, n, k):
    print()
    print("[debug] per-config performance")
    print(
        f"{'Tok/E':<8}{'Config':<40}{'Median(ms)':<14}"
        f"{'TFLOPS':<14}{'Best':<6}"
    )
    for tokens_per_expert, stats, config, swap_ab, is_best in results:
        valid_m = experts * tokens_per_expert
        tflops = tflops_from_ms(
            valid_m, n, k, stats["median_ms"]
        )
        print(
            f"{tokens_per_expert:<8}"
            f"{format_candidate(config, swap_ab):<40}"
            f"{stats['median_ms']:<14.4f}{tflops:<14.3f}"
            f"{'*' if is_best else '':<6}"
        )


class Bf16GroupedMGemmBenchmark(BenchmarkBase):
    def make_result_table(self):
        return self.make_table_printer(
            (
                ("Tok/E", 8),
                ("Pad/E", 8),
                ("M", 10),
                ("E", 8),
                ("N", 8),
                ("K", 8),
                ("KernelMed(ms)", 16),
                ("Kernel TF/s", 16),
                ("BaselineMed(ms)", 16),
                ("Baseline TF/s", 16),
                ("Speedup", 12),
                ("MaxDiff", 12),
                ("BestConfig", 40),
            )
        )

    def build_parser(self):
        parser = super().build_parser()
        self.add_timing_args(parser, warmup=2, iters=3, samples=3)
        parser.add_argument("--check", action="store_true")
        parser.add_argument(
            "--debug",
            action="store_true",
            help="print per-config performance after the benchmark",
        )
        self.add_torch_blas_arg(parser)
        parser.add_argument(
            "--baseline", choices=("torch", "deepgemm"), default="torch",
            help="reference implementation to benchmark",
        )
        self.add_cache_args(parser)
        self.add_pre_parallel_jit_arg(parser)
        self.add_sol_arg(parser)
        parser.add_argument(
            "--token",
            type=int,
            default=0,
            help="run a single sweep token instead of the default token sweep",
        )
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if args.token < 0:
            raise SystemExit("--token must be non-negative")
        return args

    def run_with_args(self, args):
        tokens_per_expert_values = (
            (args.token,) if args.token > 0 else DEFAULT_TOKENS_PER_EXPERT_VALUES
        )

        actual_blas = self.configure_torch_blas(args.torch_blas)
        print(f"[torch] preferred BLAS backend: {actual_blas}")
        if args.baseline == "deepgemm":
            print("[baseline] locally patched DeepGEMM contiguous grouped-M BF16 GEMM")
        print(
            f"[workload] experts={EXPERTS} n={N} k={K} "
            f"tokens_per_expert={tokens_per_expert_values}"
        )
        b = torch.empty((EXPERTS, N, K), device="cuda", dtype=torch.bfloat16)
        b.uniform_(-1.0, 1.0)
        self.make_result_table().print_header()
        sol_report = SolReport(debug=args.debug) if args.sol else None
        for tokens_per_expert in tokens_per_expert_values:
            a, layout, out_kernel, out_baseline = make_workload(
                tokens_per_expert, b
            )
            if args.baseline == "deepgemm":
                deep_a, deep_layout, deep_out, deep_valid_output = make_deepgemm_workload(
                    a, layout, b, tokens_per_expert
                )
                baseline = deepgemm_bf16_grouped_nt(deep_a, b, deep_out, deep_layout)
            else:
                deep_valid_output = lambda: out_baseline
                baseline = lambda: run_grouped_baseline(
                    a, b, tokens_per_expert, out_baseline
                )
            baseline_stats = benchmark_with_stats(
                baseline, args.warmup, args.iters, args.samples,
                cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            )
            selection = self.resolve_grouped_kernel(a, b, layout, out_kernel, args)
            kernel_stats = selection.stats
            best_config, best_swap_ab = selection.candidate
            grouped_m_tile = (
                best_config.tile_n
                if best_swap_ab
                else best_config.cluster_m * best_config.tile_m
            )
            padded_tokens = (
                math.ceil(tokens_per_expert / grouped_m_tile) * grouped_m_tile
            )
            diff = 0.0
            if args.check:
                selection.prepared.run()
                baseline()
                diff = max_valid_abs_diff(deep_valid_output(), out_kernel)
            self.make_result_table().print_row(
                *print_result(
                    tokens_per_expert,
                    padded_tokens,
                    EXPERTS,
                    N,
                    K,
                    kernel_stats,
                    baseline_stats,
                    diff,
                    best_config,
                    best_swap_ab,
                )
            )
            if args.sol:
                sol_report.add(
                    grouped_m_gemm_sol_bounds(
                        tokens_per_expert, EXPERTS, N, K, best_config
                    ),
                    kernel_stats["median_ms"],
                    label=f"grouped-M GEMM (tokens/expert={tokens_per_expert})",
                )
        if sol_report:
            sol_report.print()

    def resolve_grouped_kernel(self, a, b, layout, out, args):
        candidates = compatible_candidates(b.size(1), a.size(1))
        if not candidates:
            raise RuntimeError(
                f"no grouped M config supports N={b.size(1)}, K={a.size(1)}"
            )

        tokens_per_expert = layout.numel() // b.size(0)
        n = b.size(1)
        k = a.size(1)
        def prepare(candidate):
            config, swap_ab = candidate
            metadata = prepare_grouped_m_metadata(
                layout, b.size(0), b.size(1), config=config, swap_ab=swap_ab
            )
            run = lambda: bf16_grouped_m_gemm_out(
                a, b, metadata, out, config=config, swap_ab=swap_ab
            )
            run()
            torch.cuda.synchronize()
            return PreparedKernel(candidate, run)

        options = self.make_tune_options(
            args,
            role="ours",
            pre_parallel_jit=args.pre_parallel_jit,
        )
        return AutotuneRunner(options).resolve(
            candidates,
            prepare=prepare,
            benchmark=lambda prepared: benchmark_with_stats(
                prepared.run, args.warmup, args.iters, args.samples,
                cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            ),
            probe=lambda prepared: quick_probe(
                prepared.run, args.iters,
                cold_l2_flush=getattr(args, "_cold_l2_flush", None),
            ),
            format_candidate=lambda candidate: format_candidate(
                candidate[0], candidate[1]
            ),
            namespace="bf16_grouped_m_gemm", key=(n, k, tokens_per_expert),
            metric_formatter=lambda _candidate, stats: (
                f"{tflops_from_ms(a.size(0), n, k, stats['median_ms']):.1f} TF/s"
            ),
            pre_jit=lambda candidate: compile_bf16_grouped_m_gemm(candidate[0]),
        )


def main():
    Bf16GroupedMGemmBenchmark().run()


if __name__ == "__main__":
    main()
