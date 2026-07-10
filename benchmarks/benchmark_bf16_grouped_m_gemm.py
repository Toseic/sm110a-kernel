#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    benchmark_with_stats,
    summarize_samples,
    tflops_from_ms,
)
from sm110a_kernel import GroupedMConfig, bf16_grouped_m_gemm_out
from sm110a_kernel.jit import (
    compile_bf16_grouped_m_gemm,
    compile_bf16_grouped_m_gemm_configs,
)

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
N = 6144
K = 4096
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
    for mode in (0, 1)
    for direct_gmem in (False,)
    if tile_n % epilogue_tile_n == 0
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print per-config performance after the benchmark",
    )
    parser.add_argument(
        "--torch-blas",
        choices=("default", "cublas", "cublaslt"),
        default="default",
    )
    parser.add_argument(
        "--token",
        type=int,
        default=0,
        help="run a single sweep token instead of the default token sweep",
    )
    args = parser.parse_args()

    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.token < 0:
        raise SystemExit("--token must be non-negative")
    return args


def configure_torch_blas(preferred_blas):
    if not hasattr(torch.backends.cuda, "preferred_blas_library"):
        if preferred_blas != "default":
            raise SystemExit(
                "This PyTorch build does not support selecting the CUDA BLAS backend"
            )
        return "unavailable"

    if preferred_blas == "default":
        torch.backends.cuda.preferred_blas_library("default")
    elif preferred_blas == "cublas":
        torch.backends.cuda.preferred_blas_library("cublas")
    elif preferred_blas == "cublaslt":
        torch.backends.cuda.preferred_blas_library("cublaslt")
    else:
        raise SystemExit(f"unsupported --torch-blas value: {preferred_blas}")

    return str(torch.backends.cuda.preferred_blas_library()).split(".")[-1]


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
    )


def benchmark_grouped_kernel(
    a, b, layout, out, config, swap_ab, args
):
    # Compile and launch once before recording CUDA events. This also rejects a
    # configuration that cannot launch on the current device.
    bf16_grouped_m_gemm_out(
        a, b, layout, out, config=config, swap_ab=swap_ab
    )
    torch.cuda.synchronize()
    return benchmark_with_stats(
        lambda: bf16_grouped_m_gemm_out(
            a, b, layout, out, config=config, swap_ab=swap_ab
        ),
        args.warmup,
        args.iters,
        args.samples,
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


def compile_grouped_m_configs(configs, debug):
    if debug:
        configs = list(configs)
        try:
            from tqdm import tqdm
        except ImportError:
            for config in configs:
                print(f"[compile] {format_config(config)}", flush=True)
        else:
            print(f"[compile] compiling {len(configs)} configs...", flush=True)
        compile_bf16_grouped_m_gemm_configs(configs)
    else:
        compile_bf16_grouped_m_gemm_configs(configs)


def benchmark_best_grouped_kernel(a, b, layout, out, args):
    candidates = compatible_candidates(b.size(1), a.size(1))
    if not candidates:
        raise RuntimeError(
            f"no grouped M config supports N={b.size(1)}, K={a.size(1)}"
        )

    if args.debug:
        print("  [debug] per-config performance", flush=True)

    results = []
    best_stats = None
    best_config = None
    best_swap_ab = False
    for config, swap_ab in candidates:
        stats = benchmark_grouped_kernel(
            a, b, layout, out, config, swap_ab, args
        )
        results.append((stats, config, swap_ab))
        tflops = tflops_from_ms(
            a.size(0), b.size(1), a.size(1), stats["median_ms"]
        )
        marker = ""
        if best_stats is None or stats["median_ms"] < best_stats["median_ms"]:
            best_stats = stats
            best_config = config
            best_swap_ab = swap_ab
            marker = " *"
        if args.debug:
            print(
                f"  {format_candidate(config, swap_ab):<40}"
                f" min={stats['min_ms']:.4f}ms"
                f" median={stats['median_ms']:.4f}ms"
                f" {tflops:.1f} TF/s{marker}",
                flush=True,
            )
    if args.debug:
        print(
            f"  -> best: {format_candidate(best_config, best_swap_ab)}",
            flush=True,
        )
    return best_stats, best_config, best_swap_ab, results


_MODE_LABELS = {0: "np", 1: "pe", 2: "cl"}


def format_config(config):
    base = (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/e{config.epilogue_tile_n}/c{config.cluster_m}"
        f"/{_MODE_LABELS.get(config.mode, config.mode)}"
    )
    return base + "+dg" if getattr(config, "direct_gmem", False) else base


def format_candidate(config, swap_ab):
    return f"{format_config(config)}/{'swap' if swap_ab else 'noswap'}"


def max_valid_abs_diff(ref, test):
    return (ref.float() - test.float()).abs().max().item()


def print_header():
    print(
        f"{'Tok/E':<8}{'Pad/E':<8}{'M':<10}{'E':<8}{'N':<8}{'K':<8}"
        f"{'KernelMed(ms)':<16}{'Kernel TF/s':<16}"
        f"{'BaselineMed(ms)':<16}{'Baseline TF/s':<16}"
        f"{'Speedup':<12}{'MaxDiff':<12}{'BestConfig':<40}"
    )


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
    kernel_tflops = tflops_from_ms(valid_m, n, k, kernel_stats["median_ms"])
    baseline_tflops = tflops_from_ms(
        valid_m, n, k, baseline_stats["median_ms"]
    )
    speedup = baseline_stats["median_ms"] / kernel_stats["median_ms"]
    print(
        f"{tokens_per_expert:<8}{padded_tokens:<8}{valid_m:<10}"
        f"{experts:<8}{n:<8}{k:<8}"
        f"{kernel_stats['median_ms']:<16.4f}{kernel_tflops:<16.3f}"
        f"{baseline_stats['median_ms']:<16.4f}{baseline_tflops:<16.3f}"
        f"{speedup:<12.2f}{diff:<12.6f}"
        f"{format_candidate(best_config, best_swap_ab):<40}"
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


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    args = parse_args()
    tokens_per_expert_values = (
        (args.token,) if args.token > 0 else DEFAULT_TOKENS_PER_EXPERT_VALUES
    )
    torch.manual_seed(1234)

    actual_blas = configure_torch_blas(args.torch_blas)
    print(f"[torch] preferred BLAS backend: {actual_blas}")
    print(
        f"[workload] experts={EXPERTS} n={N} k={K} "
        f"tokens_per_expert={tokens_per_expert_values}"
    )
    print(
        "[autotune] configs="
        + ", ".join(
            format_candidate(config, swap_ab)
            for config, swap_ab in compatible_candidates(N, K)
        )
    )
    compile_grouped_m_configs(
        dict.fromkeys(
            config for config, _ in compatible_candidates(N, K)
        ),
        args.debug,
    )

    b = torch.empty((EXPERTS, N, K), device="cuda", dtype=torch.bfloat16)
    b.uniform_(-1.0, 1.0)
    print_header()
    for tokens_per_expert in tokens_per_expert_values:
        a, layout, out_kernel, out_baseline = make_workload(
            tokens_per_expert, b
        )
        baseline_stats = benchmark_grouped_baseline(
            a,
            b,
            tokens_per_expert,
            out_baseline,
            args,
        )
        (
            kernel_stats,
            best_config,
            best_swap_ab,
            config_results,
        ) = benchmark_best_grouped_kernel(a, b, layout, out_kernel, args)
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
            bf16_grouped_m_gemm_out(
                a,
                b,
                layout,
                out_kernel,
                config=best_config,
                swap_ab=best_swap_ab,
            )
            run_grouped_baseline(a, b, tokens_per_expert, out_baseline)
            diff = max_valid_abs_diff(out_baseline, out_kernel)
        print_result(
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


if __name__ == "__main__":
    main()
