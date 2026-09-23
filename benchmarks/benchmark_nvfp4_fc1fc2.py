#!/usr/bin/env python3
"""Fixed-shape correctness/performance test for the fused FC1+FC2 kernel."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    SolReport,
    benchmark_with_stats,
    powerlaw_router_scores,
    quick_probe,
    sol_bounds,
)

from sm110a_kernel import (  # noqa: E402
    Nvfp4Fc1Config,
    Nvfp4Fc1Fc2Config,
    Nvfp4Fc2Config,
    compile_nvfp4_fc1,
    compile_nvfp4_fc1fc2,
    compile_nvfp4_fc2,
    nvfp4_fc1_out,
    nvfp4_fc2_out,
    prepare_nvfp4_fc1_metadata,
    prepare_nvfp4_fc2_metadata,
    quantize_nvfp4_rows,
)
from sm110a_kernel.runner import (  # noqa: E402
    AutotuneRunner,
    PreparedKernel,
    TuneOptions,
)


H = 6144
INTERMEDIATE = 2048
NVFP4_MMA_PEAK_TFLOPS = 4.0 * 250.0
ROUTER_POWERLAW_ALPHA = 0.8


def fused_tflops(m, elapsed_ms):
    """Two FC1 projections plus FC2: 6*M*H*I floating-point operations."""
    flops = 6.0 * m * H * INTERMEDIATE
    return flops / (elapsed_ms * 1.0e9)


def format_config(config):
    return (
        f"buffer_i={config.buffer_i},h_split={config.h_split},"
        f"cluster_pairs={config.cluster_pairs},"
        f"fuse_routing={int(config.fuse_routing)},"
        f"cache_source_rows={config.cache_source_rows}"
    )


def compatible_configs(m, args):
    """Return config-filtered candidates that can launch for this M."""
    # Splitting I into multiple FC2 reductions rounds each partial sum to
    # BF16 and is not bit-equivalent to the unfused FP32 accumulation. Keep
    # those experimental configs addressable explicitly, but never let the
    # correctness-preserving autotuner select them.
    buffers = (args.buffer_i,) if args.buffer_i else (2048,)
    h_splits = (args.h_split,) if args.h_split else (1, 2, 3, 4)
    cluster_pairs = (
        (args.cluster_pairs,) if args.cluster_pairs else (1, 2, 4)
    )
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    row_tasks = getattr(args, "_row_task_count", m // 256)
    candidates = []
    for buffer_i in buffers:
        for h_split in h_splits:
            for pairs in cluster_pairs:
                caches = (
                    (0, 1, 2) if args.cache_source_rows < 0 else
                    (args.cache_source_rows,)
                ) if args.fuse_routing else (False,)
                for cache_source_rows in caches:
                    if h_split > 1 and (buffer_i != 2048 or pairs != 1):
                        continue
                    config = Nvfp4Fc1Fc2Config(
                        buffer_i=buffer_i,
                        h_split=h_split,
                        cluster_pairs=pairs,
                        fuse_routing=args.fuse_routing,
                        cache_source_rows=cache_source_rows,
                    )
                    try:
                        config.validate()
                    except ValueError:
                        continue
                    if h_split > 1:
                        tasks = row_tasks * h_split
                        if h_split == 3 and row_tasks == 4:
                            tasks = 10
                        if (tasks > sm_count // (2 * pairs) and h_split != 2):
                            continue
                    candidates.append(config)
    if not candidates:
        raise SystemExit("no compatible fused FC1+FC2 configs match the filters")
    return tuple(candidates)


def fused_sol_bounds(m, config, *, tokens=None, experts=1, row_tasks=None):
    """Traffic-aware SOL for the fixed H=6144, I=2048 fused pipeline."""
    nvfp4_bytes_per_element = 0.5 + 1.0 / 16.0
    flops = 6.0 * m * H * INTERMEDIATE
    tokens = m if tokens is None else tokens
    a_bytes = tokens * H * nvfp4_bytes_per_element
    w1_bytes = experts * 2 * INTERMEDIATE * H * nvfp4_bytes_per_element
    w2_bytes = experts * H * INTERMEDIATE * nvfp4_bytes_per_element
    out_bytes = tokens * H * 2
    row_tasks = math.ceil(m / 256) if row_tasks is None else row_tasks

    # FC1 reloads A once per logical 64-column output tile. Each row task
    # consumes the complete W1 and W2 weights across its distributed workers.
    fc1_a_tile_reads = row_tasks * 256 * H * nvfp4_bytes_per_element * (
        INTERMEDIATE // 64
    )
    tile_read_bytes = (
        fc1_a_tile_reads
        + row_tasks * (w1_bytes + w2_bytes)
        + out_bytes
    )
    workspace_store = 0
    streams_intermediate = (
        config.buffer_i == INTERMEDIATE and config.cluster_pairs == 1
    )
    if config.h_split > 1 or streams_intermediate:
        tasks = row_tasks * config.h_split
        if config.h_split == 3 and row_tasks == 4:
            tasks = 10
        per_row_intermediate = INTERMEDIATE * nvfp4_bytes_per_element
        workspace_store = m * per_row_intermediate
        reloads = tasks if config.h_split > 1 else row_tasks
        if config.h_split == 1 and row_tasks == 12:
            reloads = 18
        tile_read_bytes += (
            workspace_store + reloads * 256 * per_row_intermediate
        )

    compulsory_bytes = a_bytes + w1_bytes + w2_bytes + out_bytes + workspace_store
    return sol_bounds(
        flops,
        compulsory_bytes,
        a_bytes + w1_bytes + w2_bytes,
        tile_read_bytes,
        mma_peak_tflops=NVFP4_MMA_PEAK_TFLOPS,
    )


def quantize_fc1_weight(n, k):
    gate = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    up = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    interleaved = torch.stack(
        (gate.view(n // 64, 64, k), up.view(n // 64, 64, k)), dim=1
    ).reshape(2 * n, k)
    packed, scales = quantize_nvfp4_rows(interleaved)
    return packed.unsqueeze(0), scales


def quantize_expert_weights(experts, h, intermediate):
    w1q = sfw1 = w2q = sfw2 = None
    for expert in range(experts):
        w1, sfw1 = quantize_fc1_weight(intermediate, h)
        w2, sfw2 = quantize_nvfp4_rows(
            torch.randn((h, intermediate), device="cuda", dtype=torch.bfloat16)
        )
        if w1q is None:
            w1q = torch.empty(
                (experts, *w1.shape[1:]), device="cuda", dtype=w1.dtype
            )
            sfw1_all = torch.empty(
                experts * sfw1.numel(), device="cuda", dtype=sfw1.dtype
            )
            w2q = torch.empty(
                (experts, *w2.shape), device="cuda", dtype=w2.dtype
            )
            sfw2_all = torch.empty(
                experts * sfw2.numel(), device="cuda", dtype=sfw2.dtype
            )
        w1q[expert].copy_(w1[0])
        sfw1_all[expert * sfw1.numel() : (expert + 1) * sfw1.numel()].copy_(sfw1)
        w2q[expert].copy_(w2)
        sfw2_all[expert * sfw2.numel() : (expert + 1) * sfw2.numel()].copy_(sfw2)
        del w1, sfw1, w2, sfw2
    return w1q, sfw1_all, w2q, sfw2_all


def time_ms(call, warmup, iterations):
    for _ in range(warmup):
        call()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--fuse-routing", action="store_true")
    parser.add_argument(
        "--cache-source-rows", type=int, choices=(-1, 0, 1, 2), default=-1,
        help="source-row cache mode: -1 autotune, 0 none, 1 SFA, 2 A+SFA",
    )
    parser.add_argument("--tokens", type=int, default=0,
                        help="token count in routed mode; 0 derives it from M/topk")
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--buffer-i", type=int, choices=(0, 256, 512, 1024, 2048), default=0,
        help="I-buffer size, or 0 to autotune",
    )
    parser.add_argument(
        "--h-split", type=int,
        choices=(0, 1, 2, 3, 4), default=0,
    )
    parser.add_argument(
        "--cluster-pairs", type=int, choices=(0, 1, 2, 4), default=0,
        help="cooperating CTA pairs, or 0 to autotune",
    )
    parser.add_argument("--force-sweep", action="store_true")
    parser.add_argument("--cache-path")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sol", action="store_true")
    parser.add_argument(
        "--pre-parallel-jit", dest="pre_parallel_jit", action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-pre-parallel-jit", dest="pre_parallel_jit", action="store_false"
    )
    parser.add_argument("--no-accuracy", action="store_true")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--max-relative-rmse", type=float, default=5e-3)
    args = parser.parse_args()
    if args.m <= 0:
        parser.error("--m must be positive")
    if args.experts <= 0 or args.topk <= 0 or args.topk > args.experts:
        parser.error("--topk must be in [1, experts]")
    if args.fuse_routing:
        if args.tokens:
            args.m = args.tokens * args.topk
        elif args.m % args.topk:
            parser.error("routed --m must be divisible by --topk")
        args.tokens = args.m // args.topk
        if args.tokens % 128 or args.m % 128:
            parser.error("routed tokens and M=tokens*topk must be divisible by 128")
    elif args.m % 512:
        parser.error("grouped --m must be a multiple of 512")
    if args.warmup < 0 or args.iters <= 0 or args.samples <= 0:
        parser.error("invalid warmup/iteration count")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    h, intermediate = H, INTERMEDIATE
    torch.manual_seed(1234)
    tokens = args.tokens if args.fuse_routing else args.m
    experts = args.experts if args.fuse_routing else 1
    topk = args.topk if args.fuse_routing else 1
    aq, sfa = quantize_nvfp4_rows(
        torch.randn((tokens, h), device="cuda", dtype=torch.bfloat16)
    )
    w1q, sfw1, w2q, sfw2 = quantize_expert_weights(
        experts, h, intermediate
    )

    output = torch.empty((tokens, h), device="cuda", dtype=torch.bfloat16)

    fc1_config = Nvfp4Fc1Config(
        tile_n=128, tile_k=256, epilogue_tile_n=128,
        cluster_m=2, direct_gmem=True,
    )
    fc2_config = Nvfp4Fc2Config(
        tile_n=128, tile_k=256, epilogue_tile_n=64, cluster_m=2,
    )
    compile_nvfp4_fc1(fc1_config)
    compile_nvfp4_fc2(fc2_config)
    if args.fuse_routing:
        router = powerlaw_router_scores(
            tokens, experts, ROUTER_POWERLAW_ALPHA,
            device="cuda", dtype=torch.bfloat16,
        )
        topk_result = torch.topk(router.float(), topk, dim=1)
        routing_weights = torch.softmax(topk_result.values, dim=1).to(
            torch.bfloat16
        ).contiguous()
        expert_counts = torch.bincount(
            topk_result.indices.reshape(-1), minlength=experts
        )
        row_task_count = int(
            torch.div(expert_counts + 255, 256, rounding_mode="floor").sum().item()
        )
    else:
        router = torch.ones((args.m, 1), device="cuda", dtype=torch.bfloat16)
        routing_weights = torch.ones(
            (args.m, 1), device="cuda", dtype=torch.bfloat16
        )
        row_task_count = args.m // 256
    args._row_task_count = row_task_count
    fc1_metadata = prepare_nvfp4_fc1_metadata(
        router, topk, 2 * intermediate, fc1_config
    )
    grouped_layout = fc1_metadata.routing.narrow(0, 2 * experts, args.m)
    fc2_metadata = prepare_nvfp4_fc2_metadata(
        grouped_layout, experts, h, fc2_config
    )
    intermediate_packed = torch.empty(
        (args.m, intermediate // 2), device="cuda",
        dtype=torch.float4_e2m1fn_x2,
    )
    intermediate_scales = torch.empty(
        (args.m * (intermediate // 16),), device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    reference = torch.empty_like(output)

    candidates = compatible_configs(args.m, args)

    def prepare_fused(config):
        module = compile_nvfp4_fc1fc2(config)
        if args.fuse_routing:
            run = lambda: module.run_routed_out(
                aq, w1q, sfa, sfw1, w2q, sfw2, fc1_metadata,
                routing_weights, topk, output,
            )
        else:
            run = lambda: module.run_out(
                aq, w1q, sfa, sfw1, w2q, sfw2, output
            )
        run()
        torch.cuda.synchronize()
        return PreparedKernel(config, run)

    tune_options = TuneOptions(
        warmup=args.warmup,
        iters=args.iters,
        samples=args.samples,
        cache_path=args.cache_path,
        force_sweep=args.force_sweep,
        debug=args.debug,
        pre_parallel_jit=args.pre_parallel_jit,
    )
    selection = AutotuneRunner(tune_options).resolve(
        candidates,
        namespace="nvfp4_fc1fc2",
        key=("v6-routing-fusion", args.m, H, INTERMEDIATE,
             experts, topk, row_task_count, args.fuse_routing),
        prepare=prepare_fused,
        benchmark=lambda prepared: benchmark_with_stats(
            prepared.run, args.warmup, args.iters, args.samples
        ),
        probe=lambda prepared: quick_probe(prepared.run, args.iters),
        format_candidate=format_config,
        metric_formatter=lambda _config, stats: (
            f"{fused_tflops(args.m, stats['median_ms']):.1f} TF/s"
        ),
        pre_jit=compile_nvfp4_fc1fc2,
    )
    fused_config = selection.candidate

    def run_fused():
        return selection.prepared.run()

    def run_fc1():
        return nvfp4_fc1_out(
            aq, w1q, sfa, sfw1, fc1_metadata,
            intermediate_packed, intermediate_scales,
            config=fc1_config,
        )

    def run_fc2():
        return nvfp4_fc2_out(
            intermediate_packed, w2q, intermediate_scales, sfw2,
            fc1_metadata.routing, routing_weights, fc2_metadata, topk,
            reference, config=fc2_config,
        )

    def run_standalone():
        run_fc1()
        return run_fc2()

    if not args.no_accuracy:
        run_fused()
        run_standalone()
        torch.cuda.synchronize()
        difference = output.float() - reference.float()
        if args.debug and args.fuse_routing:
            repeated_reference = torch.empty_like(reference)
            original_reference = reference
            reference = repeated_reference
            run_standalone()
            torch.cuda.synchronize()
            repeat_difference = repeated_reference.float() - original_reference.float()
            print(
                f"[accuracy-baseline-repeat] max_abs="
                f"{repeat_difference.abs().max().item():.6g}; "
                f"mismatches={int((repeated_reference != original_reference).sum().item())}"
            )
            reference = original_reference
        within_tolerance = torch.isclose(
            output.float(), reference.float(), atol=args.atol, rtol=args.rtol
        )
        outside_tolerance = int((~within_tolerance).sum().item())
        rmse = difference.square().mean().sqrt()
        reference_rms = reference.float().square().mean().sqrt()
        relative_rmse = float((rmse / reference_rms).item())
        print(
            f"[accuracy] max_abs={difference.abs().max().item():.6g}; "
            f"mean_abs={difference.abs().mean().item():.6g}; "
            f"rmse={rmse.item():.6g}; relative_rmse={relative_rmse:.6g}; "
            f"mismatches={int((output != reference).sum().item())}; "
            f"outside_tol={outside_tolerance}"
        )
        if topk <= 2 and int((output != reference).sum().item()):
            raise AssertionError(
                "topk<=2 fused output must be bitwise equal to FC1+FC2"
            )
        if topk > 2 and relative_rmse > args.max_relative_rmse:
            raise AssertionError(
                "fused output exceeds the FC1+FC2 relative-RMSE bound: "
                f"{relative_rmse} > {args.max_relative_rmse}"
            )

    fused_ms = selection.stats["median_ms"]
    standalone_stats = benchmark_with_stats(
        run_standalone, args.warmup, args.iters, args.samples
    )
    standalone_ms = standalone_stats["median_ms"]
    fc1_ms = time_ms(run_fc1, args.warmup, args.iters)
    fc2_ms = time_ms(run_fc2, args.warmup, args.iters)
    print(
        f"[config] {format_config(fused_config)}; "
        f"cache_hit={selection.cache_hit}"
    )
    print(
        f"[workload] tokens={tokens}; M={args.m}; experts={experts}; "
        f"topk={topk}; input={'token-major' if args.fuse_routing else 'grouped'}; "
        f"output={'token-major-finalized' if args.fuse_routing else 'grouped'}"
    )
    print(f"[fused] {fused_ms:.6f} ms")
    print(f"[fc1+fc2] {standalone_ms:.6f} ms")
    print(f"[fc1] {fc1_ms:.6f} ms")
    print(f"[fc2] {fc2_ms:.6f} ms")
    print(
        f"[throughput] fused={fused_tflops(args.m, fused_ms):.3f} TF/s; "
        f"fc1+fc2={fused_tflops(args.m, standalone_ms):.3f} TF/s"
    )
    print(f"[ratio] {standalone_ms / fused_ms:.4f}x")
    if args.sol:
        report = SolReport(debug=args.debug)
        report.add(
            fused_sol_bounds(
                args.m, fused_config, tokens=tokens, experts=experts,
                row_tasks=row_task_count,
            ),
            fused_ms,
            label=f"M={args.m} fused",
        )
        report.print()


if __name__ == "__main__":
    main()
