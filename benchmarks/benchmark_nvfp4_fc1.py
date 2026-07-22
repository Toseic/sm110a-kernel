#!/usr/bin/env python3
"""Runner-backed config and token sweep for fused-permute NVFP4 SwiGLU FC1."""

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
    CuteDslNvfp4Fc1Runner,
    Nvfp4Fc1Config,
    compile_nvfp4_fc1,
    nvfp4_fc1_out,
    prepare_nvfp4_fc1_metadata,
    quantize_nvfp4_rows,
    from_blocked_nvfp4_scales,
    to_blocked_nvfp4_scales,
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
        cluster_m=cluster_m,
        mode=mode,
        direct_gmem=direct_gmem,
    )
    for tile_n in (128, 256)
    for tile_k in (64, 128, 256)
    for epilogue_tile_n in (64, 128)
    for cluster_m in (1, 2)
    for mode in ( 1, 2)
    for direct_gmem in (False, True)
    if tile_n % epilogue_tile_n == 0
)


def format_candidate(config):
    epilogue = "dg" if config.direct_gmem else "tma"
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/e{config.epilogue_tile_n}/c{config.cluster_m}"
        f"/{_MODE_LABELS[config.mode]}/{epilogue}"
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
    if args.cluster_m is not None:
        candidates = tuple(c for c in candidates if c.cluster_m == args.cluster_m)
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
        math.ceil(rows / (config.cluster_m * config.tile_m))
        for rows in expert_rows.tolist() if rows
    )
    task_count = m_tasks * ((2 * n) // config.tile_n)

    flops = 4 * m * n * k
    routed_a_bytes = m * (k / 2 + k / 16)
    b_bytes = active_experts * (2 * n) * (k / 2 + k / 16)
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


def fc1_tflops_from_ms(m, n, k, elapsed_ms):
    """FC1 computes both gate and up projections: 2 GEMMs, or 4MNK FLOPs."""
    return 2.0 * tflops_from_ms(m, n, k, elapsed_ms)


class TorchNvfp4Fc1Runner:
    """Pre-gather routed rows, then run per-expert scaled GEMMs and SwiGLU."""

    def __init__(self, aq, bq, sfa, sfb, metadata, experts, n, k, out):
        rows = metadata.source_rows.numel()
        grouped_layout = metadata.routing[2 * experts:2 * experts + rows]
        counts = metadata.routing[:experts].cpu().tolist()
        a_bytes = aq.view(torch.uint8).reshape(aq.size(0), -1)
        sfa_logical = from_blocked_nvfp4_scales(sfa, aq.size(0), k)
        sfb_expert_elems = (2 * n) * (k // 16)

        expert_inputs = []
        row_start = 0
        for expert, row_count in enumerate(counts):
            if not row_count:
                continue
            row_end = row_start + row_count
            if not torch.all(grouped_layout[row_start:row_end] == expert):
                raise AssertionError("metadata rows are not grouped by expert")

            source_rows = metadata.source_rows[row_start:row_end].long()
            padding = (-row_count) % 128
            if padding:
                source_rows = torch.cat(
                    (source_rows, source_rows[:1].expand(padding))
                )

            # Packed E2M1 has no advanced indexing implementation. Gather its
            # byte representation, then restore the original scalar type.
            gathered_a = a_bytes[source_rows].contiguous()
            gathered_a = gathered_a.view(torch.float4_e2m1fn_x2).reshape(
                source_rows.numel(), -1
            )
            gathered_sfa = to_blocked_nvfp4_scales(
                sfa_logical[source_rows].contiguous()
            )
            sfb_start = expert * sfb_expert_elems
            expert_inputs.append((
                gathered_a,
                gathered_sfa,
                bq[expert].t(),
                sfb[sfb_start:sfb_start + sfb_expert_elems],
                row_start,
                row_count,
            ))
            row_start = row_end

        if row_start != rows:
            raise AssertionError("expert counts do not cover all routed rows")
        self.expert_inputs = tuple(expert_inputs)
        self.n = n
        self.out = out

    def run(self):
        for aq, sfa, bq_t, sfb, row_start, row_count in self.expert_inputs:
            full_expert = torch._scaled_mm(
                aq,
                bq_t,
                sfa,
                sfb,
                # The fused kernel applies SwiGLU to FP32 MMA accumulators and
                # converts only the final result to BF16.
                out_dtype=torch.float32,
            )
            # FC1 stores every logical 32-column pair as [gate:32, up:32].
            blocks = full_expert.reshape(full_expert.size(0), self.n // 32, 64)
            gate = blocks[:, :, :32].reshape(full_expert.size(0), self.n)
            up = blocks[:, :, 32:].reshape(full_expert.size(0), self.n)
            swiglu = (
                torch.nn.functional.silu(gate.float()) * up.float()
            ).to(torch.bfloat16)
            self.out[row_start:row_start + row_count].copy_(swiglu[:row_count])

        return self.out


def make_quantized_interleaved_weights(experts, n, k):
    """Create NVFP4 weights in physical ``[gate:32, up:32]`` block order."""
    bq = None
    sfb = None
    for expert in range(experts):
        gate = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        up = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        b_expert = torch.stack(
            (gate.view(n // 32, 32, k), up.view(n // 32, 32, k)), dim=1
        ).reshape(2 * n, k)
        bq_expert, sfb_expert = quantize_nvfp4_rows(b_expert)
        if bq is None:
            bq = torch.empty(
                (experts, 2 * n, k // 2), device="cuda", dtype=bq_expert.dtype
            )
            sfb = torch.empty(
                (experts * sfb_expert.numel(),),
                device="cuda",
                dtype=sfb_expert.dtype,
            )
        bq[expert].copy_(bq_expert)
        offset = expert * sfb_expert.numel()
        sfb[offset : offset + sfb_expert.numel()].copy_(sfb_expert)
        del gate, up, b_expert, bq_expert, sfb_expert
    return bq, sfb


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
        parser.add_argument("--atol", type=float, default=2e-2)
        parser.add_argument("--rtol", type=float, default=2e-2)
        parser.add_argument("--seed", type=int, default=self.seed)
        parser.add_argument("--tile-n", type=int, choices=(64, 128, 256))
        parser.add_argument("--tile-k", type=int, choices=(64, 128, 256))
        parser.add_argument("--epilogue-tile-n", type=int, choices=(64, 128))
        parser.add_argument("--cluster-m", type=int, choices=(1, 2))
        parser.add_argument("--mode", type=int, choices=(0, 1, 2))
        parser.add_argument(
            "--direct-gmem",
            action="store_true",
            help="only tune direct-GMEM epilogue candidates",
        )
        parser.add_argument(
            "--baseline", choices=("torch", "cutedsl"), default="torch",
            help="reference implementation to benchmark (default: torch)",
        )
        parser.add_argument(
            "--cutedsl-pre-parallel-jit",
            action="store_true",
            help=(
                "compile CuTE DSL baseline configurations in parallel before "
                "benchmarking (disabled by default because CuTE DSL JIT is "
                "not thread-safe)"
            ),
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
        args.token_values = token_values
        if not selected_candidates(2 * args.n, args.k, args):
            raise SystemExit("no compatible NVFP4 FC1 configs match the filters")
        return args

    def resolve_kernel(self, aq, bq, sfa, sfb, router, out, args):
        candidates = selected_candidates(2 * args.n, args.k, args)

        def prepare(config):
            metadata = prepare_nvfp4_fc1_metadata(
                router, args.topk, 2 * args.n, config=config
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
                f"{fc1_tflops_from_ms(routed_m, args.n, args.k, stats['median_ms']):.1f} TF/s"
            ),
            pre_jit=compile_nvfp4_fc1,
        )
        return selection, prepared_metadata[selection.candidate]

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        # Retain weights only in NVFP4 form.  Quantizing a single expert at a
        # time caps the FP32 quantization workspace independently of E.
        bq, sfb = make_quantized_interleaved_weights(
            args.experts, args.n, args.k
        )

        print(
            f"[config] experts={args.experts} topk={args.topk} "
            f"n={args.n} k={args.k} tokens={args.token_values}"
        )
        print("[runner] config selection cached by tokens/experts/topk/N/K")
        if args.baseline == "cutedsl":
            print("[baseline] CuTE DSL persistent NVFP4 FC1 (autotuned)")
        else:
            print("[baseline] pre-gathered per-expert torch._scaled_mm + SwiGLU")
        table = TablePrinter((
            ("Tokens", 9), ("M", 10), ("TopK", 7), ("E", 6),
            ("N", 8), ("K", 8), ("Kernel(ms)", 14), ("Kernel TF/s", 15),
            ("Baseline(ms)", 14), ("Baseline TF/s", 15),
            ("Speedup", 10), ("MaxDiff", 12), ("Cache", 8), ("BestConfig", 34),
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

            # Build baseline
            if args.baseline == "cutedsl":
                baseline_out = torch.empty_like(out)
                baseline_runner = CuteDslNvfp4Fc1Runner(
                    aq, bq, sfa, sfb, baseline_out,
                    metadata.source_rows, metadata.tasks, metadata.state,
                )
                baseline_runner.autotune(TuneOptions(
                    warmup=args.warmup, iters=args.iters, samples=args.samples,
                    cold_l2_flush=args._cold_l2_flush, cache_path=args.cache_path,
                    force_sweep=args.force_sweep, debug=args.debug,
                    pre_parallel_jit=args.cutedsl_pre_parallel_jit,
                ))
                baseline_fn = baseline_runner.run
            else:
                baseline_out = torch.empty_like(out)
                baseline_runner = TorchNvfp4Fc1Runner(
                    aq, bq, sfa, sfb, metadata,
                    args.experts, args.n, args.k, baseline_out,
                )
                baseline_runner.run()
                baseline_fn = baseline_runner.run

            baseline_stats = benchmark_with_stats(
                baseline_fn, args.warmup, args.iters, args.samples,
                cold_l2_flush=args._cold_l2_flush,
            )

            max_diff = 0.0
            if not args.no_accuracy:
                selection.prepared.run()
                baseline_fn()
                torch.cuda.synchronize()
                max_diff = (out.float() - baseline_out.float()).abs().max().item()
                if not torch.allclose(
                    out.float(), baseline_out.float(), atol=args.atol, rtol=args.rtol
                ):
                    raise AssertionError(f"NVFP4 FC1 mismatch: max diff {max_diff}")

            kernel_ms = selection.stats["median_ms"]
            baseline_ms = baseline_stats["median_ms"]
            routed_m = tokens * args.topk
            table.print_row(
                tokens, routed_m, args.topk, args.experts, args.n, args.k,
                f"{kernel_ms:.4f}",
                f"{fc1_tflops_from_ms(routed_m, args.n, args.k, kernel_ms):.2f}",
                f"{baseline_ms:.4f}",
                f"{fc1_tflops_from_ms(routed_m, args.n, args.k, baseline_ms):.2f}",
                f"{baseline_ms / kernel_ms:.2f}x" if kernel_ms > 0 else "N/A",
                f"{max_diff:.6f}",
                "hit" if selection.cache_hit else "miss",
                format_candidate(selection.candidate),
            )
            if args.sol:
                print_sol_report(
                    nvfp4_fc1_sol_bounds(
                        router, args.topk, args.n, args.k, selection.candidate
                    ),
                    kernel_ms,
                    label="NVFP4 FC1 main kernel (routing excluded)",
                )


if __name__ == "__main__":
    Nvfp4Fc1Benchmark().run()
