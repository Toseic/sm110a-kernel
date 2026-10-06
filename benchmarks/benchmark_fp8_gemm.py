#!/usr/bin/env python3
"""Accuracy and performance benchmark for the SM110A FP8 x FP8 GEMM."""

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
    quick_probe,
    tflops_from_ms,
)
from sm110a_kernel import (
    Fp8GemmConfig,
    compile_fp8_gemm,
    fp8_gemm_reference,
    from_blocked_fp8_scales,
    quantize_fp8_gemm,
)
from sm110a_kernel.runner import AutotuneRunner, PreparedKernel


DEFAULT_SHAPES = (
    (128, 4096, 4096),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 11008, 4096),
    (11008, 4096, 4096),
)

_DEEPGEMM_MODULE = None


def load_deepgemm():
    """Load the locally patched DeepGEMM build used as an SM110 baseline."""
    global _DEEPGEMM_MODULE
    if _DEEPGEMM_MODULE is None:
        source_root = ROOT / 'baselinedir/DeepGEMM'
        if not source_root.is_dir():
            raise RuntimeError(
                f'DeepGEMM baseline is unavailable: expected {source_root}; '
                'clone and build the local DeepGEMM baseline first'
            )
        sys.path.insert(0, str(source_root))
        try:
            import deep_gemm
        except ImportError as exc:
            raise RuntimeError(
                'the DeepGEMM baseline requires its local extension; run '
                f'{source_root}/develop.sh first'
            ) from exc
        _DEEPGEMM_MODULE = deep_gemm
    return _DEEPGEMM_MODULE


def deepgemm_fp8_baseline(a8, b8, sfa, sfb, out):
    """Prepare DeepGEMM's SM110 dense FP8 x FP8 GEMM baseline.

    DeepGEMM's fp8_fp4 GEMM family accepts any FP8/FP4 operand combination
    with explicit block recipes, so both sides use the same K=32 UE8M0
    micro-scaling as the native kernel.  Scale buffers arrive in the
    TMA-swizzled hardware layout; normalize and repack once before timing.
    """
    deep_gemm = load_deepgemm()
    m, k = a8.shape
    n = b8.size(0)
    if k % 128:
        raise RuntimeError(
            f'the DeepGEMM FP8 x FP8 baseline requires K divisible by 128; got K={k}'
        )
    sfa_logical = from_blocked_fp8_scales(sfa, m, k).float()
    sfb_logical = from_blocked_fp8_scales(sfb, n, k).float()
    sfa = deep_gemm.transform_sf_into_required_layout(sfa_logical, m, k, (1, 32))
    sfb = deep_gemm.transform_sf_into_required_layout(sfb_logical, n, k, (1, 32))

    def run():
        deep_gemm.fp8_fp4_gemm_nt(
            (a8, sfa), (b8, sfb), out,
            recipe_a=(1, 32), recipe_b=(1, 32),
        )

    return run


# Mirror Fp8GemmConfig.validate without constructing invalid candidates
# during module import.
FP8_GEMM_CONFIGS = tuple(
    Fp8GemmConfig(
        tile_n=tile_n, tile_k=tile_k, cluster_m=cluster_m,
        epilogue_block_n=epilogue_block_n, split_k=split_k,
        use_prefetch=use_prefetch,
    )
    for cluster_m in (1, 2)
    for split_k in (1, 2, 4, 8)
    for tile_n in (64, 128, 256)
    for tile_k in (64, 128)
    for epilogue_block_n in (32, 64, 128)
    for use_prefetch in (False, True)
    if tile_n % epilogue_block_n == 0 and cluster_m * split_k <= 8
    if not (
        # Two-mainloop-stage SMEM check from Fp8GemmConfig.validate.
        (232448 - 1024 - 256 - 2 * 128 * epilogue_block_n * 2 -
         (128 * epilogue_block_n * 4 if split_k > 1 else 0)) //
        (((128 * 128 + (tile_n // cluster_m) * 128 + 128 * (tile_k // 8) +
           (tile_k // 32) * ((tile_n + 127) // 128) * 512 + 1023) & ~1023)) < 2
    )
)


def compatible_candidates(m, n, k):
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    return tuple(
        config
        for config in FP8_GEMM_CONFIGS
        if m % (config.tile_m * config.cluster_m) == 0
        and n % config.tile_n == 0
        and k % (config.tile_k * config.split_k) == 0
        # Rolling L2 prefetch is a tactic for large K or an incomplete CTA
        # wave, rather than a universally profitable instruction stream.
        and (not config.use_prefetch or
             k >= 8192 or
             0.5 < (
                 math.ceil(m / (config.tile_m * config.cluster_m)) *
                 math.ceil(n / config.tile_n) / sm_count
             ) < 1.0)
    )


def format_candidate(config):
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/c{config.cluster_m}/e{config.epilogue_block_n}/sk{config.split_k}"
        f"/{'prefetch' if config.use_prefetch else 'noprefetch'}"
    )


class Fp8GemmBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--m", type=int, default=0)
        parser.add_argument("--n", type=int, default=0)
        parser.add_argument("--k", type=int, default=0)
        self.add_timing_args(parser, warmup=5, iters=20, samples=7)
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--atol", type=float, default=1e-4)
        parser.add_argument("--rtol", type=float, default=1e-2)
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument(
            "--baseline",
            choices=("none", "deepgemm-fp8"),
            default="none",
            help="optional baseline: locally patched DeepGEMM",
        )
        parser.add_argument(
            "--debug", action="store_true",
            help="print per-config runner probe and timing details",
        )
        self.add_cache_args(parser)
        self.add_pre_parallel_jit_arg(parser)
        parser.add_argument(
            "--cache-path", default=None,
            help="autotune cache path (default: profiles/autotune_cache.json)",
        )
        return parser

    def validate_args(self, args):
        chosen = args.m or args.n or args.k
        if chosen and not (args.m and args.n and args.k):
            raise SystemExit("--m, --n, and --k must be specified together")
        self.validate_timing_args(args, allow_zero_warmup=True)
        shapes = ((args.m, args.n, args.k),) if chosen else DEFAULT_SHAPES
        for m, n, k in shapes:
            if m % 128 or n % 128 or k % 64:
                raise SystemExit(
                    "M and N must be multiples of 128, and K of 64"
                )
        args.shapes = shapes
        return args

    def resolve_kernel(self, a8, b8, sfa, sfb, out, m, n, k, args):
        candidates = compatible_candidates(m, n, k)
        if not candidates:
            raise RuntimeError(
                f"no FP8 GEMM config supports M={m}, N={n}, K={k}"
            )

        def prepare(config):
            module = compile_fp8_gemm(config)
            run = lambda: module.fp8_gemm_out(a8, b8, sfa, sfb, out)
            run()
            torch.cuda.synchronize()
            return PreparedKernel(config, run)

        options = self.make_tune_options(
            args,
            role="ours",
            pre_parallel_jit=args.pre_parallel_jit,
        )
        return AutotuneRunner(options).resolve(
            candidates,
            namespace="fp8_gemm_v1",
            key=(m, n, k),
            prepare=prepare,
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
            format_candidate=format_candidate,
            metric_formatter=lambda _config, stats: (
                f"{tflops_from_ms(m, n, k, stats['median_ms']):.1f} TF/s"
            ),
            pre_jit=compile_fp8_gemm,
        )

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        if args.baseline == "deepgemm-fp8":
            print("[baseline] DeepGEMM SM110 FP8 x FP8 GEMM")
        print("[runner] config selection is cached per workload and CUDA environment")
        if args.baseline != "none":
            columns = (
                ("M", 8), ("N", 8), ("K", 8),
                ("Kernel(ms)", 14), ("Kernel TF/s", 14),
                ("Baseline(ms)", 14), ("Baseline TF/s", 14),
                ("Speedup", 10), ("MaxDiff", 12), ("BaselineDiff", 14),
                ("BestConfig", 44),
            )
        else:
            columns = (
                ("M", 8), ("N", 8), ("K", 8),
                ("Kernel(ms)", 14), ("Kernel TF/s", 14),
                ("MaxDiff", 12), ("BestConfig", 44),
            )
        table = self.make_table_printer(columns, debug=args.debug)
        table.print_header()
        for m, n, k in args.shapes:
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
            a8, b8, sfa, sfb = quantize_fp8_gemm(a, b)
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            baseline_run = None
            baseline_out = None
            baseline_stats = None
            if args.baseline != "none":
                baseline_out = torch.empty_like(out)
                baseline_run = deepgemm_fp8_baseline(a8, b8, sfa, sfb, baseline_out)
                baseline_stats = benchmark_with_stats(
                    baseline_run, args.warmup, args.iters, args.samples,
                    cold_l2_flush=args._cold_l2_flush,
                )
            selection = self.resolve_kernel(
                a8, b8, sfa, sfb, out, m, n, k, args,
            )
            kernel_stats = selection.stats
            max_diff = 0.0
            baseline_diff = 0.0
            if not args.no_accuracy:
                ref = fp8_gemm_reference(a8, b8, sfa, sfb)
                selection.prepared.run()
                torch.cuda.synchronize()
                max_diff = (out.float() - ref.float()).abs().max().item()
                # The MMA and fp32 reference accumulate in different rounding
                # orders, which occasionally flips one output's final bf16
                # rounding (up to one bf16 ULP, 2^-9 relative).  rtol covers
                # that; real defects are O(1) relative.
                scaled_atol = max(args.atol, 1e-6 * ref.float().abs().max().item())
                if not torch.allclose(out.float(), ref.float(),
                                      atol=scaled_atol, rtol=args.rtol):
                    raise AssertionError(
                        f"mismatch {m}x{n}x{k}: max_diff={max_diff:.6g} "
                        f"scaled_atol={scaled_atol:.6g}")
                if baseline_run is not None:
                    baseline_run()
                    torch.cuda.synchronize()
                    baseline_diff = (
                        (baseline_out.float() - ref.float()).abs().max().item()
                    )
            kms = kernel_stats["median_ms"]
            if baseline_run is None:
                table.print_row(
                    m, n, k, f"{kms:.4f}", f"{tflops_from_ms(m,n,k,kms):.2f}",
                    f"{max_diff:.6f}", format_candidate(selection.candidate),
                )
            else:
                bms = baseline_stats["median_ms"]
                table.print_row(
                    m, n, k, f"{kms:.4f}", f"{tflops_from_ms(m,n,k,kms):.2f}",
                    f"{bms:.4f}", f"{tflops_from_ms(m,n,k,bms):.2f}",
                    f"{bms/kms:.2f}x", f"{max_diff:.6f}", f"{baseline_diff:.6f}",
                    format_candidate(selection.candidate),
                )


if __name__ == "__main__":
    Fp8GemmBenchmark().run()
