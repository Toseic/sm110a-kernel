#!/usr/bin/env python3
"""Accuracy and performance benchmark for the SM110A FP8 x MXFP4 GEMM."""

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    BenchmarkBase,
    benchmark_with_stats,
    quick_probe,
    tflops_from_ms,
)
from sm110a_kernel import (
    Fp8Mxfp4GemmConfig,
    compile_fp8mxfp4_gemm,
    fp8mxfp4_gemm_reference,
    from_blocked_fp8mxfp4_scales,
    quantize_fp8mxfp4_gemm,
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

_MODELS_MXFP4_MODULE = None
_DEEPGEMM_MODULE = None


def load_models_mxfp4_kernel():
    """Load DeepSeek V4.1 Flash's TileLang FP8 x MXFP4 kernel on demand."""
    global _MODELS_MXFP4_MODULE
    if _MODELS_MXFP4_MODULE is None:
        source = ROOT / "models/deepseek_v41_flash/inference/kernel.py"
        spec = importlib.util.spec_from_file_location(
            "_sm110a_models_deepseek_v41_mxfp4",
            source,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load models MXFP4 baseline from {source}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            if exc.name == "tilelang":
                raise RuntimeError(
                    "the models-mxfp4 baseline requires TileLang; install "
                    "models/deepseek_v41_flash/inference/requirements.txt"
                ) from exc
            raise
        _MODELS_MXFP4_MODULE = module
    return _MODELS_MXFP4_MODULE


def models_mxfp4_baseline(x8, w4, sfb, out, *, compact_b_gmem):
    """Prepare the model MXFP4 kernel for the native FP8xMXFP4 operands.

    The model kernel consumes conventional packed FP4 bytes and logical E8M0
    scales, whereas the native kernel accepts either compact or padded B
    storage and a TMA-swizzled scale buffer.  Convert these layouts once,
    outside the timed region.
    """
    model_kernel = load_models_mxfp4_kernel()
    m, k = x8.shape
    n = w4.size(0)
    if compact_b_gmem:
        packed_b = w4
    else:
        packed_b = w4.view(n, k // 16, 16)[..., :8].reshape(n, k // 2).contiguous()
    packed_b = packed_b.view(torch.float4_e2m1fn_x2)
    # The model kernel supports 128-wide activation scaling when possible,
    # and its 32-wide variant covers the native kernel's K=64 padded path.
    act_block_size = 128 if k % 128 == 0 else 32
    sfa = torch.full(
        (m, k // act_block_size), 127, device=x8.device, dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    logical_sfb = from_blocked_fp8mxfp4_scales(sfb, n, k)
    kernel = model_kernel.fp4_gemm_kernel(
        n,
        k,
        act_block_size=act_block_size,
        scale_dtype=model_kernel.FE8M0,
    )

    def run():
        kernel(x8, packed_b, out, sfa, logical_sfb)

    return run


def load_deepgemm():
    """Load the locally patched DeepGEMM build used as an SM110 baseline."""
    global _DEEPGEMM_MODULE
    if _DEEPGEMM_MODULE is None:
        source_root = ROOT / "baselinedir/DeepGEMM"
        if not source_root.is_dir():
            raise RuntimeError(
                f"DeepGEMM baseline is unavailable: expected {source_root}; "
                "clone and build the local DeepGEMM baseline first"
            )
        sys.path.insert(0, str(source_root))
        try:
            import deep_gemm
        except ImportError as exc:
            raise RuntimeError(
                "the DeepGEMM baseline requires its local extension; run "
                f"{source_root}/develop.sh first"
            ) from exc
        _DEEPGEMM_MODULE = deep_gemm
    return _DEEPGEMM_MODULE


def deepgemm_mxfp4_baseline(x8, w4, sfb, out, *, compact_b_gmem):
    """Prepare DeepGEMM's SM110 FP8 x MXFP4 dense GEMM baseline.

    DeepGEMM takes K-major packed E2M1 and TMA-aligned packed UE8M0 scale
    factors.  The native benchmark already provides the former, while its B
    scale buffer is hardware-swizzled; normalize and repack scales once before
    timing.  A is raw E4M3, so its per-32 scale is the E8M0 identity.
    """
    deep_gemm = load_deepgemm()
    m, k = x8.shape
    n = w4.size(0)
    if k % 128:
        raise RuntimeError(
            f"the DeepGEMM FP8 x FP4 baseline requires K divisible by 128; got K={k}"
        )
    if compact_b_gmem:
        packed_b = w4
    else:
        packed_b = w4.view(n, k // 16, 16)[..., :8].reshape(n, k // 2).contiguous()
    packed_b = packed_b.view(torch.int8)

    sfa_logical = torch.ones((m, k // 32), device=x8.device, dtype=torch.float32)
    sfb_logical = from_blocked_fp8mxfp4_scales(sfb, n, k).float()
    sfa = deep_gemm.transform_sf_into_required_layout(sfa_logical, m, k, (1, 32))
    sfb = deep_gemm.transform_sf_into_required_layout(sfb_logical, n, k, (1, 32))

    def run():
        deep_gemm.fp8_fp4_gemm_nt(
            (x8, sfa),
            (packed_b, sfb),
            out,
            recipe_a=(1, 32),
            recipe_b=(1, 32),
        )

    return run


# Mirror Fp8Mxfp4GemmConfig.validate without constructing invalid candidates
# during module import.  compact_b_gmem is bound to tile_k: the TMA
# b4x16_p64 expansion of compact B requires tile_k=128.
FP8MXFP4_GEMM_CONFIGS = tuple(
    Fp8Mxfp4GemmConfig(
        tile_n=tile_n,
        tile_k=tile_k,
        cluster_m=cluster_m,
        epilogue_block_n=epilogue_block_n,
        split_k=split_k,
        use_prefetch=use_prefetch,
        compact_b_gmem=tile_k == 128,
    )
    for cluster_m in (1, 2)
    for split_k in (1, 2, 4, 8)
    for tile_n in (64, 128, 256)
    for tile_k in (64, 128)
    for epilogue_block_n in (32, 64, 128)
    for use_prefetch in (False, True)
    if tile_n % epilogue_block_n == 0 and cluster_m * split_k <= 8
    if not (
        # Two-mainloop-stage SMEM check from Fp8Mxfp4GemmConfig.validate.
        (
            232448
            - 1024
            - 256
            - 2 * 128 * epilogue_block_n * 2
            - (128 * epilogue_block_n * 4 if split_k > 1 else 0)
        )
        // (
            (
                128 * 128
                + (tile_n // cluster_m) * 128
                + 128 * (tile_k // 8)
                + (tile_k // 32) * ((tile_n + 127) // 128) * 512
                + 1023
            )
            & ~1023
        )
        < 2
    )
)


def compatible_candidates(m, n, k, compact_b_gmem):
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    return tuple(
        config
        for config in FP8MXFP4_GEMM_CONFIGS
        if config.compact_b_gmem == compact_b_gmem
        and m % (config.tile_m * config.cluster_m) == 0
        and n % config.tile_n == 0
        and k % (config.tile_k * config.split_k) == 0
        # Rolling L2 prefetch is a tactic for large K or an incomplete CTA
        # wave, rather than a universally profitable instruction stream.
        and (
            not config.use_prefetch
            or k >= 8192
            or 0.5
            < (
                math.ceil(m / (config.tile_m * config.cluster_m))
                * math.ceil(n / config.tile_n)
                / sm_count
            )
            < 1.0
        )
    )


def format_candidate(config):
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/c{config.cluster_m}/e{config.epilogue_block_n}/sk{config.split_k}"
        f"/{'prefetch' if config.use_prefetch else 'noprefetch'}"
        f"/{'compact' if config.compact_b_gmem else 'padded'}"
    )


class Fp8Mxfp4GemmBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--m", type=int, default=0)
        parser.add_argument("--n", type=int, default=0)
        parser.add_argument("--k", type=int, default=0)
        self.add_timing_args(parser, warmup=5, iters=20, samples=7)
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--seed", type=int, default=1234)
        b_gmem_group = parser.add_mutually_exclusive_group()
        b_gmem_group.add_argument(
            "--compact-b-gmem",
            dest="compact_b_gmem",
            action="store_true",
            default=True,
            help=(
                "keep B in native packed FP4 storage, expanded by TMA "
                "b4x16_p64 (requires K divisible by 128; default)"
            ),
        )
        b_gmem_group.add_argument(
            "--padded-b-gmem",
            dest="compact_b_gmem",
            action="store_false",
            help="store B in the padded 16-byte-per-group layout",
        )
        parser.add_argument(
            "--baseline",
            choices=("none", "models-mxfp4", "deepgemm-mxfp4"),
            default="none",
            help="optional baseline: TileLang models kernel or locally patched DeepGEMM",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="print per-config runner probe and timing details",
        )
        self.add_cache_args(parser)
        self.add_pre_parallel_jit_arg(parser)
        parser.add_argument(
            "--cache-path",
            default=None,
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
                raise SystemExit("M and N must be multiples of 128, and K of 64")
        args.shapes = shapes
        return args

    def resolve_kernel(self, x8, w4, sfb, out, m, n, k, compact_b_gmem, args):
        candidates = compatible_candidates(m, n, k, compact_b_gmem)
        if not candidates:
            raise RuntimeError(
                f"no FP8xMXFP4 config supports M={m}, N={n}, K={k} "
                f"(compact_b_gmem={compact_b_gmem})"
            )

        def prepare(config):
            module = compile_fp8mxfp4_gemm(config)
            run = lambda: module.fp8mxfp4_gemm_out(x8, w4, sfb, out)
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
            namespace="fp8mxfp4_gemm_v1",
            key=(m, n, k, compact_b_gmem),
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
            pre_jit=compile_fp8mxfp4_gemm,
        )

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        if args.baseline == "models-mxfp4":
            print("[baseline] DeepSeek V4.1 Flash TileLang FP8 x MXFP4 GEMM")
        elif args.baseline == "deepgemm-mxfp4":
            print("[baseline] DeepGEMM SM110 FP8 x MXFP4 GEMM")
        print("[runner] config selection is cached per workload and CUDA environment")
        if args.baseline != "none":
            columns = (
                ("M", 8),
                ("N", 8),
                ("K", 8),
                ("Kernel(ms)", 14),
                ("Kernel TF/s", 14),
                ("Baseline(ms)", 14),
                ("Baseline TF/s", 14),
                ("Speedup", 10),
                ("MaxDiff", 12),
                ("BaselineDiff", 14),
                ("BestConfig", 44),
            )
        else:
            columns = (
                ("M", 8),
                ("N", 8),
                ("K", 8),
                ("Kernel(ms)", 14),
                ("Kernel TF/s", 14),
                ("MaxDiff", 12),
                ("BestConfig", 44),
            )
        table = self.make_table_printer(columns, debug=args.debug)
        table.print_header()
        for m, n, k in args.shapes:
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            w = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
            compact_b_gmem = args.compact_b_gmem and k % 128 == 0
            x8, w4, sfb = quantize_fp8mxfp4_gemm(
                x,
                w,
                compact_b_gmem=compact_b_gmem,
            )
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            baseline_run = None
            baseline_out = None
            baseline_stats = None
            if args.baseline != "none":
                baseline_out = torch.empty_like(out)
                baseline_factory = (
                    models_mxfp4_baseline
                    if args.baseline == "models-mxfp4"
                    else deepgemm_mxfp4_baseline
                )
                baseline_run = baseline_factory(
                    x8,
                    w4,
                    sfb,
                    baseline_out,
                    compact_b_gmem=compact_b_gmem,
                )
                baseline_stats = benchmark_with_stats(
                    baseline_run,
                    args.warmup,
                    args.iters,
                    args.samples,
                    cold_l2_flush=args._cold_l2_flush,
                )
            selection = self.resolve_kernel(
                x8,
                w4,
                sfb,
                out,
                m,
                n,
                k,
                compact_b_gmem,
                args,
            )
            kernel_stats = selection.stats
            max_diff = 0.0
            baseline_diff = 0.0
            if not args.no_accuracy:
                ref = fp8mxfp4_gemm_reference(x8, w4, sfb)
                selection.prepared.run()
                torch.cuda.synchronize()
                max_diff = (out.float() - ref.float()).abs().max().item()
                if not torch.equal(out, ref):
                    raise AssertionError(f"mismatch {m}x{n}x{k}: {max_diff}")
                if baseline_run is not None:
                    baseline_run()
                    torch.cuda.synchronize()
                    baseline_diff = (
                        (baseline_out.float() - ref.float()).abs().max().item()
                    )
            kms = kernel_stats["median_ms"]
            if baseline_run is None:
                table.print_row(
                    m,
                    n,
                    k,
                    f"{kms:.4f}",
                    f"{tflops_from_ms(m, n, k, kms):.2f}",
                    f"{max_diff:.6f}",
                    format_candidate(selection.candidate),
                )
            else:
                bms = baseline_stats["median_ms"]
                table.print_row(
                    m,
                    n,
                    k,
                    f"{kms:.4f}",
                    f"{tflops_from_ms(m, n, k, kms):.2f}",
                    f"{bms:.4f}",
                    f"{tflops_from_ms(m, n, k, bms):.2f}",
                    f"{bms / kms:.2f}x",
                    f"{max_diff:.6f}",
                    f"{baseline_diff:.6f}",
                    format_candidate(selection.candidate),
                )


if __name__ == "__main__":
    Fp8Mxfp4GemmBenchmark().run()
