#!/usr/bin/env python3
"""Cold-L2 RMSNorm benchmark with fixed CTA configuration and FP32 reference."""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import GMEM_BW_GBPS, BenchmarkBase, benchmark_with_stats
from sm110a_kernel import compile_rmsnorm


def reference(x, weight, eps):
    x32 = x.float()
    return (
        x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps) * weight.float()
    ).to(x.dtype)


class RmsNormBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--pdl", action="store_true", help="enable programmatic dependent launch"
        )
        parser.add_argument("--d", type=int, nargs="+", default=[1024, 4096, 8192])
        parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 256, 4096])
        parser.add_argument(
            "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
        )
        parser.add_argument(
            "--weight-dtype",
            choices=["float32", "input"],
            default="float32",
            help="FP32 by default, matching the model RMSNorm parameters",
        )
        parser.add_argument("--warps-per-row", type=int, choices=[1, 2, 4, 8])
        rows_group = parser.add_mutually_exclusive_group()
        rows_group.add_argument("--rows-per-cta", type=int, choices=[1, 2, 4, 8])
        rows_group.add_argument(
            "--sweep-rows",
            type=int,
            nargs="+",
            choices=[1, 2, 4, 8],
            help="compare fixed row counts serially at each D",
        )
        parser.add_argument("--eps", type=float, default=1e-6)
        parser.add_argument(
            "--baseline", choices=["torch", "copy", "none"], default="torch"
        )
        parser.add_argument("--peak-gbps", type=float, default=GMEM_BW_GBPS)
        parser.add_argument("--no-accuracy", action="store_true")
        self.add_timing_args(parser, warmup=5, iters=20, samples=5)
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if not math.isfinite(args.peak_gbps) or args.peak_gbps <= 0:
            raise SystemExit("--peak-gbps must be finite and positive")
        if any(d < 1 or d > 32768 for d in args.d) or any(t < 1 for t in args.tokens):
            raise SystemExit("D must be in [1, 32768] and tokens must be positive")

    def initialize(self):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required")
        torch.manual_seed(self.seed)

    def run_with_args(self, args):
        print(f"Reference GMEM bandwidth: {args.peak_gbps:g} GB/s")
        print(f"Input dtype: {args.dtype}; weight dtype: {args.weight_dtype}")
        print(
            f"D tokens warps/row rows/CTA ours(us) {args.baseline}(us) "
            "GB/s peak(%) max_abs_error"
        )
        for d in args.d:
            results = {}
            for rows in args.sweep_rows or [args.rows_per_cta]:
                try:
                    module = compile_rmsnorm(
                        d,
                        warps_per_row=args.warps_per_row,
                        rows_per_cta=rows,
                        pdl=args.pdl,
                    )
                except ValueError as error:
                    if args.sweep_rows:
                        print(f"skip D={d} rows/CTA={rows}: {error}")
                        continue
                    raise
                for tokens in args.tokens:
                    torch.manual_seed(self.seed)
                    x = torch.randn(
                        tokens, d, device="cuda", dtype=getattr(torch, args.dtype)
                    )
                    weight_dtype = (
                        torch.float32 if args.weight_dtype == "float32" else x.dtype
                    )
                    weight = torch.randn(d, device="cuda", dtype=weight_dtype)
                    y = torch.empty_like(x)
                    run = lambda: module.rmsnorm_out(x, weight, y, args.eps)
                    run()
                    error = "-"
                    if not args.no_accuracy:
                        ref = reference(x, weight, args.eps)
                        # FP32 reduction order can move a result across a final
                        # rounding tie. Allow one output ULP for low precision.
                        rtol, atol = {
                            torch.bfloat16: (0.008, 1e-6),
                            torch.float16: (0.001, 1e-6),
                            torch.float32: (2e-5, 1e-6),
                        }[x.dtype]
                        torch.testing.assert_close(y, ref, atol=atol, rtol=rtol)
                        error = f"{(y.float() - ref.float()).abs().max().item():.3g}"
                    stats = benchmark_with_stats(
                        run,
                        args.warmup,
                        args.iters,
                        args.samples,
                        cold_l2_flush=args._cold_l2_flush,
                    )
                    baseline = "-"
                    if args.baseline != "none":
                        baseline_fn = (
                            (
                                lambda: torch.nn.functional.rms_norm(
                                    x, (d,), weight, args.eps
                                )
                            )
                            if args.baseline == "torch"
                            else lambda: y.copy_(x)
                        )
                        base = benchmark_with_stats(
                            baseline_fn,
                            args.warmup,
                            args.iters,
                            args.samples,
                            cold_l2_flush=args._cold_l2_flush,
                        )
                        baseline = f"{base['median_ms'] * 1000:.3f}"
                    # Compulsory bytes: input read once, output write once, weight read once.
                    traffic = (
                        2 * x.numel() * x.element_size()
                        + weight.numel() * weight.element_size()
                    )
                    gbps = traffic / (stats["median_ms"] * 1e6)
                    c = module.config
                    print(
                        f"{d} {tokens} {c.warps_per_row} {c.rows_per_cta} "
                        f"{stats['median_ms'] * 1000:.3f} {baseline} {gbps:.2f} "
                        f"{100 * gbps / args.peak_gbps:.2f} {error}",
                        flush=True,
                    )
                    results.setdefault(tokens, []).append((stats["median_ms"], c))
            if not results:
                raise SystemExit(f"no valid configuration for D={d}")
            if args.sweep_rows:
                for tokens, candidates in results.items():
                    milliseconds, c = min(candidates, key=lambda item: item[0])
                    print(
                        f"best D={d} tokens={tokens}: warps/row={c.warps_per_row} "
                        f"rows/CTA={c.rows_per_cta} time={milliseconds * 1000:.3f} us"
                    )


if __name__ == "__main__":
    RmsNormBenchmark().run()
