#!/usr/bin/env python3
"""Cold-L2 in-place RoPE benchmark for DSV4-Flash Q, KV and inverse output."""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import GMEM_BW_GBPS, BenchmarkBase, benchmark_with_stats
from sm110a_kernel import compile_rope


def reference_(x, freqs, inverse=False):
    # FP32 .float() preserves odd outer strides; materialize those views so
    # view_as_complex also works for deliberately unaligned benchmark cases.
    values = torch.view_as_complex(x.float().contiguous().unflatten(-1, (-1, 2)))
    shape = (1, x.size(1), x.size(-1) // 2)
    if x.ndim == 4:
        shape = (1, x.size(1), 1, x.size(-1) // 2)
    frequencies = freqs.conj() if inverse else freqs
    x.copy_(torch.view_as_real(values * frequencies.reshape(shape)).flatten(-2))
    return x


class RopeBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--batch", type=int, nargs="+", default=[1])
        parser.add_argument("--seq-len", type=int, nargs="+", default=[1, 128, 1024])
        parser.add_argument("--heads", type=int, default=64)
        parser.add_argument("--head-dim", type=int, default=512)
        parser.add_argument("--rope-dim", type=int, default=64)
        parser.add_argument("--layout", choices=["q", "kv", "both"], default="both")
        parser.add_argument(
            "--direction", choices=["forward", "inverse", "both"], default="both"
        )
        parser.add_argument(
            "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
        )
        parser.add_argument(
            "--freq-stride", type=int, default=1, help="use 4 or 128 for compressed KV"
        )
        parser.add_argument("--start-pos", type=int, default=37)
        parser.add_argument(
            "--threads", type=int, choices=[64, 128, 256, 512], default=256
        )
        parser.add_argument("--pdl", action="store_true")
        parser.add_argument("--baseline", choices=["torch", "none"], default="torch")
        parser.add_argument("--peak-gbps", type=float, default=GMEM_BW_GBPS)
        parser.add_argument("--no-accuracy", action="store_true")
        self.add_timing_args(parser, warmup=5, iters=20, samples=5)
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if min(*args.batch, *args.seq_len, args.heads, args.freq_stride) < 1:
            raise SystemExit(
                "batch, sequence, heads and frequency stride must be positive"
            )
        if not 2 <= args.rope_dim <= 1024 or args.rope_dim % 2:
            raise SystemExit("rope-dim must be even and in [2, 1024]")
        if args.head_dim < args.rope_dim or args.start_pos < 0:
            raise SystemExit(
                "head-dim must cover rope-dim; start-pos must be nonnegative"
            )
        if not args.cold_l2:
            raise SystemExit("RoPE benchmark requires cold L2")
        if not math.isfinite(args.peak_gbps) or args.peak_gbps <= 0:
            raise SystemExit("peak-gbps must be finite and positive")

    def initialize(self):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required")
        torch.manual_seed(self.seed)

    def run_with_args(self, args):
        module = compile_rope(args.rope_dim, threads=args.threads, pdl=args.pdl)
        layouts = ["q", "kv"] if args.layout == "both" else [args.layout]
        directions = (
            [False, True] if args.direction == "both" else [args.direction == "inverse"]
        )
        print(
            f"SOL: compulsory RoPE read/write + unique frequencies / {args.peak_gbps:g} GB/s"
        )
        print(
            "layout B S H D rotary inverse ours(us) GB/s SOL(%) torch(us) speedup max_abs_error"
        )
        for b in args.batch:
            for s in args.seq_len:
                # Generate only selected frequencies, while retaining the model's
                # physical row stride and nonzero position offset.
                angles = (
                    args.start_pos
                    + torch.arange(s, device="cuda", dtype=torch.float32)
                    * args.freq_stride
                )[:, None]
                angles = angles * (
                    10000.0
                    ** (
                        -torch.arange(
                            0, args.rope_dim, 2, device="cuda", dtype=torch.float32
                        )
                        / args.rope_dim
                    )
                )
                storage = torch.empty(
                    (s * args.freq_stride, args.rope_dim // 2),
                    device="cuda",
                    dtype=torch.complex64,
                )
                freqs = storage[:: args.freq_stride]
                freqs.copy_(torch.polar(torch.ones_like(angles), angles))
                for layout in layouts:
                    shape = (
                        (b, s, args.heads, args.head_dim)
                        if layout == "q"
                        else (b, s, args.head_dim)
                    )
                    original = torch.randn(
                        shape, device="cuda", dtype=getattr(torch, args.dtype)
                    )
                    ours, baseline = original.clone(), original.clone()
                    x, ref = (
                        ours[..., -args.rope_dim :],
                        baseline[..., -args.rope_dim :],
                    )
                    for inverse in directions:
                        ours.copy_(original)
                        baseline.copy_(original)
                        run = lambda: module.rope_(x, freqs, inverse)
                        base = lambda: reference_(ref, freqs, inverse)
                        run()
                        error = "-"
                        if not args.no_accuracy:
                            base()
                            atol, rtol = {
                                torch.bfloat16: (1e-5, 0.008),
                                torch.float16: (1e-6, 0.001),
                                torch.float32: (1e-6, 1e-6),
                            }[x.dtype]
                            torch.testing.assert_close(x, ref, atol=atol, rtol=rtol)
                            torch.testing.assert_close(
                                ours[..., : -args.rope_dim],
                                original[..., : -args.rope_dim],
                                atol=0,
                                rtol=0,
                            )
                            error = (
                                f"{(x.float() - ref.float()).abs().max().item():.3g}"
                            )
                        ours.copy_(original)
                        baseline.copy_(original)
                        stats = benchmark_with_stats(
                            run,
                            args.warmup,
                            args.iters,
                            args.samples,
                            cold_l2_flush=args._cold_l2_flush,
                        )
                        us = stats["median_ms"] * 1000
                        traffic = (
                            2 * x.numel() * x.element_size()
                            + freqs.numel() * freqs.element_size()
                        )
                        gbps = traffic / (us * 1000)
                        sol = 100 * gbps / args.peak_gbps
                        base_us = speedup = "-"
                        if args.baseline == "torch":
                            timing = benchmark_with_stats(
                                base,
                                args.warmup,
                                args.iters,
                                args.samples,
                                cold_l2_flush=args._cold_l2_flush,
                            )
                            base_us = f"{timing['median_ms'] * 1000:.3f}"
                            speedup = f"{timing['median_ms'] * 1000 / us:.2f}"
                        print(
                            f"{layout} {b} {s} {args.heads if layout == 'q' else 1} {args.head_dim} {args.rope_dim} {inverse} {us:.3f} {gbps:.2f} {sol:.2f} {base_us} {speedup} {error}",
                            flush=True,
                        )


if __name__ == "__main__":
    RopeBenchmark().run()
