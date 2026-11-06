#!/usr/bin/env python3
"""Fused compressor norm/RoPE/FP8 cache store versus the native unfused chain."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_hc_pre import capture

from common import BenchmarkBase, benchmark_with_stats
from sm110a_kernel import (
    CompressorStoreRunner,
    Fp8ActQuantRunner,
    RmsNormRunner,
    RopeRunner,
)


def raw(t):
    return (
        t.view(torch.uint8)
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu)
        else t
    )


class CompressorStoreBenchmark(BenchmarkBase):
    def build_parser(self):
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 128])
        p.add_argument("--rows", type=int, nargs="+", default=[1, 8, 256])
        p.add_argument(
            "--scales", nargs="+", choices=["ue8m0", "fp32"], default=["ue8m0", "fp32"]
        )
        p.add_argument(
            "--modes",
            nargs="+",
            choices=["decode", "prefill"],
            default=["decode", "prefill"],
        )
        p.add_argument(
            "--input-case",
            choices=["random", "zero", "constant", "small"],
            default="random",
        )
        p.add_argument("--conjugate", action="store_true")
        p.add_argument("--pdl", action="store_true")
        p.add_argument("--baseline", choices=["native", "none"], default="native")
        p.add_argument("--no-accuracy", action="store_true")
        self.add_timing_args(p, warmup=5, iters=20, samples=5)
        return p

    def validate_args(self, a):
        self.validate_timing_args(a, allow_zero_warmup=True)
        if not a.cold_l2 or min(*a.batch_sizes, *a.rows) < 1:
            raise SystemExit("requires cold L2 and positive batch/rows")

    def initialize(self):
        torch.manual_seed(self.seed)

    @torch.inference_mode()
    def run_with_args(self, a):
        print(
            "D=512 RoPE=64; cold-L2 CUDA Graph; baseline=native RMSNorm + RoPE + FP8 quant + 3 cache copies"
        )
        print("scale B S mode fused_us unfused_us speedup")
        for scale in a.scales:
            for b in a.batch_sizes:
                for mode in a.modes:
                    for s in a.rows if mode == "prefill" else [1]:
                        self.case(a, scale, b, s, mode)

    def case(self, a, scale, b, s, mode):
        device = "cuda"
        dtype = torch.float8_e8m0fnu if scale == "ue8m0" else torch.float32
        x = torch.randn(b, s, 512, device=device, dtype=torch.bfloat16)
        weight = torch.randn(512, device=device)
        if a.input_case == "zero":
            x.zero_()
        if a.input_case == "constant":
            x.fill_(0.5)
        if a.input_case == "small":
            x.mul_(1e-4)
        # Strided prefill frequencies, single-position decode and lazy conjugate.
        stride = 4 if mode == "prefill" else 1
        angles = torch.randn(s * stride + 2, 32, device=device)
        base_freq = torch.polar(torch.ones_like(angles), angles)
        freq = base_freq[1 : 1 + s * stride : stride]
        if a.conjugate:
            freq = freq.conj()
        position, capacity = 11, 11 + s + 3

        def cache():
            values = torch.full(
                (b + 1, capacity, 448), 0x38, device=device, dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            scales = (
                torch.full(
                    (b + 1, capacity, 7), 127, device=device, dtype=torch.uint8
                ).view(dtype)
                if scale == "ue8m0"
                else torch.full((b + 1, capacity, 7), 1.0, device=device)
            )
            rope = torch.full(
                (b + 1, capacity, 64), -3.0, device=device, dtype=torch.bfloat16
            )
            return values, scales, rope

        caches, refs = cache(), cache()
        packed = (
            (
                torch.empty(b, s, 448, device=device, dtype=torch.float8_e4m3fn),
                torch.empty(b, s, 7, device=device, dtype=dtype),
                torch.empty(b, s, 64, device=device, dtype=torch.bfloat16),
            )
            if mode == "prefill"
            else None
        )
        run = CompressorStoreRunner(
            x, weight, freq, *caches, position, packed=packed, scale=scale, pdl=a.pdl
        ).run
        norm = torch.empty_like(x)
        quant_input = torch.empty(b * s, 448, device=device, dtype=torch.bfloat16)
        values = torch.empty(b * s, 448, device=device, dtype=torch.float8_e4m3fn)
        scales = torch.empty(b * s, 7, device=device, dtype=dtype)
        norm_run = RmsNormRunner(x, weight, norm, pdl=a.pdl).run
        rope_run = RopeRunner(norm[..., -64:], freq, pdl=a.pdl).run
        quant_run = Fp8ActQuantRunner(
            quant_input, values, scales, 64, scale, pdl=a.pdl
        ).run
        ref_packed = (values.view(b, s, 448), scales.view(b, s, 7), norm[..., -64:])

        def reference():
            norm_run()
            rope_run()
            quant_input.copy_(norm[..., :448].reshape(b * s, 448))
            quant_run()
            for dst, src in zip(refs, ref_packed):
                dst[:b, position : position + s].copy_(src)

        def check():
            reference()
            # Match the entire cache, including untouched positions/batch rows.
            for got, want in zip(caches, refs):
                torch.testing.assert_close(raw(got), raw(want), rtol=0, atol=0)
            if packed is not None:
                for got, want in zip(packed, ref_packed):
                    torch.testing.assert_close(raw(got), raw(want), rtol=0, atol=0)

        run()
        if not a.no_accuracy:
            check()
        graph, _ = capture(run)
        graph.replay()
        if not a.no_accuracy:
            check()
            eager = [raw(t).clone() for t in caches]
            for dst in caches:
                raw(dst).zero_()
            graph.replay()
            # Only written regions are compared to the prior eager result.
            for got, want in zip(caches, eager):
                torch.testing.assert_close(
                    raw(got)[:b, position : position + s],
                    want[:b, position : position + s],
                    rtol=0,
                    atol=0,
                )
            # Restore sentinels before changing dynamic inputs and replaying.
            for dst, saved in zip(caches, eager):
                raw(dst).copy_(saved)
            x.normal_()
            weight.mul_(0.5)
            base_freq.copy_(torch.polar(torch.ones_like(angles), angles * 0.3))
            graph.replay()
            check()
        fused = (
            benchmark_with_stats(
                graph.replay,
                a.warmup,
                a.iters,
                a.samples,
                cold_l2_flush=a._cold_l2_flush,
            )["median_ms"]
            * 1000
        )
        unfused = speed = "-"
        if a.baseline == "native":
            ref_graph, _ = capture(reference)
            timing = (
                benchmark_with_stats(
                    ref_graph.replay,
                    a.warmup,
                    a.iters,
                    a.samples,
                    cold_l2_flush=a._cold_l2_flush,
                )["median_ms"]
                * 1000
            )
            unfused, speed = f"{timing:.3f}", f"{timing / fused:.2f}x"
        print(f"{scale} {b} {s} {mode} {fused:.3f} {unfused} {speed}", flush=True)


if __name__ == "__main__":
    CompressorStoreBenchmark().run()
