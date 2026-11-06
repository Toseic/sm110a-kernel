#!/usr/bin/env python3
"""Compressor pooling/state correctness and cold-L2 CUDA Graph comparison."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_hc_pre import capture

from common import BenchmarkBase, benchmark_with_stats
from sm110a_kernel import CompressorPoolRunner


def reference(kv, score, ape, vs, ss, start, ratio):
    """Torch compressor semantics, excluding GEMM/norm/RoPE/cache writes."""
    b, seq, width = kv.shape
    overlap = ratio == 4
    d = width // (2 if overlap else 1)
    kv, score = kv.float(), score.float()
    offset = ratio if overlap else 0
    if start == 0:
        cutoff = seq - seq % ratio
        rem = seq % ratio
        if overlap and cutoff >= ratio:
            vs[:b, :ratio] = kv[:, cutoff - ratio : cutoff]
            ss[:b, :ratio] = score[:, cutoff - ratio : cutoff] + ape
        if rem:
            vs[:b, offset : offset + rem] = kv[:, cutoff:]
            ss[:b, offset : offset + rem] = score[:, cutoff:] + ape[:rem]
        if not cutoff:
            return kv.new_empty((b, 0, d), dtype=torch.bfloat16)
        kv = kv[:, :cutoff].unflatten(1, (-1, ratio))
        score = score[:, :cutoff].unflatten(1, (-1, ratio)) + ape
        if overlap:

            def expand(x, fill):
                out = x.new_full((b, cutoff // ratio, 2 * ratio, d), fill)
                out[:, :, ratio:] = x[..., d:]
                out[:, 1:, :ratio] = x[:, :-1, :, :d]
                return out

            kv, score = expand(kv, 0), expand(score, -torch.inf)
        out = (kv * score.softmax(2)).sum(2)
    else:
        slot = start % ratio
        vs[:b, offset + slot] = kv.squeeze(1)
        ss[:b, offset + slot] = score.squeeze(1) + ape[slot]
        if (start + 1) % ratio:
            return kv.new_empty((b, 0, d), dtype=torch.bfloat16)
        if overlap:
            kv = torch.cat((vs[:b, :ratio, :d], vs[:b, ratio:, d:]), 1)
            score = torch.cat((ss[:b, :ratio, :d], ss[:b, ratio:, d:]), 1)
        else:
            kv, score = vs[:b], ss[:b]
        out = (kv * score.softmax(1)).sum(1, keepdim=True)
        if overlap:
            vs[:b, :ratio] = vs[:b, ratio:]
            ss[:b, :ratio] = ss[:b, ratio:]
    return out.to(torch.bfloat16)


def compare(output, expected, states, ref_states):
    # Parallel FP32 reductions may cross a BF16 rounding boundary.
    torch.testing.assert_close(output, expected, rtol=0.016, atol=0.002)
    if output.numel():
        rel = (
            (output.float() - expected.float()).square().mean().sqrt()
            / expected.float().square().mean().sqrt().clamp_min(1e-10)
        ).item()
        if rel > 0.003:
            raise AssertionError(f"relative RMS error {rel} exceeds 0.003")
    for got, want in zip(states, ref_states):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    return (
        (output.float() - expected.float()).abs().max().item() if output.numel() else 0
    )


class CompressorPoolBenchmark(BenchmarkBase):
    def build_parser(self):
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument(
            "--ratios", type=int, nargs="+", choices=[4, 128], default=[4, 128]
        )
        p.add_argument("--dims", type=int, nargs="+", default=[128, 512])
        p.add_argument("--batch-size", type=int, default=1)
        p.add_argument(
            "--seq-lens", type=int, nargs="+", default=[1, 3, 4, 127, 128, 259, 1024]
        )
        p.add_argument(
            "--modes",
            nargs="+",
            choices=["prefill", "decode-update", "decode-boundary"],
            default=["prefill", "decode-update", "decode-boundary"],
        )
        p.add_argument("--pdl", action="store_true")
        p.add_argument("--baseline", choices=["torch", "none"], default="torch")
        p.add_argument("--no-accuracy", action="store_true")
        p.add_argument("--score-scale", type=float, default=1.0)
        self.add_timing_args(p, warmup=5, iters=20, samples=5)
        return p

    def validate_args(self, a):
        self.validate_timing_args(a, allow_zero_warmup=True)
        if not a.cold_l2 or not 1 <= a.batch_size <= 65535:
            raise SystemExit("requires cold L2 and batch in [1,65535]")
        if any(not 1 <= d <= 32768 for d in a.dims) or any(s < 1 for s in a.seq_lens):
            raise SystemExit("invalid dimensions/sequence lengths")

    def initialize(self):
        torch.manual_seed(self.seed)

    @torch.inference_mode()
    def run_with_args(self, a):
        print("BF16 projections / FP32 pooling and state; cold-L2 CUDA Graph replay")
        print("ratio D B mode S native_us torch_us speedup max_abs")
        for r in a.ratios:
            for d in a.dims:
                width = d * (2 if r == 4 else 1)
                for mode in a.modes:
                    for seq in a.seq_lens if mode == "prefill" else [1]:
                        start = (
                            0
                            if mode == "prefill"
                            else (2 * r - 1 if mode == "decode-boundary" else 2 * r + 1)
                        )
                        joint = torch.randn(
                            a.batch_size,
                            seq,
                            2 * width,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        kv, score = joint.chunk(2, -1)
                        score.mul_(a.score_scale)
                        ape = torch.randn(r, width, device="cuda") * a.score_scale
                        # One unused batch row detects writes outside active batch.
                        states = [
                            torch.randn(
                                a.batch_size + 1,
                                (2 if r == 4 else 1) * r,
                                width,
                                device="cuda",
                            )
                            for _ in range(2)
                        ]
                        saved = [t.clone() for t in states]
                        refs = [t.clone() for t in states]
                        groups = seq // r if not start else int((start + 1) % r == 0)
                        out = torch.empty(
                            a.batch_size, groups, d, device="cuda", dtype=torch.bfloat16
                        )
                        run = CompressorPoolRunner(
                            kv, score, ape, *states, out, start, ratio=r, pdl=a.pdl
                        ).run
                        ref = lambda: reference(kv, score, ape, *refs, start, r)
                        run()
                        error = "-"
                        if not a.no_accuracy:
                            expected = ref()
                            error = f"{compare(out, expected, states, refs):.3g}"
                            eager = out.clone()
                        graph, _ = capture(run)
                        for dst, src in zip(states, saved):
                            dst.copy_(src)
                        graph.replay()
                        if not a.no_accuracy:
                            torch.testing.assert_close(out, eager, rtol=0, atol=0)
                            compare(out, expected, states, refs)
                            # Graph must consume new projection, ape and state data.
                            joint.normal_()
                            ape.mul_(0.5)
                            for dst, src in zip(refs, states):
                                dst.copy_(src)
                            expected = ref()
                            graph.replay()
                            compare(out, expected, states, refs)
                            # Run consecutive steps through two overlap rotations /
                            # two ratio-128 boundaries, starting at partial prefill.
                            if mode == "decode-boundary":
                                self.check_sequence(r, d, a.batch_size, a.pdl)
                        native = (
                            benchmark_with_stats(
                                graph.replay,
                                a.warmup,
                                a.iters,
                                a.samples,
                                cold_l2_flush=a._cold_l2_flush,
                            )["median_ms"]
                            * 1000
                        )
                        baseline = speed = "-"
                        if a.baseline == "torch":
                            ref_graph, _ = capture(ref)
                            base = (
                                benchmark_with_stats(
                                    ref_graph.replay,
                                    a.warmup,
                                    a.iters,
                                    a.samples,
                                    cold_l2_flush=a._cold_l2_flush,
                                )["median_ms"]
                                * 1000
                            )
                            baseline, speed = f"{base:.3f}", f"{base / native:.2f}x"
                        print(
                            f"{r} {d} {a.batch_size} {mode} {seq} {native:.3f} {baseline} {speed} {error}",
                            flush=True,
                        )

    def check_sequence(self, r, d, b, pdl):
        width = d * (2 if r == 4 else 1)
        ape = torch.randn(r, width, device="cuda")
        states = [
            torch.zeros(b, (2 if r == 4 else 1) * r, width, device="cuda"),
            torch.full((b, (2 if r == 4 else 1) * r, width), -torch.inf, device="cuda"),
        ]
        refs = [t.clone() for t in states]
        prefill = r + 1
        for pos in [0, *range(prefill, prefill + 2 * r + 1)]:
            seq = prefill if pos == 0 else 1
            joint = torch.randn(b, seq, 2 * width, device="cuda", dtype=torch.bfloat16)
            kv, score = joint.chunk(2, -1)
            n = seq // r if pos == 0 else int((pos + 1) % r == 0)
            out = torch.empty(b, n, d, device="cuda", dtype=torch.bfloat16)
            CompressorPoolRunner(
                kv, score, ape, *states, out, pos, ratio=r, pdl=pdl
            ).run()
            compare(out, reference(kv, score, ape, *refs, pos, r), states, refs)


if __name__ == "__main__":
    CompressorPoolBenchmark().run()
