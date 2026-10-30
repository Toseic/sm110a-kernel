#!/usr/bin/env python3
"""HC=4 post mixing: accuracy and cold-L2 CUDA Graph memory-bandwidth SOL."""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_hc_pre import capture

from common import GMEM_BW_GBPS, BenchmarkBase, benchmark_with_stats
from sm110a_kernel import HcPostRunner


def reference(x, residual, post, comb):
    return (
        post.unsqueeze(-1) * x.unsqueeze(-2)
        + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(-3)
    ).to(x.dtype)


class HcPostBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--pdl", action="store_true", help="enable programmatic dependent launch"
        )
        parser.add_argument("--vector-bytes", type=int, choices=[8, 16], default=16)
        parser.add_argument("--d", type=int, default=4096)
        parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 1024])
        parser.add_argument(
            "--threads", type=int, choices=[32, 64, 128, 256], default=128
        )
        parser.add_argument(
            "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
        )
        parser.add_argument(
            "--input-case",
            choices=["random", "zero", "constant", "cancellation"],
            default="random",
        )
        parser.add_argument(
            "--unaligned",
            action="store_true",
            help="Offset data pointers by one element",
        )
        parser.add_argument("--baseline", choices=["torch", "none"], default="torch")
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--peak-gbps", type=float, default=GMEM_BW_GBPS)
        self.add_timing_args(parser, warmup=5, iters=20, samples=5)
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if not 1 <= args.d <= 32768 or any(t < 1 for t in args.tokens):
            raise SystemExit("D must be in [1, 32768]; tokens must be positive")
        if not math.isfinite(args.peak_gbps) or args.peak_gbps <= 0:
            raise SystemExit("--peak-gbps must be finite and positive")
        if not args.cold_l2:
            raise SystemExit("HC post comparisons require cold L2")

    def initialize(self):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required")
        torch.manual_seed(self.seed)

    @torch.inference_mode()
    def run_with_args(self, args):
        print(
            f"HC=4 D={args.d} dtype={args.dtype} threads={args.threads} "
            f"vector_bytes={args.vector_bytes}; cold-L2 CUDA Graph replay"
        )
        print(
            f"Compulsory traffic: x + residual + FP32 post/comb + output once; peak={args.peak_gbps:g} GB/s"
        )
        print("tokens ours(us) torch(us) speedup GB/s peak(%) max_abs")
        for tokens in args.tokens:

            def tensor(shape, dtype):
                if args.unaligned:
                    return torch.empty(
                        math.prod(shape) + 1, device="cuda", dtype=dtype
                    )[1:].view(shape)
                return torch.empty(shape, device="cuda", dtype=dtype)

            dtype = getattr(torch, args.dtype)
            x = tensor((tokens, 1, args.d), dtype).normal_()
            residual = tensor((tokens, 1, 4, args.d), dtype).normal_()
            post = torch.rand(tokens, 1, 4, device="cuda") * 2
            comb = torch.randn(tokens, 1, 4, 4, device="cuda").softmax(-1)
            if args.input_case == "zero":
                x.zero_()
                residual.zero_()
            elif args.input_case == "constant":
                x.fill_(0.5)
                residual.fill_(-0.25)
            elif args.input_case == "cancellation":
                # Signed, disparate scales expose reduction and contraction differences.
                residual[:, :, 0].mul_(128)
                residual[:, :, 1].copy_(-residual[:, :, 0])
                comb[:, :, 1].copy_(comb[:, :, 0])
            output = tensor(residual.shape, dtype)
            runner = HcPostRunner(
                x,
                residual,
                post,
                comb,
                output,
                threads=args.threads,
                vector_bytes=args.vector_bytes,
                pdl=args.pdl,
            )
            run = runner.run
            ref_fn = lambda: reference(x, residual, post, comb)
            run()
            error = "-"

            def check():
                expected = ref_fn()
                # FP32 operations follow PyTorch's separate multiply/add boundaries.
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                return (output.float() - expected.float()).abs().max().item()

            if not args.no_accuracy:
                error = f"{check():.3g}"
                eager = output.clone()
            graph, _ = capture(run)
            graph.replay()
            if not args.no_accuracy:
                torch.testing.assert_close(output, eager, rtol=0, atol=0)
                originals = [t.clone() for t in (x, residual, post, comb)]
                for t in (x, residual, post, comb):
                    t.mul_(0.5)
                graph.replay()
                check()
                updated = output.clone()
                run()
                torch.testing.assert_close(output, updated, rtol=0, atol=0)
                for t, original in zip((x, residual, post, comb), originals):
                    t.copy_(original)
            stats = benchmark_with_stats(
                graph.replay,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=args._cold_l2_flush,
            )
            ours = stats["median_ms"] * 1000
            baseline = speedup = "-"
            if args.baseline == "torch":
                ref_graph, ref_output = capture(ref_fn)
                stats = benchmark_with_stats(
                    ref_graph.replay,
                    args.warmup,
                    args.iters,
                    args.samples,
                    cold_l2_flush=args._cold_l2_flush,
                )
                baseline = f"{stats['median_ms'] * 1000:.3f}"
                speedup = f"{stats['median_ms'] * 1000 / ours:.2f}x"
            traffic = sum(t.nbytes for t in (x, residual, post, comb, output))
            gbps = traffic / (ours * 1000)
            print(
                f"{tokens} {ours:.3f} {baseline} {speedup} {gbps:.2f} {gbps / args.peak_gbps * 100:.2f} {error}",
                flush=True,
            )


if __name__ == "__main__":
    HcPostBenchmark().run()
