#!/usr/bin/env python3
"""BF16 HC post + pre + RMSNorm: exact native baseline and cold-L2 graph timing."""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_hc_pre import capture

from common import BenchmarkBase, benchmark_with_stats
from sm110a_kernel import HcPostPreNormRunner, HcPostRunner, HcPreNormRunner


class HcPostPreNormBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--d", type=int, default=4096)
        parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128])
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Use prefill shape [B, tokens/B, D]; omit for [tokens, 1, D]",
        )
        parser.add_argument("--sinkhorn-iters", type=int, default=20)
        parser.add_argument("--pdl", action="store_true")
        parser.add_argument(
            "--baseline", choices=["separate", "none"], default="separate"
        )
        parser.add_argument(
            "--input-case",
            choices=["random", "zero", "constant", "cancellation"],
            default="random",
        )
        parser.add_argument("--no-accuracy", action="store_true")
        self.add_timing_args(parser, warmup=5, iters=20, samples=5)
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if not 256 <= args.d <= 4096 or args.d % 32 or any(t < 1 for t in args.tokens):
            raise SystemExit(
                "D must be a multiple of 32 in [256,4096]; tokens must be positive"
            )
        if not 1 <= args.sinkhorn_iters <= 100:
            raise SystemExit("Sinkhorn iterations must be in [1,100]")
        if args.batch_size is not None and (
            args.batch_size < 1 or any(t % args.batch_size for t in args.tokens)
        ):
            raise SystemExit("batch-size must be positive and divide every token count")
        if not args.cold_l2:
            raise SystemExit("HC post-pre-norm comparisons require cold L2")

    def initialize(self):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required")
        torch.manual_seed(self.seed)

    @torch.inference_mode()
    def run_with_args(self, args):
        print(
            f"HC post+pre+RMSNorm D={args.d} BF16 Sinkhorn={args.sinkhorn_iters} PDL={args.pdl}; cold-L2 graph replay"
        )
        print("tokens fused(us) separate(us) speedup bitwise(mixed/y/post/comb)")
        for tokens in args.tokens:
            d = args.d
            batch = tokens if args.batch_size is None else args.batch_size
            seq = tokens // batch
            if args.batch_size is not None:
                print(f"[prefill] B={batch} S={seq} total_tokens={tokens}")
            x = torch.randn(batch, seq, d, device="cuda", dtype=torch.bfloat16)
            residual = torch.randn(batch, seq, 4, d, device="cuda", dtype=x.dtype)
            prev_post = 2 * torch.rand(batch, seq, 4, device="cuda")
            prev_comb = torch.randn(batch, seq, 4, 4, device="cuda").softmax(-1)
            if args.input_case == "zero":
                x.zero_()
                residual.zero_()
            elif args.input_case == "constant":
                x.fill_(0.5)
                residual.fill_(0.5)
            elif args.input_case == "cancellation":
                residual[..., 1, :].copy_(-residual[..., 0, :])
                residual[..., 3, :].copy_(-residual[..., 2, :])
                prev_comb.fill_(0.25)
            weight = torch.randn(24, 4 * d, device="cuda") / math.sqrt(4 * d)
            scale = torch.tensor([0.7, 1.1, 0.9], device="cuda")
            base = torch.randn(24, device="cuda") * 0.2
            norm_weight = torch.randn(d, device="cuda")

            def outputs():
                return (
                    torch.empty_like(residual),
                    torch.empty_like(x),
                    torch.empty_like(prev_post),
                    torch.empty_like(prev_comb),
                )

            result = outputs()
            runner = HcPostPreNormRunner(
                x,
                residual,
                prev_post,
                prev_comb,
                weight,
                scale,
                base,
                norm_weight,
                *result,
                sinkhorn_iters=args.sinkhorn_iters,
                pdl=args.pdl,
            )

            def fused():
                runner.run()
                return result

            reference = outputs()
            post_runner = HcPostRunner(
                x, residual, prev_post, prev_comb, reference[0], pdl=args.pdl
            )
            pre_runner = HcPreNormRunner(
                reference[0],
                weight,
                scale,
                base,
                norm_weight,
                *reference[1:],
                sinkhorn_iters=args.sinkhorn_iters,
                pdl=args.pdl,
            )

            def separate():
                post_runner.run()
                pre_runner.run()
                return reference

            fused_graph, _ = capture(fused)
            separate_graph = None
            if not args.no_accuracy or args.baseline == "separate":
                separate_graph, _ = capture(separate)

            def check():
                for name, actual, expected in zip(
                    ("mixed", "y", "post", "comb"), result, reference
                ):
                    if not torch.equal(
                        actual.view(torch.uint8), expected.view(torch.uint8)
                    ):
                        raise AssertionError(
                            f"{name}: not bitwise identical; max_abs={(actual.float() - expected.float()).abs().max().item()}"
                        )

            if not args.no_accuracy:
                fused_graph.replay()
                separate_graph.replay()
                check()
                # Verify captured launches read updated inputs, including both
                # old gates and residuals, rather than retaining a prepared post.
                saved = [t.clone() for t in (x, residual, prev_post, prev_comb)]
                for t in (x, residual, prev_post, prev_comb):
                    t.add_(0.03125)
                fused_graph.replay()
                separate_graph.replay()
                check()
                for t, old in zip((x, residual, prev_post, prev_comb), saved):
                    t.copy_(old)
            stats = benchmark_with_stats(
                fused_graph.replay,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=args._cold_l2_flush,
            )
            us = stats["median_ms"] * 1000
            baseline, speedup = "-", "-"
            if args.baseline == "separate":
                stats = benchmark_with_stats(
                    separate_graph.replay,
                    args.warmup,
                    args.iters,
                    args.samples,
                    cold_l2_flush=args._cold_l2_flush,
                )
                separate_us = stats["median_ms"] * 1000
                baseline, speedup = f"{separate_us:.3f}", f"{separate_us / us:.3f}x"
            print(
                f"{tokens} {us:.3f} {baseline} {speedup} {'pass/pass/pass/pass' if not args.no_accuracy else '-'}",
                flush=True,
            )


if __name__ == "__main__":
    HcPostPreNormBenchmark().run()
