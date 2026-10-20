#!/usr/bin/env python3
"""HC pre/head for decode/prefill, cold-L2 CUDA Graph timings."""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import GMEM_BW_GBPS, BenchmarkBase, benchmark_with_stats
from sm110a_kernel import HcPreRunner


def reference(
    x, weight, scale, base, norm_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, head=False
):
    shape, dtype = x.shape, x.dtype
    x = x.flatten(-2).float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, weight) * inv
    if head:
        pre = torch.sigmoid(mixes * scale + base) + hc_eps
        return ((pre.unsqueeze(-1) * x.view(shape)).sum(-2).to(dtype),)
    a, b, c = mixes.split((4, 4, 16), dim=-1)
    pre = torch.sigmoid(a * scale[0] + base[:4]) + hc_eps
    post = 2 * torch.sigmoid(b * scale[1] + base[4:8])
    comb = (c * scale[2] + base[8:]).unflatten(-1, (4, 4)).softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    y = (pre.unsqueeze(-1) * x.view(shape)).sum(-2).to(dtype)
    return y, post, comb


def capture(fn):
    # Eager warmup on a side stream prepares library workspaces before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    return graph, outputs


class HcPreBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--mode", choices=["pre", "head"], default="pre")
        parser.add_argument("--d", type=int, default=4096)
        parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128])
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Use prefill shape [B, tokens/B, 4, D]; omit for decode [tokens, 1, 4, D]",
        )
        parser.add_argument("--sinkhorn-iters", type=int, default=20)
        parser.add_argument(
            "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
        )
        parser.add_argument(
            "--input-case", choices=["random", "zero", "constant"], default="random"
        )
        parser.add_argument("--logit-scale", type=float, default=1.0)
        parser.add_argument("--baseline", choices=["torch", "none"], default="torch")
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--peak-gbps", type=float, default=GMEM_BW_GBPS)
        self.add_sol_arg(parser)
        self.add_timing_args(parser, warmup=5, iters=20, samples=5)
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if not 1 <= args.d <= 4096 or any(t < 1 for t in args.tokens):
            raise SystemExit("D must be in [1, 4096]; tokens must be positive")
        if args.batch_size is not None and (
            args.batch_size < 1 or any(t % args.batch_size for t in args.tokens)
        ):
            raise SystemExit(
                "--batch-size must be positive and divide every token count"
            )
        if not 1 <= args.sinkhorn_iters <= 100:
            raise SystemExit("--sinkhorn-iters must be in [1, 100]")
        if not math.isfinite(args.logit_scale) or args.logit_scale <= 0:
            raise SystemExit("--logit-scale must be finite and positive")
        if not math.isfinite(args.peak_gbps) or args.peak_gbps <= 0:
            raise SystemExit("--peak-gbps must be finite and positive")
        if not args.cold_l2:
            raise SystemExit("HC pre/head comparisons require cold L2")

    def initialize(self):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required")
        torch.manual_seed(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = False

    @torch.inference_mode()
    def run_with_args(self, args):
        head = args.mode == "head"
        kernel = (
            "SIMT threads=256; projections=4"
            if head
            else f"tcgen05 tile=128x24x128 split-K=512 buffers=2; "
            f"SIMT fallback threads=256; Sinkhorn={args.sinkhorn_iters}"
        )
        print(
            f"HC {args.mode}: HC=4 D={args.d} dtype={args.dtype}; {kernel}; "
            "cold-L2 CUDA Graph replay"
        )
        print(
            "tokens ours(us) torch(us) speedup max_abs("
            + ("y" if head else "y/post/comb")
            + ")"
        )
        if args.sol:
            print(
                f"[SOL memory] peak={args.peak_gbps:g} GB/s; compulsory traffic: "
                "input once + FP32 HC parameters once + outputs once. "
                "This is a memory-bandwidth lower bound, excluding compute, "
                "on-chip cache traffic and launch overhead."
            )
        for tokens in args.tokens:
            batch = tokens if args.batch_size is None else args.batch_size
            seq = tokens // batch
            if args.batch_size is not None:
                print(f"[prefill] B={batch} S={seq} total_tokens={tokens}")
            x = torch.randn(
                batch, seq, 4, args.d, device="cuda", dtype=getattr(torch, args.dtype)
            )
            if args.input_case == "zero":
                x.zero_()
            elif args.input_case == "constant":
                x.fill_(0.5)
            weight = torch.randn(4 if head else 24, 4 * args.d, device="cuda") * (
                args.logit_scale / math.sqrt(4 * args.d)
            )
            scale = torch.tensor([0.7] if head else [0.7, 1.1, 0.9], device="cuda")
            base = torch.randn(4 if head else 24, device="cuda") * 0.2
            y = torch.empty(batch, seq, args.d, device="cuda", dtype=x.dtype)
            post = comb = None
            if not head:
                post = torch.empty(batch, seq, 4, device="cuda")
                comb = torch.empty(batch, seq, 4, 4, device="cuda")
            outputs = (y,) if head else (y, post, comb)
            runner = HcPreRunner(
                x,
                weight,
                scale,
                base,
                y,
                post,
                comb,
                sinkhorn_iters=args.sinkhorn_iters,
                head=head,
            )
            run = runner.run
            ref_fn = lambda: reference(
                x, weight, scale, base, sinkhorn_iters=args.sinkhorn_iters, head=head
            )
            run()
            error = "-"
            if not args.no_accuracy:
                expected = ref_fn()
                rtol = {
                    torch.bfloat16: 0.008,
                    torch.float16: 0.001,
                    torch.float32: 2e-5,
                }[x.dtype]
                torch.testing.assert_close(y, expected[0], rtol=rtol, atol=2e-5)
                for actual, ref in zip(outputs[1:], expected[1:]):
                    torch.testing.assert_close(actual, ref, rtol=2e-5, atol=2e-6)
                error = "/".join(
                    f"{(a.float() - b.float()).abs().max().item():.3g}"
                    for a, b in zip(outputs, expected)
                )
                eager = tuple(t.clone() for t in outputs)
            graph, _ = capture(run)
            graph.replay()
            if not args.no_accuracy:
                for actual, ref in zip(outputs, eager):
                    torch.testing.assert_close(actual, ref, rtol=0, atol=0)
                # Verify captured work consumes new input, rather than stale intermediates.
                original = x.clone()
                x.copy_(original.roll(1, dims=-1) * 0.5)
                graph.replay()
                changed = tuple(t.clone() for t in outputs)
                run()
                for actual, ref in zip(outputs, changed):
                    torch.testing.assert_close(actual, ref, rtol=0, atol=0)
                updated_expected = ref_fn()
                torch.testing.assert_close(y, updated_expected[0], rtol=rtol, atol=2e-5)
                for actual, ref in zip(outputs[1:], updated_expected[1:]):
                    torch.testing.assert_close(actual, ref, rtol=2e-5, atol=2e-6)
                x.copy_(original)
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
                ref_graph, ref_outputs = capture(ref_fn)
                stats = benchmark_with_stats(
                    ref_graph.replay,
                    args.warmup,
                    args.iters,
                    args.samples,
                    cold_l2_flush=args._cold_l2_flush,
                )
                baseline = f"{stats['median_ms'] * 1000:.3f}"
                speedup = f"{stats['median_ms'] * 1000 / ours:.2f}x"
            print(
                f"{tokens} {ours:.3f} {baseline} {speedup} {error}",
                flush=True,
            )
            if args.sol:
                # Shared HC parameters are compulsory external-memory traffic
                # once per invocation, not once per token. Other CTAs may reuse
                # them through L1/L2 even though every CTA issues global loads.
                input_bytes = x.nbytes
                parameter_bytes = weight.nbytes + scale.nbytes + base.nbytes
                output_bytes = sum(t.nbytes for t in outputs)
                traffic_bytes = input_bytes + parameter_bytes + output_bytes
                sol_us = traffic_bytes / (args.peak_gbps * 1e3)
                effective_gbps = traffic_bytes / (ours * 1e3)
                print(
                    f"[SOL memory] tokens={tokens}: input={input_bytes / 1e6:.6f} MB; "
                    f"parameters={parameter_bytes / 1e6:.6f} MB; "
                    f"outputs={output_bytes / 1e6:.6f} MB; "
                    f"total={traffic_bytes / 1e6:.6f} MB; SOL={sol_us:.3f} us; "
                    f"measured={ours:.3f} us; effective={effective_gbps:.2f} GB/s; "
                    f"attainment={100 * sol_us / ours:.2f}%",
                    flush=True,
                )


if __name__ == "__main__":
    HcPreBenchmark().run()
