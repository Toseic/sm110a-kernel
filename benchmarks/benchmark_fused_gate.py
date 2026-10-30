#!/usr/bin/env python3
"""Cold-L2 CUDA Graph timing: fused sqrtsoftplus MoE gate versus PyTorch."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GMEM_BW_GBPS, BenchmarkBase, benchmark_with_stats
from sm110a_kernel import compile_fused_gate


def reference(logits, topk, scale, bias, ids):
    scores = F.softplus(logits).sqrt()
    if ids is None:
        ids = (scores if bias is None else scores + bias).topk(topk, dim=-1).indices
    weights = scores.gather(1, ids)
    weights /= weights.sum(-1, keepdim=True)
    weights *= scale
    return weights, ids


class GateBenchmark(BenchmarkBase):
    def build_parser(self):
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("--experts", type=int, default=256)
        p.add_argument("--topk", type=int, default=6)
        p.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128])
        p.add_argument("--mode", choices=["hash", "topk", "both"], default="both")
        p.add_argument("--route-scale", type=float, default=1.5)
        p.add_argument("--pdl", action="store_true")
        p.add_argument("--no-accuracy", action="store_true")
        p.add_argument(
            "--sol",
            action="store_true",
            help="report compulsory-byte memory SOL (not compute SOL)",
        )
        self.add_timing_args(p, warmup=5, iters=20, samples=5)
        return p

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if min(args.tokens) < 1:
            raise SystemExit("tokens must be positive")

    def run_with_args(self, args):
        torch.manual_seed(self.seed)
        module = compile_fused_gate(args.experts, args.topk, pdl=args.pdl)
        print("mode tokens fused(us) torch(us) speedup max_abs_error")
        if args.sol:
            print(
                f"Memory SOL: compulsory bytes / {GMEM_BW_GBPS:g} GB/s; excludes transaction amplification and compute/launch bounds"
            )
        for mode in ["hash", "topk"] if args.mode == "both" else [args.mode]:
            for tokens in args.tokens:
                logits = torch.randn(tokens, args.experts, device="cuda")
                bias = (
                    torch.randn(args.experts, device="cuda") if mode == "topk" else None
                )
                ids = (
                    torch.randint(
                        args.experts,
                        (tokens, args.topk),
                        device="cuda",
                        dtype=torch.int32,
                    )
                    if mode == "hash"
                    else None
                )
                weights = torch.empty(tokens, args.topk, device="cuda")
                indices = torch.empty(
                    tokens,
                    args.topk,
                    device="cuda",
                    dtype=torch.int32 if ids is not None else torch.int64,
                )
                run = lambda: module.fused_gate_out(
                    logits,
                    weights,
                    indices,
                    args.route_scale,
                    bias=bias,
                    hash_indices=ids,
                )
                baseline = lambda: reference(
                    logits, args.topk, args.route_scale, bias, ids
                )
                run()
                error = "-"
                if not args.no_accuracy:
                    ref, ref_ids = baseline()
                    torch.testing.assert_close(indices, ref_ids, rtol=0, atol=0)
                    torch.testing.assert_close(weights, ref, rtol=2e-6, atol=2e-7)
                    error = f"{(weights - ref).abs().max().item():.3g}"
                graphs = []
                for fn in (run, baseline):
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured_output = fn()
                    # Keep allocations returned by the reference alive.
                    graphs.append((graph, captured_output))
                timings = [
                    benchmark_with_stats(
                        graph.replay,
                        args.warmup,
                        args.iters,
                        args.samples,
                        cold_l2_flush=args._cold_l2_flush,
                    )["median_ms"]
                    for graph, _ in graphs
                ]
                print(
                    f"{mode} {tokens} {timings[0] * 1000:.3f} {timings[1] * 1000:.3f} {timings[1] / timings[0]:.2f}x {error}",
                    flush=True,
                )
                if args.sol:
                    # Hash: selected logits + input IDs + weights + output IDs.
                    # Top-k: full logits + bias (once) + weights + int64 IDs.
                    byte_count = (
                        tokens * args.topk * 16
                        if mode == "hash"
                        else tokens * (args.experts * 4 + args.topk * 12)
                        + args.experts * 4
                    )
                    bound_us = byte_count / (GMEM_BW_GBPS * 1000)
                    gbps = byte_count / (timings[0] * 1e6)
                    print(
                        f"  memory_SOL: bytes={byte_count} bound={bound_us:.6f} us "
                        f"effective={gbps:.4f} GB/s attainment={100 * gbps / GMEM_BW_GBPS:.4f}%",
                        flush=True,
                    )


if __name__ == "__main__":
    GateBenchmark().run()
