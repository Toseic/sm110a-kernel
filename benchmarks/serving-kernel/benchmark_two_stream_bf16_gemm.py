#!/usr/bin/env python3
"""Compare one full-M GEMM with two concurrent M-partitioned GEMM kernels."""

import argparse
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "benchmarks"))
from common import BenchmarkBase, benchmark_once
from sm110a_kernel import Bf16GemmConfig, bf16_gemm_nt_out, compile_bf16_gemm

CONFIG = Bf16GemmConfig(
    tile_m=128,
    tile_n=256,
    tile_k=64,
    cluster_m=2,
    epilogue_block_n=64,
)
STREAMS = 2
CLUSTERS_PER_STREAM = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--clusters-per-stream",
        type=int,
        default=CLUSTERS_PER_STREAM,
        help="persistent clusters allocated to each M half (default: 5)",
    )
    parser.add_argument(
        "--full-clusters",
        type=int,
        default=0,
        help="clusters for the full-M baseline (default: device SM count / cluster size)",
    )
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0:
        parser.error("--m, --n, and --k must be positive")
    if args.m % (2 * CONFIG.tile_m * CONFIG.cluster_m) or args.n % CONFIG.tile_n:
        parser.error("shape must satisfy M %% 512 == 0 and N %% 256 == 0")
    if args.k % CONFIG.tile_k:
        parser.error("K must be divisible by 64")
    if args.warmup < 0 or args.iters <= 0 or args.samples <= 0:
        parser.error("--warmup must be >= 0; --iters and --samples must be positive")
    if args.clusters_per_stream <= 0 or args.full_clusters < 0:
        parser.error(
            "--clusters-per-stream must be positive; --full-clusters must be non-negative"
        )
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    cache = BenchmarkBase()
    cache.configure_cold_l2(args)

    config = CONFIG
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    full_clusters = (
        args.full_clusters or props.multi_processor_count // CONFIG.cluster_m
    )
    split_ctas = STREAMS * args.clusters_per_stream * CONFIG.cluster_m
    full_ctas = full_clusters * CONFIG.cluster_m
    if max(split_ctas, full_ctas) > props.multi_processor_count:
        raise SystemExit(
            f"requested up to {max(split_ctas, full_ctas)} CTAs, but device has "
            f"only {props.multi_processor_count} SMs"
        )

    torch.manual_seed(1234)
    # Both streams consume disjoint M rows of one A, one shared B, and
    # disjoint C rows: together they perform exactly the full-M GEMM.
    a_full = torch.empty(
        (args.m, args.k), device="cuda", dtype=torch.bfloat16
    ).uniform_(-1.0, 1.0)
    b = torch.empty((args.k, args.n), device="cuda", dtype=torch.bfloat16).uniform_(
        -1.0, 1.0
    )
    b_nt = b.transpose(0, 1).contiguous()
    out_split = torch.empty((args.m, args.n), device="cuda", dtype=torch.bfloat16)
    out_full = torch.empty((args.m, args.n), device="cuda", dtype=torch.bfloat16)
    m_per_stream = args.m // STREAMS
    a = list(a_full.split(m_per_stream, dim=0))
    out = list(out_split.split(m_per_stream, dim=0))
    streams = [torch.cuda.Stream() for _ in range(STREAMS)]

    # Compile and execute once before timing, keeping JIT and first-use costs out.
    compile_bf16_gemm(config)
    torch.cuda.synchronize()

    def enqueue_pair():
        for stream, a_i, out_i in zip(streams, a, out):
            with torch.cuda.stream(stream):
                bf16_gemm_nt_out(
                    a_i,
                    b_nt,
                    out_i,
                    config=config,
                    max_active_clusters=args.clusters_per_stream,
                )

    def enqueue_full():
        bf16_gemm_nt_out(
            a_full,
            b_nt,
            out_full,
            config=config,
            max_active_clusters=full_clusters,
        )

    for _ in range(args.warmup):
        enqueue_pair()
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        enqueue_full()
    torch.cuda.synchronize()

    def timed_batch():
        timings = [[], []]
        for _ in range(args.iters):
            cache_flush = args._cold_l2_flush
            if cache_flush is not None:
                cache_flush()
            gate = torch.cuda.Event()
            gate.record()
            start = [torch.cuda.Event(enable_timing=True) for _ in streams]
            done = [torch.cuda.Event(enable_timing=True) for _ in streams]
            for stream, event in zip(streams, start):
                stream.wait_event(gate)
                event.record(stream)
            enqueue_pair()
            for stream, event in zip(streams, done):
                event.record(stream)
                torch.cuda.current_stream().wait_event(event)
            for i in range(STREAMS):
                timings[i].append((start[i], done[i], gate))
        torch.cuda.synchronize()
        return [
            sum(begin.elapsed_time(end) for begin, end, _gate in events) / args.iters
            for events in timings
        ]

    def timed_full():
        return benchmark_once(enqueue_full, 0, args.iters, args._cold_l2_flush)

    samples_ms = list(zip(*(timed_batch() for _ in range(args.samples))))
    full_samples_ms = [timed_full() for _ in range(args.samples)]
    median_ms = [statistics.median(values) for values in samples_ms]
    full_median_ms = statistics.median(full_samples_ms)
    full_flops = 2 * args.m * args.n * args.k
    tflops = [full_flops / STREAMS / (value * 1e9) for value in median_ms]
    full_tflops = full_flops / (full_median_ms * 1e9)

    print(f"[device] {props.name}: {props.multi_processor_count} SMs")
    print(
        f"[launch] {STREAMS} normal streams x {args.clusters_per_stream} clusters "
        f"x cluster size {CONFIG.cluster_m} = {split_ctas} CTAs"
    )
    print(
        f"[split] A[{args.m}, {args.k}] split into 2 x [{m_per_stream}, {args.k}]; B is shared"
    )
    for index, (values, value, perf) in enumerate(zip(samples_ms, median_ms, tflops)):
        print(
            f"[stream {index}] samples: "
            + ", ".join(f"{sample:.4f}" for sample in values)
            + f" ms/slice; median: {value:.4f} ms; {perf:.3f} TF/s"
        )
    print(f"[split result] sum of stream throughput: {sum(tflops):.3f} TF/s")
    print(
        f"[full result] 1 kernel x {full_clusters} clusters ({full_ctas} CTAs): "
        f"{full_median_ms:.4f} ms; {full_tflops:.3f} TF/s"
    )
    print(f"[comparison] split/full throughput: {sum(tflops) / full_tflops:.3f}x")


if __name__ == "__main__":
    main()
