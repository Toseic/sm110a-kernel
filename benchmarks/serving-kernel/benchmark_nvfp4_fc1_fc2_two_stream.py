#!/usr/bin/env python3
"""Two Green-Context streams for NVFP4 MoE FC1 or FC2, with shared weights."""

import argparse
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks"
for path in (ROOT, BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_nvfp4_fc1 import make_quantized_interleaved_weights
from benchmark_nvfp4_fc2 import make_case

from common import BenchmarkBase, benchmark_once
from sm110a_kernel import (
    Nvfp4Fc1Config,
    Nvfp4Fc2Config,
    compile_nvfp4_fc1,
    compile_nvfp4_fc2,
    create_green_contexts,
    nvfp4_fc1_out,
    nvfp4_fc2_out,
    prepare_nvfp4_fc1_metadata,
    prepare_nvfp4_fc2_metadata,
    quantize_nvfp4_rows,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", choices=("fc1", "fc2"), required=True)
    # Defaults intentionally match the largest standard point in the
    # corresponding standalone benchmark.  Keep them unset while parsing:
    # FC1 and FC2 use transposed N/K dimensions.
    p.add_argument("--tokens", type=int)
    p.add_argument("--experts", type=int)
    p.add_argument("--topk", type=int)
    p.add_argument("--n", type=int)
    p.add_argument("--k", type=int)
    p.add_argument("--sm-per-context", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--samples", type=int, default=7)
    a = p.parse_args()
    defaults = (
        dict(tokens=1024 * 32, experts=256, topk=8, n=2048, k=6144)
        if a.op == "fc1"
        else dict(tokens=1024 * 32, experts=256, topk=8, n=6144, k=2048)
    )
    for name, value in defaults.items():
        if getattr(a, name) is None:
            setattr(a, name, value)
    if (
        min(a.tokens, a.experts, a.topk, a.n, a.k, a.iters, a.samples) <= 0
        or a.topk > a.experts
    ):
        p.error("invalid positive dimensions/topk/iterations")
    if a.tokens % 128 or a.n % 128 or a.k % 64 or a.warmup < 0:
        p.error(
            "tokens/N/K must be divisible by 128/128/64; warmup must be non-negative"
        )
    return a


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cache = BenchmarkBase()
    cache.configure_cold_l2(args)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm_per_context = args.sm_per_context or props.multi_processor_count // 2
    if 2 * sm_per_context > props.multi_processor_count:
        raise SystemExit("two Green Context partitions exceed device SM count")

    torch.manual_seed(1234)
    if args.op == "fc1":
        # Matches the standalone FC1 autotune winner for this workload.
        config = Nvfp4Fc1Config(
            tile_n=128,
            tile_k=256,
            epilogue_tile_n=128,
            cluster_m=2,
        )
        bq, sfb = make_quantized_interleaved_weights(args.experts, args.n, args.k)
        work = []
        for _ in range(2):
            a = torch.randn((args.tokens, args.k), device="cuda", dtype=torch.bfloat16)
            aq, sfa = quantize_nvfp4_rows(a)
            router = torch.randn(
                (args.tokens, args.experts), device="cuda", dtype=torch.bfloat16
            )
            meta = prepare_nvfp4_fc1_metadata(
                router, args.topk, 2 * args.n, config=config
            )
            out = torch.empty(
                (args.tokens * args.topk, args.n // 2),
                device="cuda",
                dtype=torch.float4_e2m1fn_x2,
            )
            out_scale = torch.empty(
                (args.tokens * args.topk * (args.n // 16),),
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
            work.append((aq, sfa, meta, out, out_scale))
        compile_nvfp4_fc1(config)

        def launch(item, clusters):
            aq, sfa, meta, out, out_scale = item
            return nvfp4_fc1_out(
                aq,
                bq,
                sfa,
                sfb,
                meta,
                out,
                out_scale,
                config=config,
                max_active_clusters=clusters,
            )

        flops = 4 * args.tokens * args.topk * args.n * args.k
    else:
        # Matches the standalone FC2 autotune winner for this workload.
        config = Nvfp4Fc2Config(
            tile_n=128,
            tile_k=256,
            epilogue_tile_n=64,
            cluster_m=2,
        )
        cases = [
            make_case(args.tokens, args.experts, args.topk, args.n, args.k)
            for _ in range(2)
        ]
        bq, sfb = cases[0][1], cases[0][3]
        work = []
        for aq, _, sfa, _, routing, layout, weights, _ in cases:
            meta = prepare_nvfp4_fc2_metadata(
                layout, args.experts, args.n, config=config
            )
            work.append(
                (
                    aq,
                    sfa,
                    routing,
                    weights,
                    meta,
                    torch.empty(
                        (args.tokens, args.n), device="cuda", dtype=torch.bfloat16
                    ),
                )
            )
        compile_nvfp4_fc2(config)

        def launch(item, clusters):
            aq, sfa, routing, weights, meta, out = item
            return nvfp4_fc2_out(
                aq,
                bq,
                sfa,
                sfb,
                routing,
                weights,
                meta,
                args.topk,
                out,
                config=config,
                max_active_clusters=clusters,
            )

        flops = 2 * args.tokens * args.topk * args.n * args.k

    contexts = create_green_contexts((sm_per_context, sm_per_context))
    try:
        streams = [ctx.torch_stream() for ctx in contexts]
        clusters = [
            ctx.sm_count // (config.cluster_m * getattr(config, "cluster_n", 1))
            for ctx in contexts
        ]

        def enqueue():
            for stream, item, count in zip(streams, work, clusters):
                with torch.cuda.stream(stream):
                    launch(item, count)

        for _ in range(args.warmup):
            enqueue()
        for ctx in contexts:
            ctx.synchronize()
        samples = [[], []]
        for _ in range(args.samples):
            events = [[], []]
            for _ in range(args.iters):
                # Evict on the first Green Context, then gate both launches.
                with torch.cuda.stream(streams[0]):
                    if args._cold_l2_flush is not None:
                        args._cold_l2_flush()
                    gate = torch.cuda.Event()
                    gate.record()
                start = [torch.cuda.Event(True) for _ in contexts]
                end = [torch.cuda.Event(True) for _ in contexts]
                for stream, event in zip(streams, start):
                    stream.wait_event(gate)
                    event.record(stream)
                enqueue()
                for stream, event in zip(streams, end):
                    event.record(stream)
                    streams[0].wait_event(event)
                for i in range(2):
                    events[i].append((start[i], end[i], gate))
            for ctx in contexts:
                ctx.synchronize()
            for i in range(2):
                samples[i].append(
                    sum(s.elapsed_time(e) for s, e, _gate in events[i]) / args.iters
                )
    finally:
        for ctx in contexts:
            ctx.close()
    med = [statistics.median(x) for x in samples]
    perf = [flops / (x * 1e9) for x in med]
    full_clusters = props.multi_processor_count // (
        config.cluster_m * getattr(config, "cluster_n", 1)
    )

    def serial_pair():
        for item in work:
            launch(item, full_clusters)

    baseline = [
        benchmark_once(serial_pair, 0, args.iters, args._cold_l2_flush)
        for _ in range(args.samples)
    ]
    baseline_ms = statistics.median(baseline)
    baseline_tflops = 2 * flops / (baseline_ms * 1e9)
    print(f"[op] {args.op}; shared B/SFB; config={config}")
    print(f"[green] 2 contexts x {sm_per_context} SM; active clusters={clusters}")
    for i, (x, p) in enumerate(zip(med, perf)):
        print(f"[stream {i}] {x:.4f} ms; {p:.3f} TF/s")
    print(f"[result] aggregate: {sum(perf):.3f} TF/s")
    print(
        f"[baseline] serial 20-SM kernels: {baseline_ms:.4f} ms/pair; {baseline_tflops:.3f} TF/s"
    )
    print(f"[comparison] green/serial baseline: {sum(perf) / baseline_tflops:.3f}x")


if __name__ == "__main__":
    main()
