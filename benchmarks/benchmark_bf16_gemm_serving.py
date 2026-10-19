#!/usr/bin/env python3

"""Persistent BF16 GEMM serving correctness and latency benchmark."""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import B300Sim, BenchmarkBase
from sm110a_kernel import (
    Bf16GemmServingConfig,
    compile_bf16_gemm_serving,
    create_green_contexts,
    sm_resource_info,
)

SERVING_SMS = 16
PRODUCER_SMS = 2
CONSUMER_SMS = 2
M = 256
N = 4096
K = 4096


def create_serving_contexts():
    partitions = (SERVING_SMS, PRODUCER_SMS, CONSUMER_SMS)
    info = sm_resource_info()
    alignment = info["sm_coscheduled_alignment"]
    if sum(partitions) > info["sm_count"] or any(
        count < info["min_sm_partition_size"] or count % alignment
        for count in partitions
    ):
        raise RuntimeError(
            "benchmark requires valid 16-SM serving, 2-SM producer, and "
            f"2-SM consumer Green Contexts; device info is {info}"
        )
    return create_green_contexts(partitions)


def release_green_resources(contexts):
    for context in contexts:
        context.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    for context in contexts:
        context.close()


def run_prepublished_case(module, groups, b_nt, *, slots, clusters, bypass=False):
    queue = module.create_queue(b_nt, slot_count=slots, max_active_clusters=clusters)
    for rows in groups:
        queue.test_publish(rows, flush=rows.shape[0] != M, bypass=bypass)
    queue.request_drain()
    queue.start()
    queue.wait_stopped()
    outputs = [queue.test_consume(bypass=bypass) for _ in groups]
    torch.cuda.synchronize()
    return queue, outputs


def check_outputs(groups, outputs, b_nt):
    max_abs_diff = 0.0
    for rows, output in zip(groups, outputs):
        reference = rows @ b_nt.T
        max_abs_diff = max(
            max_abs_diff,
            (output.float() - reference.float()).abs().max().item(),
        )
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=1.0)
    return max_abs_diff


def run_persistent_serving_tests():
    print("[test] persistent serving queue correctness")
    module = compile_bf16_gemm_serving(Bf16GemmServingConfig())
    cases = (
        ("single-full", [M], 256, 2, 1, False),
        ("single-flush", [173], 256, 2, 1, False),
        ("multi-group-multi-n", [M, 119], 512, 4, 4, False),
        ("bypass-state-machine", [M, 137], 256, 4, 2, True),
    )
    for case_index, (name, row_counts, n, slots, clusters, bypass) in enumerate(cases):
        torch.manual_seed(9000 + case_index)
        b_nt = torch.randn((n, K), device="cuda", dtype=torch.bfloat16)
        groups = [
            torch.randn((count, K), device="cuda", dtype=torch.bfloat16)
            for count in row_counts
        ]
        queue, outputs = run_prepublished_case(
            module, groups, b_nt, slots=slots, clusters=clusters, bypass=bypass
        )
        max_abs_diff = (
            check_outputs(groups, outputs, b_nt) if not bypass else float("nan")
        )
        expected_work = len(groups) * (n // 256)
        dispatch_head = int(queue.control[256:264].view(torch.int64).item())
        recycle_head = int(queue.control[384:392].view(torch.int64).item())
        lifecycle = queue.control[512:524].view(torch.int32).cpu().tolist()
        if dispatch_head != expected_work or recycle_head != len(groups):
            raise AssertionError(
                f"{name}: dispatch={dispatch_head}, recycle={recycle_head}"
            )
        if lifecycle != [3, clusters, clusters]:
            raise AssertionError(f"{name}: invalid lifecycle {lifecycle}")
        diff_text = (
            f"max_diff={max_abs_diff:.3f}"
            if not bypass
            else "outputs unchecked (bypass)"
        )
        print(
            f"  PASS {name}: groups={len(groups)}, N={n}, K={K}, "
            f"work_items={expected_work}, {diff_text}"
        )

    contexts = create_serving_contexts()
    persistent_stream = contexts[0].torch_stream()
    producer_stream = contexts[1].torch_stream()
    consumer_stream = contexts[2].torch_stream()
    torch.manual_seed(9010)
    b_nt = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    groups = [
        torch.randn((M, K), device="cuda", dtype=torch.bfloat16),
        torch.randn((137, K), device="cuda", dtype=torch.bfloat16),
        torch.randn((M, K), device="cuda", dtype=torch.bfloat16),
    ]
    outputs = [
        torch.empty((rows.shape[0], N), device="cuda", dtype=torch.bfloat16)
        for rows in groups
    ]
    torch.cuda.synchronize()
    queue = None
    try:
        queue = module.create_queue(
            b_nt,
            slot_count=2,
            max_active_clusters=SERVING_SMS // 2,
            persistent_stream=persistent_stream,
            control_stream=producer_stream,
        )
        queue.start()
        with torch.cuda.stream(producer_stream):
            for rows in groups:
                queue.test_publish(rows, flush=rows.shape[0] != M)
            queue.request_drain()
        with torch.cuda.stream(consumer_stream):
            for index in range(len(groups)):
                queue.test_consume(out=outputs[index])
        queue.wait_stopped()
        contexts[1].synchronize()
        contexts[2].synchronize()

        max_abs_diff = check_outputs(groups, outputs, b_nt)
        state_word = int(queue.control[640:648].view(torch.int64).item())
        dispatch_head = int(queue.control[256:264].view(torch.int64).item())
        recycle_head = int(queue.control[384:392].view(torch.int64).item())
        expected_work = len(groups) * (N // 256)
        if (
            state_word != len(groups) << 16
            or dispatch_head != expected_work
            or recycle_head != len(groups)
        ):
            raise AssertionError(
                "online-wrap: invalid final queue state "
                f"state={state_word}, dispatch={dispatch_head}, recycle={recycle_head}"
            )
        print(
            "  PASS online-empty-wakeup-wrap: serving=16 SM, producer=2 SM, "
            "consumer=2 SM, N=4096, K=4096, 16 N tiles/group, "
            f"capacity=1 group, tickets=0,1,2, max_diff={max_abs_diff:.3f}"
        )
    finally:
        queue = None
        persistent_stream = producer_stream = consumer_stream = None
        b_nt = groups = outputs = None
        release_green_resources(contexts)

    print(f"[test] PASS ({len(cases) + 1} cases)")


def warm_gpu_clocks(b_nt, iterations=20):
    """Ramp SM clocks to the loaded state before timing.

    Thor DVFS drops the SM clock to a fraction of the loaded rate when the GPU
    is lightly loaded, and paced producer phases start from that state. Without
    this warmup the first ~30ms of a phase run 5x slower and poison 16-group
    averages (the "500us knee" measurement artifact).
    """
    warm = torch.empty((N, N), dtype=torch.bfloat16, device=b_nt.device)
    for _ in range(iterations):
        torch.mm(b_nt, b_nt.T, out=warm)
    torch.cuda.synchronize()


def measure_sm_clock_hz():
    """Calibrate the running SM clock by timing a known cycle-count spin.

    torch's reported clock_rate matches neither the idle nor the loaded clock,
    so producer delay_cycles computed from it are off by ~1.5x in wall time.
    """
    cycles = 8_000_000
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.cuda._sleep(cycles)
    torch.cuda.synchronize()
    return cycles / (time.perf_counter() - started)


def run_producer_speed_sweep(args):
    module = compile_bf16_gemm_serving(Bf16GemmServingConfig())
    torch.manual_seed(args.seed)
    b_nt = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    rows = [
        torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        for _ in range(args.groups)
    ]
    outputs = [
        torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        for _ in range(args.groups)
    ]
    references = [group @ b_nt.T for group in rows] if not args.bypass_io else []
    torch.cuda.synchronize()

    # Force CUDA module/function/TMA setup before all 20 SMs are assigned to
    # Green Contexts. This prepublished compute is outside measurement.
    warmup_queue = module.create_queue(b_nt, slot_count=2, max_active_clusters=1)
    warmup_queue.test_publish(rows[0])
    warmup_queue.request_drain()
    warmup_queue.start()
    warmup_queue.wait_stopped()
    warmup_queue.test_consume(out=outputs[0])
    torch.cuda.synchronize()
    warmup_queue = None
    gc.collect()

    contexts = create_serving_contexts()
    persistent_stream = contexts[0].torch_stream()
    producer_stream = contexts[1].torch_stream()
    consumer_stream = contexts[2].torch_stream()
    queue = None
    eviction = torch.empty(
        max(64 << 20, 2 * torch.cuda.get_device_properties(b_nt.device).L2_cache_size),
        device=b_nt.device,
        dtype=torch.uint8,
    )
    queues = [
        module.create_queue(
            b_nt,
            slot_count=2 * args.capacity,
            max_active_clusters=SERVING_SMS // 2,
            persistent_stream=persistent_stream,
            control_stream=producer_stream,
        )
        for _ in args.producer_delays_us
    ]
    if args.bypass_io:
        # Bypass skips the producer payload copy, so torch.empty() can leave
        # the A ring backed by zero-filled, highly compressible fresh pages.
        # Initialize it outside the timed region to keep the GEMM's logical
        # work unchanged without giving A loads and the resulting C stores an
        # unrealistic data-compression advantage.
        for queue in queues:
            queue.a_ring.normal_()
    torch.cuda.synchronize()
    try:
        warm_gpu_clocks(b_nt)
        clock_hz = measure_sm_clock_hz()
        print(f"[clock] calibrated SM clock during pacing: {clock_hz / 1.0e6:.0f} MHz")
        print("[serving] producer arrival-rate sweep (cold L2 at phase start)")
        print("[resources] serving=16 SM (8 clusters), producer=2 SM, consumer=2 SM")
        print(
            f"[queue] capacity={args.capacity} groups, groups/run={args.groups}, "
            "M=256, N=4096, K=4096, 16 N tiles/group"
        )
        if args.bypass_io:
            print(
                "[mode] payload bypass: producer/consumer skip A/C-ring copies; "
                "A-ring is prefilled with random data; outputs are not validated"
            )
        print(
            "Delay(us)     Pace ceiling/s    Total groups/s   Steady groups/s  "
            "Steady ms/group  Steady TF/s      MaxDiff"
        )
        throughputs = []
        for delay_us, queue in zip(args.producer_delays_us, queues):
            warm_gpu_clocks(b_nt)
            # The complete streaming phase starts cold; intra-phase reuse is part of the workload.
            eviction.zero_()
            torch.cuda.synchronize()
            queue.start()
            delay_cycles = round(delay_us * clock_hz / 1.0e6)
            started = time.perf_counter()
            completions = [
                torch.cuda.Event(enable_timing=True) for _ in range(args.groups)
            ]
            with torch.cuda.stream(producer_stream):
                for group in range(args.groups):
                    queue.test_publish(
                        rows[group], delay_cycles=delay_cycles, bypass=args.bypass_io
                    )
                queue.request_drain()
            with torch.cuda.stream(consumer_stream):
                for group in range(args.groups):
                    queue.test_consume(out=outputs[group], bypass=args.bypass_io)
                    completions[group].record()
            queue.wait_stopped()
            contexts[1].synchronize()
            contexts[2].synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            if args.bypass_io:
                max_abs_diff = float("nan")
            else:
                max_abs_diff = max(
                    (output.float() - reference.float()).abs().max().item()
                    for output, reference in zip(outputs, references)
                )
                for output, reference in zip(outputs, references):
                    torch.testing.assert_close(output, reference, rtol=2e-2, atol=1.0)
            groups_per_second = args.groups * 1000.0 / elapsed_ms
            intervals_ms = [
                completions[index - 1].elapsed_time(completions[index])
                for index in range(1, args.groups)
            ]
            steady_intervals = intervals_ms[len(intervals_ms) // 4 :]
            steady_groups_per_second = (
                1000.0 * len(steady_intervals) / sum(steady_intervals)
            )
            throughputs.append(steady_groups_per_second)
            steady_latency_ms = sum(steady_intervals) / len(steady_intervals)
            steady_tflops = 2.0 * M * N * K * steady_groups_per_second / 1.0e12
            target = "unlimited" if delay_us == 0 else f"{1.0e6 / delay_us:.1f}"
            diff_text = "n/a (bypass)" if args.bypass_io else f"{max_abs_diff:.3f}"
            print(
                f"{delay_us:<14.1f}{target:<18}{groups_per_second:<17.1f}"
                f"{steady_groups_per_second:<17.1f}{steady_latency_ms:<17.4f}"
                f"{steady_tflops:<17.3f}{diff_text}"
            )
        print(
            f"[compute] peak steady completion rate: {max(throughputs):.1f} "
            "groups/s across the producer pacing sweep"
        )
    finally:
        queue = None
        queues = None
        persistent_stream = producer_stream = consumer_stream = None
        outputs = references = rows = b_nt = module = None
        release_green_resources(contexts)


def parse_args():
    parser = argparse.ArgumentParser()
    BenchmarkBase.add_b300_sim_args(parser)
    parser.add_argument(
        "--test",
        action="store_true",
        help="run persistent serving correctness tests instead of timing",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument(
        "--producer-delays-us",
        default="0,500,1000,2000,4000,8000",
        help="comma-separated producer pacing delays in microseconds",
    )
    parser.add_argument(
        "--bypass-io",
        action="store_true",
        help="producer/consumer skip A/C-ring payload copies; measures the "
        "queue and serving kernel without test I/O overhead",
    )
    args = parser.parse_args()
    if args.groups <= 0 or args.capacity <= 0:
        raise SystemExit("--groups and --capacity must be positive")
    try:
        args.producer_delays_us = tuple(
            float(value) for value in args.producer_delays_us.split(",")
        )
    except ValueError as exc:
        raise SystemExit(
            "--producer-delays-us must be comma-separated numbers"
        ) from exc
    if not args.producer_delays_us or any(
        value < 0 for value in args.producer_delays_us
    ):
        raise SystemExit("producer delays must be non-negative")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    with B300Sim(args):
        if args.test:
            run_persistent_serving_tests()
        else:
            run_producer_speed_sweep(args)


if __name__ == "__main__":
    main()
