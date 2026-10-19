#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import queue
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_rank(rank, world_size, unique_id, count, warmup, iters, results, barrier):
    import torch

    from sm110a_kernel.kernels.nccl_lsa import NcclLsaAllReduce, NcclLsaRunner

    torch.cuda.set_device(0)
    values = torch.arange(1, count + 1, device="cuda", dtype=torch.float32)
    values.add_(100.0 * rank)
    output = torch.empty_like(values)

    communicator = NcclLsaAllReduce(
        unique_id,
        rank=rank,
        world_size=world_size,
        capacity=count,
        device=0,
    )
    try:
        runner = NcclLsaRunner(communicator, values, output)
        runner.run()
        for _ in range(warmup):
            runner.run()
        torch.cuda.synchronize()

        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        eviction = torch.empty(max(64 << 20, 2 * l2), device="cuda", dtype=torch.uint8)
        timings = []
        for _ in range(iters):
            eviction.zero_()
            torch.cuda.synchronize()
            # Complete both ranks' evictions before either records a timed event.
            barrier.wait(timeout=30)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runner.run()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))

        expected = world_size * torch.arange(
            1, count + 1, device="cuda", dtype=torch.float32
        )
        expected.add_(100.0 * world_size * (world_size - 1) / 2)
        max_abs_diff = (output - expected).abs().max().item()
        results.put((rank, sum(timings) / iters, max_abs_diff))
    finally:
        torch.cuda.synchronize()
        communicator.close()


def main():
    parser = argparse.ArgumentParser(
        description="Minimal NCCL Device API LSA all-reduce"
    )
    parser.add_argument("--count", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if args.count <= 0 or args.warmup < 0 or args.iters <= 0:
        parser.error("count and iters must be positive; warmup must be non-negative")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("NCCL_MULTI_RANK_GPU_ENABLE", "1")
    os.environ.setdefault("NCCL_MAX_CTAS", "4")

    from sm110a_kernel.kernels.nccl_lsa import nccl_unique_id

    world_size = 2
    unique_id = nccl_unique_id()
    context = mp.get_context("spawn")
    results = context.Queue()
    barrier = context.Barrier(world_size)
    processes = [
        context.Process(
            target=_run_rank,
            args=(
                rank,
                world_size,
                unique_id,
                args.count,
                args.warmup,
                args.iters,
                results,
                barrier,
            ),
        )
        for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
        for process in processes:
            if process.is_alive():
                raise SystemExit(f"rank process {process.pid} timed out")
            if process.exitcode != 0:
                raise SystemExit(
                    f"rank process {process.pid} exited with {process.exitcode}"
                )
    finally:
        # A failed rank can leave its peer blocked inside NCCL initialization.
        # Reap every child before exiting, including when the first rank fails.
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join()

    rows = []
    try:
        for _ in range(world_size):
            rows.append(results.get(timeout=5))
    except queue.Empty as exc:
        raise SystemExit("rank result queue was incomplete") from exc

    print("[cache] cold L2: evicted on both ranks before each collective")
    print("Rank    Time (ms)    Max abs diff")
    for rank, elapsed_ms, max_abs_diff in sorted(rows):
        print(f"{rank:<8}{elapsed_ms:<13.4f}{max_abs_diff:.6f}")
    if any(diff != 0.0 for _, _, diff in rows):
        raise SystemExit("accuracy check failed")


if __name__ == "__main__":
    main()
