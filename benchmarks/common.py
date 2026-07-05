"""Shared benchmarking utilities for sm110a-kernel benchmarks."""

import math
import statistics

import torch


TARGET_SAMPLE_MS = 50.0
PROBE_TARGET_MS = 10.0       # quick probe target — small enough to be fast
EARLY_STOP_RATIO = 1.3       # skip config if probe > best_probe * this


def tflops_from_ms(m, n, k, elapsed_ms):
    flops = 2.0 * m * n * k
    return flops / (elapsed_ms * 1.0e-3) / 1.0e12


def calibrate_iters(initial_ms, requested_iters, target_ms=TARGET_SAMPLE_MS):
    if initial_ms <= 0.0:
        return requested_iters
    total_ms = initial_ms * requested_iters
    if total_ms >= target_ms:
        return requested_iters
    scale = target_ms / total_ms
    return max(requested_iters, math.ceil(requested_iters * scale))


def summarize_samples(samples, measured_iters):
    samples = sorted(samples)
    return {
        "min_ms": samples[0],
        "median_ms": statistics.median(samples),
        "max_ms": samples[-1],
        "measured_iters": measured_iters,
    }


def benchmark_once(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_once_batched(fn, warmup, iters):
    for _ in range(warmup):
        fn(1)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn(iters)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_with_stats(fn, warmup, iters, samples):
    probe_ms = benchmark_once(fn, warmup, iters)
    measured_iters = calibrate_iters(probe_ms, iters)

    timings = [benchmark_once(fn, warmup, measured_iters)]
    for _ in range(1, samples):
        timings.append(benchmark_once(fn, 0, measured_iters))
    return summarize_samples(timings, measured_iters)


def benchmark_with_stats_batched(fn, warmup, iters, samples):
    probe_ms = benchmark_once_batched(fn, warmup, iters)
    measured_iters = calibrate_iters(probe_ms, iters)

    timings = [benchmark_once_batched(fn, warmup, measured_iters)]
    for _ in range(1, samples):
        timings.append(benchmark_once_batched(fn, 0, measured_iters))
    return summarize_samples(timings, measured_iters)


def quick_probe(fn):
    """Lightweight probe (~10 ms GPU time) for early-stop decisions."""
    ms = benchmark_once(fn, warmup=2, iters=5)
    iters = calibrate_iters(ms, 5, target_ms=PROBE_TARGET_MS)
    return benchmark_once(fn, warmup=0, iters=iters)
