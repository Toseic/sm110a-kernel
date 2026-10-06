---
name: benchmark
description: Use this skill when the user asks to benchmark this repository, run performance measurements, compare against a baseline, check accuracy during benchmarking, or add a new benchmark entry for this project.
---

# Benchmark

Use this skill when the user wants benchmark commands or benchmark execution for this repository.

## Entry points

- Import check: `python3 -c "import sm110a_kernel"`
- Benchmark scripts live under `benchmarks/` and follow the pattern `benchmark_<name>.py`

## Common commands

- List available benchmarks:
  `rg --files benchmarks -g 'benchmark_*.py'`
- Run a selected benchmark:
  `/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_<name>.py`
- Skip accuracy checks when the selected benchmark supports it:
  `/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_<name>.py --no-accuracy`
- Pass custom problem sizes or iteration counts directly to the selected benchmark script:
  `/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_<name>.py --m 4096 --n 4096 --k 4096 --warmup 5 --iters 20`
- Compare against a baseline when the selected benchmark supports it:
  `/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_<name>.py --baseline <baseline>`

## Architecture override

If the benchmark or import should target a different CUDA architecture, run commands with:

```bash
CUDA_ARCH=sm_90a COMPUTE_CUDA_ARCH=compute_90a \
python3 -c "import sm110a_kernel"
```

Apply the same environment override pattern to benchmark commands when needed.

## Workflow

1. Confirm whether the user wants a quick import check, a benchmark run, a baseline comparison, or a new benchmark script.
2. Discover the available benchmark scripts under `benchmarks/` and choose the one that matches the target kernel or user request.
3. Pass benchmark arguments directly to the selected script instead of wrapping it in another entry point.
4. If the user needs architecture-specific validation, set `CUDA_ARCH` and `COMPUTE_CUDA_ARCH` explicitly before running.
5. Report the exact command used and summarize the important performance or accuracy results.

## Adding benchmarks

Add new benchmark scripts as `benchmarks/benchmark_<name>.py`.

## Guardrails

- Prefer the documented Python entry points from this repository over ad hoc wrappers.
- Do not assume there is only one benchmark script or only one kernel.
- Do not silently change benchmark dimensions, warmup counts, iteration counts, or baseline choice.
- If a benchmark cannot run because the environment lacks the required GPU, CUDA setup, or built extension, report the blocker clearly.
