# BF16 GEMM Scaffold

## Build

```bash
make
```

Override architecture if needed:

```bash
make CUDA_ARCH=sm_90a
```

## Run

Default benchmark sweep:

```bash
make bench
make bench BENCH=bf16_gemm
make bench-list
```

This runs a small `512 x 512 x 512` accuracy check first, then the default
performance sweep with `--skip-check`.

Disable the built-in accuracy phase:

```bash
make bench BENCH_ARGS="--no-accuracy"
```

Single shape:

```bash
make bench BENCH=bf16_gemm BENCH_ARGS="--m 4096 --n 4096 --k 4096 --warmup 5 --iters 20"
```

Skip correctness check:

```bash
make bench BENCH=bf16_gemm BENCH_ARGS="--m 8192 --n 4096 --k 4096 --skip-check"
```

## Contents

- `src/bf16_gemm.cu`: BF16 GEMM implementation
- `../../benchmarks/benchmark_bf16_gemm.cu`: benchmark driver and cuBLASLt baseline
- `include/common.h`: shared CUDA/cuBLAS helpers
- `include/bf16_gemm.h`: reusable BF16 GEMM declarations
