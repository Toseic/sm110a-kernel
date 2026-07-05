# BF16 GEMM Scaffold

## Build

```bash
python3 -c "import sm110a_kernel"
```

Override architecture if needed:

```bash
CUDA_ARCH=sm_90a COMPUTE_CUDA_ARCH=compute_90a \
python3 -c "import sm110a_kernel"
```

This builds the PyTorch CUDA extension through `torch.utils.cpp_extension`.

## Run

Default benchmark sweep:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py
```

This runs a small `512 x 512 x 512` accuracy check first, then the default
performance sweep.

Disable the built-in accuracy phase:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --no-accuracy
```

Single shape:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --m 4096 --n 4096 --k 4096 --warmup 5 --iters 20
```

Skip correctness check:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --m 8192 --n 4096 --k 4096 --skip-check
```

Use the old-style cuBLASLt baseline from Python:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --baseline cublaslt
```

Dynamic-schedule benchmark:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm_dynamic_schedule.py
```

Grouped M benchmark:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_grouped_m_gemm.py
```

The grouped M kernel uses the following K-major interface:

- `A`: `[M, K]`
- `B`: `[E, N, K]`
- `grouped_layout`: `[M]`

`grouped_layout` is the row-to-expert assignment for the top-k-permuted `A`.
Each aligned 128-row tile must contain one valid expert.

The current grouped benchmark keeps `M` aligned to `128` rows per tile and
uses the same `N`/`K` alignment constraints as the existing BF16 GEMM
baseline.

## Contents

- `src/bf16_gemm.cu`: BF16 GEMM implementation
- `src/bf16_gemm_dynamic_schedule.cu`: BF16 GEMM implementation with dynamic schedule entry point
- `src/torch_binding.cpp`: PyTorch binding entry points
- `../../benchmarks/benchmark_bf16_gemm.py`: Python benchmark driver
- `../../benchmarks/benchmark_bf16_gemm_dynamic_schedule.py`: Python benchmark driver for the dynamic schedule entry point
- `../../benchmarks/benchmark_bf16_grouped_m_gemm.py`: Python benchmark driver for grouped M GEMM
- `../../sm110a_kernel/__init__.py`: lazy JIT loader for the CUDA extension
- `include/common.h`: shared CUDA/cuBLAS helpers
- `include/bf16_gemm.h`: reusable BF16 GEMM declarations
