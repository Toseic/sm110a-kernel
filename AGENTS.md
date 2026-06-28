# Python Entry Points

```bash
python3 -c "import sm110a_kernel"
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py
```

Override architecture when needed:

```bash
CUDA_ARCH=sm_90a COMPUTE_CUDA_ARCH=compute_90a \
python3 -c "import sm110a_kernel"
```

Pass benchmark arguments directly to the benchmark script:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --no-accuracy
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --m 4096 --n 4096 --k 4096 --warmup 5 --iters 20
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --baseline cublaslt
```

Add new benchmarks as `benchmarks/benchmark_<name>.py`.
