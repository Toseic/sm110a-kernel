# Makefile

```bash
make
make bench
make bench-list
make clean
```

Override architecture when needed:

```bash
make CUDA_ARCH=sm_90a
```

Pass benchmark arguments through `BENCH_ARGS`:

```bash
make bench BENCH=bf16_gemm
make bench-list
make bench BENCH_ARGS="--no-accuracy"
make bench BENCH_ARGS="--m 4096 --n 4096 --k 4096 --warmup 5 --iters 20"
```

Add new benchmarks as `benchmarks/benchmark_<name>.cu` so `make bench-list`
and `make bench BENCH=<name>` pick them up automatically.
