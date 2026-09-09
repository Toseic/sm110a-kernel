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

Autotune is resolved by the shared runner before the benchmark's timed loop.
To prewarm the same cache explicitly, invoke the runner with the workload and
the normal benchmark arguments:

```bash
python3 -m sm110a_kernel.runner bf16-gemm -- --m 4096 --n 4096 --k 4096
```

The runner records results in `profiles/autotune_cache.json`, partitioned by
the CUDA device, software stack, architecture settings, and source revision.
Use `--force-sweep` to re-run selection for an existing workload.  When using
the autotuned CuTE DSL baseline, `--force-sweep-ours` and
`--force-sweep-baseline` can re-run only the local-kernel or baseline sweep.

This runs a small `512 x 512 x 512` accuracy check first, then the default
performance sweep.

Disable the built-in accuracy phase:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --no-accuracy
```

Single shape:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --m 4096 --n 4096 --k 4096 --warmup 5 --iters 20

# Launch only four persistent clusters (instead of all available SMs)
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --m 4096 --n 4096 --k 4096 --max-active-clusters 4
```

Skip correctness check:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --m 8192 --n 4096 --k 4096 --skip-check
```

Use the old-style cuBLASLt baseline from Python:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --baseline cublaslt
```

Grouped M benchmark:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_grouped_m_gemm.py
```

The grouped M kernel is compiled on its first Python call. The default
configuration selects `cluster_m=1` for at most 128 average rows per expert and
`cluster_m=2` otherwise. A configuration can also be precompiled or supplied
explicitly:

```python
from sm110a_kernel import (
    GroupedMConfig,
    bf16_grouped_m_gemm_out,
    compile_bf16_grouped_m_gemm,
)

config = GroupedMConfig(
    tile_m=128,
    tile_n=256,
    tile_k=128,
    epilogue_tile_n=64,
    cluster_m=2,
)
compile_bf16_grouped_m_gemm(config)  # optional warm-up
bf16_grouped_m_gemm_out(
    a, b, grouped_layout, out, config=config, swap_ab=True
)
```

`epilogue_tile_n` supports `32`, `64`, and `128`.
The grouped kernel supports native MMA N shapes from 8 through 256 for
`cluster_m=1` and from 16 through 256 for `cluster_m=2`; the benchmark tunes
`tile_n` over `64`, `128`, `192`, and `256`.
With `swap_ab=True`, each expert computes the equivalent transposed
`B_e @ A_e.T` problem and transposes the epilogue back to the original
`(M, N)` output layout.

Compiled modules are reused in-process and through PyTorch's extension cache
across processes. Set `SM110A_JIT_CACHE_DIR` to choose the disk cache directory,
or `SM110A_JIT_VERBOSE=1` to print compilation commands.

The grouped M kernel uses the following K-major interface:

- `A`: `[M, K]`
- `B`: `[E, N, K]`
- `grouped_layout`: `[M]`

`grouped_layout` is the row-to-expert assignment for the top-k-permuted `A`.
Each aligned 128-row tile must contain one valid expert.

The current grouped benchmark keeps `M` aligned to `128` rows per tile and
uses the same `N`/`K` alignment constraints as the existing BF16 GEMM
baseline.

## Persistent-cluster limit

The BF16 GEMM entry points accept `max_active_clusters`. They launch
`max_active_clusters * config.cluster_m` CTAs. Leaving it as `None` uses the
tensor device's SM count divided by `config.cluster_m`, preserving the former
one-CTA-per-SM launch size.

```python
bf16_gemm_nt_out(a, b_nt, out, config=config, max_active_clusters=4)
```

The benchmark exposes the same control as `--max-active-clusters`; `0` selects
the device-derived default.

## Contents

- `src/bf16_gemm_kernel.cu`: BF16 GEMM implementation
- `src/bf16_gemm_split_torch_binding.cpp`: stable PyTorch binding for the
  configuration-specialized CUDA libraries
- `src/bf16_gemm_torch_binding.cpp`: cuBLASLt baseline binding
- `../../benchmarks/benchmark_bf16_gemm.py`: Python benchmark driver
- `../../benchmarks/benchmark_bf16_grouped_m_gemm.py`: Python benchmark driver for grouped M GEMM
- `../../sm110a_kernel/__init__.py`: Python API and core extension loader
- `../../sm110a_kernel/jit.py`: configuration-specialized grouped M JIT loader
- `include/common.h`: shared CUDA/cuBLAS helpers
- `include/bf16_gemm.h`: reusable BF16 GEMM declarations
