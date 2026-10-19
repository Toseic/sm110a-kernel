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
The default sweep includes `M=1, 4, 8, 16, 32` at `N=K=4096`. These shapes
run directly with `cluster_m=1`; TMA fills rows beyond the logical M with zero
and suppresses out-of-bounds output stores. Pass `--m 1` (or another listed M)
to run just that default shape.

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

For `split_k=1`, the BF16 GEMM entry points accept `max_active_clusters`. They launch
`max_active_clusters * config.cluster_m` CTAs. Leaving it as `None` uses the
tensor device's SM count divided by `config.cluster_m`, preserving the former
one-CTA-per-SM launch size.

```python
bf16_gemm_nt_out(a, b_nt, out, config=config, max_active_clusters=4)
```

The benchmark exposes the same control as `--max-active-clusters`; `0` selects
the device-derived default.

## Split-K

Set `split_k` on `Bf16GemmConfig` to `1`, `2`, `4`, or `8`. The product
`cluster_m * split_k` must be at most `8`; `split_k=8` therefore requires
`cluster_m=1`. For `split_k > 1`, `swap_ab` must be `False`, and `K` must be
divisible by `tile_k * split_k`. The cluster shape is `(cluster_m, split_k, 1)`: each
cluster-N rank computes one K partition, then the CTAs exchange FP32 partial
tiles through distributed shared memory. Each rank reduces and stores a
different group of output rows, converting to BF16 after reduction.

```python
from sm110a_kernel import Bf16GemmConfig, bf16_gemm_nt_out

config = Bf16GemmConfig(cluster_m=1, split_k=8)
bf16_gemm_nt_out(a, b_nt, out, config=config)
```

The split-K path launches one cluster per output tile. `max_active_clusters`
continues to control only the persistent `split_k=1` path.
The benchmark autotuner considers compatible split-K configurations and
`swap_ab` only with `split_k=1`.

## Tile-M

`Bf16GemmConfig.tile_m` supports `64` and `128`. `tile_m=64` uses the
Blackwell non-`.ws` M=64 tensor-memory layout and requires `cluster_m=1`;
the autotuner considers both tile sizes where the workload is compatible.

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

## FP32 output

Set `Bf16GemmConfig(output_fp32=True)` to store the FP32 accumulator directly,
without rounding through BF16. Inputs remain BF16; the default output is BF16.
This option specializes the epilogue and is included in the JIT cache key.
It supports both B layouts, `swap_ab`, and split-K (whose existing restriction
against `swap_ab` still applies). Preallocated outputs must match the option.
FP32 doubles the epilogue buffers, so some large epilogues need a smaller
`tile_k` or `tile_m` to retain the required two mainloop stages; validation
rejects configurations that exceed shared memory.

```python
import torch
from sm110a_kernel import Bf16GemmConfig, bf16_gemm_nt_out

config = Bf16GemmConfig(cluster_m=1, output_fp32=True)
out = torch.empty((a.size(0), b_nt.size(0)), device=a.device, dtype=torch.float32)
bf16_gemm_nt_out(a, b_nt, out, config=config)
```

FP32 output preserves accumulator precision; it does not change the BF16 MMA
arithmetic or guarantee bitwise equality with an FP32-input cuBLAS operation.

Benchmark and autotune FP32 output with the PyTorch BF16-to-FP32 baseline:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --output-fp32 --baseline torch --m 1 --n 4096 --k 4096
```

Autotune keys include output dtype, so BF16 and FP32 selections are independent.
FP32 candidates are validated with their larger epilogue buffers before tuning.
The benchmark reports the output dtype and uses it in the SOL memory estimate.
