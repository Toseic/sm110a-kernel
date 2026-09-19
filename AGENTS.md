# Development guide

## Python entry points

```bash
python3 -c "import sm110a_kernel"
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py
```

## CUDA architecture override

The default JIT target is `sm_110a`. Override both architecture variables
together when needed:

```bash
CUDA_ARCH=sm_90a COMPUTE_CUDA_ARCH=compute_90a \
python3 -c "import sm110a_kernel"
```

## Benchmarks

Pass benchmark arguments directly to the benchmark script:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --no-accuracy
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --m 4096 --n 4096 --k 4096 --warmup 5 --iters 20
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py --baseline cublaslt
```

Add new benchmarks as `benchmarks/benchmark_<name>.py`.

When investigating a pipeline, fix the workload and kernel configuration so
autotuning and routing distribution do not obscure an injected delay.

## Pipeline delay probes

`include/pipeline_delay.h` provides compile-time device probes that perturb a
kernel-local cycle constant. Point names and defaults belong in that kernel's
local `*_pipeline_delay.cuh`; call the point from the lane/warp owning it:

```cpp
#include "pipeline_delay.h"
#include "my_kernel_pipeline_delay.cuh"

sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MY_KERNEL_STAGE>();
```

The JIT reads delay settings from the environment, passes them as CUDA defines,
and includes them in its in-process and persistent cache keys. Changing a
value produces a distinct kernel variant; do not manually clear
`SM110A_JIT_CACHE_DIR`. Zero is the default and compiles away.

Set multiple points with `SM110A_PIPELINE_DELAYS`:

```bash
SM110A_PIPELINE_DELAYS='producer_issue_tma=512,consumer_issue_mma=2048' \
  /home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py ...
```

Or set/override one point with an uppercase per-point variable:

```bash
SM110A_PIPELINE_DELAY_NVFP4_FC1_LOAD_A=1024 \
  /home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_nvfp4_fc1.py ...
```

Values must be non-negative integers. Per-point variables override packed
settings. The JIT forwards configured point names without a global registry;
the target kernel's local `.cuh` owns its supported macros. Use the lowercase
name in the packed form, or its uppercase form after `SM110A_PIPELINE_DELAY_`
for an individual variable. Kernel-specific documentation belongs in
`docs/pipeline_delay.md`.

For critical-path experiments, record a zero-delay baseline and perturb one
point at a time; start near 512 cycles, then try 2048 and 8192. No timing
change means the work is overlapped, while an immediate increase identifies a
critical or tightly coupled edge. The `clock64()` busy wait has small loop and
scheduling overhead, so interpret relative changes rather than exact latency.

Confirm a nonzero probe in SASS with `cuobjdump --dump-sass <kernel.so>`: look
for `CS2R ... SR_CLOCKLO`, comparison with the configured immediate, and a
backward branch. A zero-delay binary should not contain that loop. See
`docs/pipeline_delay.md` for the public API reference.

## Exclusive GPU launch handling

When GPU 0 is configured with `EXCLUSIVE_PROCESS`, a launch may fail with
`CUDA-capable device(s) is/are busy or unavailable` while another process is
using the device. Treat this as a transient condition: poll the GPU state at a
reasonable interval (for example, every 2 seconds) and retry the launch after
the active compute process exits. Do not start competing retries in a tight
loop; preserve the original command and report a timeout if a bounded wait is
required.
