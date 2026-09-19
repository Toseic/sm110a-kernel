# Pipeline delay probes

`include/pipeline_delay.h` provides a device-side primitive for perturbing a
software pipeline by a requested number of SM clock cycles. Kernel-specific
`*_pipeline_delay.cuh` files own point names and default values. The probe is
meant for finding the critical path: increase one stage's delay while
measuring total kernel time; delay hidden by overlap produces little change.

The values are compile-time constants because device code cannot safely read a
process environment.  The JIT reads the environment, passes nonzero settings
as CUDA defines, and includes the settings in its cache key.  Change an
environment variable before compiling/running the kernel (a fresh Python
process is the simplest workflow).

```bash
# Several points in one variable.
SM110A_PIPELINE_DELAYS='producer_issue_tma=200,consumer_issue_mma=400' \
  /home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py

# This form overrides the matching value in SM110A_PIPELINE_DELAYS.
SM110A_PIPELINE_DELAY_CONSUMER_WAIT_TMA=500 \
  /home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_gemm.py
```

All local point defaults are zero. Invalid cycle counts fail early during JIT
configuration. The JIT forwards configured point names without maintaining a
global registry, so an environment name must match a macro recognized by the
kernel being compiled.

NVFP4 FC1 exposes the complete pipeline as individually configurable points:
`nvfp4_fc1_task_wait`, `nvfp4_fc1_task_publish`,
`nvfp4_fc1_producer_wait_reuse`, `nvfp4_fc1_load_a`,
`nvfp4_fc1_b_wait_issue`, `nvfp4_fc1_load_b`,
`nvfp4_fc1_consumer_wait_epilogue`, `nvfp4_fc1_consumer_wait_a`,
`nvfp4_fc1_consumer_wait_b`, `nvfp4_fc1_mma`,
`nvfp4_fc1_epilogue_wait`, `nvfp4_fc1_epilogue_tmem_to_smem`, and
`nvfp4_fc1_epilogue_store`. For example, set
`SM110A_PIPELINE_DELAY_NVFP4_FC1_LOAD_A=1024`.

To add a probe, create a local `*_pipeline_delay.cuh` that defines the default
macro, include it with the public header, and place a call in the control
lane/warp that owns the stage:

```cpp
#include "pipeline_delay.h"
#include "my_kernel_pipeline_delay.cuh"

sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MY_KERNEL_STAGE>();
```

The probe busy-waits on `clock64()`, so the elapsed time is at least the
requested cycle count with a small architecture-dependent loop overhead.
Measure a zero-delay baseline and use differences rather than treating it as
an exact instruction-level latency injector.  Delaying every lane separately
is usually unnecessary; use the elected leader that issues or consumes the
pipeline operation unless the stage itself requires warp-wide participation.
