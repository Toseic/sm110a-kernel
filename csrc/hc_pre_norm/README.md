# HC pre + RMSNorm

BF16 pre uses two launches: tensor-core projection, then gates/Sinkhorn,
collapse and weighted RMSNorm. Collapse is rounded to BF16 before RMSNorm,
matching separate HC pre and RMSNorm calls. FP16/FP32 and head use SIMT.

The tensor-core path uses K chunks of 1024 for D=4096 and at least 256 tokens;
other shapes retain chunks of 512 for split-K parallelism. Both projection and
finish specialize on the same chunk size. Scratch capacity remains sized for
512, allowing reuse across the runtime dispatch threshold; 1024 uses half of
that capacity. There is no persistent packed-weight cache.

RMSNorm loads aligned FP32 weights using two 128-bit loads per eight elements.
Weights with a nonzero storage offset retain a scalar fallback when unaligned.

## Cold-L2 measurements

NVIDIA Thor, GPC=1575 MHz, EMC=4266 MHz, HC=4, D=4096, BF16,
Sinkhorn=20, decode shape [tokens, 1, 4, D]. CUDA Graph replay, 64 MiB dirty
write eviction before each timed invocation, 5 warmups, 50 iterations per
sample, median of 5 samples. Eviction is outside the timed interval.
2026-10-30 measurements; timing varies slightly across runs.

| Tokens | Previous fusion (us) | Optimized fusion (us) | Separate HC pre + norm (us) | Fusion optimization speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 33.558 | 33.512 | 33.776 | 1.00x |
| 16 | 37.855 | 36.645 | 37.535 | 1.03x |
| 128 | 83.610 | 78.634 | 81.742 | 1.06x |
| 256 | 135.274 | 119.295 | 130.269 | 1.13x |
| 512 | 228.416 | 206.226 | 226.301 | 1.11x |
| 1024 | 394.467 | 381.228 | 414.161 | 1.03x |
| 2048 | 759.874 | 722.522 | 833.563 | 1.05x |
| 4096 | 1441.287 | 1358.908 | 1700.867 | 1.06x |

The separate column was measured in the optimized run. Small differences in
separate timings across runs indicate measurement variability. All benchmark
reference and separate-path accuracy checks passed, including updated-input
CUDA Graph replay. Additional checks covered 255/256/257-token routing,
D=256/258, FP16/FP32 fallback, head, and unaligned FP32 norm weights.
Compute Sanitizer memcheck at 1/257 tokens reported zero errors.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_hc_pre_norm.py \
  --tokens 1 16 128 256 512 1024 2048 4096 --baseline none \
  --warmup 5 --iters 50 --samples 5
```

## Profiling and experiments

Nsight Systems on the original 4096-token fusion showed roughly equal time
in projection and finish. NCU reported finish at 40 registers/thread and
98.43% achieved occupancy, despite low compute utilization. Source sampling
on an experimental finish showed substantial long-scoreboard stalls and
scalar norm-weight loads, motivating explicit vector loads.

Rejected experiments: 128-thread finish with direct global input loads,
one warp per token with packed BF16 retention, and overlapping Sinkhorn with
collapse. None produced a stable overall improvement. K chunks of 2048
regressed small shapes and did not consistently improve on 1024.

Further opportunities include projection operand preparation, scratch layout,
and remaining finish memory latency. The chunk-1024 dispatch is restricted
to the measured D=4096 shape; other dimensions need measurements before
extending it.
