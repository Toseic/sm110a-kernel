# RMSNorm

`y = (x * rsqrt(mean(x²) + eps)) * weight`, reducing along the last axis.
The default model configuration is BF16 input/output with FP32 weight.
FP16 and FP32 inputs and input-dtype weights are also supported. Accumulation
and arithmetic use FP32, followed by one cast to the input dtype. Weight has
shape `[D]` on the input device. FP32 weight is consumed directly without
converting either the input tensor or weight tensor before the kernel. Input is contiguous with `1 <= D <= 32768`; preceding
dimensions are flattened into rows, including zero rows.

```python
from sm110a_kernel import rmsnorm, rmsnorm_out, compile_rmsnorm

y = rmsnorm(x, weight, eps=1e-6)
rmsnorm_out(x, weight, y, warps_per_row=2, rows_per_cta=4)
module = compile_rmsnorm(x.size(-1), warps_per_row=4, rows_per_cta=1)
module.rmsnorm_out(x, weight, y)
```

Each CTA owns one or multiple complete rows. Input is loaded **once from
global memory directly into registers**, retained across reduction, and reused
for the output. BF16/FP16 inputs retain their packed 16-bit representation
through reduction, rather than keeping FP32 expansions live. An opaque
register identity prevents the compiler from extending the unpacked values'
lifetimes. Input never goes through shared memory. A single-warp row
uses shuffle reduction only; multi-warp rows exchange one partial sum per
warp and the inverse RMS through shared memory. No atomics or inter-CTA
synchronization are needed.

Vectorizable rows use 16-byte loads/stores and require 16-byte aligned input,
weight, and output pointers. Other row widths use masked scalar accesses.
Exact in-place input/output is supported; shifted overlap and output/weight
overlap are rejected. Precompile before CUDA graph capture. The out API does
not allocate or autotune.

`warps_per_row` and `rows_per_cta` each support 1, 2, 4, or 8, with at most
32 warps (1024 threads) per CTA and at most 128 input values per thread. A
conservative register budget rejects large CTAs that could force spills in
FP32. Fixed defaults, selected for bandwidth on large inputs, are:

| D | Warps/row | Rows/CTA |
|---|---:|---:|
| <= 1024 | 1 | 4 |
| 1025–2048 | 4 | 4 |
| 2049–8192 | 8 | 4 |
| 8193–16384 | 8 | 2 |
| 16385–32768 | 8 | 1 |

The optimal row count also depends on input row count. For example, at
D=4096 with 4096 input rows, 1 row/CTA outperforms 4 rows/CTA; use an explicit
configuration for that workload. At D=1024, 2 and 4 rows/CTA are nearly tied
on large inputs. There is no runtime autotuning.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_rmsnorm.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_rmsnorm.py --d 4096 8192 --tokens 4096 16384 --sweep-rows 1 2 4 --baseline none
```

Timing uses the repository's cold-L2 harness by default. Reported GB/s uses
compulsory traffic (one input read, one output write, and one weight read);
it is effective bandwidth rather than measured DRAM traffic. Benchmark weights
are FP32 by default, matching DeepSeek-V4-Flash; `--weight-dtype input` selects
the input dtype for comparison. The Torch
baseline is `torch.nn.functional.rms_norm`; accuracy uses a reference with
FP32 intermediates and a single final cast.

On Thor / CUDA 13.2, ptxas inspection of D=8192 with 8 warps/row and
4 rows/CTA, D=16384 with 8 warps/row and 2 rows/CTA, and D=32768 with
8 warps/row and 1 row/CTA reported zero stack and zero spill loads/stores for
all three dtypes. At D=8192, the packed 8-warp variant uses 38 registers per
thread (1 row/CTA), versus 79 for the original FP32-retention 4-warp variant.
NCU achieved occupancy increased from 49.1% to 95.3% for these configurations.

## Row-count measurements

These earlier measurements use BF16 input **and BF16 weight** (reproduce with
`--weight-dtype input`). The benchmark now defaults to FP32 weight to match
the model. Thor GPC=1575 MHz / EMC=4266 MHz locked, cold L2 with the default dirty
write eviction, 16384 input rows, 5 warmups, 20 iterations, 7 timing samples.
The table is a reverse-order confirmation sweep; each case uses identical
seeded inputs. Values are effective GB/s, with the best value in bold.

| D | Warps/row | 1 row/CTA | 2 rows/CTA | 4 rows/CTA |
|---|---:|---:|---:|---:|
| 1024 | 1 | 227.16 | **229.15** | 229.03 |
| 4096 | 8 | 247.30 | 248.51 | **248.98** |
| 8192 | 8 | 248.89 | 250.99 | **252.10** |
| 16384 | 8 | 248.31 | **249.39** | exceeds register budget |

D=32768 uses 1 row/CTA under the register budget, reaching 248.77 GB/s in the
initial sweep. D=8192's winning 4-row configuration reached 251.74 and
252.10 GB/s in the forward and reverse sweeps, or 92.2–92.3% of the
repository's 273 GB/s GMEM reference. Original defaults reached 245.72 GB/s
on the same 16384x8192 workload. D=4096's 1/2/4-row configurations are within
roughly 1% of each other on the large workload; their exact ranking varies.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_rmsnorm.py --d 1024 4096 8192 16384 --tokens 16384 --sweep-rows 4 2 1 --weight-dtype input --warmup 5 --iters 20 --samples 7 --baseline none
```

`--baseline copy` measures a same-size tensor copy under the same cache
conditions. The smaller 4096x4096 and 4096x8192 workloads remain below 90%
while closely matching the copy bandwidth: the dirty eviction introduces
additional writeback traffic which is not included in compulsory bytes.
No warm-L2 results are used.

The exact NCU metrics
`mcc__dram_throughput_op_read.avg.pct_of_peak_sustained_elapsed` and
`mcc__dram_throughput_op_write.avg.pct_of_peak_sustained_elapsed` return `nan`
on this stack. Peak percentages above are effective bandwidth divided by
273 GB/s, not hardware-counter measurements of DRAM utilization. NCU reports
are `/tmp/ncu-reports/rmsnorm-baseline.ncu-rep` and
`/tmp/ncu-reports/rmsnorm-packed.ncu-rep`; they compare the original 4-warp,
1-row configuration to packed 8-warp, 1-row, before the final row-count sweep.

## Model-aligned weight precision

BF16 input/output with FP32 weight is the default benchmark mode and the
DeepSeek-V4-Flash runtime path. With maximum clocks and dirty cold-L2 eviction,
5 warmups, 20 iterations and 7 samples, the default configurations measured:

| Input shape | BF16 weight (us) | FP32 weight (us) | FP32 weight GB/s | Reference peak |
|---|---:|---:|---:|---:|
| 16384 x 512 | 161.371 | 161.136 | 208.25 | 76.28% |
| 16384 x 1024 | 292.206 | 289.915 | 231.49 | 84.80% |
| 4096 x 4096 | 293.533 | 291.915 | 229.95 | 84.23% |
| 16384 x 4096 | 1078.840 | 1081.530 | 248.21 | 90.92% |
| 16384 x 8192 | 2131.691 | 2127.288 | 252.39 | 92.45% |

The large-workload latency differences are within about 1%. Each FP32 weight
is read directly; input remains packed in registers, and input/output
precision is unchanged. D=8192, 8 warps/row, 4 rows/CTA uses 40 registers per
thread for BF16 input / FP32 weight and has no spills. Bandwidth includes
one compulsory weight read at its actual element size.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_rmsnorm.py --d 512 1024 4096 8192 --tokens 128 4096 16384 --weight-dtype input --warmup 5 --iters 20 --samples 7 --baseline none
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_rmsnorm.py --d 512 1024 4096 8192 --tokens 128 4096 16384 --weight-dtype float32 --warmup 5 --iters 20 --samples 7 --baseline none
```
