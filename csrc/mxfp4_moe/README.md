# SM110A MXFP4 MoE FC1 / FC2

The directory contains standalone MXFP4 FC1 and FC2 kernels and the fused
`mxfp4_fc1fc2_kernel.cu` implementation. Standalone entry points are
`mxfp4_fc1_out` / `mxfp4_fc2_out`, with routing prepared by
`prepare_mxfp4_fc1_metadata` / `prepare_mxfp4_fc2_metadata`. Their benchmarks
are `benchmarks/benchmark_mxfp4_fc1.py` and `benchmark_mxfp4_fc2.py`.

## Fused FC1/SwiGLU/FC2 inputs

The fused W4A8 kernel targets SM110a and fixes `H=6144`, `I=2048`.
It computes two FC1 projections, SwiGLU with optional clipping, FP8/E8M0
requantization, and FC2 in one kernel. FP32 accumulation is retained across
FC2's entire reduction dimension; output is BF16.

| Tensor | Type | Shape / storage |
|---|---|---|
| A | FP8 E4M3 | `[rows, 6144]` |
| W1 | `float4_e2m1fn_x2` | `[experts, 4096, 3072]`, gate/up interleaved in 64-row blocks |
| W2 | `float4_e2m1fn_x2` | `[experts, 6144, 1024]` |
| SFA | E8M0 | `rows * 768` elements, blocked padded scales |
| SFW1 | E8M0 | `experts * 4096 * 768` elements |
| SFW2 | E8M0 | `experts * 6144 * 256` elements |
| Output | BF16 | `[rows, 6144]` |

All tensors must be contiguous and on the same CUDA device. Scales use groups
of 32 with the same replicated blocked layout as the standalone MXFP4 FC1/FC2.
Grouped mode supports one expert and `M % (128 * cluster_m) == 0`, with
`cluster_m=1` supporting M=128. Routed mode consumes
unexpanded token-major A, requires `tokens % 128 == 0`, and returns reduced
token-major output. FC2-position routing weights are BF16; FC1-position weights
are FP32. `activation_bf16_rounding=True` matches the corresponding standalone
FC1 numerical mode.

## Python and JIT

```python
from sm110a_kernel import (
    Mxfp4Fc1Fc2Config,
    compile_mxfp4_fc1fc2,
    compile_mxfp4_fc1fc2_configs,
    mxfp4_fc1fc2,
    mxfp4_fc1fc2_out,
    mxfp4_fc1fc2_routed_out,
    prepare_mxfp4_fc1fc2_metadata,
)

# Already quantized inputs; grouped, single expert.
output = mxfp4_fc1fc2(a, w1, sfa, sfw1, w2, sfw2, swiglu_limit=7.0)

# Compile and allocate output/workspace before capturing a CUDA Graph.
config = Mxfp4Fc1Fc2Config(fuse_routing=True)
module = compile_mxfp4_fc1fc2(config)
metadata = prepare_mxfp4_fc1fc2_metadata(router_scores, topk, config)
row_tasks = int(metadata.state[0].item()) // 16
module.run_routed_out(
    a, w1, sfa, sfw1, w2, sfw2, metadata, routing_weights, topk,
    output, 7.0, row_task_count=row_tasks,
)
```

`module.run`, `module.run_out`, and `module.run_routed_out` provide bound entry
points. Batch compilation accepts an iterable of configs and an optional
`max_workers`. The split JIT compiles a common Torch bridge and separate CUDA
variants; configs, architecture and source dependencies participate in cache keys. The module exposes `library_path` for inspection.

Config fields:

- `cluster_m`: 1 or 2 CTAs per cluster.
- `h_split`: 1, 2, or 4. Splits FC1's intermediate columns and FC2's output
  columns across resident clusters per row. Each row's partitions must fit
  concurrently; additional rows execute in waves.
- `stages`: 2 through 5 shared-memory stages; single-CTA configs support 2/3.
- `accumulator_stages`: 1 or 2; two stages overlap TMEM accumulation and
  epilogue reads, with explicit protection of aliased columns.
- `gather_warps`: 2 or 4 producer warps for routed A/SFA gathering. The
  two-CTA/two-gather path reuses the peer MMA warp to forward readiness.
- `reduce_tile_n`: 64 (default) or 128 columns per FC2 reduction slab.
  Both choices preserve the 16-column TMEM overlap protection.
- `compact_weight_scales`: false by default. Set true when both weight scale
  inputs were packed with `compact_mxfp4_fc1fc2_weight_scales`.
- `tiled_schedule`: distribute FC1 and FC2 tiles across resident workers,
  with one grid handoff between phases. Requires `h_split=1`.
- `fuse_routing`, `routing_weight_position`, `activation_bf16_rounding`:
  choose routing and numerical semantics.

### Compact static weight scales

```python
from sm110a_kernel import compact_mxfp4_fc1fc2_weight_scales

# Do this once for static weights, outside inference timing and graph capture.
packed_sfw1 = compact_mxfp4_fc1fc2_weight_scales(sfw1)
packed_sfw2 = compact_mxfp4_fc1fc2_weight_scales(sfw2)
config = Mxfp4Fc1Fc2Config(compact_weight_scales=True)
module = compile_mxfp4_fc1fc2(config)
output = module.run(a, w1, sfa, packed_sfw1, w2, packed_sfw2, 7.0)
```

Packing keeps every live E8M0 byte exactly and removes the three unused bytes
per scale. SFW1/SFW2 use one quarter of the sizes in the input table; A, SFA,
weights and intermediate images retain their formats. The kernel expands each
weight scale tile in its existing shared-memory slot before MMA. Changes to
weights require repacking their scales. The binding rejects a storage size
that does not match the compiled format.

The benchmark's `--weight-scales auto` default searches padded and compact
variants. Use `--weight-scales padded` or `compact` for a fixed experiment.
JSON records the one-time packing cost separately from repeated execution.
Compact-format measurements require static prepacked scales; packing cost
must also be considered for workloads that change weights every invocation.

`h_split=1` uses persistent workers and supports multiple waves of row tasks.
Convenience functions choose the largest resident split. Explicit compilation
defaults to split 1. Performance tuning happens in the benchmark before timing.

Routing metadata must use FC1's matching `cluster_m`, `tile_n=256`, and
`macro_swizzle_n_major=True`; the dedicated preparation helper supplies these
settings. Eager routed calls read the current state count on each launch.
Graph capture requires an explicit `row_task_count` matching the captured
metadata. Warm up once before capture to allocate the workspace. Workspaces
are cached per device, stream, and task capacity.

Each CTA writes a reusable global intermediate image of 294,912 bytes
(262,144 FP8 values plus 32,768 padded scales), then streams it through FC2's
TMA ring. This image is larger than SM110a's shared-memory capacity. The fused
path therefore still moves intermediate data through global memory. Routed
execution includes output zeroing and BF16 atomic reduction; the benchmark
includes zeroing in the measured call.

In two-CTA routed configs, A/SFA gather producers advance independently. A
ready warp forwards local TMA/gather completion to the leader MMA warp,
allowing producers to prefetch subsequent stages. With two gather warps,
the peer MMA warp performs this role; four-gather configs use a separate warp.

## Benchmarks and validation

All correctness checks live in `benchmarks/benchmark_mxfp4_fc1fc2.py`.
No UT script is required.

```bash
PYTHON=/home/bifrost/miniforge3/bin/python3

# Correctness matrix: grouped/routed, tails, empty experts, multiple waves,
# FC1/FC2 weighting, BF16 rounding, zeros, eager repeats, CUDA Graph replay,
# JIT config validation and in-process/batch cache reuse.
$PYTHON benchmarks/benchmark_mxfp4_fc1fc2.py --validate

# Measure every compatible config, without probe pruning.
$PYTHON benchmarks/benchmark_mxfp4_fc1fc2.py --m 512 \
  --force-sweep --warmup 5 --iters 20 --samples 5 --graph-check \
  --json-output profiles/mxfp4_fc1fc2/m512.json

# Complete preset: 6 grouped + 9 routed workloads, serial GPU execution.
$PYTHON benchmarks/benchmark_mxfp4_fc1fc2.py --sweep all \
  --warmup 5 --iters 20 --samples 5 --graph-check

# Routed case; change weight position / rounding to measure other semantics.
$PYTHON benchmarks/benchmark_mxfp4_fc1fc2.py \
  --fuse-routing --tokens 512 --experts 8 --topk 6 --force-sweep

# Also accessible through the common runner.
$PYTHON -m sm110a_kernel.runner mxfp4-fc1fc2 -- --m 512

# Memory checks run the same benchmark entry point.
compute-sanitizer --tool memcheck --error-exitcode 99 \
  $PYTHON benchmarks/benchmark_mxfp4_fc1fc2.py \
  --fuse-routing --tokens 128 --experts 2 --topk 2 \
  --cluster-m 2 --schedule tiled --h-split 1 --stages 5 \
  --accumulator-stages 2 --gather-warps 2 --reduce-tile-n 128 \
  --warmup 0 --iters 1 --samples 1 --no-pre-parallel-jit \
  --baseline-cluster-m 2 --weight-scales compact --graph-check --force-sweep
```

The default config search covers cluster sizes 1/2, splits 1/2/4, stages
2/3/4/5, single/double accumulator buffers, row/tiled schedules, and routed
gather counts 2/4, reduction slabs 64/128, and padded/compact weight scales. Zero selects tuning for the corresponding
CLI field; an explicit value restricts the search. The standalone baseline
independently tunes FC1/FC2 cluster sizes 1/2 and FC2 epilogue sizes 32/64,
with FC1 `direct_gmem=True`. `--force-sweep` measures every candidate. Other
runs can reuse the selected config or prune clearly slower probe results.
Metadata preparation, quantization, JIT, and tuning are excluded from timing.

Timing is cold L2 only, by default a dirty write flush of twice the GPU's
reported L2 size before each measurement. `--clean-l2` selects a read flush.
The complete FC1+FC2 baseline flushes before the pair, so its own intermediate
naturally stays in cache. Individually cold FC1/FC2 times are supplemental and
should not be summed to replace the pair measurement.

Grouped/top-1 results must match standalone FC1+FC2 bitwise. For top-k
reduction, BF16 atomic summation can occur in different orders. The default
acceptance bound is relative RMSE <= 0.005, with finite-output checks; the
script also reports absolute errors, mismatches and pointwise tolerances.
The validation matrix requires bitwise equality for its top-2 cases as well.

`--sweep all` writes per-workload JSON/log files and an incremental
`summary.json` under `profiles/mxfp4_fc1fc2/`. JSON includes every measured
candidate, timing statistics, baseline choices, selected-config accuracy,
shape, device, software versions, and measurement parameters. The preset is:

- Grouped: M = 256, 512, 1024, 2048, 4096, 8192.
- Routed: tokens = 128, 512, 2048 for (experts, top-k) = (2,2), (8,6), (32,6).

Each sweep writes source hashes and command arguments to `manifest.json` and
rejects source changes during the run. Use a new output directory for a new
revision. The current benchmark validation matrix passes 91/91 cases, including
17 compact-format cases, plus 502,295 exact scale encoder comparisons. Encoder
checks compile their own bench-only CUDA unit; production modules export only
the inference launcher.



## Recorded performance

The completed v3 sweep used padded weight scales on NVIDIA Thor (20 SMs),
H=6144, I=2048, SwiGLU limit=7, FC2-position routing weights and no BF16
activation rounding. Cold L2 uses a 64 MiB dirty write flush; times are
medians of 5 samples, with 5 warmups and 20 iterations per sample.

It measured 2304 fused candidates and 120 unfused pairs: **13/15 wins,
1.1243x geometric mean speedup**. Every selected configuration passed accuracy
and graph replay; the maximum relative RMSE was 0.00415997.
These are historical v3 results, not a completed sweep of the compact variant.

| Workload | Fused ms | Unfused ms | Speedup | Schedule / cluster / split / stages / accumulators / gather / reduction |
|---|---:|---:|---:|---|
| grouped_m256 | 0.213085 | 0.229454 | 1.0768x | tiled / 2 / 1 / 4 / 2 / 4 / 128 |
| grouped_m512 | 0.237101 | 0.334824 | 1.4122x | row / 2 / 4 / 5 / 2 / 4 / 128 |
| grouped_m1024 | 0.354101 | 0.486123 | 1.3728x | row / 2 / 2 / 5 / 2 / 4 / 128 |
| grouped_m2048 | 0.583406 | 0.761514 | 1.3053x | tiled / 2 / 1 / 5 / 2 / 4 / 128 |
| grouped_m4096 | 1.254630 | 1.502902 | 1.1979x | tiled / 2 / 1 / 5 / 2 / 4 / 128 |
| grouped_m8192 | 2.575406 | 3.154230 | 1.2248x | tiled / 2 / 1 / 3 / 2 / 4 / 128 |
| routed_t128_e2_k2 | 0.319422 | 0.326414 | 1.0219x | row / 2 / 4 / 5 / 2 / 4 / 128 |
| routed_t512_e2_k2 | 0.409728 | 0.473899 | 1.1566x | row / 2 / 2 / 5 / 2 / 2 / 128 |
| routed_t2048_e2_k2 | 1.374501 | 1.364203 | 0.9925x | tiled / 2 / 1 / 5 / 2 / 2 / 128 |
| routed_t128_e8_k6 | 0.891285 | 0.937178 | 1.0515x | row / 2 / 1 / 5 / 2 / 2 / 128 |
| routed_t512_e8_k6 | 1.319408 | 1.435558 | 1.0880x | tiled / 2 / 1 / 5 / 2 / 2 / 128 |
| routed_t2048_e8_k6 | 4.619853 | 4.941992 | 1.0697x | row / 1 / 1 / 3 / 2 / 2 / 64 |
| routed_t128_e32_k6 | 3.345422 | 3.214963 | 0.9610x | row / 1 / 1 / 3 / 1 / 2 / 64 |
| routed_t512_e32_k6 | 3.432291 | 3.484733 | 1.0153x | row / 1 / 1 / 3 / 1 / 2 / 128 |
| routed_t2048_e32_k6 | 5.911992 | 6.117056 | 1.0347x | row / 1 / 1 / 3 / 2 / 2 / 128 |

The baseline search covers FC1/FC2 cluster sizes 1/2, FC2 epilogue sizes 32/64,
FC1 direct-GMEM and FC2 reduction width 64. It is not an exhaustive search of
all standalone FC2 settings. Source hashes and raw per-candidate timings are
in `profiles/mxfp4_fc1fc2_tuned_v3/manifest.json` and `summary.json`.
Earlier sweep data remains in the profiles directories.

### Final targeted measurements

The retained implementation was remeasured after compact-format integration:

| Workload | Format / fused search | Fused ms | Unfused ms | Speedup |
|---|---|---:|---:|---:|
| T128 / E32 / topk6 | compact, fixed CM1 row / split1 / stages3 / acc1 / gather2 / reduce64 | 2.944842 | 3.219197 | 1.0932x |
| T2048 / E2 / topk2 | 384 configs, best padded CM2 tiled / split1 / stages5 / acc2 / gather2 / reduce64 | 1.370653 | 1.368173 | 0.9982x |

Both selected configurations passed accuracy and graph replay. Sparse relative
RMSE was 0.00367529; dense output matched bitwise. Packing sparse W1/W2 scales
once cost 1.007104 ms and reduced scale storage from 144 MiB to 36 MiB.
Repeated-execution timings exclude that one-time packing cost. The dense case
is near parity and does not establish a stable speedup.

The final targeted JSON, logs, source manifest, validation and memcheck logs
are in `profiles/mxfp4_fc1fc2_final/`. The 15-workload table above remains the
last complete sweep; the final compact/padded implementation has not had a
new complete sweep. For a future full run, use a new output directory:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_mxfp4_fc1fc2.py \
  --sweep all --weight-scales auto --sweep-output profiles/mxfp4_fc1fc2_next \
  --warmup 5 --iters 20 --samples 5 --graph-check
```

### NCU / SASS observations

One sparse kernel per report, fixed configuration and cold-L2 benchmark
launch. Reports use LaunchStats, SpeedOfLight, MemoryWorkloadAnalysis,
SchedulerStats, WarpStateStats and SourceCounters. Compare the original padded
v3 configuration with the retained compact configuration:

| Metric | Padded | Compact |
|---|---:|---:|
| `launch__registers_per_thread` | 254 | 254 |
| `derived__local_spilling_requests` | 0 | 0 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 21.40% | 25.16% |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 27.87% | 27.89% |
| `lts__t_sector_hit_rate.pct` | 18.83% | 21.43% |
| `lts__t_sectors_lookup_miss.sum` | 23,888,673 | 20,345,464 |

L2-miss sectors fell 14.83%; they indicate reduced memory pressure, not exact
DRAM read/write traffic. DRAM metrics returned NaN on this stack and are
unavailable. GPU frequencies were not locked, so utilization comparisons are
indicative. Profiling timings are excluded from performance results.

Read-only SASS inspection confirms REG=254, STACK=0, LOCAL=0, no LDL/STL and
no extra diagnostic kernel. The two-gather/two-CTA path reuses the peer MMA
warp for readiness, avoiding spills seen in an earlier tiled configuration.
Unsuccessful resident-image, intermediate-scale compression, row-order,
gather-padding and contiguous-TMA trials were reverted. Production delay
probes, the unused allocating binding and repeated baseline diagnostics were
removed. Encoder boundary checks compile only inside the benchmark.

Reports and disassembly remain under `/tmp/ncu-reports/`. The retained
compact CM2 tiled / stages5 / acc2 / gather2 / reduce128 path passed memcheck
with zero errors, bitwise accuracy and graph replay; sanitizer timings are
excluded.
