---
name: ncu
description: Use this skill when the user asks to measure or compare CUDA kernel metrics with NVIDIA Nsight Compute, especially L2, DRAM, launch, occupancy, or throughput metrics. Run requested cases serially, restrict collection to the actual kernel, and report exact metric names and availability limitations.
---

# NCU

Use this skill for reproducible Nsight Compute measurements in the current repository.

## Default behavior

Unless the user gives more specific instructions, run NCU cases one at a time on the same GPU, save `.ncu-rep` reports under `/tmp/ncu-reports`, and restrict collection to the requested kernel only after discovering its actual name. Never treat `nan` as zero. Do not add reports to the repository unless requested. Export raw data with `ncu --import <report>.ncu-rep --csv --page raw` when structured post-processing is needed.

Verify metric suffixes on the active GPU with:

```bash
ncu --query-metrics --query-metrics-mode suffix \
  --metrics lts__throughput,lts__t_sector_hit_rate,lts__t_request_hit_rate,mcc__dram_throughput_op_read,mcc__dram_throughput_op_write
```

## Workflow

1. Inspect `ncu --version`, the benchmark help, and the repository's documented Python entry point.
2. Select full metric names, not only base names. For example:

   ```text
   lts__throughput.avg.pct_of_peak_sustained_elapsed
   lts__t_sector_hit_rate.pct
   lts__t_request_hit_rate.pct
   mcc__dram_throughput_op_read.avg.pct_of_peak_sustained_elapsed
   mcc__dram_throughput_op_write.avg.pct_of_peak_sustained_elapsed
   ```

   For a sustained-active comparison, use the corresponding `.avg.pct_of_peak_sustained_active` suffixes. Use `lts__t_sectors_lookup_miss.sum` only as an L2-miss/DRAM-pressure proxy, never as an exact DRAM read-byte or write-byte counter.

3. Run each case only after the previous NCU process exits. Discover kernels first when a filter is needed. `launch` is not an NCU section set; use `LaunchStats`:

   ```bash
   ncu --section LaunchStats --launch-count 1 \
     -- /path/to/program <args>
   ```

   Then use the actual name or a user-provided regular expression:

   ```bash
   ncu -f --csv --page raw \
     --target-processes all \
     --kernel-name-base function \
     --kernel-name 'regex:<specific-kernel-regex>' \
     --metrics <full-metric-names> \
     -o /tmp/ncu-reports/case-1 \
     -- /path/to/program <args>
   ```

   Omit `--kernel-name` when all kernels are in scope. Do not use broad patterns such as `.*kernel.*`: they can capture PyTorch allocation, initialization, or L2-eviction kernels before the target. If NCU reports no kernels, inspect its printed kernel names or the JIT fatbin and update the filter.
4. Import each report with `ncu --import` and summarize by actual problem shape, launch parameters, kernel, and configuration. One program argument may exercise multiple shapes or kernel variants.
5. State whether cold-L2 eviction was enabled, list the exact metrics, distinguish throughput from hit rate, and include report paths.

## Source-level hotspot analysis

Use this workflow when a metric such as `long_scoreboard`, `barrier`, or
`lg_throttle` needs to be attributed to a source operation rather than merely
reported at kernel level.

### Preserve and verify source correlation

Compile CUDA code with `-lineinfo`. For this repository, first verify the
actual JIT command or cache manifest:

```bash
rg -n -- '-lineinfo|cuda_cflags' \
  ~/.cache/sm110a_jit ~/.cache/torch_extensions 2>/dev/null
```

Verify the resulting binary contains debug line sections:

```bash
cuobjdump --dump-elf /path/to/kernel.so | \
  rg '\.debug_line|\.nv_debug_line_sass'
```

When collecting the report, import source files into the report. `-lineinfo`
alone is not enough for a portable `.ncu-rep` source view:

```bash
ncu --set full \
  --import-source on \
  --source-folders "$PWD" \
  --kernel-name 'regex:.*target_kernel.*' \
  -o /tmp/ncu-reports/source-hotspots \
  -- /path/to/program <args>
```

Inspect both CUDA and SASS views:

```bash
ncu --import /tmp/ncu-reports/source-hotspots.ncu-rep \
  --page source --print-source cuda,sass
```

Use `--print-source cuda,sass`, not only the default SASS view. If the CUDA
source column is empty, check that `--import-source on` was used during the
profiling run and that `--source-folders` contains the exact source root.

### Collect the stall metrics

Use `SchedulerStats`, `WarpStateStats`, and source sampling in addition to the
memory sections:

```bash
ncu --set full \
  --section SchedulerStats --section WarpStateStats \
  --section SourceCounters \
  --import-source on --source-folders "$PWD" \
  --kernel-name 'regex:.*target_kernel.*' \
  -o /tmp/ncu-reports/target-stalls \
  -- /path/to/program <args>
```

Useful aggregate metrics include:

```text
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__warps_eligible.avg.per_cycle_active
smsp__warps_active.avg.per_cycle_active
```

Do not equate `long_scoreboard` with an L2 miss. NCU's generic scoreboard
classification can include dependencies on asynchronous memory operations,
barriers, and polling instructions. Confirm the actual instruction in the
CUDA/SASS source table (`mbarrier.try_wait`, `cp.async.bulk.wait_group`, a
global load, etc.) before naming L2 as the cause.

### Inline helpers and source-line aggregation

An inlined helper such as:

```cpp
__device__ inline void wait(...) { mbarrier.try_wait...; }
```

can cause all call sites to collapse into one source line. If the report shows
one large hotspot but the helper has multiple producer/consumer roles, and the
user has authorized an analysis-only source change, split the roles into
separate inline helpers with identical PTX, for example:

```cpp
wait_mma_slot(...);
wait_tma(...);
wait_epilogue(...);
wait_mainloop(...);
```

Keep the synchronization instructions, operands, phase handling, and call
order unchanged. Rebuild with `-lineinfo`, profile again, and aggregate
`stall_long_sb` (and the other stall columns) by the distinct helper lines.
Report that this instrumentation can perturb code generation and should not be
used for final performance numbers. Remove or clearly label the analysis-only
diff after the investigation.

For a source CSV, the relevant columns are `Line No`, `Source`,
`stall_long_sb`, `stall_barrier`, `stall_sleep`, and `stall_wait`. Prefer
source-level sample counts for locating a hotspot, then validate the diagnosis
with the corresponding aggregate cycles-per-issue metric.

## Persistent FC1 / CLC SourceCounters recipe

For `benchmark_fc1_block_interleaved_swiglu.py`, profile **one deterministic
candidate**. A multi-candidate autotune changes the launch sequence and can
silently profile the wrong specialization. In particular, the FC1 autotune
cache key does not distinguish the direct-GMEM filter, so use a temporary cache
path together with `--force-sweep` for direct-only experiments.

The following command was verified on Thor for the 32k-token direct-GMEM
control path. `--launch-skip 3` selects the final timed FC1 launch after the
initial invocation, one warmup, and two timing iterations:

```bash
ncu --force-overwrite \\
  --section SourceCounters \\
  --import-source on --source-folders "$PWD" \\
  --kernel-name-base demangled \\
  --kernel-name 'regex:fc1_block_interleaved_swiglu_kernel' \\
  --launch-skip 3 --launch-count 1 \\
  --export /tmp/fc1_32k_source \\
  -- /home/bifrost/miniforge3/bin/python3 \\
  benchmarks/benchmark_fc1_block_interleaved_swiglu.py \\
  --token 32768 --tile-n 256 --tile-k 64 \\
  --cluster-m 2 --cluster-n 1 --direct-gmem \\
  --warmup 1 --iters 2 --samples 1 \\
  --force-sweep --cache-path /tmp/fc1_source_cache.json \\
  --no-cold-l2
```

For the cached winning TMEM/SMEM/TMA candidate, use a cache file whose entry is
known to be `tmem-smem-tma`, omit `--direct-gmem` and `--force-sweep`, and keep
the same warmup/iteration/skip sequence. Confirm the specialization in NCU's
kernel name: its final boolean is `0` for TMEM/SMEM/TMA and `1` for direct
GMEM.

### Replay and report lifecycle

For this persistent CLC kernel, NCU can print `0%` and the warning that launch
replay is taking longer than expected while the target is still progressing.
Do not treat that line as a failure and do not check the report path
immediately: NCU writes the `.ncu-rep` only after all replay passes complete.
Wait for the report file to appear, then import it:

```bash
while [ ! -f /tmp/fc1_32k_source.ncu-rep ]; do sleep 5; done
ncu --import /tmp/fc1_32k_source.ncu-rep --page details
ncu --import /tmp/fc1_32k_source.ncu-rep \\
  --page source --print-source cuda,sass \\
  --resolve-source-file "$PWD/csrc/bf16-moe/src/fc1_block_interleaved_swiglu.cu"
```

### Splitting a collapsed mbarrier hotspot

`mbarrier.try_wait` helpers are frequently inlined, causing all producer,
consumer, TMA, and epilogue waits to collapse onto one helper source line. If
that line is the SourceCounters hotspot, make an **analysis-only** source diff:
duplicate the identical inline-PTX helper into separately named helpers for
each pipeline edge (for example `mma_reuse`, `consumer_a_ready`,
`consumer_cluster_a_ready`, `tma_b_ready`, `epilogue_slot`, and
`mainloop_output`) and replace each call site with its semantic helper.

Keep the PTX, operands, phase argument, suspend interval, memory clobber, and
call order identical. Rebuild and rerun SourceCounters; report the per-helper
stall samples rather than the aggregate helper total. This instrumentation may
change code generation slightly, so validate it compiles and runs but do not
use its timing as a final performance number. Clearly retain or remove the
analysis-only diff intentionally after the investigation.

Interpretation guardrail: a high `mainloop_output` wait sample count belongs to
epilogue warps waiting for MMA output and is not itself a wall-time percentage.
Compare it with `tma_b_ready`, local-A, cluster-A, and epilogue-slot waits. Low
A/B-ready waits plus low epilogue-slot waits indicate that A/B transfer and
epilogue store are hidden; the next investigation belongs in MMA/mainloop
issue, commit, and stage-reuse cadence.

## BF16 GEMM on Thor (GB10B, CC 11.0)

For `benchmarks/benchmark_bf16_gemm.py`, select only the JIT GEMM with `--kernel-name-base function --kernel-name 'regex:.*bf16_gemm_kernel.*'`. This is required because the benchmark initializes tensors and evicts L2 before timing.

Use this confirmed section collection for common metrics:

```bash
ncu --force-overwrite --target-processes all \
  --kernel-name-base function --kernel-name 'regex:.*bf16_gemm_kernel.*' \
  --launch-count 1 \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  -o /tmp/ncu-reports/bf16-gemm-1024 \
  -- /home/bifrost/miniforge3/bin/python3 \
  benchmarks/benchmark_bf16_gemm.py --debug --m 1024 --no-accuracy
```

On the tested Thor software stack, the confirmed useful outputs are the SpeedOfLight duration, SM and L2 throughput, MemoryWorkloadAnalysis L2 hit rate and busy rate, LaunchStats block/grid/cluster/shared-memory values, and Occupancy. The confirmed case is `1024x1024x1024` with `--debug --m 1024 --no-accuracy`. Keep the NCU execution time separate from the benchmark's timing: profiling replay perturbs runtime.

### Confirmed metric names

These exact raw metrics were collected successfully for the target kernel on Thor. Prefer the section collection above for a human-readable report; use this list with `--metrics` when a compact CSV/raw report is needed.

```text
# Time and utilization
gpu__time_duration.avg
sm__throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
l1tex__throughput.avg.pct_of_peak_sustained_active

# L2: throughput, hit rate, pressure, and internal busy rates
lts__throughput.avg.pct_of_peak_sustained_elapsed
lts__throughput.avg.pct_of_peak_sustained_active
lts__t_sector_hit_rate.pct
lts__t_sectors_lookup_miss.sum
lts__d_sectors.avg.pct_of_peak_sustained_elapsed
lts__lts2xbar_cycles_active.avg.pct_of_peak_sustained_elapsed

# Launch geometry and resource footprint
launch__grid_size
launch__block_size
launch__cluster_size
launch__registers_per_thread
launch__shared_mem_per_block_dynamic
launch__shared_mem_per_block_static
launch__waves_per_multiprocessor

# Occupancy
sm__maximum_warps_per_active_cycle_pct
sm__warps_active.avg.pct_of_peak_sustained_active
sm__warps_active.avg.per_cycle_active
launch__occupancy_limit_shared_mem
```

`lts__t_sector_hit_rate.pct` is a cache hit ratio. The `throughput` and `cycles_active` metrics are peak-relative utilization. `lts__t_sectors_lookup_miss.sum` counts L2-miss sectors and is only a DRAM-pressure proxy.

The queried `mcc__dram_throughput_op_read/write.*` metrics returned `nan` for this kernel. Treat both as unavailable on this stack, even though the metric names are queryable. Do not replace them with L2 misses, and do not claim an exact DRAM read/write bandwidth. Re-test them after a driver or Nsight Compute upgrade; use them only if the collected values are finite.

## Guardrails

- Do not run NCU jobs in parallel; they interfere through the shared GPU.
- Do not modify benchmark or kernel source just to collect metrics unless the
  user explicitly authorizes source-level instrumentation; if authorized,
  keep the change analysis-only and disclose its possible code-generation
  perturbation.
- L2 throughput is pipeline utilization, not cache residency. Use `lts__t_sector_hit_rate.pct` or `lts__t_request_hit_rate.pct` for cache hit rate.
- A working set fitting in nominal L2 capacity does not imply a 100% hit rate because of cold-L2 eviction, compulsory misses, C traffic, sector granularity, and cache mapping.
- DRAM throughput counters can return `nan` on some integrated or newer GPUs. Report them as unavailable and do not infer zero traffic.
- On Thor, `lts__t_sectors_lookup_miss.sum` is a useful DRAM-pressure proxy, not a DRAM read-byte counter and never a DRAM write-byte counter.
