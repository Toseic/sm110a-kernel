# FC1 → FC2 programmatic dependent launch

The independent NVFP4 and MXFP4 FC1/FC2 kernels support opt-in PDL with
`pdl=True` in both configurations. Defaults remain `False`. This applies to
the two-kernel path; the fused `*_fc1fc2` kernels have no inter-kernel edge.
PDL requires compute capability 9.0 or later.

FC1 signals launch completion from every CTA at a configurable point
(`pdl_trigger="entry"` by default). FC2 is launched with
`cudaLaunchAttributeProgrammaticStreamSerialization`, initializes its shared
memory, TMEM and descriptors, then calls `cudaGridDependencySynchronize()`
from every thread before the task loop can consume FC1 activations/scales
(`pdl_wait="init"`, the conservative default). With `pdl_wait="before_a"`,
the TMA issuer first issues independent B/SFB loads, then waits once before
its first A/SFA access. Existing TMA/SFA barriers gate downstream consumers;
this hides more weight-loading work behind FC1.
The wait provides visibility of FC1 writes, including asynchronous stores.
PDL permits early scheduling; concurrent execution depends on available SM
resources and is not guaranteed. It does not allow FC2 to consume partial
FC1 output.

## Calling the kernels

```python
from sm110a_kernel import (
    Nvfp4Fc1Config, Nvfp4Fc2Config,
    compile_nvfp4_fc1, compile_nvfp4_fc2,
)

fc1 = compile_nvfp4_fc1(Nvfp4Fc1Config(pdl=True))
fc2 = compile_nvfp4_fc2(Nvfp4Fc2Config(pdl=True, pdl_wait="before_a"))
# Prepare metadata and allocate intermediate/output tensors first.
# All three calls below must use the same current CUDA stream.
fc2.prepare_pdl_out(output, metadata2)
fc1.nvfp4_fc1_out(a, w1, sfa, sfw1, metadata1, hidden, hidden_scales)
fc2.nvfp4_fc2_out(
    hidden, w2, hidden_scales, sfw2, metadata1.routing, routing_weights,
    metadata2, topk, output,
)
```

For MXFP4 use `Mxfp4Fc1Config`, `Mxfp4Fc2Config`, `compile_mxfp4_fc1`,
`compile_mxfp4_fc2` and the corresponding `mxfp4_*_out` methods. Match the
routing weight position across FC1/FC2 as usual. MXFP4 deterministic FC2 is
also supported; its final reduction remains after FC2.

`prepare_pdl_out` resets FC2's scheduler counter and clears its atomic
reduction output (the deterministic variant needs only the counter reset).
Call it **before FC1, once per invocation**. When capturing a CUDA graph,
include it inside capture so every replay resets state. PDL FC2 skips its
ordinary pre-launch memsets because those would serialize FC1 and FC2.
Use the output API: allocating FC2 calls reject `pdl=True` because their
output cannot be prepared before FC1. Keep FC1/FC2 adjacent; inserting an
event, memset or unrelated kernel between them can block overlap or change
the dependency being waited on. Metadata and tensors must remain alive and
must not be reused concurrently on other streams.

## FC1 trigger placement

Both FC1 config classes accept `pdl_trigger` with these values. It is part of
the JIT configuration/cache key and only affects code when `pdl=True`.

| Value | Per-CTA trigger point |
|---|---|
| `entry` | Thread 0 at kernel entry (default) |
| `post_init` | Thread 0 after shared-memory/TMEM/descriptor initialization and cluster barrier |
| `mainloop_quarter` | TMA lane 0 after issuing ceil(K-stages / 4) stages of its first local task |
| `mainloop_mid` | TMA lane 0 after issuing ceil(K-stages / 2) stages of its first local task |
| `mainloop_three_quarters` | TMA lane 0 after issuing ceil(3 K-stages / 4) stages of its first local task |
| `mainloop_end` | TMA lane 0 after issuing every K stage of its first local task |
| `tail` | Thread 0 after the persistent task loop, before final C-store drain and TMEM teardown |

The mainloop points describe **TMA issue progress within the first task**,
not half of all persistent work or completion of half the grid's MMA work.
Each CTA signals once. Both CTA ranks run the TMA issuer, so two-CTA clusters
signal correctly; only rank 0 runs the MMA issuer. CTAs without tasks exit,
which also fulfills their launch-completion obligation. FC2 still waits for
all FC1 output before its dependent work. Triggering earlier can hide FC2
startup but can also introduce scheduling/resource contention; the best
point must be measured for the intended workload.

## Wait placement and controls

Use `--wait-points init before_a` to compare FC2 dependency waits. Add
`--prepared-control` to include `serial_signaled` (FC1 signals but FC2 runs
serially with normal preparation) and `serial_prepared` (same early preparation
and FC1 signal as PDL, but no dependent launch). The former isolates trigger
cost; the latter isolates overlap benefit from the cost of relocating output
clearing. `external_prepare=True` is the corresponding FC2 control option;
it requires `prepare_pdl_out` before FC1 even with PDL disabled.

## Benchmark

These commands fix the MoE workload at approximately 100–200 us per pair
on Thor. CUDA Graph is enabled by default.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_pdl.py \
  --tokens 128 --experts 4 --topk 2 --hidden 2048 --intermediate 1024 \
  --config1 '{"cluster_m":2,"tile_k":256,"epilogue_tile_n":128}' \
  --config2 '{"cluster_m":2,"tile_n":128,"epilogue_tile_n":64}' \
  --trigger-points entry mainloop_mid --wait-points init before_a \
  --prepared-control --warmup 10 --iters 50 --samples 17 \
  --output-json /tmp/pdl_nv_short_moe.json

/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_pdl.py \
  --kernel1 mxfp4_fc1 --kernel2 mxfp4_fc2 \
  --tokens 128 --experts 4 --topk 2 --hidden 2048 --intermediate 1024 \
  --config1 '{"cluster_m":2,"direct_gmem":true}' --config2 '{"cluster_m":2}' \
  --trigger-points entry post_init mainloop_mid tail --wait-points init before_a \
  --prepared-control --warmup 10 --iters 50 --samples 17 \
  --output-json /tmp/pdl_mx_short_moe.json
```

For the NVFP4 timeline, use the same dimensions/configs with
`--trigger-points entry mainloop_mid --wait-points before_a --warmup 2 --iters 3
--samples 1 --profile-iters 10 --timeline /tmp/pdl_nv_short_timeline`, omitting
`--prepared-control`. CUDA Graph is on by default in all commands.

The two kernel arguments select compatible producer/consumer kernels; mixed
NVFP4/MXFP4 layouts are rejected. JSON configuration overrides and
`--max-active-clusters1/2` fix the kernel configuration and resident grid
limits; there is no autotuning. The same routing, weights, tensors and
configuration are used in both modes. Accuracy compares FC1 bytes exactly
and FC2 against the serial result over three PDL graph replays. FC2's BF16
atomic reduction permits rounding differences (`rtol=0.03`, `atol=2.0`).
This validates PDL against the serial implementation, not against a separate
mathematical reference. `--no-accuracy` skips this comparison.

CUDA graph capture/replay is enabled by default (`--cuda-graph`); the default
workload has 128 tokens for decoding. `--no-cuda-graph` is available for eager
launch debugging. The output explicitly prints `cuda_graph=True/False`.

All variants compile and pass accuracy checks before timing starts. Sample
order rotates across serial and selected trigger placements to reduce drift.
The table reports median/min/max sample times and speedup versus serial;
`--output-json` retains raw timing samples and configuration for comparison.
`--timeline` supports the same list and reports interval overlap separately
for serial and each selected trigger point.

Timing uses CUDA Events surrounding the complete graph replay: elapsed GPU
time from pair start to completion, including preparation and any final
reduction. It is not the sum of standalone kernel durations and not CPU
`perf_counter()` wall time.

Timing uses cold L2 before each measured invocation.
For a pair, L2 is flushed before FC1; FC2 naturally receives the cache state
left by FC1. Preparation uses the same `cudaMemsetAsync` operations as the serial path,
and costs are included in both pair timings. Individual
FC1/FC2 timings are separate cold-cache measurements and need not sum to the
serial pair time. Warmup/iteration/sample counts are configurable.

The default report shows standalone times, serial/PDL pair times, signed
saved microseconds, saved percentage and speedup. Savings are the net PDL
benefit and can be negative. They are not an estimate of physical kernel
interval intersection.

`--timeline PREFIX` launches a fresh process under `nsys`, saves
`PREFIX.nsys-rep` and `PREFIX.sqlite`, and directly measures interval overlap
from CUPTI timestamps. It reports the median of
`max(0, min(end1,end2) - max(start1,start2))` and its percentage of the shorter
kernel duration for serial and PDL pairs. This interval includes FC2 startup
and dependency waiting; it does not measure simultaneous MMA execution. The final `--profile-iters` pairs
(default 5 in this mode) are identified by their kernel names and ordered
serial/trigger-point NVTX ranges. Profiling may change execution timing. The saved
report can be opened in Nsight Systems to inspect startup/wait/resource
contention. `--profile-iters N` alone emits these marked replays when running
under an external profiler.

## Measurements on Thor (2026-10-19)

CUDA Graph and cold L2 are enabled. GPU clocks were not pinned. Routing
uses seed 1234 and power-law alpha 0.8. Both formats use tokens=128, experts=4, topk=2, H=2048, I=1024.
Both kernels use cluster_m=2. NVFP4 FC1 uses tile_k=256 and
epilogue_tile_n=128; FC2 uses tile_n=128 and epilogue_tile_n=64.
MXFP4 FC1 uses direct_gmem=True; other tile settings use defaults.
Each result has 10 warmups, 50 iterations per sample and 17 interleaved
samples. All variants passed exact FC1 bytes/scales and tolerant FC2
comparisons against serial over three replays.

| Mode | NVFP4 (us) | MXFP4 (us) |
|---|---:|---:|
| Serial | 159.196 | 186.460 |
| Serial, FC1 trigger enabled | 159.728 | 186.352 |
| Serial, preparation moved before FC1 | 160.029 | 187.062 |
| Entry trigger, wait after initialization | 158.113 | 185.130 |
| **Entry trigger, wait before A** | **155.180** | **182.696** |
| Mid-mainloop trigger, wait after initialization | 161.457 | 185.331 |
| Mid-mainloop trigger, wait before A | 158.888 | 182.458 |
| Post-init trigger, wait before A | — | 182.934 |
| Tail trigger, wait before A | — | 186.830 |

Entry + before-A saves **4.016 us / 2.52%** for NVFP4 and
**3.764 us / 2.02%** for MXFP4 versus the original serial implementation.
All 17 corresponding sample pairs improve in both cases: savings range
2.919–6.116 us for NVFP4 and 2.896–5.791 us for MXFP4. These are paired
sample averages, not per-invocation tail latency or confidence intervals.
For MXFP4 the midpoint is 0.238 us faster than entry by median (2.15% versus
serial), but the sample ranges overlap; this is not a convincing universal
reason to change the default trigger.

`pdl_wait="before_a"` lets the TMA issuer issue independent B/SFB loads
before waiting on FC1, then waits before the first dependent A/SFA load.
Existing completion barriers prevent other warps from consuming those data
prematurely. In these cases it saves another 2.4–2.9 us over the init wait.
The prepared serial control shows that early output clearing has a cost;
comparing only against that control would overstate the net application gain.
An earlier experiment with a separate PDL preparation kernel did not remove
that cost reliably; that experimental implementation was removed.

## Physical overlap

A separate Nsight Systems run of the NVFP4 workload, 10 marked pairs per
mode, measured these median intersections of the FC1 and FC2 intervals:

| Mode | Interval intersection (us) | Fraction of shorter interval |
|---|---:|---:|
| Serial | 0.000 | 0.00% |
| Entry / before_a | 9.312 | 16.61% |
| Mainloop midpoint / before_a | 11.360 | 19.55% |

These intervals include FC2 startup and dependency waiting, not just useful
compute. More interval overlap does not necessarily produce more net savings:
the midpoint has more measured intersection here but worse unprofiled latency.
Profiling also perturbs execution; use the unprofiled table for speedup.

## Earlier workloads

The initial seven-trigger sweep used the initialization wait, tokens=128,
H=6144 and I=2048. With 256 experts/top-8, serial latency was 20.108 ms for
NVFP4 and 22.309 ms for MXFP4; no trigger showed a stable gain. With 8
experts/top-2, serial latency was 843.646 us and 933.458 us respectively,
and PDL was about 1–2% slower. Moving the trigger into the first task's
mainloop did not recover that cost.

The subsequent 100–200 us workloads above use deferred FC2 waiting and
show positive net savings. Performance depends on workload, available SM
resources, wait placement and preparation cost; interval overlap alone is
insufficient to establish a speedup.

## Validation

`compute-sanitizer --tool memcheck --error-exitcode 99` completed with
**0 errors** for NVFP4 cluster_m=2 (the benchmark configs above) and MXFP4
cluster_m=1 with `deterministic=True` in FC2 and `direct_gmem=True` in FC1.
Both used the fixed MoE dimensions, entry trigger, before-A wait, CUDA Graph,
0 warmups, 1 iteration and 1 sample, with accuracy enabled. Sanitizer timing
is excluded from performance conclusions. NVFP4 cluster_m=1 also passed the
three-replay accuracy check with `--no-cuda-graph` and before-A waiting.
The main MXFP4 benchmark validates the cluster_m=2 atomic-reduction path.
Python compilation and `git diff --check` passed.

## DeepSeek-V4-Flash decode chains

`models/deepseek_v4_flash_sm110a` supports opt-in PDL for fixed CUDA Graph
single-token decode with `Transformer(args, pdl=True)`. The model benchmark
accepts `--pdl` / `--no-pdl` (default off). Both modes use quantizers that
write padded, blocked activation scales directly, so the switch compares
PDL on the same decode computation/layout rather than introducing scale
conversion kernels in the serial control. Prefill retains its serial path.

The enabled edges are:

- `fp8_act_quant -> fp8_gemm`: `compile_fp8_act_quant(blocked=True, pdl=True)`
  signals every CTA at entry; `Fp8GemmConfig(pdl=True, cluster_m=1, split_k=1)`
  enables dependent launch and waits after shared memory/TMEM initialization,
  before activation accesses (including optional data prefetches).
- `bf16tofp8_quant -> mxfp4_fc1`:
  `compile_bf16tofp8_quant(blocked=True, pdl=True)` signals at entry;
  `Mxfp4Fc1Config(pdl_input=True)` enables dependent launch, skips the normal
  scheduler memset, and waits after initialization before consuming A/SFA.
- `mxfp4_fc1 -> mxfp4_fc2`: the existing `pdl=True` switches apply, with
  FC2's `pdl_wait="before_a"`. Deterministic finalize remains a normal
  completion dependency after FC2.

Blocked quantization requires 128-aligned rows and K; the generalized
quantizer additionally requires block size 128 and UE8M0 scales. It returns
one-dimensional scale storage of `rows * K / 8` bytes, including replicated
K=32 scales and zero padding. The `_out` APIs accept this storage as well.

For repeated calls with reused FC1 metadata, call
`fc1.prepare_pdl_input(metadata1)` once per invocation **before the quantizer**,
inside graph capture. Preparing fresh FC1 metadata already resets the same
scheduler, so the model needs no additional reset. Prepare FC2 metadata,
output and routing-weight conversions before the quantizer as well, and
call `fc2.prepare_pdl_out(output, metadata2)` there. Keep all three kernels
adjacent on the same stream. Enabling FC1's incoming `pdl_input` is separate
from its outgoing `pdl` trigger so either edge can be used independently.
