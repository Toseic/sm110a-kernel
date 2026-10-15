# SM110A row Top-K

`topk(scores, k)` selects the largest K scores in each row and returns CUDA
INT32 indices `[Qlen, k]`, unordered by score. It is a separate selection
step after the MXFP4 indexer score kernel.

```python
from sm110a_kernel import compile_topk, topk, topk_out

indices = topk(scores)  # K is fixed at 512

# Precompile before CUDA graph capture; reuse the output buffer.
module = compile_topk()
module.topk_out(scores, indices, 512)
# Equivalent public out entry point:
topk_out(scores, indices, 512)
```

The API accepts:

- `scores`: contiguous CUDA BF16 or FP32 `[Qlen, KVlen]`.
- `k`: defaults to 512; only 512 is supported. `KVlen` must be >= 512.
- `out` for `topk_out`: contiguous CUDA INT32 `[Qlen, k]`, on the scores
  device, without aliasing scores.

Empty query batches are supported. Scores are unchanged; ties choose
smaller indices, signed zeros compare equal, and NaNs rank above +inf. The
out entry point allocates no GPU memory after autotune prewarming and launches on the input device's
current CUDA stream.

Decode scores `[B, KVlen]` can be passed directly. Flatten a contiguous
batched prefill output `[B, Qlen, KVlen]` with
`scores.view(-1, scores.size(-1))`. For causal or ragged prefill, initialize
the score output to `-inf` before calling `indexer_score_prefill_out`, because
that producer leaves masked positions untouched. Top-K does not add a mask
or KV-length metadata. If fewer than K unmasked entries exist, masked `-inf`
positions fill the remaining slots; this API does not emit `-1` padding.

## Buffered radix implementation

`src/topk_kernel.cu` implements the full loop in
[`docs/note/.topk.design.md`](../../docs/note/.topk.design.md), with a cluster
of 1, 2, 4, or 8 256-thread CTAs per row. Each iteration reads 1024 scores and
stably appends values above the threshold, carrying original indices.
Before the first selection, all values participate, including `-inf`.

Input uses a configurable ring of 2–8 shared-memory stages. The CTA primes
multiple slots before processing and consumes them in input order. After
filtering and any radix selection, it refills the consumed slot for the block
`stages` positions ahead. Loads overlap computation through asynchronous
copies; the compute threads also issue prefetches. There is no independent
producer warp or per-stage free barrier.

`compile_topk(stages=2, copy_engine="ldgsts")` selects a fixed JIT
configuration. The default is two stages with LDGSTS. `copy_engine="tma"`
enables single-thread 1D TMA bulk copies for full blocks of 16-byte-aligned
rows, with one completion mbarrier per stage and phase tracking across reuse.
This copy requires no tensor-map descriptor. Other blocks use LDGSTS:
16-byte copies for aligned rows and 4-byte copies for other 4-byte-aligned
rows. BF16 rows with only 2-byte alignment use ordinary loads into the ring.
Tail copies are zero-filled and masked by logical row length.

Input staging costs `stages * 1024 * sizeof(dtype)` bytes, in addition to the
candidate buffer, scratch, and `stages * sizeof(uint64_t)` barrier storage.
Larger configurations opt in to shared memory above the default 48 KiB block
limit. Stage count and copy engine are JIT cache keys; prepare the desired
module before CUDA graph capture.

```bash
# Compare identical workloads; shared benchmark logic locks GPC/EMC by default.
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --baseline deepselect --stages 4 --copy-engine tma
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --baseline deepselect --stages 4 --copy-engine ldgsts
```

### Mask filter

Each thread owns four consecutive elements, retains them in registers, and
builds a four-bit `uint32_t` mask. One CTA count scan assigns output ranges;
`ffs` and `mask &= mask - 1` visit selected elements. BF16 uses packed
`set.gtu.s32.bf16x2`, `prmt.b32`, and `dp4a.s32.s32`; FP32 uses unordered
floating greater-than without FTZ, avoiding per-element key generation.
A NaN threshold keeps no later candidates. Tail bits are masked, NaNs rank first, and signed zeros tie. This is the
only Filter implementation; `filter_mode` and `--filter-mode` were removed.

### CGA split-K and autotuning

`module.topk(scores, k, split_k=4)` and `topk_out(..., split_k=4)` fix the
number of CTAs per row to 1, 2, 4, or 8. Each CTA streams a contiguous range
of 1024-element input blocks and retains at most K local candidates. A binary
tree merges these candidates within the cluster using distributed shared
memory. Each merge appends at most 512 peer candidates and selects once from
at most 1024 entries through the same radix device function.
There is no global candidate workspace or second merge kernel. Empty
partitions participate in synchronization; ties preserve smaller indices.
Split 1 uses a normal CTA launch and has no cluster synchronization.

The default `split_k=None` selects the fastest valid split with
`AutotuneRunner` and cold-L2 CUDA event timings. The cache key includes device,
shape, dtype, input alignment, stages, and copy engine. K is fixed at 512;
a dedicated tuning namespace keeps earlier variable-K choices separate. Candidates that
would have empty input partitions are excluded from tuning; explicit splits
still support them. Results use the shared persistent cache plus a module
memory cache. First-time tuning allocates eviction and optional output
buffers. After prewarming, `topk_out` allocates no GPU memory.

Before CUDA graph capture, prewarm the same shape with `topk_out`, or specify
an explicit split. `module.select_split_k(scores, k, out, options=TuneOptions(...))`
exposes runner options. Benchmark selection uses common clock locking,
cache controls, and L2 eviction. Selected splits appear in logs and each JSON
result; timing uses a fixed selected split outside the runner.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --baseline deepselect --force-sweep-ours --json reports/topk_split_auto.json
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --baseline deepselect --split-k 8 --json reports/topk_split8.json
```

### Candidate selection

K is a CUDA compile-time constant of 512; candidates live in shared memory. When their count reaches
K+1024, the device function `radixselectTopk` selects the K largest candidates
using two 8-bit histogram passes for BF16 or four for FP32, compacts values and indices in place, and
returns the kth-largest value as the new threshold. A final selection handles
the tail. Later values equal to the threshold are discarded because retained
ties have smaller original indices. The buffer reserves K+2048 entries to
cover an entire block append before compaction.

`radixselectTopk` and its scratch storage are local to the `.cu` file. CTA
prefix sums use ballot/popcount, warp reductions, shuffles and shared memory;
there is no CUB dependency.
BF16 candidates are exact FP32 expansions, so two high-byte passes determine
the entire key; the low half is reconstructed, including negative values and
canonical NaNs. Compaction reads four adjacent candidates per thread into
registers, packs equal/greater counts into two 11-bit fields, and scans once
per 1024 candidates. Each thread emits its selected candidates in input order,
preserving stable ties. General shuffle scans handle these packed counts;
filter counts use three ballot bit planes with `popc` and `%lanemask_le`. General histogram counts use shuffle prefix steps with the
hardware validity predicate. Each warp uses `redux.sync.add` for the total
and for the sum of preceding warps, requiring two CTA barriers per prefix sum. Histogram selection reuses its bucket-publication barrier to
protect scan scratch, requiring only one barrier inside that scan. In-place
compaction publishes output once at the end rather than after every tile.
The primitive takes a runtime candidate count so it also handles partial
final blocks. Compaction is safe in place because each tile loads its values
and indices into registers before writing earlier output positions.
Candidate loops use 32-bit offsets within the bounded shared buffer; global
row and input addresses retain 64-bit arithmetic.

`src/topk_torch_binding.cpp` validates tensors and launches the kernel;
`sm110a_kernel/kernels/topk.py` owns the split CUDA JIT and Python facade.
Output follows input index order; consumers should rely only on unordered
Top-K semantics. FP32/BF16 support, NaN ordering, signed-zero ties, allocation
behavior, and stream/graph contracts are unchanged.

## Benchmark

All correctness checks live in `benchmarks/benchmark_topk.py`. The default
run executes 29 radix contract check groups, then checks every measured
implementation for exact selected-value multisets, distinct indices, and
index bounds. Tie positions may differ between implementations.

The contract checks cover odd widths, KVlen=512, empty rows, ties, masks,
infinities, NaNs, signed zeros, invalid inputs, a nondefault stream, and CUDA
graph replay. `--check-only` runs these checks without timing or loading
baselines; `--no-accuracy` skips correctness checks.

The default sweep is:

| Parameter | Values |
|---|---|
| Qlen | 1, 8, 64, 256 |
| KVlen | 4096, 16384, 65536, 131072 |
| K | 512 |
| dtype | BF16, FP32 |
| Distribution | normal |

This gives 32 configurations and 144 implementation measurements with all
baselines enabled. L2 eviction runs before every measured invocation, outside
the CUDA event pair. Compilation, allocations, and initialization occur
before timing. Default performance comparisons use cold L2.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py --check-only
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py --json reports/topk.json

# Custom shape and timing.
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --q-lens 1 64 --kv-lens 16384 --topks 512 --dtypes bf16 \
  --warmup 5 --iters 20 --samples 5

# Adversarial distributions.
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --q-lens 1 8 --kv-lens 4096 --topks 512 --dtypes bf16 fp32 \
  --distributions ties constant ascending masked spikes
```

Use `--baseline none|torch|gvr-v2|deepselect|all` to select baselines (`all`
is the default). `--seed` controls input generation; `--json` exports device,
source revisions, workload/configuration metadata, accuracy status, and
min/median/max timing statistics. `--no-lock-clocks` leaves clocks unchanged.
The reported speedup is radix time divided by the measured implementation's
time, or `score+radix` time divided by chain time for producer chains.

## Baselines

| Measurement | Input | Output and adapter |
|---|---|---|
| `torch` | BF16/FP32 | Preallocated values and INT64 indices, `sorted=False`; time includes writing values |
| `gvr-v2-kernel` | FP32 | Preallocated INT32 indices; excludes BF16 conversion |
| `cast+gvr-v2` | BF16 | BF16→FP32 copy into a preallocated buffer, then GVR v2 |
| `deepselect-kernel` | BF16/FP32 | Preallocated INT32 indices, no value output |
| `pad+deepselect` | BF16/FP32 | Copy into an aligned buffer, then DeepSelect; present for unaligned row strides |

TRT-LLM GVR v2 uses
`gvr_topk_decode_self_sampling_host.run_varlen` with `next_n=1`,
`compress_ratio=1`, device KV lengths, a fixed envelope, and a reusable
workspace. It is the hint-free self-sampling version. The benchmark uses
upstream's standalone import path from `baselinedir/TensorRT-LLM`, avoiding
the full TRT runtime import. Set `--trtllm-root` for another checkout. GVR
requires KVlen divisible by four in this benchmark.

[DeepSelect](https://github.com/deepseek-ai/DeepSelect) uses `sorted=False`,
`sorted_index=False`, `indices_type=torch.int32`, `return_value=False`, and
`output_idx` to reuse output memory. It supports K in `[1, 4096]` and KVlen
below 2^23. Input row strides must be aligned to 1024 bytes; output strides
must be aligned to 32 bytes. The adapter handles arbitrary widths/K with
aligned storage and logical views. Padding copies are excluded from the
kernel-only measurement and included in adapter and chain measurements.
Set `--deepselect-root` for another checkout; no global package installation
is required after an in-place build.

GVR's NaN selection differs from Torch, and DeepSelect's NaN check aborts by
default. NaN contract checks exercise only radix; measured distributions
contain no NaNs.

### Building DeepSelect on Thor

Upstream `setup.py` targets SM100a/103a. The local
[`deepselect_sm110.patch`](../../baselinedir/deepselect_sm110.patch) adds the
repository's paired architecture override while retaining upstream defaults.
Run these commands from the repository root on a fresh checkout:

```bash
git clone --recursive https://github.com/deepseek-ai/DeepSelect.git baselinedir/DeepSelect
git -C baselinedir/DeepSelect apply ../deepselect_sm110.patch
cd baselinedir/DeepSelect
CUDA_ARCH=sm_110a COMPUTE_CUDA_ARCH=compute_110a \
  MAX_JOBS=4 NVCC_THREADS=2 DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA \
  /home/bifrost/miniforge3/bin/python3 setup.py build_ext --inplace
cd ../..

/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py --baseline deepselect
```

The current local checkout is already patched and built for SM110a with CUDA
13.2. `cuobjdump --list-elf` confirms SM110a cubins; upstream's build-time
spill check reports no register spills.

## Indexer score → Top-K

`--score-source indexer-decode` generates real BF16 decode scores; Qlen is
the batch size. `--score-source indexer-prefill` uses one dense, noncausal
sequence. Both modes require `--dtypes bf16 --distributions normal` and use
fixed producer configurations (`--split-kv` or `--cluster-m`), without
producer autotuning.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --score-source indexer-decode --q-lens 1 8 --kv-lens 16384 \
  --topks 512 --dtypes bf16 --split-kv 1

/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --score-source indexer-prefill --q-lens 64 256 --kv-lens 16384 \
  --topks 512 --dtypes bf16 --cluster-m 1
```

The benchmark measures selection alone and the complete score→selection
chain for each baseline. Selection-only timing flushes L2 before selection;
chain timing flushes once before the producer, so selection sees the cache
state left by score production. Chains include necessary conversion/padding
copies. Top-K checks compare with Torch on produced scores; indexer numerical
accuracy is covered by its own benchmarks.

## Current four-candidate compaction sweep

2026-10-15, relative to `813de09`, two complete 32-case cold-L2 sweeps measured
1.066x and 1.064x geometric mean speedups. The final repeat gives BF16 1.068x,
FP32 1.061x, with 29/32 cases faster (16/16 FP32). Three BF16 long-sequence
cases regress 1.0–1.4%. Conditions: stage2 LDGSTS, K512, freshly autotuned
split-k=1/2/4/8, normal seed1234, GPC1575MHz/EMC4266MHz, 64MiB dirty L2
flush, warmup3/iters10/samples5. Final sweep enables accuracy.

The retained change processes four adjacent candidates per thread in radix
compaction. Register loads precede the scan barriers, protecting in-place
compaction; scan frequency falls from once per 256 candidates to once per
1024. Two 11-bit fields prevent carry between equal/greater counts. The
filter keeps its bit-plane scan. The unused single-item boolean-pair scan
specialization was removed; namespace is `topk512_cga_mask_items4_v4`.

Experiments with 8 or 10 candidates per thread, double-buffered scan scratch,
a filter barrier removal, a scan shortcut before the initial threshold,
and threshold-warp-only histogram scans were not retained.

An explicit `bfe.u32` byte-extraction experiment follows PTX ISA 9.4 section
9.7.1.20 in `docs/ptx-isa-9.4/`. On SM110a it compiled to PRMT selections.
An isolated 32-case comparison against the same four-candidate implementation
measured 0.9946x (12/32 faster), so explicit BFE was reverted. Source snapshots
and A/B JSON are in `/tmp/topk-scan2-opt/`, including `items4bfe.cu` and
`items4bfe_sweep.json`; SASS is `/tmp/ncu-reports/topk_bfe.sass`.

Validation: 192/192 exact selected-value workload/distribution cases and 29
contract groups for each split-k=1/2/4/8 with stage2 LDGSTS and stage8 TMA.
Racecheck on the 29 split1 contract groups reports zero errors/warnings.

NCU source counters confirm dynamic BAR instructions fell 9.67% for
Q256/N131072/BF16/split1, 8.96% for FP32 at the same configuration, and
25.95% for Q1/N131072/BF16/split8. Registers are 37/40/39 respectively,
with no stack or local spills in any variant. Stall ratios do not uniformly
improve; BF16 large-Q dynamic instructions increase, consistent with its
small regressions. Compare NCU replay diagnostics separately from benchmark
timing. Metric names and complete before/after tables are in the results file.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --baseline deepselect --force-sweep-ours \
  --json /tmp/topk-scan2-opt/final_sweep.json
```

Detailed results and validation commands: `/tmp/topk-scan2-opt/results.md`.
NCU/SASS and resource reports: `/tmp/ncu-reports/topk_items4*`.
These artifacts are temporary and not committed.

## Previous scan optimization sweep

2026-10-15, relative to commit `732ba82`, two full 32-case cold-L2 sweeps
with split-K autotuning measured 1.076x and 1.067x geometric mean speedups.
The final repeat gives BF16 1.075x (16/16 faster), FP32 1.059x (15/16 faster),
and 31/32 faster overall; the remaining case regresses 0.5%. Configuration:
stage-2 LDGSTS, K=512, normal inputs, seed=1234, GPC=1575 MHz/EMC=4266 MHz,
64 MiB dirty L2 eviction, warmup=3, iters=10, samples=5.

Retained changes use `redux.sync.add.s32` for warp-total reductions and
`vote.sync.ballot`/`popc`/`%lanemask_le` for the bounded compaction and filter
prefixes. PTX ISA 9.4 sections 9.7.15.13, 9.7.15.10, 9.7.1.15 and 10.19
in `docs/ptx-isa-9.4/` describe these operations. Reduction does not replace
the ordered general histogram scan. The filter-only increment was smaller:
1.008x overall relative to the reduction + compaction-ballot version.
The autotune namespace is `topk512_cga_mask_scan_v3`.

Accuracy passed 192/192 shape/distribution cases and 29 contract groups for
each split-k=1/2/4/8 with stage-2 LDGSTS and stage-8 TMA, both before and
after the scan changes. These include tails, stable ties, special values,
streams and CUDA Graphs. The final timed sweep also enabled accuracy.

NCU/SASS confirms REDUX, VOTE and POPC with no spills. At Q=256,N=131072,
split=1, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`
fell from 2.658 to 1.215 (BF16) and 2.554 to 1.166 (FP32). Dynamic warp
instruction counts increased 5.84%/2.58%; the benefit is a shorter dependency
chain, and total instruction count alone does not predict performance.
At Q=1,N=131072,BF16,split=8, short-scoreboard fell 2.879 to 1.808.
Barrier stalls remain significant. NCU replay durations are diagnostic,
not benchmark timings.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --baseline deepselect --force-sweep-ours \
  --json /tmp/topk-scan-opt/final_sweep.json
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --baseline none --distributions normal ties constant ascending masked spikes \
  --warmup 0 --iters 1 --samples 1 \
  --json /tmp/topk-scan-opt/accuracy_after.json
```

Detailed A/B, full tables, accuracy logs and commands:
`/tmp/topk-scan-opt/results.md`. Source snapshots and JSON are alongside it.
NCU reports, raw/source CSVs, resource usage and SASS:
`/tmp/ncu-reports/topk_scan*`. These artifacts are temporary and not committed.

## Previous compute optimization sweep

2026-10-15, two complete 32-workload sweeps of the optimized kernel, compared
with a freshly measured pre-optimization baseline. All runs used stages=2,
LDGSTS, normal inputs, seed=1234, freshly autotuned split-K, locked GPC=1575
MHz / EMC=4266 MHz, and cold L2 with a 64 MiB dirty write-flush. Warmup=3,
iters=10, samples=5; accuracy checks were skipped. The final repeat gives:

| dtype | speedup vs pre-optimization | current / DeepSelect time |
|---|---:|---:|
| BF16 | 1.370× | 1.929× |
| FP32 | 1.169× | 1.647× |
| All | 1.265× | 1.782× |

All 32 workloads improved versus the pre-optimization baseline. The retained
changes are BF16 two-pass radix, fused compaction scans, shuffle predicates,
direct FP32 filter comparisons, 32-bit candidate offsets, and simpler CGA
merges. Complete-block copy/filter specializations were reverted after small-Q
regressions. The tuning namespace is `topk512_cga_mask_compute_v2`.

NCU/SASS on fixed Q=256, N=131072, split=1 configurations confirms dynamic
warp instruction reductions of 19.66% for BF16 and 21.94% for FP32. No local
spills were introduced. Scan dependencies remain a bottleneck; reduced
instruction counts do not imply lower stall cycles per issued instruction.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --baseline deepselect --no-accuracy --force-sweep-ours --debug \
  --json /tmp/topk-compute-opt/final_sweep.json
```

Detailed A/B results, source snapshots and logs: `/tmp/topk-compute-opt/`,
especially `results.md`, `baseline_sweep.json`, and `final_sweep.json`.
NCU analysis, reports, raw/source CSVs and SASS: `/tmp/ncu-reports/`,
especially `topk_compute_analysis.md` and `topk_compute_after*.ncu-rep`.
These artifacts are temporary and are not committed.

## Previous K=512 sweep after rollback

2026-10-15, all 32 default workloads completed with stages=2, LDGSTS,
cooperative 256-thread prefetch, and freshly autotuned split-K. Common
benchmark logic locked GPC=1575 MHz / EMC=4266 MHz and used cold L2 with a
64 MiB dirty write-flush. Seed=1234, normal inputs, warmup=3, iters=10,
samples=5. This performance sweep used `--no-accuracy`.

| dtype | current / DeepSelect time | speedup vs producer/consumer |
|---|---:|---:|
| BF16 | 2.669× | 1.209× |
| FP32 | 1.919× | 1.183× |
| All | 2.263× | 1.196× |

Ratios are geometric means; producer/consumer comparisons use separate runs
with the same measurement settings. All 32 workloads remain slower than
DeepSelect. Split=1/2/4/8 is selected for 16/6/6/4 workloads respectively.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --baseline deepselect --no-accuracy --force-sweep-ours --debug \
  --json reports/topk512_rollback_sweep.json
```

Full results and candidate timings are in ignored `reports/`:
`topk512_rollback_sweep.md`, `topk512_rollback_sweep.json`, and
`topk512_rollback_sweep.log`.

## Reverted producer/consumer experiment

2026-10-15, NVIDIA Thor. Compared against the former cooperative pipeline
with **K=512 as a compile-time constant in both versions**. Each sweep covers
32 default workloads, with split-K freshly autotuned. GPU runs are serial,
GPC=1575 MHz / EMC=4266 MHz locked, cold L2 with 64 MiB dirty write-flush,
warmup=3, iters=10, samples=5, normal inputs, seed=1234. This measurement used
`--no-accuracy`.

| Copy engine | Stages | Before/after time (speedup) | Faster workloads |
|---|---:|---:|---:|
| LDGSTS | 2 | 0.832× | 1/32 |
| LDGSTS | 8 | 0.932× | 13/32 |
| TMA | 2 | 0.990× | 15/32 |
| TMA | 8 | 1.003× | 19/32 |

The independent producer provides no stable overall gain in this sweep.
LDGSTS takes about 20% longer with two stages and 7% longer with eight;
TMA stays approximately unchanged. Matching split-K candidate comparisons
also show the LDGSTS regression. The full report includes commands, per-case
medians, split choices, and source snapshots in ignored `reports/`:
`topk_pc_comparison.md`, `topk_{before_pc,pc}_{ldgsts,tma}_s{2,8}.json/.log`,
and `topk_pc_ab_sources/`. The prior implementation was restored only in a
temporary directory for comparison. The producer/consumer experiment has
been reverted; the workspace now uses the cooperative prefetch pipeline.

## Previous variable-K split-K sweep

These historical results predate fixing K=512; they include K=128/512/1024.

2026-10-15, NVIDIA Thor, stages=2, LDGSTS, mask filter, normal inputs,
seed=1234. The full default 96-workload sweep remeasured split-K choices
1/2/4/8. Common benchmark logic locked GPC=1575 MHz / EMC=4266 MHz and used
cold L2 with 64 MiB dirty write-flush. Warmup=3, iters=10, samples=5.
This performance run used `--no-accuracy`; no correctness checks were run.

| dtype | speedup vs previous fixed-split mask | current / DeepSelect time |
|---|---:|---:|
| BF16 | 1.162× | 2.762× |
| FP32 | 1.200× | 1.996× |
| All | 1.181× | 2.348× |

Ratios are geometric means. The old-mask comparison uses separate runs with
the same clock/cache/timing settings. All 96 current workloads remain slower
than DeepSelect. Qlen=1 and Qlen=8 improve by 1.434× and 1.280×; Qlen=256
selects split=1 for every workload. Overall split choices are
1:47, 2:19, 4:17, 8:13 workloads.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --baseline deepselect --no-accuracy --force-sweep-ours --debug \
  --json reports/topk_split_sweep.json
```

Local artifacts in ignored `reports/`: `topk_split_sweep.json`,
`topk_split_sweep.log` (including candidate timings), and
`topk_split_sweep.md` (all 96 rows and split choices).

## Thor measurements

These historical results measure the former naive implementation, before the
buffered radix replacement described above.

Measured on NVIDIA Thor (SM110a), 2026-10-14, seed 1234, cold L2 with a 64 MiB
dirty write-flush buffer and unlocked clocks (`--no-lock-clocks`). Source
revisions:

- TRT-LLM: `edc41150e6bd51095b207efa2c4a4eeee7b6ce63`.
- DeepSelect: `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18`, with the local build patch.
- DeepSelect CUTLASS: `ae6bccf341fb4410241f696ba06873023d5ce4ed`.

The default 96-configuration sweep passes all 432 implementation checks.
An additional 72 configurations at KVlen=513 cover Qlen={1,8}, K={1,129,513},
both dtypes, and all six distributions; all 216 implementation checks pass.
K=4096 passes on Qlen={1,8}, KVlen=8193, both dtypes, and normal/ties/constant
inputs. Both indexer chains, including an unaligned decode chain, pass.
CUDA memcheck reports **0 errors** on Qlen={1,8}, KVlen={513,8193}, K=129,
both dtypes, including the padding adapter.

Representative chain times below are median microseconds, BF16 scores,
KVlen=16384, K=512, 2 warmups, 5 invocations per sample, and 3 samples. Decode
uses split_kv=1; prefill uses cluster_m=1. DeepSelect needs no input adapter
for these aligned BF16 scores; GVR includes BF16→FP32 conversion.

| Producer | Query rows | score+naive | score+Torch | score+cast+GVR v2 | score+DeepSelect |
|---|---:|---:|---:|---:|---:|
| Decode | 1 | 171.699 | 110.515 | 50.554 | 47.072 |
| Decode | 8 | 215.290 | 154.368 | 95.866 | 90.579 |
| Prefill | 64 | 472.614 | 260.710 | 277.344 | 152.403 |
| Prefill | 256 | 1327.232 | 671.571 | 486.432 | 416.371 |

These are initial unlocked-clock measurements. The complete selection sweep
used 1 warmup, 2 invocations, and 1 sample. Local JSON artifacts live in the
ignored `reports/` directory: `topk_all_deepselect_full_sm110a.json`,
`topk_deepselect_padding_sm110a.json`,
`topk_deepselect_indexer_decode_sm110a.json`, and
`topk_deepselect_indexer_prefill_sm110a.json`.

```bash
# Reproduce the complete selection sweep.
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --baseline all --warmup 1 --iters 2 --samples 1 --no-lock-clocks \
  --json reports/topk_all_deepselect_full_sm110a.json

# Reproduce the memory check.
compute-sanitizer --tool memcheck --error-exitcode 1 \
  /home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_topk.py \
  --baseline deepselect --q-lens 1 8 --kv-lens 513 8193 --topks 512 \
  --dtypes bf16 fp32 --warmup 0 --iters 1 --samples 1 --no-lock-clocks
```

## Compute optimization accuracy validation

The compute optimization passed all 192 benchmark cases (Q=1/8/64/256,
N=4096/16384/65536/131072, BF16/FP32, normal/ties/constant/ascending/masked/spikes)
with exact selected-value checks. Contract checks passed 29 groups for each
split-k=1/2/4/8 with stage-2 LDGSTS and stage-8 TMA, including partial tiles,
NaNs, infinities, signed zeros, subnormals, stable ties, streams, and graphs.
Logs and JSON: `/tmp/topk-scan-opt/accuracy_before*` and `check_before*`.
