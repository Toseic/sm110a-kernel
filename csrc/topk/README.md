# SM110A row Top-K

`topk(scores, k)` selects the largest K scores in each row and returns CUDA
INT32 indices `[Qlen, k]`, unordered by score. It is a separate selection
step after the MXFP4 indexer score kernel.

```python
from sm110a_kernel import compile_topk, topk, topk_out

indices = topk(scores, 512)

# Precompile before CUDA graph capture; reuse the output buffer.
module = compile_topk()
module.topk_out(scores, indices, 512)
# Equivalent public out entry point:
topk_out(scores, indices, 512)
```

The API accepts:

- `scores`: contiguous CUDA BF16 or FP32 `[Qlen, KVlen]`.
- `k`: integer in `[0, KVlen]`.
- `out` for `topk_out`: contiguous CUDA INT32 `[Qlen, k]`, on the scores
  device, without aliasing scores.

Empty query batches and K=0 are supported. Scores are unchanged; ties choose
smaller indices, signed zeros compare equal, and NaNs rank above +inf. The
out entry point allocates no GPU memory and launches on the input device's
current CUDA stream.

Decode scores `[B, KVlen]` can be passed directly. Flatten a contiguous
batched prefill output `[B, Qlen, KVlen]` with
`scores.view(-1, scores.size(-1))`. For causal or ragged prefill, initialize
the score output to `-inf` before calling `indexer_score_prefill_out`, because
that producer leaves masked positions untouched. Top-K does not add a mask
or KV-length metadata. If fewer than K unmasked entries exist, masked `-inf`
positions fill the remaining slots; this API does not emit `-1` padding.

## Naive implementation

`src/topk_kernel.cu` implements the design in
[`docs/note/.topk.design.md`](../../docs/note/.topk.design.md): one 256-thread
CTA per row. The CTA finds the kth-largest score with 32 binary radix passes,
then emits indices using block prefix scans. Output happens to follow input
index order; consumers should rely only on unordered top-k semantics.

This is the functional starting point for optimization. There is no
autotuning, multi-CTA partitioning, or fusion with score production.
`src/topk_torch_binding.cpp` validates tensors and launches the kernel;
`sm110a_kernel/kernels/topk.py` owns the split CUDA JIT and Python facade.

## Benchmark

All correctness checks live in `benchmarks/benchmark_topk.py`. The default
run executes 31 naive contract check groups, then checks every measured
implementation for exact selected-value multisets, distinct indices, and
index bounds. Tie positions may differ between implementations.

The contract checks cover odd widths, K=0/1/KVlen, empty rows, ties, masks,
infinities, NaNs, signed zeros, invalid inputs, a nondefault stream, and CUDA
graph replay. `--check-only` runs these checks without timing or loading
baselines; `--no-accuracy` skips correctness checks.

The default sweep is:

| Parameter | Values |
|---|---|
| Qlen | 1, 8, 64, 256 |
| KVlen | 4096, 16384, 65536, 131072 |
| K | 128, 512, 1024 |
| dtype | BF16, FP32 |
| Distribution | normal |

This gives 96 configurations and 432 implementation measurements with all
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
  --q-lens 1 8 --kv-lens 4096 --topks 128 1024 --dtypes bf16 fp32 \
  --distributions ties constant ascending masked spikes
```

Use `--baseline none|torch|gvr-v2|deepselect|all` to select baselines (`all`
is the default). `--seed` controls input generation; `--json` exports device,
source revisions, workload/configuration metadata, accuracy status, and
min/median/max timing statistics. `--no-lock-clocks` leaves clocks unchanged.
The reported speedup is naive time divided by the measured implementation's
time, or `score+naive` time divided by chain time for producer chains.

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
default. NaN contract checks exercise only naive; measured distributions
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

## Thor measurements

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
  --baseline deepselect --q-lens 1 8 --kv-lens 513 8193 --topks 129 \
  --dtypes bf16 fp32 --warmup 0 --iters 1 --samples 1 --no-lock-clocks
```
