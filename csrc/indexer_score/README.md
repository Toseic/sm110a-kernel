# Indexer score prefill

`indexer_score_prefill(q, k, q_scales, k_scales, head_weight)` returns an
FP32 `[Qs, KVs]` candidate score matrix:

```text
score[q, kv] = sum(h=0..63) head_weight[q, h]
               * max(dot(Q[q, h, :], K[kv, :]), 0)
```

The kernel launches up to one CTA per SM. Each CTA owns a fixed, strided
subset of two-query tiles. For each query tile it loads Q and Q scales once,
then traverses all 256-row KV tiles. Warp 0 loads Q and KV data with TMA;
warps 1 and 3 store one output row each; warp 2 issues the MMAs; and warps
4–7 perform ReLU and the weighted head reduction.
Two 256-row KV stages have separate shared-memory and output slots. Their TMEM
accumulators overlap by 16 columns, and the epilogue drains the shared range
before signaling the next MMA, following the capacity-constrained NVFP4 GEMM
scheme. Q and K scales reuse one 12-column TMEM scratch slot. This keeps the
full allocation at 508 of 512 columns while loading, MMA, epilogue, and stores
overlap across KV tiles. The store
warps use TMA when `KVs` is divisible by four (the tensor-map row-stride
alignment requirement), and direct stores otherwise. This is the full
candidate matrix from the design note; top-k selection is a separate operation.

The epilogue stores ReLU results by candidate column and head in shared
memory after reading 256 candidate columns per head with four TMEM loads.
Four loader warps materialize the ReLU values, then eight reduction warps use
two lanes per candidate; each lane accumulates 32 heads and the pair combines
with a shuffle. Each reduction warp keeps its output row's query weights in
registers across KV tiles.
The head stride is padded to spread neighboring candidates across shared-memory
banks.

Inputs use the repository's MXFP4 format:

- `q`: contiguous CUDA `torch.float4_e2m1fn_x2`, shape `[Qs, 64, 64]`.
- `k`: contiguous CUDA `torch.float4_e2m1fn_x2`, shape `[KVs, 64]`.
- `q_scales`: contiguous CUDA `torch.float8_e8m0fnu`, `ceil(Qs/2)*1024`
  bytes in the blocked 128-row format from `to_blocked_mxfp4_scales`.
- `k_scales`: same format, `ceil(KVs/128)*1024` bytes.
- `head_weight`: contiguous CUDA FP16 `[Qs, 64]`.

For a partial final Q or KV scale tile, pad the logical rows with zeros to
the next multiple of 128 before calling `to_blocked_mxfp4_scales`. The packed
Q and K tensors themselves contain only their logical rows. The kernel masks
the final output tile using `Qs` and `KVs`.

The public Python entry points are `indexer_score_prefill`,
`indexer_score_prefill_out`, and `compile_indexer_score_prefill`.

Benchmark the full prefill shape (`Qs = KVs = seq`) at 1K, 2K, 4K, 8K,
16K, and 32K with cold L2 by default:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_indexer_score.py
```

Use `--seq-lens` to select lengths, `--no-accuracy` to skip the sampled
PyTorch reference, and `--warmup`, `--iters`, and `--samples` to adjust timing.
Add `--sol` for MXFP4 MMA, compulsory HBM, and tile-traffic L2 rooflines.
The roofline omits epilogue instruction throughput and launch overhead.

## Decoding

`indexer_score_decode(q, k, q_scales, k_scales, head_weight, split_kv=None)`
computes the same score for one query per batch and returns FP32 `[B, KVs]`.
`indexer_score_decode_out` writes into a supplied contiguous FP32 output.
The CUDA implementation and binding are in separate `indexer_score_decode_*`
files; `compile_indexer_score_decode` precompiles their JIT module.

- `q`: contiguous CUDA MXFP4 `[B, 64, 64]` (packed D=128).
- `k`: contiguous CUDA MXFP4 `[B, KVs, 64]`.
- `q_scales`: contiguous blocked E8M0, `B * 1024` bytes. Pad each batch's
  64 query rows to 128 before calling `to_blocked_mxfp4_scales`.
- `k_scales`: contiguous blocked E8M0, `B * ceil(KVs/128) * 1024` bytes.
  Pad each batch's KV rows to a multiple of 128 before blocking scales.
- `head_weight`: contiguous CUDA FP16 `[B, 64]`.

`split_kv` accepts 1, 2, 4, or 8 and cannot exceed `ceil(KVs/256)`.
The default `None` measures valid splits with cold L2 on the first cache miss
for each `(device, B, KVs)`, caches the winner, and reuses it on later calls.
Pass an integer to bypass tuning. `select_indexer_score_decode_split_kv`
returns the selected integer without running the requested output call.
Each batch launches one cluster of `split_kv` CTAs. The split count is a
compile-time kernel template parameter; each CTA computes and stores its own
nonoverlapping KV range. Since the output is the full candidate vector, no
cross-CTA score reduction is needed. Top-k selection remains a separate step.

Benchmark decoding at `KVs=16384` for batch sizes 1, 2, 4, 8, 16, 32, 64,
and 128 across all supported split counts, with cold L2 by default:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_indexer_score_decode.py
```

Use `--batches`, `--kv-lens`, and `--split-kvs` to select cases. The benchmark
checks sampled batch rows against the FP32 PyTorch reference unless
`--no-accuracy` is passed. It reports latency in microseconds, useful TF/s,
and maximum absolute error; `--sol` adds approximate rooflines.
The default sweep also reports the autotuned split for each batch size. Use
`--no-auto` for fixed splits only, `--force-sweep` to remeasure the tuning
choice, and `--cache-path` to set the persistent cache location.
