# Indexer score prefill

`indexer_score_prefill(q, k, q_scales, k_scales, head_weight, causal=True)`
returns a BF16 candidate score matrix, `[Qs, KVs]` for a single sequence or
`[B, Qs, max_kv_len]` for fixed-length batched Q:

```text
score[q, kv] = sum(h=0..63) head_weight[q, h]
               * max(dot(Q[q, h, :], K[kv, :]), 0)
```

By default, the batch-local, top-left-aligned causal mask matches MHA: entries
with `kv > q` are left untouched. Positions beyond each batch's `kv_seqlens`
are also left untouched. Pass `causal=False` to compute the full valid matrix.
The `_out` entry point preserves existing values at masked positions; the
allocating entry point leaves those positions undefined. The `_out` entry point
and cluster selector accept the same keyword.

The kernel launches up to one CTA per SM. Each CTA owns a fixed, strided
subset of two-query tiles. For each query tile it loads Q and Q scales once,
then traverses only the KV tiles visible under the causal mask. It skips
fully masked output tiles without loading KV data or issuing MMA.
The default entry point autotunes
`cluster_m=1` and `cluster_m=2` with cold L2 on the first call for each
`(device, batch, Qs, KVs, layout, causal)` and caches the winner. Pass
`cluster_m=1` or `2` to select
a tactic explicitly; `select_indexer_score_prefill_cluster_m` returns the
selected value. With two CTAs, the producer warps take turns issuing KV TMA
multicast loads to both CTAs, halving those load requests per query pair.
Warp 0 loads Q and KV data with TMA;
warps 1 and 3 store one output row each; warp 2 issues the MMAs; and warps
4–7 perform ReLU and the weighted head reduction.
Two 256-row KV stages have separate shared-memory and output slots. Their TMEM
accumulators overlap by 16 columns, and the epilogue drains the shared range
before signaling the next MMA, following the capacity-constrained NVFP4 GEMM
scheme. Q and K scales reuse one 12-column TMEM scratch slot. This keeps the
full allocation at 508 of 512 columns while loading, MMA, epilogue, and stores
overlap across KV tiles. The store
warps use TMA for fully valid 256-column tiles of a single fixed-length
sequence when `KVs` is divisible by eight (the tensor-map row-stride alignment
requirement), and direct stores for boundary tiles and other layouts. This is
the candidate matrix from the design note; top-k selection is separate.

The epilogue stores ReLU results by candidate column and head in shared
memory after reading 256 candidate columns per head with four TMEM loads.
Four loader warps materialize the ReLU values, then eight reduction warps use
two lanes per candidate. ReLU values are rounded to BF16 before the weighted
reduction. Each lane keeps two BF16 FMA accumulators for alternating head pairs;
the two accumulators and then the two lanes are added in BF16, matching the
DeepGEMM SM100 MQA logits reduction order. Each reduction warp keeps its output
row's query weights in registers across KV tiles.
The head stride is padded to spread neighboring candidates across shared-memory
banks.

Inputs use the repository's MXFP4 format. The original unbatched inputs remain
supported:

- `q`: contiguous CUDA `torch.float4_e2m1fn_x2`, shape `[Qs, 64, 64]`.
- `k`: contiguous CUDA `torch.float4_e2m1fn_x2`, shape `[KVs, 64]`.
- `q_scales`: contiguous CUDA `torch.float8_e8m0fnu`, `ceil(Qs/2)*1024`
  bytes in the blocked 128-row format from `to_blocked_mxfp4_scales`.
- `k_scales`: same format, `ceil(KVs/128)*1024` bytes.
- `head_weight`: contiguous CUDA BF16 `[Qs, 64]`.

For a partial final Q or KV scale tile, pad the logical rows with zeros to
the next multiple of 128 before calling `to_blocked_mxfp4_scales`. The packed
Q and K tensors themselves contain only their logical rows. The kernel masks
the final output tile using `Qs` and `KVs`.

For batched dense input, use `q` `[B, Qs, 64, 64]`, `k` `[B, KVs, 64]`,
`head_weight` `[B, Qs, 64]`, and output `[B, Qs, KVs]`. Each batch's Q scales
occupy `ceil(Qs/2)*1024` bytes and K scales occupy `ceil(KVs/128)*1024` bytes;
concatenate these batch-local blocked scale buffers. Optional CUDA int32
`kv_seqlens` `[B]` masks each batch's KV tail.

For paged KV, pass `block_table` as contiguous CUDA int32 `[B, max_pages]`,
`kv_seqlens` as CUDA int32 `[B]`, and `max_kv_len`. K is
`[num_pages, page_size, 64]`, where `page_size` is 128 or 256. Each physical
page's blocked K scales occupy `(page_size/128)*1024` bytes. The output has
`max_kv_len` columns, and page-table entries map logical pages to physical
pages. The table must cover `max_kv_len` per batch.

Paged KV also accepts packed Q `[sum_q, 64, 64]`, `head_weight`
`[sum_q, 64]`, and CUDA int32 `cu_seqlens_q` `[B+1]`; pass `max_q_len`.
The output is `[sum_q, max_kv_len]`. Packed Q scales remain **batch-local**:
each batch contributes `ceil(max_q_len/2)*1024` bytes of blocked scales,
including padding after shorter sequences. This preserves the MXFP4 128-row
scale-block layout even when a packed batch begins at an odd query offset.

The public Python entry points are `indexer_score_prefill`,
`indexer_score_prefill_out`, `select_indexer_score_prefill_cluster_m`, and
`compile_indexer_score_prefill`.

The default cold-L2 sweep keeps the original 1K, 2K, 4K, 8K, 16K, and 32K
sequence lengths at `B=1` (using batched tensor shapes), then adds 8K at
`B=2, 4, 8, 16`:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_mxfp4_indexer_score_prefill.py
```

Use `--batch` and `--seq-lens` to select one dense-batch workload, or
`--batches` to sweep batch sizes across selected lengths. Add
`--paged-kv --page-size 128` (or 256) to use a shuffled physical page table;
add `--packed-q --q-lengths ... --kv-lengths ...` to test variable lengths
with packed Q. Explicit per-batch lengths require one `--seq-lens` value.
Use `--non-causal` to disable the causal mask,
`--no-accuracy` to skip the sampled PyTorch reference, and `--warmup`,
`--iters`, and `--samples` to adjust timing.
Use `--sweep-cluster-m` to measure both tactics again and update the autotune
cache, or `--cluster-m 1` / `--cluster-m 2` to benchmark a fixed tactic.
Add `--sol` for MXFP4 MMA, compulsory HBM, and tile-traffic L2 rooflines.
The reported TF/s and MMA roofline count the executed 2-by-256 tiles,
including partial tiles.
The roofline omits epilogue instruction throughput and launch overhead.

## Decoding

`indexer_score_decode(q, k, q_scales, k_scales, head_weight, split_kv=None,
*, block_table=None, kv_seqlens=None, max_kv_len=None)`
computes the same score for one query per batch and returns BF16 `[B, KVs]`.
`indexer_score_decode_out` writes into a supplied contiguous BF16 output.
The CUDA implementation and binding are in separate `indexer_score_decode_*`
files; `compile_indexer_score_decode` precompiles their JIT module.

- `q`: contiguous CUDA MXFP4 `[B, 64, 64]` (packed D=128).
- `k`: contiguous CUDA MXFP4 `[B, KVs, 64]`, or paged
  `[num_pages, page_size, 64]` with `page_size` 128 or 256.
- `q_scales`: contiguous blocked E8M0, `B * 1024` bytes. Pad each batch's
  64 query rows to 128 before calling `to_blocked_mxfp4_scales`.
- `k_scales`: contiguous blocked E8M0, `B * ceil(KVs/128) * 1024` bytes.
  Pad each batch's KV rows to a multiple of 128 before blocking scales.
- `head_weight`: contiguous CUDA BF16 `[B, 64]`.
- `kv_seqlens`: optional contiguous CUDA int32 `[B]` containing each batch's
  valid KV length. Paged K requires this argument. Scores beyond the valid
  length are `-inf`, and fully invalid KV tiles skip computation.
- `block_table`: optional contiguous CUDA int32 `[B, max_pages]` mapping
  logical pages to physical pages. With paged K, also pass `max_kv_len`; K
  scales contain `num_pages * (page_size/128) * 1024` bytes in physical-page
  order.

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
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_mxfp4_indexer_score_decode.py
```

Use `--batches`, `--kv-lens`, and `--split-kvs` to select cases. By default,
each case uses reproducible random KV lengths (controlled by `--seed`) between
half and one-and-a-half times `KVs`, paired so their sum remains `B * KVs`.
The dense KV/output capacity (`Kcap`) is `1.5 * KVs` to allow longer sequences;
`B=1` remains uniform. `--uniform` restores full-length sequences and `Kcap=KVs`.
`--paged-page-size {128,256}` enables shuffled physical-page accuracy checks.
The benchmark
checks sampled batch rows against a PyTorch reference with BF16 head FMA
rounding unless
`--no-accuracy` is passed. It reports latency in microseconds, useful TF/s,
and maximum absolute error; `--sol` adds approximate rooflines.
The default sweep also reports the autotuned split for each batch size. Use
`--no-auto` for fixed splits only, `--force-sweep` to remeasure the tuning
choice, and `--cache-path` to set the persistent cache location.
