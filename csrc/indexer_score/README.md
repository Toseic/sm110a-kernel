# Indexer score prefill

`indexer_score_prefill(q, k, q_scales, k_scales, head_weight)` returns an
FP32 `[Qs, KVs]` candidate score matrix:

```text
score[q, kv] = sum(h=0..63) head_weight[q, h]
               * max(dot(Q[q, h, :], K[kv, :]), 0)
```

The kernel launches up to two CTAs per SM. Each CTA owns a fixed, strided
subset of two-query tiles. For each query tile it loads Q and Q scales once,
then traverses all 64-row KV tiles. Warp 0 loads Q and KV data with TMA and
stores the first output row; warp 1 stores the second row; warp 2 issues the
MMAs; warp 3 is idle; warps 4–7 perform ReLU and the weighted head reduction.
Two KV stages have separate shared-memory, TMEM accumulator, and output slots
so loading, MMA, epilogue, and stores can overlap across KV tiles. The store
warps use TMA when `KVs` is divisible by four (the tensor-map row-stride
alignment requirement), and direct stores otherwise. This is the full
candidate matrix from the design note; top-k selection is a separate operation.

The epilogue stores ReLU results by candidate column and head in shared
memory after reading all 64 candidate columns per head with one TMEM load.
Each lane accumulates all 64 heads for one candidate. Each epilogue
warp keeps only its output row's query weights in registers across KV tiles.
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
