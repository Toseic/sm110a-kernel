# BF16 FA4 sparse MLA, H=64

This is a separate one-token decode kernel for the DeepSeek V4 Flash attention
shape. It retains the existing 64-head CTA and QK/PV MMA schedule, adds four
warps for indexed KV gather, and includes the per-head attention sink in the
softmax denominator.

Inputs are contiguous CUDA tensors:

- Q: BF16 `[B,64,1,512]` (BHSD).
- KV: BF16 `[B,N,512]`, shared by all heads.
- Indices: int32 `[B,K]`, with `-1` for invalid entries. Other entries must be
  in `[0,N)`; out-of-range entries are also masked by the kernel.
- Sink: FP32 `[64]`, one logit per head.
- Output: BF16 `[B,64,1,512]`.

The operation is `sum(softmax(scale * QK, sink) * KV)`, where the sink has no
value vector. The default scale is `1/sqrt(512)`. This kernel takes already
selected cache indices; indexer scoring and cache updates stay with the model.

```python
from sm110a_kernel import bf16_fa4_sparse_mla_out

bf16_fa4_sparse_mla_out(q, kv, indices, sink, out)
```

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_sparse_mla_h64.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_sparse_mla_h64.py \
  --batch 1 --kv-len 8192 --selected-len 640 --invalid 16 --sol
```

The benchmark uses cold L2 by default, following the repository policy.
`--sol` reports an idealized bound that counts one KV row read per valid
selected index, plus Q, output, indices, and sink traffic. It does not model
extra index loads issued by the gather warps or cache reuse from repeated
indices.

## Prefill

`src/bf16_fa4_sparse_mla_prefill_kernel.cu` is a separate 16-warp prefill
kernel. Each CTA handles one query token and all 64 heads. Four warps gather
that token's indexed KV rows; the remaining warp groups issue MMA, softmax,
and the output epilogue. Query tokens run as independent CTAs because their
index lists may differ.

Inputs are contiguous Q/output `[B,64,Q,512]`, KV `[B,Q,512]`, int32 indices
`[B,Q,K]`, and FP32 sink `[64]`. Q and KV must have the same original sequence
length. An index outside `[0,Q)` is ignored. The
caller supplies the selected positions for each token, including any desired
causal restriction; the kernel does not infer token positions from cache rows.

```python
from sm110a_kernel import bf16_fa4_sparse_mla_prefill_out

bf16_fa4_sparse_mla_prefill_out(q, kv, indices, sink, out)
```

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_sparse_mla_prefill_h64.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_sparse_mla_prefill_h64.py \
  --single --batch 1 --q-len 1024 --selected-len 640 --sol
```

By default the prefill benchmark sweeps Q=KV lengths 1024, 2048, and 4096
at batch 1, then length 1024 at batch 4. `--single` uses the requested
batch and Q=KV length. Each query selects only from its causal KV prefix; slots
beyond the available prefix are padded with -1. It checks a PyTorch reference
and measures cold L2 by default. `--sol` reports an idealized bound using one KV read per valid
selection, with compulsory KV bytes counted once per distinct cache row in
each batch. It omits repeated index loads by the gather warps.
