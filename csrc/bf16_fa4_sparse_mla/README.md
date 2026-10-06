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
