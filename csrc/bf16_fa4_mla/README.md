# BF16 FA4 MLA decoding

This directory implements the one-token decoding path, adapted from MQA's
12-warp FA4-style schedule. H=128, D_KV=512, D_rope=64 are fixed.

The contiguous BF16 CUDA inputs use the same BHSD convention as MQA:

- Q: `[B,128,1,576]`, concatenating Q_absorb and Q_rope.
- K: `[B,1,S,576]`, concatenating cKV and K_rope along the last axis.
  Its logical transpose is K_input `[576,S]`.
- V: `[B,1,S,512]`, the same cKV used to form K.
- O: `[B,128,1,512]`.

The kernel computes `softmax(scale * (Q_absorb @ cKV.T + Q_rope @ K_rope.T)) @ cKV`.
The default scale is `1/sqrt(576)`; pass `scale=` explicitly for a model's
scaling convention. All S cache entries are visible. With Q length 1,
`causal=True` has the same bottom-right aligned semantics. Prefill, paged
caches, and the final value up-projection are outside this decoding path.

## Schedule

M=N=128 for QK; all 128 heads fill M. Each batch uses two CTAs, each producing
256 of the 512 output features. This duplicates QK/softmax work but keeps
128 score columns plus 256 FP32 output columns within the 512-column TMEM
allocation. PV uses M=128, N=256, K=128.

Q stays in 144 KiB of shared memory. K streams through nine 64-feature
chunks, each released by an MMA completion barrier. A shared 64 KiB K/V
buffer then receives this CTA's V slice. After the final PV, the output
epilogue reuses Q storage. Total dynamic shared memory is 208 KiB.

The MQA mechanisms retained here are TMA, tcgen05 QK/PV, BF16 P in TMEM,
online softmax, conditional O rescaling, partial-P completion, and separate
load/MMA, softmax, and rescale/epilogue warpgroups.

## Usage

```python
from sm110a_kernel import bf16_fa4_mla, bf16_fa4_mla_out

out = bf16_fa4_mla(q, k, ckv)
bf16_fa4_mla_out(q, k, ckv, out, scale=576 ** -0.5)
```

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mla.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mla.py \
  --single --batch 1 --kv-len 8192 --warmup 5 --iters 20
```

The default benchmark covers B=32/64/128 at S=8192. It constructs K and V
from the same cKV and checks against a batch-wise FP32 PyTorch reference
that evaluates the content and RoPE score terms separately. `--no-accuracy`
skips that check. TF/s reports useful attention FLOPs, excluding duplicated
QK computation. Timing uses the repository's cold-L2 policy by default.
