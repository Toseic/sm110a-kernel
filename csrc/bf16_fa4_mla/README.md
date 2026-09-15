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

Each batch uses two independent CTAs. Each CTA owns 64 query heads and emits
their complete 512-feature output, so QK and softmax are no longer duplicated
over an output-D split. QK uses two interleaved M64N32 instructions for each
K=16 step; PV uses eight M64N64 output chunks with K=64.

The M=64 tcgen05 layout packs pairs of logical N tiles into alternate 16-row
datapath groups. Consequently S0 and S1 need 32 TMEM columns each, while the
logical O `[64,512]` needs 256 columns. P is kept in shared memory so PV can
use the interleaved accumulator layout; a TMEM A operand would force the
non-interleaved layout.

Q `[64,576]` occupies 72 KiB. Two `[64,576]` K/V stages occupy 144 KiB. P0
has a dedicated 8 KiB slot, while P1 aliases stage 1's dead 64-feature RoPE
chunk, for 224 KiB total dynamic shared memory. PV releases each of the first
eight 64x64 V chunks independently to the following K load. Stage 0's RoPE
chunk is released after QK; stage 1's aliased chunk is released after PV1.

The steady-state tensor-core order is `QK0, PV1, QK1, PV0`. WG0 retains the
TMA/MMA roles, WG1 uses four warps for online softmax, and WG2 uses four warps
for conditional O rescaling and the epilogue. The final epilogue reuses Q
shared memory as its 64 KiB output staging buffer.

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
