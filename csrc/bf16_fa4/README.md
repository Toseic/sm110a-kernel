# BF16 FA4 MHA

This directory contains the SM110a BF16 FlashAttention-4-style MHA kernel.

`fa_mha` consumes paged K/V. Q and output use `[B, H, S, D]`, or packed
`[sum_q, H, D]` with `cu_seqlens_q`. It computes
`softmax(scale * Q @ K.T) @ V` in FP32 and stores BF16 output. With
`causal=True`, query position `i` attends to keys `0..i`.

Paged KV uses K/V `[num_pages, H, 16, D]`, an int32 `block_table`
`[B, max_pages]`, and int32 `kv_seqlens` `[B]`. Physical page IDs in the
table may be shuffled. Packed Q/O use `[sum_q, H, D]` and int32
`cu_seqlens_q` `[B+1]`; pass `max_q_len` and `max_kv_len` for the launch
grid and page table capacity. The 16-warp tactic uses the 12-warp epilogue
for packed Q because its TMA output store cannot mask a batch-local tail.

The benchmark can compare this kernel with the official FlashAttention-4 CuTe
DSL implementation. On CUDA 13, install that baseline with:

```bash
/home/bifrost/miniforge3/bin/python3 -m pip install "flash-attn-4[cu13]"
```

Run the default performance suite (32 heads, head dimension 128). It covers
causal prefill at 1K, 2K, 4K, 8K, and 16K, four-sequence packed prefill with
reproducible random lengths at 1K–16K maxima, plus non-causal Q=1/KV=8K
decode workloads at batch sizes 32, 64, and 128:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mha.py --baseline fa4
```

Add `--sol` to print compute and memory roofline estimates for the actual
lengths and paged tile reads.

It also exposes the official TRTLLM-Gen FMHA cubins through FlashInfer's
standalone binding:

```bash
/home/bifrost/miniforge3/bin/python3 -m pip install "flashinfer-python[cu13]>=0.6.18"
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mha.py \
  --baseline trtllm-gen
```

The current TRTLLM-Gen cubin distribution supports SM100, SM103, and SM107.
It does not contain an SM110 binary for NVIDIA Thor; the benchmark rejects
that combination instead of reporting fallback or invalid timing results.

Use a quick explicit paged shape with an FP32 correctness check while iterating:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mha.py \
  --single --batch 1 --heads 32 --q-len 128 --kv-len 128 --head-dim 128 --causal
```

Test shuffled 16-token pages and variable-length packed Q in the same suite:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mha.py \
  --single --batch 2 --heads 2 --q-len 137 --kv-len 145 --causal \
  --batched-q --q-lengths 19,137 --kv-lengths 19,137
```

The FA4 baseline consumes the same shuffled page table and sequence lengths
through its paged-KV interface, with KV pages rearranged to FA4's
`[num_pages, 16, H, D]` layout before timing. Packed prefill uses one FA4
call with `cu_seqlens_q`. The suite compares both on the same logical Q/K/V;
`--single` also checks both against FP32 attention. Causal cases with
different Q and KV lengths are rejected because the baseline uses a different
mask alignment.

The extension is JIT-compiled on the first call and cached by PyTorch. Set
`CUDA_ARCH` and `COMPUTE_CUDA_ARCH` to override the target architecture.

## Contents

- `src/bf16_fa4_mha_kernel.cu`: SM110a BF16 MHA kernel
- `src/bf16_fa4_mha_torch_binding.cpp`: PyTorch validation and binding
- `../../sm110a_kernel/kernels/bf16_fa4_mha.py`: Python API and extension loader
- `../../benchmarks/benchmark_bf16_fa4_mha.py`: correctness and timing harness
