# BF16 FA4 Scaffold

This directory is the development home for a future FlashAttention-4 kernel.
It currently contains a correctness-first, intentionally slow CUDA baseline.

`fa_mha` uses the contiguous tensor layout `[B, H, S, D]` for Q, K,
V, and output.  It computes `softmax(scale * Q @ K.T) @ V` in FP32 and stores
BF16 output.  With `causal=True`, query position `i` attends to keys `0..i`.
It supports independent Q and KV sequence lengths, but the causal convention
is the simple upper-left mask (`key_index <= query_index`).

Build and run the default performance suite (32 heads, head dimension 128).
It covers causal prefill at 1K, 2K, 4K, and 8K sequence lengths, plus
non-causal decode-equivalent Q=1/KV=8K workloads at batch sizes 32, 64, and
128:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mha.py
```

Use a quick explicit shape with an FP32 correctness check while iterating:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mha.py \
  --single --batch 1 --heads 32 --q-len 128 --kv-len 128 --head-dim 128 --causal
```

The extension is JIT-compiled on the first call and cached by PyTorch. Set
`CUDA_ARCH` and `COMPUTE_CUDA_ARCH` to override the target architecture.

## Contents

- `src/bf16_fa4_mha_kernel.cu`: one-thread-per-output-element reference CUDA MHA kernel
- `src/bf16_fa4_mha_torch_binding.cpp`: PyTorch validation and binding
- `../../sm110a_kernel/kernels/bf16_fa4_mha.py`: Python API and extension loader
- `../../benchmarks/benchmark_bf16_fa4_mha.py`: correctness and timing harness
