# BF16 FA4 MHA

This directory contains the SM110a BF16 FlashAttention-4-style MHA kernel.

`fa_mha` uses the contiguous tensor layout `[B, H, S, D]` for Q, K,
V, and output.  It computes `softmax(scale * Q @ K.T) @ V` in FP32 and stores
BF16 output.  With `causal=True`, query position `i` attends to keys `0..i`.
It supports independent Q and KV sequence lengths, but the causal convention
is the simple upper-left mask (`key_index <= query_index`).

The benchmark can compare this kernel with the official FlashAttention-4 CuTe
DSL implementation. On CUDA 13, install that baseline with:

```bash
/home/bifrost/miniforge3/bin/python3 -m pip install "flash-attn-4[cu13]"
```

Run the default performance suite (32 heads, head dimension 128). It covers
causal prefill at 1K, 2K, 4K, and 8K sequence lengths, plus non-causal
decode-equivalent Q=1/KV=8K workloads at batch sizes 32, 64, and 128:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mha.py --baseline fa4
```

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

Use a quick explicit shape with an FP32 correctness check while iterating:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mha.py \
  --single --batch 1 --heads 32 --q-len 128 --kv-len 128 --head-dim 128 --causal
```

The extension is JIT-compiled on the first call and cached by PyTorch. Set
`CUDA_ARCH` and `COMPUTE_CUDA_ARCH` to override the target architecture.

## Contents

- `src/bf16_fa4_mha_kernel.cu`: SM110a BF16 MHA kernel
- `src/bf16_fa4_mha_torch_binding.cpp`: PyTorch validation and binding
- `../../sm110a_kernel/kernels/bf16_fa4_mha.py`: Python API and extension loader
- `../../benchmarks/benchmark_bf16_fa4_mha.py`: correctness and timing harness
