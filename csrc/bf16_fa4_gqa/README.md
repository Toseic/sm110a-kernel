# BF16 FA4 GQA

The GQA kernel uses contiguous BF16 tensors: Q is `[B, Hq, Q, D]`, while K
and V are `[B, Hkv, KV, D]`. `Hq` must be divisible by `Hkv`; query heads in
each consecutive group map to one K/V head. The only supported head dimension
is 128.

This initial version deliberately keeps the MHA CTA schedule. Every query-head
CTA independently TMA-loads its mapped K/V head, so GQA-group K/V traffic is
repeated. This is intentional: it is the correctness baseline for later
cross-query-head K/V sharing work.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_gqa.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_gqa.py \
  --single --batch 1 --q-heads 32 --kv-heads 8 --q-len 128 --kv-len 128 --causal
```

Contents:

- `src/bf16_fa4_gqa_kernel.cu`: standalone GQA CUDA kernel
- `src/bf16_fa4_gqa_torch_binding.cpp`: PyTorch validation and binding
- `../../sm110a_kernel/kernels/bf16_fa4_gqa.py`: Python API/JIT recipe
- `../../benchmarks/benchmark_bf16_fa4_gqa.py`: GQA timing and correctness suite
