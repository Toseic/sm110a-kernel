# BF16 FA4 MQA

The MQA kernel uses contiguous BF16 tensors: Q is `[B, Hq, Q, D]`, while K
and V are `[B, 1, KV, D]`. The query-head count must be a multiple of four and
the supported head dimension is 128.

There is one 16-warp CTA schedule with a fixed pack size of 32. Each CTA folds
32 adjacent query heads into the logical query-row dimension while all groups
in a batch reuse its single K/V head.

The benchmark uses the same shapes as the GQA benchmark except that `Hkv=1`:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mqa.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mqa.py \
  --single --batch 1 --q-heads 32 --kv-heads 1 --q-len 128 --kv-len 128 \
  --causal
```

The pack size is compiled directly into the CUDA source and is not a runtime or
JIT configuration option.

Contents:

- `src/bf16_fa4_mqa_kernel.cu`: standalone 16-warp MQA CUDA kernel
- `src/bf16_fa4_mqa_torch_binding.cpp`: PyTorch validation and binding
- `../../sm110a_kernel/kernels/bf16_fa4_mqa.py`: Python API/JIT recipe
- `../../benchmarks/benchmark_bf16_fa4_mqa.py`: correctness and timing harness
