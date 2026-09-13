# BF16 FA4 GQA

The GQA kernel uses contiguous BF16 tensors: Q is `[B, Hq, Q, D]`, while K
and V are `[B, Hkv, KV, D]`. It is specialized for `Hq == 4 * Hkv`; query
heads in each consecutive group map to one K/V head. The only supported head
dimension is 128.

The kernel has two orthogonal schedule axes. `tactic` selects the warpgroup
organization (`16warp` with dedicated softmax/rescale/TMA-store warpgroups,
or `12warp` with one fused per-Q-slot warpgroup). `pack_gqa` selects the CTA
decomposition: unpacked assigns one query head per CTA (K/V tiles are
re-read per head, with L2 serving the GQA-group reuse), while packed folds
the four query heads of a K/V head into the logical query-row dimension of
a single CTA and scatters output rows back to the original BHSD layout.
Packed quarters the task count, which mainly benefits decode.

The benchmark autotunes the 2x2 schedule space per workload by default
(`--tactic auto`) through the shared `AutotuneRunner`, caching the winner in
`profiles/autotune_cache.json`; pass an explicit `--tactic [--pack-gqa]` to
pin one schedule and skip the sweep.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_gqa.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_gqa.py \
  --single --batch 1 --q-heads 32 --kv-heads 8 --q-len 128 --kv-len 128 --causal
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_gqa.py \
  --single --batch 4 --q-heads 32 --kv-heads 8 --q-len 1 --kv-len 1024 \
  --tactic 12warp --pack-gqa
```

Contents:

- `src/bf16_fa4_gqa_kernel.cu`: standalone GQA CUDA kernel
- `src/bf16_fa4_gqa_torch_binding.cpp`: PyTorch validation and binding
- `../../sm110a_kernel/kernels/bf16_fa4_gqa.py`: Python API/JIT recipe
- `../../benchmarks/benchmark_bf16_fa4_gqa.py`: GQA timing and correctness suite
