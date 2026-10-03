# SM110A MXFP4 GEMM

The kernel computes a BF16 output from packed E2M1 operands and UE8M0
block-32 scales using `tcgen05.mma.cta_group::2.kind::mxf4`
with `.scale_vec::2X`.

Python usage:

```python
from sm110a_kernel import quantize_mxfp4_gemm, mxfp4_gemm

aq, bq, sfa, sfb = quantize_mxfp4_gemm(a, b)
out = mxfp4_gemm(aq, bq, sfa, sfb)
```

The low-level API accepts:

- `A`: contiguous `torch.float4_e2m1fn_x2`, physical shape `[M, K/2]`.
- `B`: contiguous `torch.float4_e2m1fn_x2`, physical shape `[N, K/2]`.
- `SFA`, `SFB`: contiguous `torch.float8_e8m0fnu` arrays in Blackwell's
  `SWIZZLE_32_4_4` blocked format.  Each K=64 slab stores two live scale
  bytes per row and two layout-padding bytes.
- Output: contiguous BF16 `[M,N]`.

`M` must be a multiple of `tile_m * cluster_m` (the default is 256), `N` a
multiple of 128, and `K` a multiple of `tile_k * split_k`. `tile_n` and
`tile_k` support 64, 128, and 256. `tile_n=64/128` use two
TMEM accumulator stages and the remaining TMEM capacity to
give every mainloop SMEM stage and K=64 MMA slab a distinct scale-factor slot.
This lets each `tile_k` stage prefetch all of its scales without aliasing an
in-flight stage. `tile_n=256` uses one accumulator stage so its accumulator
and scale factors occupy disjoint TMEM columns.
`to_blocked_mxfp4_scales` converts logical `[rows,K/32]` E8M0 scales to the
required physical format. This is the same scale format consumed by
cuBLASLt's 1x32 E8M0 block-scaling interfaces.

Run the benchmark and comparison against its PyTorch dequantize+BF16 reference:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_mxfp4_gemm.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_mxfp4_gemm.py \
  --m 4096 --n 4096 --k 4096 --warmup 5 --iters 20 --sol
```

PyTorch does not currently expose an E2M1+E8M0 MXFP4 `_scaled_mm` path, so
the benchmark uses an FP32 dequantize plus BF16 GEMM reference for correctness
and its baseline timing.

The benchmark autotunes `tile_n` and `tile_k` over 64, 128, and 256, and
`epilogue_block_n` over the compatible values 32, 64, and 128 through the
shared runner. Selections are cached by workload and CUDA environment. Force
a new sweep or prewarm the cache with:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_mxfp4_gemm.py \
  --m 4096 --n 4096 --k 4096 --force-sweep --debug
/home/bifrost/miniforge3/bin/python3 -m sm110a_kernel.runner mxfp4-gemm -- \
  --m 4096 --n 4096 --k 4096
```


## Split-K and small-M

`Mxfp4GemmConfig(split_k=...)` supports 1, 2, 4, and 8, provided
`cluster_m * split_k <= 8`.  Each cluster-N rank accumulates a K partition;
the kernel reduce-scatters FP32 partial tiles through DSMEM before converting
to BF16 and storing.  Split-K launches one cluster per output tile, while
`split_k=1` retains the persistent path.  `cluster_m=1` supports M=128
workloads (the small-M tile for MXFP4); mxf4 itself has an M=128 minimum,
so MXFP4 does not expose BF16's M=64 tile variant.
