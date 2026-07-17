# SM110A NVFP4 GEMM

The kernel computes a BF16 output from packed E2M1 operands and UE4M3
block-16 scales using `tcgen05.mma.cta_group::2.kind::mxf4nvf4`.

Python usage:

```python
from sm110a_kernel import quantize_nvfp4_gemm, nvfp4_gemm

aq, bq, sfa, sfb = quantize_nvfp4_gemm(a, b)
out = nvfp4_gemm(aq, bq, sfa, sfb)
```

The low-level API accepts:

- `A`: contiguous `torch.float4_e2m1fn_x2`, physical shape `[M, K/2]`.
- `B`: contiguous `torch.float4_e2m1fn_x2`, physical shape `[N, K/2]`.
- `SFA`, `SFB`: contiguous `torch.float8_e4m3fn` arrays in Blackwell's
  `SWIZZLE_32_4_4` blocked format.
- Output: contiguous BF16 `[M,N]`.

`M` must be a multiple of 256, `N` a multiple of 128, and `K` a multiple of
64.  `tile_n` and `tile_k` support 64, 128, and 256. All variants use two
TMEM accumulator stages; `tile_n=256` overlaps the stages by 16 columns and
drains that region first in the epilogue so the next mainloop can start early.
`to_blocked_nvfp4_scales` converts logical `[rows,K/16]` scales to the
required physical format. This is the same scale format consumed by
`torch._scaled_mm`/cuBLASLt.

Run the benchmark and bit-exact comparison against PyTorch NVFP4:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_nvfp4_gemm.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_nvfp4_gemm.py \
  --m 4096 --n 4096 --k 4096 --warmup 5 --iters 20 --sol
```

The benchmark autotunes `tile_n` and `tile_k` over 64, 128, and 256, and
`epilogue_block_n` over the compatible values 32, 64, and 128 through the
shared runner. Selections are cached by workload and CUDA environment. Force
a new sweep or prewarm the cache with:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_nvfp4_gemm.py \
  --m 4096 --n 4096 --k 4096 --force-sweep --debug
/home/bifrost/miniforge3/bin/python3 -m sm110a_kernel.runner nvfp4-gemm -- \
  --m 4096 --n 4096 --k 4096
```
