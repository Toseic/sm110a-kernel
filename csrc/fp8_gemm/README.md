# SM110A FP8 x FP8 GEMM

This independent kernel computes BF16 `C[M,N] = A[M,K] @ B[K,N]`, with both
operands as E4M3 and one UE8M0 scale per 32 K values on each side. It uses
`tcgen05.mma.kind::mxf8f6f4.block_scale.scale_vec::1X` with the instruction
descriptor's atype and btype both set to E4M3.

`A` is contiguous `torch.float8_e4m3fn [M,K]` and `B` is the transposed
mathematical weight, contiguous `torch.float8_e4m3fn [N,K]`. `SFA` and `SFB`
are contiguous `torch.float8_e8m0fnu`, packed from logical `[rows,K/32]` with
`to_blocked_fp8_scales`. Both scales are user inputs; there is no identity
scale caching.

```python
from sm110a_kernel import quantize_fp8_gemm, fp8_gemm
a8, b8, sfa, sfb = quantize_fp8_gemm(a, b)
out = fp8_gemm(a8, b8, sfa, sfb)
```

Benchmark with:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_fp8_gemm.py \
  --m 4096 --n 4096 --k 4096
```

M must be a multiple of 128; N must be a multiple of 128 (and of the selected
tile_n); K must be a multiple of 64 for the current pipeline tiles. Run
`benchmarks/benchmark_fp8_gemm.py` for correctness, timing, and cached
autotuning.
