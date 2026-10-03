# SM110A FP8 x MXFP4 GEMM

This independent kernel computes BF16 `C[M,N] = A[M,K] @ B[K,N]`, with an
unscaled E4M3 activation and an E2M1 MXFP4 weight with one UE8M0 scale per
32 K values. It uses `tcgen05.mma.kind::mxf8f6f4.block_scale.scale_vec::1X`.

`A` is contiguous `torch.float8_e4m3fn [M,K]`. By default `B` is contiguous
native packed E2M1 `torch.uint8 [N,K/2]` (the transposed mathematical weight).
`16U4_ALIGN16B` TMA expands each compact 16-value group to the eight data bytes
plus eight padding bytes required by `mxf8f6f4`. `SFB` is contiguous
`torch.float8_e8m0fnu`, packed from logical `[N,K/32]` with
`to_blocked_fp8mxfp4_scales`. The public facade caches an internal E8M0=1
A-side scale tensor, so the user API has no activation-scale argument.

The packed path requires `tile_k=128` (and hence K divisible by 128). K=64
automatically falls back to padded `torch.uint8 [N,K]` storage. Pass
`compact_b_gmem=False` or `--padded-b-gmem` to force the padded path.

```python
from sm110a_kernel import quantize_fp8mxfp4_gemm, fp8mxfp4_gemm
a8, b4, sfb = quantize_fp8mxfp4_gemm(a, b)
out = fp8mxfp4_gemm(a8, b4, sfb)
```

Benchmark the compact variant with:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_fp8mxfp4_gemm.py \
  --m 4096 --n 4096 --k 4096 --compact-b-gmem
```

M and N must be multiples of 128; K must be a multiple of 64 for the current
pipeline tiles. Run `benchmarks/benchmark_fp8mxfp4_gemm.py` for correctness,
timing, and cached autotuning.
