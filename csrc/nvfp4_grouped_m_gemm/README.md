# SM110A NVFP4 grouped-M GEMM

This kernel computes BF16 grouped output from packed E2M1 activations and
expert weights with blocked UE4M3 scales. Expert segments in grouped A must
start on 128-row boundaries.

Run the default tokens-per-expert and kernel-config sweep:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_nvfp4_grouped_m_gemm.py
```

Run a smaller workload, force every compatible config to be measured, and
print the per-config runner table:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_nvfp4_grouped_m_gemm.py \
  --experts 8 --n 4096 --k 4096 --tokens 256 --force-sweep --debug --sol
```

The sweep covers compatible combinations of tile-N 64/128, tile-K
64/128/256, epilogue tile-N 32/64/128, cluster-M 1/2, and non-persistent,
persistent, or CLC scheduling. Results are cached by CUDA environment and
`(experts, N, K, tokens_per_expert)`.

`--sol` reports NVFP4 MMA, compulsory HBM, repeated tile-read GMEM/L2, and
mixed HBM/L2 bounds for the runner-selected configuration. The traffic model
accounts for packed E2M1 operands, UE4M3 block scales, BF16 output, and the
current grouped kernel's per-CTA SFB scale loads.

The same cache can be populated through the shared runner:

```bash
/home/bifrost/miniforge3/bin/python3 -m sm110a_kernel.runner \
  nvfp4-grouped-m-gemm -- --experts 8 --n 4096 --k 4096 --tokens 256
```
