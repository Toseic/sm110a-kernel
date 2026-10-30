# Weightless QK normalization

`qk_norm` normalizes each last-axis row independently, without learned weights:

```python
y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
```

For model Q shaped `[B, S, H, D]`, each token/head is a separate row. This is
the normalization after `wq_b` and before RoPE, distinct from the weighted
low-rank `q_norm` before `wq_b`. The SM110A runtime uses it in both regular
and MTP attention, with counters `sm110a.qk_norm` and
`torch.qk_norm_fallback` for unsupported inputs.

The implementation shares the vectorized load, register retention and row
reduction code in `../rmsnorm/src`, compiled with `SM110A_QK_NORM=1` into a
separate `qk_norm_kernel` variant. No weight tensor is allocated or read.
Unlike weighted RMSNorm, it preserves input-dtype rounding after square,
mean, epsilon addition and reciprocal square root, matching the model's
unfused Torch expression. The reduction accumulates rounded squares in FP32;
different reduction orders can still cause rounding differences. This API
does not promise bitwise equivalence to Torch for all inputs.

Contiguous CUDA BF16/FP16/FP32 inputs with `1 <= D <= 32768` are supported.
Other dimensions are flattened into rows. Vectorizable rows require 16-byte
aligned input/output pointers. Exact in-place execution is supported; shifted
overlap is rejected. Fixed configuration defaults and register-budget limits
are shared with RMSNorm. Compile/warm up before CUDA Graph capture.

```python
from sm110a_kernel import QkNormRunner, qk_norm, qk_norm_out

y = qk_norm(q, eps=1e-6)
qk_norm_out(q, q, eps=1e-6)  # in place
QkNormRunner(q, q, eps=1e-6, pdl=True).run()
```

The benchmark checks Torch agreement, in-place execution, eager/graph equality
and graph replay after input changes. Timings use cold L2; reported bandwidth
counts one input read and one output write, excluding eviction traffic.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_qk_norm.py --tokens 1 128 1024
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_qk_norm.py --tokens 1 128 --pdl
```
