# MXFP4 MoE precision modes

FC1 and FC2 have a compile-time `routing_weight_position` field (`"fc1"`
or `"fc2"`). Use the same position for both projections to apply the expert
weight exactly once. The default `"fc2"` preserves the original behavior.

The DeepSeek-V4 activation path applies FP32 routing weights to SwiGLU before
BF16 rounding and FP8 quantization. Moving this multiply past W2 changes the
quantized operands, even though it would commute with W2 in exact arithmetic.
Select the following configurations to follow the model's activation nodes:

```python
fc1 = compile_mxfp4_fc1(Mxfp4Fc1Config(
    routing_weight_position="fc1", activation_bf16_rounding=True,
))
fc2 = compile_mxfp4_fc2(Mxfp4Fc2Config(
    routing_weight_position="fc1", deterministic=True,
))
h, h_scale = fc1.mxfp4_fc1(
    a, b1, a_scale, b1_scale, metadata1,
    swiglu_limit=7.0, routing_weights=weights.float().contiguous(),
)
out = fc2.mxfp4_fc2(
    h, b2, h_scale, b2_scale, metadata1.routing,
    weights.bfloat16().contiguous(), metadata2, topk,
)
```

FC1 weights must be contiguous FP32 `[tokens, topk]` in the rank order encoded
in its routing metadata. FC1 first rounds gate/up to BF16, evaluates SwiGLU
and the router multiply in FP32, rounds the weighted hidden to BF16, then
quantizes to FP8. The E8M0 scale uses `ceil(log2(amax / 448))`, with the
official `amax >= 1e-4` floor in this mode. The FC2 routing-weight argument
retains its BF16 shape contract but is not multiplied in the `"fc1"` mode.

Deterministic FC2 writes each expert assignment once into BF16 scratch and
sums the ranks in FP32 before one final BF16 conversion. Its additional
workspace is `tokens * topk * output_dim * 2` bytes; it adds one finalize
kernel. The default BF16 atomic reduction can change rounding across launches.
The deterministic rank sum can still differ slightly from the official
expert-index sum because their FP32 summation orders differ.

## Validation and cold-L2 measurements

The one-off combined MoE precision benchmark has been removed. Use the
existing FC1/FC2 benchmarks for ongoing kernel validation:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_mxfp4_fc1.py \
  --experts 8 --topk 2 --n 128 --k 128 --token 128 --cluster-m 1 \
  --tma-epilogue --routing-weight-position fc1 --activation-bf16-rounding
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_mxfp4_fc2.py \
  --experts 8 --topk 2 --n 256 --k 128 --token 128 --cluster-m 1 \
  --routing-weight-position fc1 --deterministic
```

The historical combined measurements below used NVIDIA Thor,
GPC 1575 MHz / EMC 4266 MHz, CUDA Graph, cold L2 with a 64 MiB
dirty-write eviction buffer. Inputs and tile configurations remain fixed;
metadata preparation is outside timing. One-token inputs pad to 128 rows.
Times below are milliseconds, comparing default FC2 weighting/BF16 atomics
against FC1 weighting/BF16 activation nodes/deterministic FP32 summation:

| Tokens | FC1 default | FC1 precise | FC2 default | FC2 precise |
| --- | ---: | ---: | ---: | ---: |
| 1 (pad 128) | 0.5903 | 0.5867 | 0.3359 | 0.3905 |
| 128 | 10.8026 | 10.7498 | 5.1254 | 5.2148 |
| 1024 | 11.5604 | 11.4228 | 5.8480 | 5.9910 |

The precise path's FC1 relative L2 errors against identical quantized operands
were `0`, `0`, and `3.336e-5`; FC2 errors were `0`, `5.345e-6`, and `6.730e-6`.
Five eager repeats and graph replay were bitwise equal for deterministic FC2
at every shape. Default FC2's one-token result had relative L2 error `0.00401`
and changed across eager and graph executions.

Small cluster-M 1/2 tests included zero, tiny (`1e-8`), negative, and positive
routing weights. Both projections matched the reference exactly in these
cases. Compute Sanitizer memcheck reported zero errors for those historical
combined tests.

## Regression against the original kernels

The subsequent cold-L2 regression compared the original revision `80c89bc`
with the modified kernels using the existing benchmarks, fixed cluster-M=1,
power-law routing (alpha 0.8), E=256, top-6, dim=4096, intermediate dim=2048,
5 warmups, 20 iterations and 5 samples. These timings use the benchmark's
direct launch measurements, not the historical graph measurements above.

| Kernel / tokens | Original default ms | Current default ms | FC1-weighted precise mode ms |
| --- | ---: | ---: | ---: |
| FC1 / 128 | 2.6325 | 2.5736 | 2.5793 |
| FC1 / 1024 | 3.9972 | 3.9898 | 4.1358 |
| FC2 / 128, epilogue 32 | 1.3321 | 1.3196 | 1.3154 |
| FC2 / 1024, epilogue 32 | 2.2175 | 2.2102 | 2.6487 |

Default FC1/FC2 showed no material slowdown. The optional precise FC1 cost
about 3.7% at 1024 tokens; deterministic FC2 cost about 19.8% in this routing
distribution. Reduction overhead depends on workload and must not be inferred
from the uniform-routing measurements alone.

Across 48 small edge cases, FC1 matched its quantized reference exactly.
Deterministic FC2's maximum relative L2 was 1.333e-5, with bitwise-equal
repeats and CUDA Graph output. Cases included cluster-M 1/2, FC1 TMA/direct
stores, FC2 epilogues 32/64, and zero, tiny and negative weights. The default
FC1 output and logical scales were byte-identical to the original revision
at 128/384 tokens for both cluster sizes. Compute Sanitizer reported zero
errors for the tested FC1/FC2 deterministic cluster-2 path.

Eight atomic stress cases exceeded the elementwise tolerance near
cancellation, despite relative L2 below 0.004. Their failing values were
within the range of valid BF16 summation permutations. The same limitation
was reproduced in the original kernel on identical operands (using unit
weights to emulate the unweighted specialization). It is not a new regression
and is not covered by a blanket claim of strict atomic reference equality.
