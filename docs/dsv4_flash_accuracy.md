# DSV4 SM110A precision re-evaluation

Measured on NVIDIA Thor / SM110A, 2026-10-15. Configuration: batch 1,
1024-token prefill, 128-token decode, one CSA and one HCA layer, dim 4096,
256 experts, top-6, intermediate dim 2048. These are deterministic random
benchmark weights (seed 1), not a real checkpoint quality evaluation.

## Results after the FC1/FC2 change

The model defaults to FP32 expert weighting inside FC1, before hidden BF16
rounding and FP8 quantization. FC1 also rounds gate/up to BF16 before SwiGLU.
FC2 defaults to fixed-rank FP32 accumulation of BF16 expert outputs.
The standalone kernel APIs retain their legacy defaults for compatibility.
See [kernel modes](mxfp4_moe_precision.md) for API and kernel-level validation.

On one frozen first-layer decode MoE input, the routed output versus the
official expert computation improved from relative L2 approximately **3.48%**
to **0** (bitwise equal in this case). This is a local result, not a claim of
bitwise equality for all routed MoE inputs or the full model.

The new full model produced identical logits in five fixed-input eager
repeats. All 128 autoregressive tokens and every step's logits were bitwise
equal between eager and each of three CUDA Graph replays, restoring the same
cache state before each run.

With FC1 weighting retained but FC2 changed back to BF16 atomics, fixed-input
logits varied by up to 0.0788. In that run, graph/eager autoregressive tokens
first diverged at zero-based step 10; only 10/128 tokens matched. This is not a
Graph-specific error: eager repetitions are also nondeterministic. Later
logits after token divergence no longer compare identical model inputs.

Deterministic reduction is useful for reproducibility and diagnosing graph
correctness. It is **not required to move the routing multiply**, nor does
bitwise self-consistency establish agreement with the official model or
better checkpoint quality.

## Remaining differences from the official model

The full native path is **not yet precision-validated against the official
model**. Same prompt and weights give prefill logits relative L2 **6.6401%**,
cosine **0.9977957**. Across 128 teacher-forced decode inputs, minimum logits
cosine is **0.9974265**, and argmax agrees on **106/128** steps. Teacher forcing
keeps input tokens identical and avoids feedback from different predictions.
Long-vector diagnostic metrics use CPU FP64 reductions.

The first significant difference appears before routed MoE:

| Prefill output | Relative L2 vs official |
| --- | ---: |
| Embedding / first attention norm input | 0 |
| First attention | 1.0083% |
| First FFN norm input | 0.5191% |
| First FFN | 3.8185% |
| Second block | 6.5677% |
| Final logits | 6.6401% |

Same-operand probes found dense projection errors around 0–2.3e-5 relative
L2 (BF16 indexer weights projection approximately 1.55e-4). Native sparse
attention versus official sparse attention on identical Q/KV/index operands
was approximately 0.15–0.24% relative L2. Replacing only sparse attention with
official attention did not improve overall error (a 32-step control had
6.76% prefill relative L2). These results do not identify one faulty kernel;
small differences cross BF16/FP8 quantization thresholds and propagate.

Two further numerical semantics differences are confirmed:

1. Native indexer top-k is exact but **unordered**; official top-k is sorted
   by score. Even an identical selected set changes the low-precision
   attention reduction order. With all other backends replaced by official
   operators, native unordered top-k yielded 6.5916% prefill logits relative
   L2. Sorting only those native-selected indices by score reduced it to
   1.3857%, exactly matching the official-indexer control; first attention
   then became bitwise equal. However, adding the same sort to the **full
   native** backend yielded 6.6281%, essentially unchanged. Sorting alone is
   therefore not a complete fix. These interventions are diagnostic only;
   the production native top-k path remains unchanged.
2. FC2 currently converts the routed FP32 sum to BF16 **before** adding the
   shared expert. Official MoE adds the shared BF16 output into the FP32 sum
   and rounds only afterward. The frozen decode input's complete MoE output
   has 0.0732% relative L2 against an otherwise identical reference, despite
   the separately rounded routed outputs being identical. FC2's fixed rank
   order also differs from official expert-id summation order. Removing
   these differences requires a separate epilogue/API change; they have not
   been silently treated as fixed by deterministic reduction.

## End-to-end cost

The following commands used cold-L2 full-model passes, always-on decode
CUDA Graph, 1 warmup and 3 measured iterations:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py --warmup 1 --iters 3
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py --fc2-reduction atomic --warmup 1 --iters 3
```

| FC1 weighting, FC2 reduction | Prefill ms | Decode ms / 128 tokens | Decode tok/s | Peak GiB |
| --- | ---: | ---: | ---: | ---: |
| FC1, deterministic | 115.15 | 1800.04 | 71.11 | 17.97 |
| FC1, atomic | 114.13 | 1790.16 | 71.50 | 17.97 |

The measured end-to-end difference is approximately 0.55%; this limited run
does not establish statistical significance. Isolated cold-L2 FC2 decode
increased from 0.3359 to 0.3905 ms: assignment scratch writes/reads and an
extra finalize launch cost more when the GEMM itself is small.

## Diagnostic artifacts and next checks

The one-off diagnostic scripts have been removed after the investigation.
Retained local, ignored results under `reports/` include:

- `dsv4_flash_accuracy_fc1_deterministic.json`, `...fc1_atomic.json`,
  `...fc2_atomic.json`: full 128-step comparisons.
- `dsv4_flash_operator_accuracy.json`: same-operand backend probes.
- `...official_backend_sorted_topk.json`, `...official_backend_official_indexer.json`,
  `...sorted_topk.json`: controlled interventions described above.

Next accuracy work should preserve FP32 routed sums through shared-expert
addition, investigate quantization-sensitive attention intermediates with
fixed operands, and eventually evaluate real-checkpoint layerwise and
teacher-forced outputs. Do not lower the graph self-consistency check or
claim official precision acceptance from a single cosine threshold.
