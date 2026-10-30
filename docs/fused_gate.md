# Fused sqrtsoftplus MoE gate

`sm110a_kernel.fused_gate(logits, topk, route_scale=1.0, bias=None,
hash_indices=None, pdl=False)` returns `(weights, expert_ids)` from contiguous
CUDA FP32 `[tokens, experts]` logits. Optional arguments after `route_scale`
are keyword-only. Supported sizes: 1–1024 experts, 1–min(experts, 32) selections.

The hash path assigns one warp per token and four warps per CTA. It reads only
selected logits and preserves the supplied IDs, order and int32/int64 dtype.
The hash-table lookup remains outside the kernel.

The top-k path assigns one CTA per token. Threads cooperatively compute
activations and stage scores in shared memory; warp 0 retains selection keys
in registers and selects experts. Selected scores use shared-memory broadcast.
Ordered integer keys preserve float score order, rank NaNs
above +Inf and treat signed zeros equally. Each selection uses inline PTX
`redux.sync.max.u32` followed by `redux.sync.min.s32` for ascending-ID ties,
replacing branch-heavy float/shuffle comparisons. Removed keys become zero,
below the key for -Inf. IDs are int64, in descending selection-score order.
There are no global intermediate score tensors.

CTA size is fixed by shape: at most 32 rows use 32/64/128/256 threads for up to
32/64/128/1024 experts respectively; larger row counts cap this at 128 threads.
At E=256 this reduces the serial activation chain for small decode batches,
while retaining more resident CTAs for larger batches. This is a fixed heuristic,
with no autotuning. Softplus/sqrt arithmetic and FP32 division are unchanged.

Both paths compute `a = sqrt(softplus(logit))`, with beta 1 and threshold 20,
then `(a_selected / sum(a_selected)) * route_scale`. Bias affects top-k selection
only; it never affects weights, and hash mode ignores it. Equal selection
scores use ascending expert ID; PyTorch topk does not guarantee this tie order.
FP32 reduction/math can differ in the last bits from eager PyTorch. Hash IDs
must be in range; invalid IDs generate NaN weights without reading out of bounds.

`compile_fused_gate(experts, topk, pdl=False)` returns a cached JIT module with
`fused_gate` and allocation-free `fused_gate_out(logits, weights, indices, ...)`
methods. Outputs must not overlap each other or any input. The PDL variant uses
`pdl_entry()` and the common `launch_kernel()` helper; PDL is part of the JIT key.

`Sm110aGate` uses this kernel for sqrtsoftplus-compatible scoring and supported
sizes in prefill and decode. Softmax, sigmoid and larger configurations retain
the existing reference path. Router GEMM and downstream MoE metadata are unchanged.

Cold-L2 measurements capture a single gate invocation in a CUDA Graph for each
implementation, including all PyTorch scoring/selection/normalization kernels:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_fused_gate.py --tokens 1 16 128
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_fused_gate.py --tokens 1 16 128 --pdl
```

Pass `--sol` to print the compulsory-byte memory lower bound using the shared
273 GB/s reference bandwidth. This excludes compute and launch latency; it is
not a complete SOL model for small gate workloads.
