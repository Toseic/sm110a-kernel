# Kernel runners

The public runners live alongside their kernel APIs in `sm110a_kernel/kernels/`.
`BaseKernelRunner` in `sm110a_kernel/runner.py` supplies selection, cold-L2 timing,
persistent caching, and process-local configuration reuse. Kernel subclasses
supply compatible candidates, cache keys, compilation and argument binding.

All native kernel families have public runners:

| Module | Runner |
| --- | --- |
| `bf16_fa4_gqa.py` | `Bf16Fa4GqaRunner` |
| `bf16_fa4_mha.py` | `Bf16Fa4MhaRunner` |
| `bf16_fa4_mla.py` | `Bf16Fa4MlaRunner` |
| `bf16_fa4_mqa.py` | `Bf16Fa4MqaRunner` |
| `bf16_fa4_sparse_mla.py` | `Bf16Fa4SparseMlaRunner`, `Bf16Fa4SparseMlaPrefillRunner` |
| `bf16_gemm.py` | `Bf16GemmRunner` |
| `bf16_gemm_serving.py` | `Bf16GemmServingRunner` |
| `bf16tofp4_quant.py` | `Bf16ToFp4QuantRunner` |
| `bf16tofp8_quant.py` | `Bf16ToFp8QuantRunner` |
| `bf16tonvfp4_quant.py` | `Bf16ToNvfp4QuantRunner` |
| `fc1_block_interleaved_swiglu.py` | `Fc1BlockInterleavedSwiGLURunner` |
| `fc2_finalize.py` | `Fc2FinalizeRunner` |
| `fp8_act_quant.py` | `Fp8ActQuantRunner` |
| `fp8_gemm.py` | `Fp8GemmRunner` |
| `fp8mxfp4_gemm.py` | `Fp8Mxfp4GemmRunner` |
| `grouped_m_gemm.py` | `Bf16GroupedMGemmRunner` |
| `mxfp4_fc1.py` | `Mxfp4Fc1Runner` |
| `mxfp4_fc1fc2.py` | `Mxfp4Fc1Fc2Runner`, `Mxfp4MoeRunner`, `Mxfp4StandaloneRunner` |
| `mxfp4_fc2.py` | `Mxfp4Fc2Runner` |
| `mxfp4_gemm.py` | `Mxfp4GemmRunner` |
| `mxfp4_indexer_score.py` | `IndexerScorePrefillRunner`, `IndexerScoreDecodeRunner` |
| `nccl_lsa.py` | `NcclLsaRunner` |
| `nvfp4_fc1.py` | `Nvfp4Fc1Runner` |
| `nvfp4_fc1fc2.py` | `Nvfp4Fc1Fc2Runner` |
| `nvfp4_fc2.py` | `Nvfp4Fc2Runner` |
| `nvfp4_gemm.py` | `Nvfp4GemmRunner` |
| `nvfp4_grouped_m_gemm.py` | `Nvfp4GroupedMGemmRunner` |
| `topk.py` | `TopKRunner` |

The classes are also exported from `sm110a_kernel`. Existing `compile_*` and
functional APIs retain their existing behavior; choosing a runner explicitly
enables measured configuration selection.

```python
from sm110a_kernel import Bf16GemmRunner

# A[M,K], B_nt[N,K], out[M,N] are contiguous CUDA tensors.
runner = Bf16GemmRunner(a, b_nt, out)
runner.run()  # First eager call autotunes; subsequent calls launch the winner.
```

`run()` uses a process-local selection when another instance already tuned the
same workload. It binds each instance's current tensors. The shared cache holds
configurations only; a live runner instance owns its prepared tensors/callable.
The persistent cache defaults to `profiles/autotune_cache.json` and uses the
existing CUDA/environment fingerprint.

Benchmarks call `tune(TuneOptions(...), candidates=..., benchmark=..., probe=...,
metric_formatter=...)` to obtain a `Selection`. They retain control of CLI
filters, timing counts, reporting and baseline/accuracy comparisons. Candidate
policies and native-kernel execution bindings live in the kernel modules.
`selection.prepared.run()` launches the chosen workload without selection work.
Default runner timing evicts L2 before each measured invocation, outside the
CUDA event pair. Pipeline runners time CUDA Graph replay to exclude Python
launch gaps while preserving producer/consumer adjacency.

Warm up every intended shape and semantic mode eagerly before CUDA Graph
capture. A cache miss inside capture raises an explicit warmup error. Once
prepared, changing tensor *contents* works without retuning; changing shapes,
dtypes, or semantic settings requires a new runner. Use
`clear_kernel_runner_cache()` to forget process-local selections; it does not
delete persistent entries or invalidate already prepared instances.

Routing, rounding, scale layout and reduction semantics are held fixed during a
sweep. FC1/FC2 runners rebuild routing/task metadata by default; their benchmark
paths use `dynamic_metadata=False` to measure only the projection. Grouped GEMM
uses `dynamic_metadata=True` in the model so graph replay consumes the current
layout. `Mxfp4MoeRunner` measures the complete routed quantization/FC1/FC2 chain,
preparing metadata and output state ahead of adjacent PDL launches.
`Fp8GemmRunner(..., input=bf16_input, pdl=True)` similarly tunes the quantizer/GEMM
chain rather than launching a PDL consumer in isolation.

Top-k and indexer retain automatic selection through their existing functional
APIs; their selectors now delegate to the public runners. Quantizers, MQA,
sparse MLA and NCCL LSA inherit `FixedKernelRunner`, which prepares their sole
implementation without running a timing sweep. Calling `tune()` explicitly is
still available. NCCL requires an existing communicator and the same collective
call order on every rank.

`Bf16GemmServingRunner.run()` tunes ordinary GEMM execution. Its `create_queue()`
uses the sole persistent-queue tactic supported by the native ABI
(`128x256x128/c2/e32`); ordinary GEMM timing does not select a queue tactic.
The serving queue lifecycle benchmark keeps its explicit configuration controls.

`Nvfp4Fc1Fc2Runner` accepts fixed routing metadata. Its underlying native facade
caches a host-read row-task count per metadata pointer, so warm up that exact
metadata eagerly before graph capture and keep its routing/task count fixed.
Use a new metadata allocation and warmup when changing routing. This restriction
does not apply to the dynamic MXFP4 MoE runners.

Standalone FC1/FC2 runners use serial configurations. PDL requires a complete
producer/consumer chain with preparation ahead of the launches, as provided by
`Mxfp4MoeRunner` and the FP8 input chain. Do not pass a PDL-only consumer as a
standalone tuning candidate.

The `cutedsl/` classes remain external-library baseline runners with their
existing autotuning API. `cublaslt.py` wraps the vendor library; `moe_routing.py`
and `pdl.py` contain helpers/constants rather than independent native kernel
families.

`Mxfp4StandaloneRunner` binds a serial FC1→FC2 pair, including intermediate
buffers and candidate-specific routing/task metadata. Its
`Mxfp4StandaloneConfig` holds both configurations and survives persistent-cache
serialization. `selection.prepared.fc1` and `.fc2` expose individual stages for
benchmark reporting. Routing is fixed for the lifetime of a prepared instance;
use `Mxfp4MoeRunner` when routing must be rebuilt on each invocation.

Native parity baselines in the MXFP4 FC1/FC2 benchmarks use
`Nvfp4Fc1Runner`/`Nvfp4Fc2Runner` with `dynamic_metadata=False`. Indexer prefill,
indexer decode, and top-k benchmarks also time prepared runner callables. Fixed
CLI tactics call `prepare(config)` directly; automatic tactics call `tune()`
with the benchmark's timing and cache options.

The runner CLI forwards arguments to each kernel family's benchmark:

```bash
python3 -m sm110a_kernel.runner indexer-score-decode -- --batches 2 --kv-lens 4096
python3 -m sm110a_kernel.runner topk -- --q-lens 16 --kv-lens 8192
```

## Model integration

`models/deepseek_v4_flash_sm110a/model.py` invokes public runners directly for
dense GEMMs, grouped output projection, quantization, sparse attention, routed
MoE, and indexer top-k. It allocates outputs and binds current tensors to each
runner; compilation and candidate selection remain inside the kernel modules.
The model does not call `compile_*` or hold compiled module objects.

Warm up the model eagerly for each intended input shape and mode before graph
capture. Model weight preparation only packs static tensors; it does not tune
input-dependent workloads. Unsupported operations keep their explicit PyTorch
or reference fallbacks. See the [model guide](../models/deepseek_v4_flash_sm110a/README.md)
for supported modes and numerical limitations.
