# DeepSeek-V4-Flash SM110A runtime

This directory contains the repository-native DeepSeek-V4-Flash runtime.  It
keeps checkpoint compatibility with the official readable implementation in
`../deepseek_v4_flash/inference`, but executes supported operators with the
SM110A kernels from `sm110a_kernel`.

The 32-GiB benchmark configuration contains one ratio-4 CSA layer and one
ratio-128 HCA layer. Routed experts default to separate W4A8 `mxfp4_fc1` and
`mxfp4_fc2` kernels. An optional `fused` backend executes both in one
`mxfp4_fc1fc2` kernel. Dense projections, activation quantization and
sparse MLA use the corresponding SM110A kernels. Missing fused model-level
operators currently use explicit PyTorch fallbacks marked with
`TODO(sm110a)` in `model.py`.

All RMSNorm modules, including attention/FFN, compressor, final and MTP norms,
use `RmsNormRunner` by default. Weights stay FP32 as in the reference model;
BF16/FP16 inputs are normalized and multiplied by the FP32 weight in FP32,
with one final cast to input dtype. No weight conversion is needed per call,
and state-dict keys/dtypes are unchanged. CPU and unsupported input formats
retain the reference implementation. Backend statistics count native calls
under `sm110a.rmsnorm` and fallbacks under `torch.rmsnorm_fallback`.

Prefill and decode use `HcPreRunner` for HC pre in attention and FFN, including
MTP blocks. Aligned BF16 inputs with even D >= 256
use a split-K `tcgen05.mma` projection with instruction N=24. Its fixed tile is
M=128, N=24, K=128; each CTA handles 512 K elements with two shared-memory
buffers, 512 producer threads and one consumer warp. Producers load and convert
the next tile while MMA processes the current tile. Each input load supplies
both the FP32 RMS sum and projection. FP32 weights are decomposed into three
BF16 components on device, with separate FP32 high/residual accumulators.
A second kernel reduces partials, computes pre/post gates and 4x4 Sinkhorn,
and collapses the HC streams. It overlaps asynchronous input prefetch with
Sinkhorn. The runner retains its scratch allocation across graph replays
(512 KiB for batch=128, D=4096).

FP16/FP32 inputs, odd/small D, and unaligned input/output buffers retain the
single-CTA SIMT kernel with 256 threads per token. The implementation supports
HC=4, D up to 4096, contiguous CUDA BF16/FP16/FP32 inputs, and FP32 HC parameters
(16-byte aligned projection weights). Only the collapsed output is cast back
to input dtype; post/comb stay FP32. FP32 reduction
order differs from PyTorch, so agreement is numerical
rather than bitwise. Unsupported formats use the reference path. The fixed
configuration is tuned for decode; large-M prefill tuning remains future work.
Dispatch counts are
`sm110a.hc_pre` and `torch.hc_pre_fallback`. Attention/FFN RMSNorm remains a
separate operator after HC pre, and checkpoint keys/dtypes are unchanged.

Compare the complete fused operation with the current PyTorch implementation
using cold-L2 CUDA Graph replay:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_hc_pre.py --tokens 1 16 128 --sol
```

`--dtype`, `--d`, `--sinkhorn-iters`, `--input-case`, and `--logit-scale`
control numerical cases. Pass `--batch-size B` to use `[B, tokens/B, 4, D]`
prefill inputs (token counts must be divisible by B):

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_hc_pre.py --batch-size 1 --tokens 128 512 1024 4096 --sol
```

Accuracy checks cover all three outputs and eager/graph reproducibility,
including input updates between replays.

On Thor at GPC 1575 MHz / EMC 4266 MHz, BF16 D=4096, B=1, Sinkhorn=20,
5 warmups, 20 iterations and 5 samples, dirty cold-L2 CUDA Graph replay measured:

| Prefill S | Native (us) | PyTorch (us) | Speedup | GMEM SOL |
| --- | ---: | ---: | ---: | ---: |
| 128 | 77.00 | 481.41 | 6.25x | 32.47% |
| 512 | 210.17 | 1569.33 | 7.47x | 39.36% |
| 1024 | 384.93 | 3085.19 | 8.02x | 41.49% |
| 4096 | 1447.97 | 11809.64 | 8.16x | 42.92% |

All three outputs passed comparison with the FP32 PyTorch reference at the
existing tolerances, including graph replay after changing the input.

`--sol` reports a memory-bandwidth lower bound: input bytes plus FP32 HC
parameter bytes plus the three output tensors, each counted once per invocation.
Shared projection weights are counted once across tokens because on-chip caches
can reuse them. The default peak is the repository's 273 GB/s, overridable with
`--peak-gbps`. Attainment is `(traffic / peak_bandwidth) / measured_time`.
This is a compulsory-memory SOL only; it excludes compute, repeated L1/L2
traffic, temporary buffers, synchronization, and launch costs, and does not
use tensor-core peaks.

HC post uses `HcPostRunner` in attention and FFN, for prefill and decode
(including MTP). It fuses the FP32 post-gated expansion, 4x4 residual mixing,
addition and final cast into one kernel, reusing each loaded residual vector
across all four output streams. FP32 multiplies and adds remain separate to
preserve reference rounding. Contiguous CUDA BF16/FP16/FP32 inputs with HC=4
and D in [1, 32768] are supported; unsupported inputs use the reference path.
Aligned rows use streaming vector loads/stores; other layouts use a scalar
kernel. Output must not overlap inputs. Dispatch statistics are
`sm110a.hc_post` and `torch.hc_post_fallback`.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_hc_post.py --tokens 1 4 16 128 8192
```

The benchmark checks reference agreement, eager/graph agreement, and graph
replay after updating all four inputs. `--dtype`, `--d`, `--unaligned` and
`--input-case` select numerical/layout cases; `--threads` and `--vector-bytes`
allow fixed-configuration comparisons. Bandwidth is effective compulsory
traffic `(x + residual + post + comb + output) / time`, relative to 273 GB/s.
This is not a hardware DRAM-counter measurement. The default dirty cold-L2
flush also causes unrelated dirty cache lines to be written back; these bytes
are excluded from compulsory traffic. Conversely, output may remain in L2,
so clean-cold-L2 effective bandwidth can exceed DRAM peak for small tensors.
Small-token bandwidth percentages include launch/latency costs and should not
be interpreted as saturation throughput. No warm-L2 measurements are used.

On Thor with GPC 1575 MHz and EMC 4266 MHz, BF16 D=4096, 5 warmups,
20 iterations and 5 samples, a cold dirty-L2 sweep of 32/64/128/256 threads
measured 6.1–6.8 us for 1 token and 12.9–14.0 us for 16 tokens. Smaller
blocks did not provide a consistent improvement, so the default stays at
128 threads and 16-byte vectors. A comparison against the reference measured
6.19 vs 21.62 us at 1 token, 13.45 vs 91.53 us at 16 tokens, and 65.00 vs
613.34 us at 128 tokens. The 128-token effective bandwidth was 145 GB/s
(53% of peak); the 90% target is not reached for small-token decode. At 8192
tokens, repeated measurements were around 243–246 GB/s (89–90%); this does
not establish stable 90% bandwidth attainment.

Dense BF16/FP8/FP8xMXFP4 GEMMs, grouped wo_a projection, routed MoE,
quantizers, RMSNorm, sparse MLA, and indexer top-k execute through public runner
classes in `sm110a_kernel/kernels/*.py`; `model.py` does not call `compile_*`.
These inherit `BaseKernelRunner` in `sm110a_kernel/runner.py`, which uses
`AutotuneRunner` for selection and persistent caching. Benchmarks share the
same candidate policies and execution bindings.
The first eager call for each shape and numerical/PDL mode selects a compatible
configuration using cold-L2 timings (CUDA Graph timings for pipelines) and persists it in
`profiles/autotune_cache.json`. Later calls reuse the configuration with the
current tensors; the shared selection cache retains only configurations.
MoE tuning measures the complete quantization/routing/FC1/FC2 chain, rebuilding
configuration-specific metadata on every invocation. Routing weights, BF16
rounding, compact scales, and deterministic reduction remain fixed during tuning.
Top-k uses `TopKRunner` for configuration selection. Quantizers, RMSNorm and sparse MLA
use fixed runners, which prepare their single implementation without a timing
sweep. See the [runner guide](../../docs/kernel_runners.md) for the full API.

Warm up each intended shape and mode eagerly before capturing a CUDA Graph.
`prepare_for_inference()` packs static weights; input-dependent tuning happens
on the first forward. The benchmark already runs eager decode before capture.
`clear_backend_cache()` also clears in-process kernel runner selections; the
persistent autotune cache remains available.

Both MoE backends compact static weight scales once during preparation.
`Transformer(args, compact_moe_scales=False)` or the benchmark flag
`--moe-weight-scales padded` selects padded scales for comparison.
The default is `--moe-weight-scales compact`; input and intermediate activation
scale formats are unchanged.

The default expert path applies FP32 routing weights inside FC1 before
activation quantization, with the official BF16 rounding boundaries. FC2
uses a fixed-order FP32 reduction and rounds to BF16 once after summing the
expert outputs. `--routing-weight-position fc2 --fc2-reduction atomic`
selects the previous output-weighted path for comparison.

Select the fused backend with `Transformer(args, moe_backend="fused")` or:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py --moe-backend fused
```

`--moe-backend unfused` remains the default. The fused backend keeps the same
FP4 weight values, gate ordering, FC1 routing weights and BF16 activation
rounding. It autotunes row scheduling across compatible cluster and hidden-split
configurations, with statically compacted weight scales, releasing the padded
scale buffers after preparation.
It currently uses BF16 atomic expert accumulation, so its default reduction is
`atomic`; explicitly requesting deterministic reduction or PDL is rejected.
Use `--moe-backend unfused --fc2-reduction atomic` for a comparison with the same
reduction semantics. Fused output can vary between replays and is validated
with a numerical tolerance rather than bitwise equality. It supports positive
hidden dimensions divisible by 256 and expert intermediate dimension 2048.
Routing task counts are read on device each invocation, including graph replay.

The default path is bitwise reproducible between eager and CUDA Graph in the
tested decode runs, but full-model agreement with the official model is not
yet validated. Graph self-consistency does not establish agreement with the
official model or real-checkpoint quality.

The LM head keeps the checkpoint's BF16 weights and uses the native BF16
GEMM with FP32 output. It projects only the last sequence position, so GEMM M
is the batch size. The epilogue writes the FP32 accumulator directly without
an intermediate BF16 rounding or per-step weight conversion. Unsupported
shapes/dtypes use a PyTorch FP32 fallback. Loading FP32 head weights rounds
them to BF16; checkpoint key names and the head shared with MTP stay unchanged.

The CSA indexer uses the native SM110A row top-k kernel for both prefill and
decode. Its score producer currently retains the reference BF16/QAT path.

Decode is always captured as one fixed CUDA Graph and benchmarked through
graph replay. By default the benchmark runs decode only with batch 128,
context capacity 8192, and one step per replay (position 8191). KV caches are
zero-initialized; no prefill is run. Warmup and measurement replay counts remain
2 and 5. Run the default workload with:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py
```

To include prefill and reproduce the previous batch-1, 1024-token prompt,
128-step workload:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py --no-decode-only --batch-size 1 --prefill-len 1024 --decode-len 128 --seq-len 1152
```

Decode PDL is opt-in with `Transformer(args, pdl=True)` or the benchmark's
`--pdl` flag (`--no-pdl`, the default, is the serial control):

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py --decode-only --seq-len 2184 --decode-len 16 --warmup 1 --iters 3 --no-pdl
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py --decode-only --seq-len 2184 --decode-len 16 --warmup 1 --iters 3 --pdl
```

The single-token decode path directly writes blocked activation scales in
both modes. PDL enables `fp8_act_quant -> fp8_gemm` and
`bf16tofp8_quant -> mxfp4_fc1 -> mxfp4_fc2`. Routing metadata, weight
conversions and output/scheduler preparation precede the chain on every
CUDA Graph replay. Prefill uses the existing serial backend. Eager decode
also executes the selected kernels to warm up and validate graph capture;
performance comparisons target fixed decode CUDA Graphs. The deterministic
FC2 finalize kernel remains serial after FC2.
