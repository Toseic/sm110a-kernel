# DeepSeek-V4-Flash SM110A runtime

This directory contains the repository-native DeepSeek-V4-Flash runtime.  It
keeps checkpoint compatibility with the official readable implementation in
`../deepseek_v4_flash/inference`, but executes supported operators with the
SM110A kernels from `sm110a_kernel`.

The 32-GiB benchmark configuration contains one ratio-4 CSA layer and one
ratio-128 HCA layer. Routed experts default to separate W4A8 `mxfp4_fc1` and
`mxfp4_fc2` kernels. An optional `fused` backend executes both in one
`mxfp4_fc1fc2` kernel. Dense projections and activation quantization use
SM110A kernels. Main attention uses native FP8-cache sparse MLA as described
below. Missing fused model-level operators use explicit PyTorch fallbacks
marked with `TODO(sm110a)` in `model.py`.

Q, KV, compressed KV and indexer RoPE, plus attention output inverse RoPE,
use the native [RoPE kernel](../../csrc/rope/README.md). It consumes strided
head-tail views and precomputed complex64 frequency slices directly, with
FP32 arithmetic, BF16/FP16/FP32 input and optional decode PDL. Native counters
are `sm110a.rope` and `sm110a.rope_inverse`; unsupported formats use
`torch.rope_fallback`.

Main-attention KV caches now store E4M3 FP8 non-RoPE values, one scale per
64 values, and separate BF16 RoPE values. This covers the sliding window and
both CSA/HCA compressed caches, including MTP. With the default 512-dimensional
KV and E8M0 scales, each entry uses 448 + 7 + 128 = 583 bytes instead of
1024 bytes (43.1% less persistent main KV storage). FP32 scales are also
supported. The separate indexer retains its existing MXFP4/QAT cache path.
All cache buffers are non-persistent, preserving checkpoint keys.

`sparse_attn_fp8` consumes these packed tensors in both prefill and decode.
For CUDA BF16 Q with H=64/D=512 it uses the separate native
[FP8 KV sparse MLA kernel](../../csrc/fp8_kv_fa4_sparse_mla/README.md).
Warp 4–7 gather selected FP8 values/scales, dequantize in registers and write
BF16 shared tiles; warp 2 issues BF16 MMA. RoPE loads remain BF16, while
accumulation, sink and online softmax remain FP32. This preserves the former
QAT KV operands without a BF16 global-cache mirror or Torch dequantization.
It does not introduce FP8 Q quantization or FP8 MMA. The existing BF16 sparse
MLA kernel is unchanged. Unsupported layouts retain the chunked Torch fallback.

Native dispatch counters are `sm110a.sparse_mla_fp8_kv_prefill` and
`sm110a.sparse_mla_fp8_kv_decode`; the fallback is
`torch.sparse_mla_fp8_cache_todo`. Quantization uses `sm110a.fp8_act_quant`,
and cache writes still use `torch.fp8_kv_cache_store_todo`. Fused FP8 cache
writes and additional head/layout support remain in `TODO(sm110a)`.
Benchmark CUDA Graph snapshots restore FP8 bytes, scales, RoPE and
compressor/indexer state together. Memory budgets use the packed layout.
HBM SOL counts logical selected-cache bytes, excluding repeated loads and
fallback temporary traffic, so it remains a lower bound.

MoE sqrtsoftplus routing uses a [fused gate kernel](../../docs/fused_gate.md)
for scoring, top-k/hash selection, normalization and scaling. Bias affects
selection only. Decode passes the model PDL flag to the fused gate.

Compressor KV/gate and MoE router projections use native BF16 GEMM in both
prefill and decode, including MTP. Compressor `wkv`/`wgate` parameters retain
their checkpoint keys but are stored in BF16. `prepare_for_inference()` joins
them into a non-persistent weight buffer for one GEMM with BF16 output.
Compression state, positional scores and pooling accumulation remain FP32.
Router weights are also stored in BF16; its GEMM writes FP32 logits directly,
and scoring, selection and routing-weight normalization retain the reference
FP32 computation. Dispatch counts are `sm110a.compressor_bf16_gemm` and
`sm110a.router_bf16_gemm_fp32`. Unsupported shapes/devices use PyTorch with
the same BF16 operands. Load checkpoints before preparing packed weights.
These projections change the previous FP32 computation: compressor output
now rounds to BF16, and FP32 checkpoint projection weights round to BF16 on
load. Prior full-model precision measurements do not validate this path.

Shared experts use `Fp8SharedFc1Runner` by default for CUDA BF16 inputs
with supported E4M3 weights, E8M0 scales and 128-aligned dimensions. This
applies to both decoder and MTP blocks. `prepare_for_inference()` packs
static gate/up weights once into 64-row interleaved pairs and expands the
checkpoint's 128x128 scales into the blocked MMA layout. Original w1/w3
parameters and state-dict keys are retained; derived packed buffers are
non-persistent. As with routed expert preparation, load the checkpoint before
preparing the model.

The shared FC1 quantizes its BF16 input once, then fuses both FP8 projections,
intermediate BF16 rounding, FP32 clamp/SiLU/multiply, and final BF16 conversion.
Decode supports the existing optional PDL setting. Small token workloads pad
to 128 rows, and the padded activation feeds the existing native dense FC2
before trimming the result. Shared/routed output addition remains PyTorch.
Backend statistics are `sm110a.fp8_shared_fc1` and
`torch.shared_expert_fallback`; unsupported formats or explicit per-expert
routing weights use the inherited reference path. See the
[shared FC1 kernel guide](../../csrc/fp8_shared_fc1/README.md) for benchmark
results and the corrected L2 SOL model; isolated FC1 gains do not establish
an end-to-end model speedup.

Standalone RMSNorm modules, including compressor and MTP input norms,
use `RmsNormRunner` by default. Attention/FFN and final norms are fused into
HC pre/head when the input and parameters support `HcPreNormRunner`. Weights stay FP32 as in the reference model;
BF16/FP16 inputs are normalized and multiplied by the FP32 weight in FP32,
with one final cast to input dtype. No weight conversion is needed per call,
and state-dict keys/dtypes are unchanged. CPU and unsupported input formats
retain the reference implementation. Backend statistics count native calls
under `sm110a.rmsnorm` and fallbacks under `torch.rmsnorm_fallback`.

Prefill and decode use `HcPreNormRunner` for HC pre + attention/FFN RMSNorm,
including MTP blocks. BF16 inputs with even D >= 256 use a split-K
`tcgen05.mma` projection with instruction N=24. Its tile is M=128, N=24,
K=128 with two shared-memory buffers, 512 producer threads and one consumer
warp. Each CTA handles 1024 K elements for D=4096 and at least 256 tokens;
other shapes retain 512. FP32 projection weights are decomposed into three
BF16 components on device, with separate FP32 high/residual accumulators.
A second kernel reduces partials, computes gates and Sinkhorn, collapses the HC
streams, and applies weighted RMSNorm without a global collapse intermediate.
The collapse is rounded to input dtype before RMSNorm to preserve the separate
operators' semantics. HC projection epsilon and the target norm's epsilon are
passed independently. Post/comb stay FP32. Scratch is sized for split-K=512
and reused during graph replay (512 KiB for batch=128, D=4096).

FP16/FP32 inputs and odd/small D use the single-CTA SIMT projection and fused
collapse/norm. Fusion supports HC=4, D up to 4096, contiguous CUDA
BF16/FP16/FP32 inputs, contiguous FP32 HC and norm parameters, and 16-byte
aligned projection weights. Custom norm implementations and unsupported formats
retain separate HC pre and norm calls. Checkpoint keys and parameter dtypes
are unchanged. Native fusion counts are `sm110a.hc_pre_norm`; separate-path
fallbacks count `unfused.hc_pre_norm` and their individual operator counters.
Explicit `hc_pre` calls still expose the original standalone operator.

The shared final/MTP head fuses HC head collapse and its passed RMSNorm using
`HcPreNormRunner(..., head=True)`, followed by the existing logits projection
and distributed gather. This mode uses FP32 `[4, 4*D]` projection weights,
`[1]` scale, and `[4]` base, with no post/comb outputs, Sinkhorn, or scratch.
Dispatch statistics are `sm110a.hc_head_norm` and `unfused.hc_head_norm`.
Explicit `hc_head` calls retain their standalone behavior.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_hc_pre_norm.py --tokens 1 16 128 256 512 1024 2048 4096 --baseline none --warmup 5 --iters 50 --samples 5
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_hc_pre_norm.py --mode head --tokens 1 16 128 --baseline none
```

See [HC pre + RMSNorm measurements](../../csrc/hc_pre_norm/README.md) for the
current fused-kernel comparison. The standalone HC measurements below predate
model integration of RMSNorm fusion.

Integration validation ran one-layer prefill (4 tokens) and decode with the
standard benchmark config at batch=1, context=16. CUDA Graph/eager logits had
cosine 1.0 and identical final tokens; backend counts included fused HC pre/norm
and head/norm. A same-weight fused/unfused comparison with D=4096, one regular
layer plus one MTP layer, vocab=128, eight experts and expert width=128 produced
identical logits for prefill, decode and MTP in that random-weight case. This
is an integration check, not a checkpoint accuracy or end-to-end speedup claim.
CPU head fallback also matched the existing forward exactly.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py \
  --batch-size 1 --seq-len 16 --prefill-len 4 --decode-len 1 \
  --layers 1 --no-decode-only --warmup 1 --iters 1
```

On Thor at GPC 1575 MHz / EMC 4266 MHz, BF16 D=4096, dirty cold-L2
CUDA Graph replay with a 64 MiB eviction buffer, 5 warmups, 20 iterations
and 5 samples measured the following medians. Inputs were `[tokens, 1, 4, D]`
and parameters were random FP32 tensors. Both implementations include RMS
normalization, projection, sigmoid gates and weighted collapse.

| Tokens | Native (us) | PyTorch (us) | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 17.342 | 35.778 | 2.06x |
| 16 | 20.861 | 98.442 | 4.72x |
| 128 | 59.208 | 278.302 | 4.70x |
| 512 | 167.994 | 1324.010 | 7.88x |
| 1024 | 294.298 | 2866.018 | 9.74x |

All cases passed the existing BF16 tolerance (`rtol=0.008`, `atol=2e-5`),
bitwise eager/graph agreement, and replay checks after changing the input.
Maximum absolute output differences ranged from `1.19e-7` to `0.015625`.
These are isolated HC head timings, excluding the final RMSNorm and LM head.

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
quantizers, RMSNorm, FP8 KV sparse MLA, and indexer top-k execute through public
runner classes in `sm110a_kernel/kernels/*.py`; `model.py` does not call `compile_*`.
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
Top-k uses `TopKRunner` for configuration selection. Quantizers, RMSNorm and
FP8 KV sparse MLA use fixed runners, preparing their single implementation
without a timing sweep. See the [runner guide](../../docs/kernel_runners.md)
for the full API.

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

Select the fused backend with `Transformer(args, moe_backend="fused", pdl=False)` or:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py --moe-backend fused --no-pdl
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

The CSA indexer uses native MXFP4 score runners and row top-k for both prefill
and decode on single-GPU CUDA BF16 inputs with 64 index heads of dimension 128.
Rotated Q is quantized directly to block-32 E2M1/E8M0; the valid prefix of the
BF16/QAT KV cache is repacked on every invocation. Scale tiles are padded
independently per batch, including odd query and KV lengths. Repacking keeps
cache resets and CUDA Graph replay consistent without a second persistent KV
state; fusing packing into compressor cache writes remains future work.
Prefill computes scores with the kernel's ordinary causal mask disabled and
applies compressed visibility `kv_index < floor((query_position + 1) / ratio)`
before top-k. Empty compressed prefixes return empty indices. Other index
head configurations and distributed execution retain the reference score path.
Dispatch counts are `sm110a.indexer_score_prefill`,
`sm110a.indexer_score_decode`, and `torch.indexer_score_fallback`.

The native score epilogue uses the existing DeepGEMM-style BF16 FMA/reduction
order, which differs from the reference PyTorch multiply and sum. Scores are
numerically rather than bitwise equivalent; near-tied candidates may select
different top-k indices. Full-model/checkpoint quality agreement still needs
validation.

Supported configurations always use the native indexer score kernels. The
full-model benchmark uses the shared GPC/EMC clock-lock helper by default;
`--no-lock-clocks` keeps existing clock settings.

On Thor at GPC 1575 MHz / EMC 4266 MHz, cold-L2 CUDA Graph measurements gave:

| Workload | Previous PyTorch | Native MXFP4 | Previous/native |
| --- | ---: | ---: | ---: |
| Indexer decode B=1, KV=2048 | 31.47 us | 38.18 us | 0.824x |
| Indexer decode B=16, KV=2048 | 155.08 us | 159.43 us | 0.973x |
| Indexer decode B=128, KV=2048 | 1245.98 us | 828.65 us | 1.504x |
| Indexer prefill B=1, S=1024, KV=256 | 1207.61 us | 315.67 us | 3.826x |
| Complete two-layer model decode B=128, context=8192 | 40.30 ms | 39.83 ms | 1.012x |

Indexer timings include Q QAT or Q/K packing, score, compressed causal masking
and top-k, excluding projections, RoPE/Hadamard and compressor. Inputs are fixed
random BF16/QAT values; timings are medians of five samples of 20 replays after
five warmups, with a dirty cold-L2 flush before each measured replay. Relative
score RMSE was about 0.49% for B=128 decode and prefill; decode top-k overlap
was 99.72%. Prefill selects all available keys in this case, so its 100% overlap
does not establish agreement for a truncated candidate selection.

Full-model timings are means of 20 replays after five warmups, with the existing
default random weights and zero-initialized history, unfused deterministic MoE,
and PDL disabled. Full model traffic exceeds L2 capacity. Both score backends
passed the benchmark's eager/graph checks; these are not checkpoint-quality
comparisons. Small-batch decode currently regresses; repacking KV and dispatch
overhead remain optimization work.

Run the current native full-model workload with:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py --batch-size 128 --seq-len 8192 --decode-len 1 --warmup 5 --iters 20
```

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

Decode PDL is enabled by default in `Transformer(args)` and the benchmark.
Use `Transformer(args, pdl=False)` or `--no-pdl` for the serial control.
The fused MoE backend requires explicitly disabling PDL. Compare serial and
PDL decode with the default unfused backend:

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
FC2 finalize kernel also uses PDL to wait on FC2 and trigger a downstream consumer.


The decode PDL switch also covers RMSNorm, HC pre/head + norm (including
its internal projection/finish edge), HC post, BF16 GEMM projections and LM
head, grouped wo_a GEMM and its metadata kernels, FP4 packing/QAT including
tails, FP8 KV sparse MLA, indexer scoring, and TopK. FP8 quantizers wait on
incoming dependencies, while FP8 GEMM/shared FC1 and routed FC2 trigger
outgoing consumers. New operators share the conservative input-wait/output-
trigger implementation in `include/pdl.h`; existing deferred GEMM/MoE waits
remain. PDL participates in JIT and runner selection keys. Standalone kernel
benchmarks accept `--pdl`, while model prefill retains its serial setting.
PyTorch processing between native operators still introduces ordinary stream
dependencies; fused compressor/cache/router work is outside this support.
See [the shared PDL interface](../../docs/pdl.md#shared-kernel-interface).
