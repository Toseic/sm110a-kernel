# Fused compressor norm, RoPE, FP8 quantization and cache store

`CompressorStoreRunner` implements the post-pooling path for main-attention
DeepSeek-V4-Flash compressors. It consumes contiguous BF16 `[B,S,512]` pooled
KV, FP32 `[512]` RMSNorm weights and complex64 `[S,32]` frequencies. Each
128-thread CTA handles four rows, one warp per row. RMSNorm, tail RoPE,
block-64 FP8 quantization and the three persistent cache stores share one
launch, without a global normalized or rotated intermediate.

The cache consists of contiguous tensors:

- E4M3 `[max_batch,capacity,448]` non-RoPE values;
- E8M0 or FP32 `[max_batch,capacity,7]` scales;
- BF16 `[max_batch,capacity,64]` RoPE values.

The kernel writes active batch rows at `[position:position+S]`. All other
positions/batches retain their previous bytes. Optional separate packed output
tensors `[B,S,448/7/64]` are written in the same launch. Prefill uses them for
attention over the original prompt plus compressed KV. Decode only writes
persistent cache and the model returns a cache view. Cache and optional packed
outputs must be disjoint from each other and from inputs.

The RMSNorm follows the existing D=512 native runner's vector/reduction order:
each lane processes two groups of eight contiguous elements, with FP32 FMA
squared accumulation and a warp sum. It rounds to BF16 after weighted norm,
then performs adjacent-pair FP32 complex multiplication and rounds RoPE to
BF16. Non-RoPE quantization matches the existing `Fp8ActQuantRunner` block-64
scale and conversion rules, including BF16 power-of-two scaling for E8M0 and
the 1e-4 amax floor for FP32 scales. Strided frequency slices and lazy conjugates
are supported. Epsilon must be finite and positive.

The fixed runner uses the shared split-binding JIT/cache and optional PDL.
Warm up each workload eagerly before CUDA Graph capture. Position is a host
scalar, so a graph replays a fixed position, matching the model's existing API.
Supported main-attention compressors dispatch this kernel for both ratio-4
and ratio-128 (including MTP); other formats and custom norms keep the separate
operators. Pooling remains a separate kernel. Indexer Hadamard/FP4 processing
and sliding-window/ring cache writes are outside this kernel's scope.

Benchmark against the previous **native** RMSNorm + RoPE + block-64 quantizer
+ three cache copies, using dirty cold-L2 CUDA Graph medians:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_compressor_store.py --batch-sizes 1 16 128 --rows 8 256 --pdl --warmup 5 --iters 20 --samples 5
```

The unfused reference includes packing the non-RoPE input into a contiguous
quantizer buffer, as the model did. Its quantizer outputs and normalized RoPE
view are reused as prefill packed output; no artificial extra copies are
added to that reference. The fused prefill path writes both cache and packed
outputs, while decode writes only cache.

Accuracy compares complete cache contents and prefill packed output with the
native unfused chain exactly (FP8/E8M0 compared as bytes). It checks untouched
cache positions and inactive batch rows, eager/graph agreement, and replay
after changing pooled input, weights and frequency data. Additional inputs:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_compressor_store.py --batch-sizes 1 16 --rows 3 --pdl --conjugate --input-case zero --warmup 1 --iters 3 --samples 1
```

On Thor with GPC 1575 MHz / EMC 4266 MHz, E8M0, PDL, 5 warmups,
20 iterations and 5 samples, the command above measured:

| B | S | Mode | Fused (us) | Unfused native (us) | Speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1 | Decode | 8.138 | 14.763 | 1.81x |
| 16 | 1 | Decode | 8.554 | 18.976 | 2.22x |
| 128 | 1 | Decode | 9.771 | 23.574 | 2.41x |
| 1 | 256 | Prefill | 13.163 | 23.962 | 1.82x |
| 16 | 256 | Prefill | 87.598 | 146.904 | 1.68x |
| 128 | 256 | Prefill | 522.373 | 1136.059 | 2.17x |

Both scale modes passed exact comparisons across B=1/16/128, prefill S=3/8/256,
decode, PDL, strided/conjugated frequencies and zero/random inputs. These are
isolated post-pooling measurements, excluding pooling/GEMM and downstream
attention. They do not establish an end-to-end speedup.

Integration used the existing full two-layer benchmark (B=1, prefill=127,
three decode steps at positions 127–129, PDL enabled), with native store
counters present in both phases and exact eager/graph decode agreement:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py --batch-size 1 --seq-len 130 --prefill-len 127 --decode-len 3 --no-decode-only --warmup 1 --iters 1
```

A same-process fused/unfused comparison additionally used one shared random
model with D=4096, two ratio-4/128 layers, vocab=128, eight routed experts,
expert width=2048, B=1 and capacity=260. Only the two main-compressor norm
classes were temporarily subclassed to select the unchanged separate path;
weights, inputs and already-selected kernel configurations were shared.
After restoring all caches/states, both paths produced identical eager logits
and complete cache/state contents for prefill lengths 127 and 257 followed by
three fixed-input decode steps. Separate CUDA Graphs also produced identical
first/final logits and final tokens. This is integration equivalence for random
weights, not checkpoint-quality validation. Historical hashes from a different
commit were not used as an equivalence criterion: autotune cache partitions
include the git revision and may select different GEMM configurations.

Compute Sanitizer memcheck, restricted to the compressor store kernel, reported
zero errors for both scale modes, PDL, B=1/5, decode and three-row prefill.
Racecheck with the same filter/workloads also reported zero hazards, errors or
warnings. Sanitizer-instrumented timings are excluded from performance results.
