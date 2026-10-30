# RoPE

In-place adjacent-pair rotary embedding for DeepSeek-V4-Flash. The kernel
reads precomputed complex64 frequencies and rotates BF16, FP16 or FP32
activations with FP32 arithmetic, including inverse RoPE. YaRN/base selection
stays in the model's existing frequency precomputation.

```python
from sm110a_kernel import rope_, compile_rope, RopeRunner

rope_(q[..., -64:], freqs_cis)
rope_(output[..., -64:], freqs_cis, inverse=True)
module = compile_rope(64, pdl=True)
module.rope_(kv[..., -64:], freqs_cis)
RopeRunner(q[..., -64:], freqs_cis, pdl=True).run()
```

Inputs have shape `[B,S,D]` (KV) or `[B,S,H,D]` (Q/output), with unit last
stride and nonoverlapping positive outer strides. Sliced head tails and
permuted outer axes are accepted without copying. Only the supplied view is
modified. Frequencies have shape `[S,D/2]`, complex64, unit last stride;
strided sequence slices (including compressor ratios 4 and 128), nonzero
storage offsets and lazy conjugates are supported. Both tensors must share
the CUDA device and must not overlap. Empty inputs are a no-op.

The optimized path uses a batch/sequence/head grid to avoid runtime 64-bit
division in address calculation. For aligned inputs whose rotary pair count
is divisible by four, each thread owns four adjacent pairs and uses vector
loads/stores. Other alignments use pair or scalar accesses; dimensions beyond
the grid's Y/Z limits retain the original flattened kernel.
The kernel has no shared memory,
reductions or auxiliary allocations. D is an even compile-time constant in
`[2,1024]`; CTA threads are fixed at 256 by default, with 64/128/512 available.
There is no autotuning. The split JIT caches the binding and specialized
kernel libraries separately; D, CTA threads, PDL, architecture and source
dependencies participate in cache selection. Precompile/warm up before CUDA
Graph capture. The module API performs no device allocations on each call.

The SM110A model backend installs this operator within its existing scoped
patch context, covering inherited compressor/indexer paths and explicit
attention calls, including inverse output RoPE. Unsupported devices/formats
retain the original implementation. Counters are `sm110a.rope`,
`sm110a.rope_inverse` and `torch.rope_fallback`. Checkpoint keys are unchanged.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_rope.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_rope.py --layout kv --freq-stride 128 --batch 1 16 --seq-len 1 128 --pdl
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_rope.py --dtype float32 --head-dim 73 --heads 3 --seq-len 17
```

The benchmark compares against the model's Torch complex multiply and copy
under cold L2, with fixed kernel configuration. By default it checks Q/KV,
forward/inverse rotation, a nonzero position offset, and preservation of
non-RoPE dimensions. Use `--no-accuracy` to skip checks or `--baseline none`
to skip Torch timing. Accuracy allows rounding differences in FP32 complex
arithmetic and one output ULP for BF16/FP16. Timing repeatedly rotates the
same tensor in place, as does the Torch baseline; it excludes tensor reset.
The benchmark also prints effective GB/s and bandwidth SOL, using compulsory
RoPE input read + output write + one unique frequency-table read, divided by
the reference 273 GB/s (`--peak-gbps` overrides it). Unaccessed head prefixes
and gaps in strided frequency storage do not count. This is logical bandwidth,
not a hardware DRAM utilization counter.

## Retained optimization

For BF16 `[1,1024,64,512]`, rotary width 64, 256 threads and PDL off, with
the same maximum clocks and dirty cold-L2 conditions, the original kernel
measured 142.112 us. Removing address divisions reduced this to 104.731 us;
the retained four-pair vector version's best measured run was 90.851 us
(187.55 GB/s, 68.70% SOL). A final forward/inverse confirmation measured
94.618 / 93.376 us (65.97% / 66.84%). The 80% target was not reached;
it requires at most 78.019 us with this exact traffic convention.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_rope.py --layout q --seq-len 1024 --direction both --warmup 5 --iters 20 --samples 5 --baseline none
```

NCU baseline/optimized reports are `/tmp/ncu-reports/rope-baseline.ncu-rep`
and `/tmp/ncu-reports/rope-vector4.ncu-rep`; the ordinary-launch timeline is
`/tmp/ncu-reports/rope-vector4-timeline.nsys-rep`. NCU replay timings differ
from the benchmark's dirty eviction conditions and are not used for SOL
claims. `mcc__dram_throughput_op_read.avg.pct_of_peak_sustained_elapsed` and
`mcc__dram_throughput_op_write.avg.pct_of_peak_sustained_elapsed` returned
`nan` and are unavailable. Later experiments with larger per-thread work,
frequency vectorization, cross-head sharing, flat mapping and alternate
cache policies were discarded in favor of the measured best candidate.

## Initial measurements

Thor / CUDA 13.2, GPC 1575 MHz, EMC 4266 MHz, dirty cold-L2 eviction (64 MiB),
BF16 `[1,S,64,512]`, rotary width 64, 256 threads, PDL off. These are quick
measurements with 2 warmups, 5 iterations and 2 samples, not full-model gains.

| S | Direction | Native (us) | Torch (us) | Speedup |
|---|---|---:|---:|---:|
| 1 | forward | 5.043 | 12.102 | 2.40x |
| 1 | inverse | 5.360 | 13.142 | 2.45x |
| 128 | forward | 22.522 | 53.296 | 2.37x |
| 128 | inverse | 23.507 | 57.472 | 2.44x |
| 1024 | forward | 144.010 | 304.979 | 2.12x |
| 1024 | inverse | 143.472 | 284.397 | 1.98x |

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_rope.py --seq-len 1 128 1024 --warmup 2 --iters 5 --samples 2
```

All Q/KV and forward/inverse checks passed. The largest BF16 absolute
difference was 0.0078125. A separate FP32/PDL run with batch 2, sequence
lengths 1/17, 3 heads, head dimension 73 and frequency stride 128 passed with
maximum absolute difference 4.77e-7. Compiled D=64 kernels for all three
dtypes use 32 registers and zero shared/local memory and stack.

The two-layer random-weight model smoke run below completed real prefill and
PDL CUDA Graph decode. Eager/replay first-step logits cosine was 1.0 and final
tokens matched. Eager/capture counters recorded 34 forward and 10 inverse
native RoPE calls, with no RoPE fallback (replays do not re-enter Python).
This verifies integration and replay, not checkpoint-level model accuracy.

Compute Sanitizer memcheck also completed with zero errors for BF16/FP16/FP32,
D=6/64/128, sliced/permuted inputs, offset/strided/conjugate frequencies,
forward/inverse, empty tensors, a non-default stream and PDL Runner graph
replay. These checks used temporary verification code, with no added unit tests.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py --batch-size 1 --seq-len 256 --prefill-len 128 --decode-len 1 --layers 2 --no-decode-only --warmup 1 --iters 1
```
