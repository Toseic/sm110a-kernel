# Compressor pooling

`CompressorPoolRunner` fuses positional scoring, gated pooling and incremental
FP32 state updates for DeepSeek-V4-Flash ratio-4 and ratio-128 compressors.
The model invokes it after its joint BF16 KV/gate GEMM. RMSNorm, RoPE,
quantization and final cache stores remain separate.

Inputs are BF16 KV and score projections `[B,S,W]`, including identically
strided halves of a joint GEMM output. Positional scores are FP32 `[ratio,W]`.
For ratio-4 overlap, `W=2*D` and states have shape `[max_batch,8,2*D]`;
for ratio-128, `W=D` and states have shape `[max_batch,128,D]`.
The caller initializes KV state to zero and score state to negative infinity,
matching the model. State and output writes must not alias any other argument.
D in `[1,32768]`, B up to 65535, and fewer than 65535 prefill groups are supported.

Prefill (`start_pos=0`) emits `floor(S/ratio)` BF16 rows and saves the last
overlap group and incomplete tail. An extra CTA per channel tile handles state
writes in the same launch; pooling reads projections directly, independently
of those state writes. Short prefills update state and emit no rows.
Decode (`start_pos>0`, S=1) updates state on every step and only emits a row
when `(start_pos+1)%ratio==0`. Ratio-4 rolls the complete current window into
the previous window after pooling. Unused state entries and inactive batch
rows retain their prior values, as in the reference.

Ratio-4 assigns a channel to each thread and keeps eight scores/values in
registers. It reads the previous group's first D channels and the current
group's second D channels without materializing overlap tensors. Ratio-128
assigns 32 adjacent channels to a 128-thread CTA; four warps each process 32
tokens and reduce through shared memory. Both use max-subtracted softmax and
separate FP32 probability multiply/add operations, with a BF16 output cast
before downstream RMSNorm. Parallel reduction order and exponential/division
approximations mean output agreement is numerical, not bitwise with Torch.
Repeated native execution with identical input/state is deterministic.

The runner uses the standard split-binding JIT and fixed-runner cache. PDL is
optional and follows the model decode setting. Warm up each shape/start position
and PDL mode before CUDA Graph capture. The start position is a host scalar;
a graph replay repeats that fixed position, matching the existing model API.

Run accuracy checks and dirty cold-L2 CUDA Graph comparisons:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_compressor_pool.py
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_compressor_pool.py --batch-size 128 --dims 128 512 --seq-lens 128 1024 --warmup 5 --iters 20 --samples 5
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_compressor_pool.py --batch-size 16 --dims 65 512 --seq-lens 3 127 129 --pdl
```

The benchmark compares all state entries exactly and BF16 outputs numerically,
checks eager/graph bitwise agreement from restored state, changes inputs before
graph replay, and runs consecutive decode updates through two compression
boundaries after partial prefill. Timings include pooling and state updates,
excluding GEMM, norm, RoPE, quantization and final cache writes. Repeated fixed
boundary replays settle overlap state during warmup; sequential validation
separately checks changing positions.

Integration validation used the existing two-layer model benchmark with random
weights, B=1, a 127-token prefill and three decode steps (positions 127–129),
covering both ratio-4 and ratio-128 boundaries with decode PDL enabled:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py --batch-size 1 --seq-len 130 --prefill-len 127 --decode-len 3 --no-decode-only --warmup 1 --iters 1
```

Native pooling dispatched for main-attention and indexer compressors with no
pooling fallbacks. Eager/graph first-step logits cosine was 0.99999994, and final
tokens matched. This checks integration and graph state handling, not
checkpoint quality or an end-to-end speedup against the previous implementation.

On Thor with GPC 1575 MHz / EMC 4266 MHz, B=16, D=512, PDL enabled,
5 warmups, 20 iterations and 5 samples, dirty cold-L2 CUDA Graph medians were:

| Ratio | Operation | Native (us) | Torch (us) | Speedup |
| --- | --- | ---: | ---: | ---: |
| 4 | Prefill S=1024 | 374.446 | 5063.402 | 13.52x |
| 4 | Decode state update | 8.280 | 21.101 | 2.55x |
| 4 | Decode compression boundary | 17.547 | 55.578 | 3.17x |
| 128 | Prefill S=1024 | 475.798 | 1850.456 | 3.89x |
| 128 | Decode state update | 8.974 | 19.624 | 2.19x |
| 128 | Decode compression boundary | 80.659 | 196.152 | 2.43x |

Command (also measures D=128 and S=128):

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_compressor_pool.py --batch-size 16 --dims 128 512 --seq-lens 128 1024 --pdl --warmup 5 --iters 20 --samples 5
```

Primitive comparisons passed for D=65/128/512, B=1/16, incomplete and complete
prefills, continuous decode and changed-input graph replay, including PDL.
All state comparisons were exact. Compute Sanitizer memcheck and racecheck,
filtered to compressor kernels, reported zero errors/hazards for both ratios
with D=65, short/partial prefills and consecutive boundary updates. These
isolated gains do not establish an end-to-end model speedup.
