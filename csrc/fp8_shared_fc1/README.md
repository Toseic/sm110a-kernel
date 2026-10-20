# FP8 shared-expert FC1

Independent SM110A kernel and JIT target for dense shared-expert gate/up
projection followed by clamp and SwiGLU. The DSV4 Flash SM110A model uses it
by default for supported shared experts; the existing `fp8_gemm` kernel
continues to serve shared FC2 and other dense projections.

## Semantics

For each token and intermediate channel:

```text
gate = float32(bfloat16(dequant(A) @ dequant(W_gate).T))
up   = float32(bfloat16(dequant(A) @ dequant(W_up).T))
if swiglu_limit > 0:
    gate = min(gate, swiglu_limit)
    up = clamp(up, -swiglu_limit, swiglu_limit)
out = bfloat16(silu(gate) * up)
```

Both projections share one quantized activation and one Tensor Core GEMM
launch. The epilogue reads FP32 accumulators from TMEM and performs BF16
rounding, FP32 conversion, clamp, SiLU, multiplication and final BF16
conversion in registers. Neither projection is written to global memory.
There is no routing-weight multiplication in this shared expert.

## API and layouts

- `A`: contiguous CUDA E4M3 `[M,K]`.
- `B`: contiguous CUDA E4M3 `[2*I,K]`, interleaved gate[64], up[64].
- `SFA`, `SFB`: E8M0 K=32 blocked scales, same physical layout as FP8 GEMM.
- Output: contiguous CUDA BF16 `[M,I]`.
- `M` must be divisible by `128 * cluster_m`; `I` by 64; `K` by `tile_k`.
- Configurations: `tile_k` 64/128, `cluster_m` 1/2. PDL requires cluster 1.
- `swiglu_limit=0` disables clamp. Positive finite limits specialize the JIT.

```python
from sm110a_kernel import (
    Fp8SharedFc1Config,
    compile_fp8_shared_fc1,
    pack_fp8_shared_fc1_weights,
)

# Logical per-row K=32 E8M0 scales; pack once for static weights.
b, sfb = pack_fp8_shared_fc1_weights(w_gate, w_up, gate_scales, up_scales)
module = compile_fp8_shared_fc1(Fp8SharedFc1Config(swiglu_limit=10.0))
y = module.fp8_shared_fc1(a, b, sfa, sfb)
# For preallocated output:
module.fp8_shared_fc1_out(a, b, sfa, sfb, y)
```

Official checkpoint FP8 weight scales are per 128x128 block. Expand them to
logical `[I,K/32]` before calling the packing helper, repeating each scale
128 times along rows and four times along K.

`Fp8SharedFc1Runner(a, b, sfa, sfb, out, input=x, swiglu_limit=10.0)` optionally
runs the existing BF16-to-blocked-FP8 quantizer once before the fused FC1.
Here `x` is BF16 `[M,K]`, and `a`/`sfa` are preallocated quantizer outputs.
Input quantization is a separate launch; `input=None` consumes existing FP8.

## Benchmark

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_fp8_shared_fc1.py \
  --tokens 128 256 512 1024 4096 --warmup 5 --iters 20 --samples 5 --sol
```

Measured on NVIDIA Thor, GPC 1575 MHz / EMC 4266 MHz locked, dirty 64 MiB
L2 eviction before every CUDA Graph replay (eviction excluded from timing).
K=4096, I=2048, clamp=10, BF16 outputs; FP8 inputs and static packed weights
are prepared before timing. M is the physical GEMM row count, without
padding smaller token workloads. This isolates FC1 and excludes input
quantization and FC2.

The unfused baseline is two native FP8 GEMMs followed by PyTorch FP32
conversions, clamp, SiLU, multiply and BF16 conversion in one CUDA Graph.
Both projections share the prequantized A. Fused sweeps tile_n=128,
tile_k=64/128, cluster_m=1/2; baseline additionally sweeps tile_n=256.
Both exclude split-K and PDL; these results do not represent an exhaustive
sweep of the dense GEMM's full tactic space. All candidate configurations
are checked against independently dequantized FP32 matmuls with the same
intermediate BF16 rounding.

| M | Fused (us) | Unfused (us) | Speedup | TF/s | SOL attainment |
|---:|---:|---:|---:|---:|---:|
| 128 | 166.438 | 186.267 | 1.12x | 25.81 | 44.0% |
| 256 | 169.901 | 212.104 | 1.25x | 50.56 | 45.5% |
| 512 | 185.862 | 311.894 | 1.68x | 92.43 | 46.0% |
| 1024 | 239.550 | 498.038 | 2.08x | 143.43 | 58.6% |
| 4096 | 673.680 | 2498.584 | 3.71x | 204.01 | 83.4% |

Best fused configuration: tile_k=128 throughout, cluster_m=1 for M=128,
cluster_m=2 otherwise. Relative output RMSE is 2.58e-6 to 2.27e-5;
maximum absolute error is 0.03125. Every tested fused and baseline
configuration passes rtol=0.016, atol=0.002 against the reference.
Eager versus graph output, and replay after changing A and scales versus
eager output, agree bitwise. An additional M=128/256, clamp=0.5 run
exercises clipping and also passes, with zero error versus the reference.

SOL attainment is `max(MMA_time, compulsory_GMEM_time, tiled_L2_time) /
measured_time`. The model uses a 500 TF/s FP8 peak before clock scaling,
273 GB/s GMEM bandwidth, and 4*M*I*K GEMM FLOPs. Compulsory traffic is A,
packed B, both physical blocked-scale buffers, and output, each once.
The tiled read model includes A/SFA per N tile and B/SFB per M cluster,
accounting for SFB multicast. It excludes output-store sectors and launch,
SFU and conversion costs, so it is an analytical lower-bound model rather
than a complete execution-time prediction.

### L2 peak correction verified with NCU

The previous 1281 GB/s constant was incompatible with the locked 1575 MHz
GPC/LTS clock. Excluding L2 from the selected SOL to hide the >100% result
was incorrect. `benchmarks/common.py` now uses the NCU-confirmed
1024 bytes/LTS-cycle: 1612.8 GB/s at the 1575 MHz reference clock.
Full MMA/GMEM/L2 SOL selection is restored.
No percentages are capped at 100%.

NCU 2026.1.1, fixed M=4096, tile=128x128x128, cluster_m=2,
normalized FP8 weights (same 1/sqrt(K) distribution as the benchmark),
CUDA Graph replay, `--clock-control none --cache-control none
--pipeline-boost-state dynamic`:

| Metric | 1575 MHz lock | 468 MHz lock |
|---|---:|---:|
| `lts__cycles_elapsed.avg.per_second` | 1.574456 GHz | 467.960120 MHz |
| `lts__t_bytes.sum.peak_sustained_elapsed.per_second` | 1.612243 TB/s | 479.191163 GB/s |
| `lts__t_bytes.sum.pct_of_peak_sustained_elapsed` | 91.882732% | 96.274188% |

Both profiled runs give 1024 bytes/cycle. These are NCU profiling conditions.
The low-clock unprofiled run takes about 1.60 ms versus 2.03 ms profiled;
a GPC-scaled L2 denominator again produces >100% when mixed with unprofiled
timing. Automatic GPC-to-L2 scaling is therefore not justified by these
measurements and has not been retained. The shared model keeps the
maximum-clock L2 peak for low-clock runs, providing a conservative bound
instead of claiming a verified per-clock peak.

At 1575 MHz, NCU reports 28,015,243 read sectors (896.488 MB) and
1,050,345 write sectors (33.611 MB), using 32 bytes/sector. The analytical
905.970 MB tiled-read model is within 1.1% of the read counter. The wrong
bandwidth denominator is the main cause of the original >100% result.

The corrected M=4096 L2 lower bound is 561.7 us; measured cold-L2 benchmark
time is 673.7 us, giving 83.4% analytical SOL attainment. NCU's 91.88% is
its separate hardware counter utilization during a profiled launch;
profiling reports 627.9 us and is not substituted for unprofiled timing.
Cold L2 eviction precedes the selected profiled launch. No warm-L2 timing
is used for performance comparisons.

The earlier 71.55% NCU measurement used unnormalized random weights and
default profiler controls. That setup is superseded by the matching
normalized-weight Graph measurements above.

NCU reports are stored outside the repository:
`/tmp/ncu-reports/fp8_shared_fc1_sol_graph_normalized_1575mhz.ncu-rep` and
`/tmp/ncu-reports/fp8_shared_fc1_sol_graph_normalized_468mhz.ncu-rep`.
Ordinary benchmark log: `/tmp/fp8_shared_fc1_benchmark_corrected_sol.log`.
