# HC pre launch modes

`hc_pre`, `hc_pre_out`, `compile_hc_pre`, and `HcPreRunner` default to
`single_kernel=False`, using the original two-launch BF16 tensor-core path:
projection followed by gates, Sinkhorn, and collapse.
FP16/FP32 and unsupported tensor-core shapes use the existing SIMT fallback.

Pass `single_kernel=True` to opt into the experimental single-launch version.
For one BF16 token it uses SIMT split-K=1024; otherwise supported BF16 shapes
use tensor cores with 32 active tokens per tile and split-K=512. Each CTA
publishes its partials, and the last CTA computes gates, Sinkhorn, and collapse.

Single-launch scratch arrival counters must initially be zero; the module and
runner allocate zeroed scratch automatically for this mode. The last CTA resets
its counter before returning, so serial graph replay can reuse scratch.
Concurrent launches must use separate scratch buffers.

The benchmark measures the default two-stage path and the optional single-launch
path on identical inputs with cold L2. `fusion_speedup` is two-stage latency
divided by single-launch latency; values above 1 favor single launch. The Torch
speedup and SOL comparison use the default two-stage path.

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_hc_pre.py --tokens 1 16 128 512 1024 --warmup 5 --iters 20 --samples 5
```

Single launch has improved one-token latency in measurements, but larger token
counts still regress because finalization is concentrated in fewer CTAs.
The HC pre + RMSNorm implementation remains separate.
