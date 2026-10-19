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
rounding. It uses row scheduling with cluster_m=1 and statically compacted
weight scales, releasing the padded scale buffers after preparation.
It currently uses BF16 atomic expert accumulation, so its default reduction is
`atomic`; explicitly requesting deterministic reduction or PDL is rejected.
Use `--moe-backend unfused --fc2-reduction atomic` for a comparison with the same
reduction semantics. Fused output can vary between replays and is validated
with a numerical tolerance rather than bitwise equality. It supports positive
hidden dimensions divisible by 256 and expert intermediate dimension 2048.
Routing task counts are read on device each invocation, including graph replay.

The default path is bitwise reproducible between eager and CUDA Graph in the
tested decode runs, but full-model agreement with the official model is not
yet validated. See [precision re-evaluation](../../docs/dsv4_flash_accuracy.md)
for measured residual differences and controlled experiments.

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
