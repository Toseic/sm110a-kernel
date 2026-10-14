# DeepSeek-V4-Flash SM110A runtime

This directory contains the repository-native DeepSeek-V4-Flash runtime.  It
keeps checkpoint compatibility with the official readable implementation in
`../deepseek_v4_flash/inference`, but executes supported operators with the
SM110A kernels from `sm110a_kernel`.

The 32-GiB benchmark configuration contains one ratio-4 CSA layer and one
ratio-128 HCA layer. Routed experts use the fused W4A8 `mxfp4_fc1` and
`mxfp4_fc2` kernels directly. Dense projections, activation quantization and
sparse MLA use the corresponding SM110A kernels. Missing fused model-level
operators currently use explicit PyTorch fallbacks marked with
`TODO(sm110a)` in `model.py`.

The default expert path applies FP32 routing weights inside FC1 before
activation quantization, with the official BF16 rounding boundaries. FC2
uses a fixed-order FP32 reduction and rounds to BF16 once after summing the
expert outputs. `--routing-weight-position fc2 --fc2-reduction atomic`
selects the previous output-weighted path for comparison.

The default path is bitwise reproducible between eager and CUDA Graph in the
tested 128-step run, but full-model agreement with the official model is not
yet validated. See [precision re-evaluation](../../docs/dsv4_flash_accuracy.md)
for measured residual differences and controlled experiments.

The CSA indexer uses the native SM110A row top-k kernel for both prefill and
decode. Its score producer currently retains the reference BF16/QAT path.

Decode is always captured as one fixed CUDA Graph and benchmarked through
graph replay. Run the SM110A model with:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py
```
