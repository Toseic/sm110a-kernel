# DeepSeek-V4-Flash, 32-GiB reference runtime

This directory contains the official DeepSeek-V4-Flash reference inference
modules, copied from `deepseek-ai/DeepSeek-V4-Flash` revision
`60d8d70770c6776ff598c94bb586a859a38244f1`.  The only model alteration is
the local `benchmark_config_32g.json`; the files in `inference/` and `encoding/` are
unmodified official code.

The upstream 43-layer model cannot fit in a 32-GiB GPU.  This benchmark uses
6 transformer layers (14% of the original), disables the unused MTP auxiliary
layer, and limits the runtime to a batch of one and 2,048 tokens.  It retains
the original hidden size, attention layout, 256 routed FP4 experts per layer,
and FP8 dense weights, so kernel shapes and quantization paths match the
official model.

Run the non-allocating estimate before launching:

```bash
/home/bifrost/miniforge3/bin/python3 models/deepseek_v4_flash/benchmark_utils.py
```

The default six-layer configuration reports 24.18 GiB model-resident storage.
That total does not include CUDA workspaces or temporary activations. The
configuration reserves 6 GiB for those allocations (30.18 GiB total) and must
remain below 32 GiB. Lower `n_layers` if the local PyTorch/CUDA stack requires
more workspace.

The benchmark entry point supports direct random-weight kernel runs and a
converted official checkpoint.  Its shape options are `--batch-size` / `--bs`,
`--seq-len` / `--seq`, `--prefill-len` / `--p`, and `--decode-len` / `--d`.
For example:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py \
  --bs 1 --p 1024 --d 128 --iters 5
```

Random weights are the default; pass `--checkpoint /path/to/converted` to load
official converted weights instead.  The script performs a meta-device memory
check before allocating CUDA memory. Use `--dry-run` to inspect a shape without
loading weights.

Pass `--sol` to print the same HBM roofline style used by the other benchmarks.
It counts model-weight reads (one full read for prefill and one per decode step)
plus persistent KV-cache reads selected by the official sparse-attention path.
It excludes KV writes, activations, inputs/outputs, and compute.

For decode throughput at a large context without allocating a matching large
prefill activation, use `--decode-only --seq 4096`. This starts decode from
the allocated zero-initialized KV buffers and exercises the same sparse KV
gather path; it is not a numerical end-to-end generation test.

Convert an official HF checkpoint, then invoke the original `generate.py`
through the wrapper:

```bash
cd models/deepseek_v4_flash/inference
/home/bifrost/miniforge3/bin/python3 convert.py --hf-ckpt-path /path/to/hf --save-path /path/to/converted \
  --n-experts 256 --model-parallel 1 --expert-dtype fp4

cd ../../..
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_dsv4_flash.py \
  --checkpoint /path/to/converted --p 1024 --d 128
```

The full HF checkpoint is only needed during conversion; no model weights are
stored in this repository.  The current repository ignores `models/`, but this
benchmark directory is tracked normally.
