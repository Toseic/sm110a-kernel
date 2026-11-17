# SM110A Kernel

A collection of CUDA kernels for LLM inference, targeting NVIDIA GPUs with
the `sm_110a` architecture. The project provides Python interfaces, JIT
compilation, autotuning, and benchmarks for developing and evaluating kernels.

It includes BF16, FP8, MXFP4, and NVFP4 matrix multiplication; dense and sparse
attention; mixture-of-experts operations; quantization; normalization; and RoPE.
DeepSeek model integrations demonstrate how these kernels fit into inference
workloads.

## Getting started

Use a Python environment with CUDA-enabled PyTorch, the CUDA toolkit (including
`nvcc`), and a compatible NVIDIA GPU. Run from the repository root:

```bash
python3 -c "import sm110a_kernel"
python3 benchmarks/benchmark_bf16_gemm.py --m 4096 --n 4096 --k 4096
```

Kernels are compiled on demand. The default JIT target is `sm_110a`.
Benchmarks measure cold-L2-cache performance by default.

## Repository layout

- `csrc/` and `include/`: CUDA implementations and shared headers.
- `sm110a_kernel/`: Python APIs, JIT compilation, and kernel runners.
- `benchmarks/`: correctness checks and performance comparisons.
- `models/`: model implementations and integrations.

See the READMEs under `csrc/` and `models/` for individual kernel and model
details, and [AGENTS.md](AGENTS.md) for development instructions.
