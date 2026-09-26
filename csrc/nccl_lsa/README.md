# Minimal NCCL LSA communication kernel

This kernel is a deliberately small NCCL Device API example. Each rank owns a
symmetric float32 buffer. One CTA synchronizes the load/store-accessible (LSA)
team, loads every rank's value, sums it, and writes the result back to every
rank before a release barrier.

The PyTorch-facing `NcclLsaAllReduce` copies a contiguous input into the
symmetric buffer, launches the communication kernel on the current PyTorch
stream, and copies the result into the requested output tensor. It is intended
as a starting point for compute/communication fusion, not as a peak-bandwidth
collective.

All ranks must:

- construct with the same unique ID, world size, and capacity;
- invoke collectives with identical element counts and ordering;
- call `close()` after outstanding work has completed.

For two ranks sharing one physical GPU, start CUDA MPS and set:

```bash
export NCCL_MULTI_RANK_GPU_ENABLE=1
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
```

Then run `benchmarks/benchmark_nccl_lsa.py`.
