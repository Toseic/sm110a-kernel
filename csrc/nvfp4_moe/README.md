# NVFP4 MoE FC1

`nvfp4_fc1_kernel.cu` is an independent routed FC1 implementation derived
from the NVFP4 grouped-M GEMM pipeline. It consumes unpermuted token rows and
fuses `source_rows` gathering for both packed E2M1 activations and their
blocked UE4M3 scales into the mainloop producer stage.

Weights use a 64-column block-interleaved `[gate, up]` layout. The main kernel
applies `silu(gate) * up` and quantizes it to packed E2M1 plus blocked UE4M3
scales in the epilogue, so the logical result width is half the physical
weight width. The public Python API
mirrors the BF16 FC1 metadata/main split (pass the physical, interleaved N to
metadata preparation):

```python
metadata = prepare_nvfp4_fc1_metadata(router_scores, topk, 2 * hidden, config)
out, out_scale = nvfp4_fc1(aq, bq, sfa, sfb, metadata, config)
```

The fused gather path supports `cluster_m=1` and `cluster_m=2` (with B/SFB
cluster multicast). Only `cluster_n=1` is supported.
Tile K may be 64, 128, or 256. Tasks support configurable N-major or
M-major macro traversal without serpentine ordering. When all 128 routed
rows in a task are consecutive, FC1 switches A to a swizzled TMA load; sparse
or fragmented routing continues to use the vectorized `cp.async` gather.

Routing metadata is produced entirely on GPU with a cooperative stable top-k,
stable expert compaction, and a parallel per-expert task builder. The mainloop
automatically uses the maximum stage count that fits in shared memory. Scheduler
traversal and direct/TMA epilogue are available to the benchmark autotuner.
SiLU always uses the precise implementation.

## FC1 → FC2 PDL

Standalone FC1/FC2 configs accept `pdl=True`. Call the FC2 module's
`prepare_pdl_out(output, metadata)` before FC1, then launch FC1 and FC2
adjacently on the same stream. Use `benchmarks/benchmark_pdl.py --timeline`
to inspect overlap; include output preparation before FC1 on every replay.
