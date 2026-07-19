# NVFP4 MoE FC1

`nvfp4_fc1_kernel.cu` is an independent routed FC1 implementation derived
from the NVFP4 grouped-M GEMM pipeline. It consumes unpermuted token rows and
fuses `source_rows` gathering for both packed E2M1 activations and their
blocked UE4M3 scales into the mainloop producer stage.

Weights use the same 32-column block-interleaved `[gate, up]` layout as the
BF16 FC1 path. The main kernel applies `silu(gate) * up` in its epilogue, so
the result width is half the physical weight width. The public Python API
mirrors the BF16 FC1 metadata/main split (pass the physical, interleaved N to
metadata preparation):

```python
metadata = prepare_nvfp4_fc1_metadata(router_scores, topk, 2 * hidden, config)
out = nvfp4_fc1(aq, bq, sfa, sfb, metadata, config)
```

The fused gather path supports `cluster_m=1` and `cluster_m=2` (with B/SFB
cluster multicast), tile N of 64, 128, or 256, tile K of 64, 128, or 256,
and all three grouped scheduler modes.  Tasks use the same configurable
N-major/serpentine macro traversal as grouped-M GEMM.  When all 128 routed
rows in a task are consecutive, FC1 switches A to a swizzled TMA load; sparse
or fragmented routing continues to use the vectorized `cp.async` gather.
