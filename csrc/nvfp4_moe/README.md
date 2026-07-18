# NVFP4 MoE FC1

`nvfp4_fc1_kernel.cu` is an independent routed FC1 implementation derived
from the NVFP4 grouped-M GEMM pipeline. It consumes unpermuted token rows and
fuses `source_rows` gathering for both packed E2M1 activations and their
blocked UE4M3 scales into the mainloop producer stage.

The public Python API mirrors the BF16 FC1 metadata/main split:

```python
metadata = prepare_nvfp4_fc1_metadata(router_scores, topk, n, config)
out = nvfp4_fc1(aq, bq, sfa, sfb, metadata, config)
```

The current fused gather path supports `cluster_m=1`, tile N of 64 or 128,
tile K of 64, 128, or 256, and all three grouped scheduler modes.
