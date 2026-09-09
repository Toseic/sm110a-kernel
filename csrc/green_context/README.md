# CUDA Green Context helpers

`sm110a_kernel.green_context` provides a small PyTorch-facing wrapper over the
CUDA Driver Green Context API.  It creates exact, disjoint SM partitions and
an owned nonblocking stream for every partition.

```python
import torch
from sm110a_kernel import create_green_contexts, sm_resource_info

info = sm_resource_info()
alignment = info["sm_coscheduled_alignment"]
contexts = create_green_contexts((2 * alignment, 4 * alignment))
try:
    with torch.cuda.stream(contexts[0].torch_stream()):
        output = model(input)
    contexts[0].synchronize()
finally:
    for context in contexts:
        context.close()
```

`sm_counts` are exact counts, not targets: each must satisfy the device's
`min_sm_partition_size` and `sm_coscheduled_alignment` returned by
`sm_resource_info()`.  SMs not assigned to a helper-created Green Context are
left for the primary PyTorch context.  A context must outlive all work issued
to its stream; `close()` destroys that stream and releases the SM partition.
