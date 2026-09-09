import pytest


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_create_exact_sm_green_context_and_torch_stream():
    from sm110a_kernel.green_context import (
        create_green_contexts,
        sm_resource_info,
    )

    info = sm_resource_info()
    sm_count = info["sm_coscheduled_alignment"]
    if sm_count < info["min_sm_partition_size"]:
        pytest.skip("device does not expose an SM partition at its alignment")
    requested_counts = (
        (sm_count, sm_count)
        if info["sm_count"] >= 2 * sm_count
        else (sm_count,)
    )
    contexts = create_green_contexts(requested_counts)
    try:
        context = contexts[0]
        assert tuple(context.sm_count for context in contexts) == requested_counts
        assert context.sm_count == sm_count
        stream = context.torch_stream()
        with torch.cuda.stream(stream):
            value = torch.ones(1, device="cuda")
            value.add_(1)
        context.synchronize()
        assert value.item() == 2
    finally:
        for context in contexts:
            context.close()


def test_split_green_contexts_rejects_misaligned_sm_count():
    from sm110a_kernel.green_context import create_green_context

    with pytest.raises(RuntimeError, match="minimum partition size|alignment"):
        create_green_context(1)
