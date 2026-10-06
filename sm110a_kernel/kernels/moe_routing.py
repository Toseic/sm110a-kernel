"""Reusable routing metadata construction for routed-MoE kernels.

The packed layout is ``counts[E], offsets[E], grouped_layout[M],
grouped_assignments[M], assignment_to_grouped[M]``.
"""

from __future__ import annotations

import torch


def pack_routing_metadata(selected_experts: torch.Tensor, experts: int) -> torch.Tensor:
    """Pack stable expert grouping metadata from token-major assignments."""
    if selected_experts.ndim != 1 or selected_experts.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            "selected_experts must be a 1D torch.int32 or torch.int64 tensor"
        )
    if experts <= 0:
        raise ValueError("experts must be positive")
    selected = selected_experts.to(dtype=torch.int32)
    counts = torch.bincount(selected, minlength=experts).to(dtype=torch.int32)
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32) - counts
    grouped_assignments = torch.argsort(selected, stable=True).to(dtype=torch.int32)
    grouped_layout = selected.index_select(0, grouped_assignments)
    assignment_to_grouped = torch.empty_like(grouped_assignments)
    assignment_to_grouped[grouped_assignments] = torch.arange(
        selected.numel(), device=selected.device, dtype=torch.int32
    )
    return torch.cat(
        (counts, offsets, grouped_layout, grouped_assignments, assignment_to_grouped)
    )
