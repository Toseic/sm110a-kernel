"""Fused sqrt(softplus) MoE gate, selection, normalization and scaling."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from .pdl import validate_pdl

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/fused_gate/src"


@dataclass(frozen=True)
class FusedGateConfig:
    experts: int
    topk: int
    pdl: bool = False


class FusedGateModule:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self.config = config

    def fused_gate_out(
        self, logits, weights, indices, route_scale=1.0, *, bias=None, hash_indices=None
    ):
        self._binding.fused_gate_out(
            logits,
            bias,
            hash_indices,
            weights,
            indices,
            route_scale,
            self.config.experts,
            self.config.topk,
            self._launcher,
        )
        return weights, indices

    def fused_gate(self, logits, route_scale=1.0, *, bias=None, hash_indices=None):
        shape = (logits.size(0), self.config.topk)
        weights = torch.empty(shape, device=logits.device, dtype=torch.float32)
        indices = torch.empty(
            shape,
            device=logits.device,
            dtype=torch.int64 if hash_indices is None else hash_indices.dtype,
        )
        return self.fused_gate_out(
            logits, weights, indices, route_scale, bias=bias, hash_indices=hash_indices
        )


_JIT = SplitCudaJit(
    name="sm110a_fused_gate",
    root=_ROOT,
    binding_sources=(_SOURCE / "fused_gate_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "fused_gate_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_fused_gate_launch",
    config_key=lambda c: (c.experts, c.topk, c.pdl),
    kernel_cflags=lambda c, _: (
        f"-DSM110A_GATE_EXPERTS={c.experts}",
        f"-DSM110A_GATE_TOPK={c.topk}",
        f"-DSM110A_PDL={int(c.pdl)}",
    ),
    make_facade=FusedGateModule,
)


def compile_fused_gate(experts, topk, *, pdl=False):
    """Fixed gate kernel for 1..1024 experts and 1..32 selected IDs.

    Equal selection scores use ascending expert ID (torch.topk tie order is
    unspecified). Bias affects selection only. Hash IDs must be in range;
    invalid IDs produce NaN weights without an out-of-bounds read.
    """
    validate_pdl(pdl)
    if type(experts) is not int or not 1 <= experts <= 1024:
        raise ValueError("experts must be an integer in [1, 1024]")
    if type(topk) is not int or not 1 <= topk <= min(experts, 32):
        raise ValueError("topk must be an integer in [1, min(experts, 32)]")
    return _JIT.compile(FusedGateConfig(experts, topk, pdl))


def fused_gate(
    logits, topk, route_scale=1.0, *, bias=None, hash_indices=None, pdl=False
):
    """Return FP32 weights and expert IDs, preserving hash ID dtype/order.

    Computes (sqrt(softplus(selected logits)) / selected_score_sum) * scale.
    Softplus uses beta=1, threshold=20. Reduction uses FP32; last-bit differences
    from eager PyTorch are possible. Hash mode ignores selection bias.
    """
    return compile_fused_gate(logits.size(-1), topk, pdl=pdl).fused_gate(
        logits,
        route_scale,
        bias=bias,
        hash_indices=hash_indices,
    )
