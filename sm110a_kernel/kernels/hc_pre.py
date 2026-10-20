"""Decode HC pre: fused FP32 projection/RMS, split-Sinkhorn and HC collapse."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/hc_pre/src"


@dataclass(frozen=True)
class HcPreConfig:
    d: int
    sinkhorn_iters: int = 20


class HcPreModule:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self.config = config

    def hc_pre_out(
        self, input, weight, scale, base, output, post, comb, norm_eps=1e-6, hc_eps=1e-6
    ):
        self._binding.hc_pre_out(
            input,
            weight,
            scale,
            base,
            output,
            post,
            comb,
            norm_eps,
            hc_eps,
            self.config.d,
            self._launcher,
        )
        return output, post, comb

    def hc_pre(self, input, weight, scale, base, norm_eps=1e-6, hc_eps=1e-6):
        prefix = input.shape[:-2]
        output = torch.empty(
            (*prefix, self.config.d), device=input.device, dtype=input.dtype
        )
        post = torch.empty((*prefix, 4), device=input.device, dtype=torch.float32)
        comb = torch.empty((*prefix, 4, 4), device=input.device, dtype=torch.float32)
        return self.hc_pre_out(
            input, weight, scale, base, output, post, comb, norm_eps, hc_eps
        )


_JIT = SplitCudaJit(
    name="sm110a_hc_pre",
    root=_ROOT,
    binding_sources=(_SOURCE / "hc_pre_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "hc_pre_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_hc_pre_launch",
    config_key=lambda c: (c.d, c.sinkhorn_iters),
    kernel_cflags=lambda c, _variant: (
        f"-DSM110A_HC_PRE_D={c.d}",
        f"-DSM110A_HC_PRE_ITERS={c.sinkhorn_iters}",
    ),
    make_facade=HcPreModule,
)


def compile_hc_pre(d, *, sinkhorn_iters=20):
    """Fixed decode configuration; HC=4, contiguous BF16/FP16/FP32, FP32 weights."""
    if type(d) is not int or not 1 <= d <= 4096:
        raise ValueError("d must be an integer in [1, 4096]")
    if type(sinkhorn_iters) is not int or not 1 <= sinkhorn_iters <= 100:
        raise ValueError("sinkhorn_iters must be an integer in [1, 100]")
    return _JIT.compile(HcPreConfig(d, sinkhorn_iters))


def hc_pre(
    input,
    weight,
    scale,
    base,
    norm_eps=1e-6,
    hc_eps=1e-6,
    *,
    sinkhorn_iters=20,
):
    return compile_hc_pre(
        input.size(-1),
        sinkhorn_iters=sinkhorn_iters,
    ).hc_pre(input, weight, scale, base, norm_eps, hc_eps)


def hc_pre_out(
    input,
    weight,
    scale,
    base,
    output,
    post,
    comb,
    norm_eps=1e-6,
    hc_eps=1e-6,
    *,
    sinkhorn_iters=20,
):
    return compile_hc_pre(
        input.size(-1),
        sinkhorn_iters=sinkhorn_iters,
    ).hc_pre_out(input, weight, scale, base, output, post, comb, norm_eps, hc_eps)


class HcPreRunner(FixedKernelRunner):
    def __init__(
        self,
        input,
        weight,
        scale,
        base,
        output,
        post,
        comb,
        norm_eps=1e-6,
        hc_eps=1e-6,
        *,
        sinkhorn_iters=20,
    ):
        super().__init__(
            input.device,
            "hc_pre",
            (
                tuple(input.shape),
                str(input.dtype),
                norm_eps,
                hc_eps,
                sinkhorn_iters,
            ),
            lambda: compile_hc_pre(input.size(-1), sinkhorn_iters=sinkhorn_iters),
            lambda module: (
                lambda: module.hc_pre_out(
                    input, weight, scale, base, output, post, comb, norm_eps, hc_eps
                )
            ),
        )
