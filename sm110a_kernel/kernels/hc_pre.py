"""HC pre and head with fused projection, gates and collapse."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/hc_pre/src"
_SPLIT_K = 512


def _allocate_scratch(input, d):
    return torch.empty(
        (input.numel() // (4 * d), (4 * d + _SPLIT_K - 1) // _SPLIT_K, 32),
        device=input.device,
        dtype=torch.float32,
    )


@dataclass(frozen=True)
class HcPreConfig:
    d: int
    sinkhorn_iters: int = 20
    head: bool = False


class HcPreModule:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self.config = config

    def hc_pre_out(
        self,
        input,
        weight,
        scale,
        base,
        output,
        post=None,
        comb=None,
        norm_eps=1e-6,
        hc_eps=1e-6,
        *,
        scratch=None,
    ):
        if scratch is None and not self.config.head:
            scratch = _allocate_scratch(input, self.config.d)
        self._binding.hc_pre_out(
            input,
            weight,
            scale,
            base,
            output,
            post,
            comb,
            scratch,
            norm_eps,
            hc_eps,
            self.config.d,
            self.config.head,
            self._launcher,
        )
        return output if self.config.head else (output, post, comb)

    def hc_pre(self, input, weight, scale, base, norm_eps=1e-6, hc_eps=1e-6):
        prefix = input.shape[:-2]
        output = torch.empty(
            (*prefix, self.config.d), device=input.device, dtype=input.dtype
        )
        post = comb = None
        if not self.config.head:
            post = torch.empty((*prefix, 4), device=input.device, dtype=torch.float32)
            comb = torch.empty(
                (*prefix, 4, 4), device=input.device, dtype=torch.float32
            )
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
    config_key=lambda c: (c.d, c.sinkhorn_iters, c.head),
    kernel_cflags=lambda c, _variant: (
        f"-DSM110A_HC_PRE_D={c.d}",
        f"-DSM110A_HC_PRE_ITERS={c.sinkhorn_iters}",
        f"-DSM110A_HC_PRE_HEAD={int(c.head)}",
    ),
    make_facade=HcPreModule,
)


def compile_hc_pre(d, *, sinkhorn_iters=20, head=False):
    """Compile HC=4 pre or head; head uses FP32 [4, 4*D]/[1]/[4] parameters."""
    if type(d) is not int or not 1 <= d <= 4096:
        raise ValueError("d must be an integer in [1, 4096]")
    if type(sinkhorn_iters) is not int or not 1 <= sinkhorn_iters <= 100:
        raise ValueError("sinkhorn_iters must be an integer in [1, 100]")
    if type(head) is not bool:
        raise TypeError("head must be bool")
    return _JIT.compile(HcPreConfig(d, 1 if head else sinkhorn_iters, head))


def hc_pre(
    input,
    weight,
    scale,
    base,
    norm_eps=1e-6,
    hc_eps=1e-6,
    *,
    sinkhorn_iters=20,
    head=False,
):
    return compile_hc_pre(
        input.size(-1),
        sinkhorn_iters=sinkhorn_iters,
        head=head,
    ).hc_pre(input, weight, scale, base, norm_eps, hc_eps)


def hc_pre_out(
    input,
    weight,
    scale,
    base,
    output,
    post=None,
    comb=None,
    norm_eps=1e-6,
    hc_eps=1e-6,
    *,
    sinkhorn_iters=20,
    head=False,
):
    return compile_hc_pre(
        input.size(-1),
        sinkhorn_iters=sinkhorn_iters,
        head=head,
    ).hc_pre_out(input, weight, scale, base, output, post, comb, norm_eps, hc_eps)


class HcPreRunner(FixedKernelRunner):
    def __init__(
        self,
        input,
        weight,
        scale,
        base,
        output,
        post=None,
        comb=None,
        norm_eps=1e-6,
        hc_eps=1e-6,
        *,
        sinkhorn_iters=20,
        head=False,
    ):
        scratch = None if head else _allocate_scratch(input, input.size(-1))
        super().__init__(
            input.device,
            "hc_pre",
            (
                tuple(input.shape),
                str(input.dtype),
                norm_eps,
                hc_eps,
                1 if head else sinkhorn_iters,
                head,
            ),
            lambda: compile_hc_pre(
                input.size(-1), sinkhorn_iters=sinkhorn_iters, head=head
            ),
            lambda module: (
                lambda: module.hc_pre_out(
                    input,
                    weight,
                    scale,
                    base,
                    output,
                    post,
                    comb,
                    norm_eps,
                    hc_eps,
                    scratch=scratch,
                )
            ),
        )
