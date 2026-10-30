"""BF16 HC post + next HC pre + RMSNorm, retaining both BF16 rounding boundaries.

Two launches: post is evaluated inside split-K projection, then finish reduces
the projections and computes Sinkhorn/collapse/RMSNorm. The mixed residual is
returned because the next sublayer still needs it for its residual connection.
For D=4096 and at least 512 rows, a CTA produces all four post streams from
one set of source vectors; smaller shapes retain the decode-oriented path.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner
from .pdl import validate_pdl

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/hc_post_pre_norm/src"
_IMPL = _ROOT / "csrc/hc_pre_norm/src/hc_pre_norm_impl.cuh"


def _allocate_scratch(x, d):
    return torch.empty(
        (x.numel() // d, (4 * d + 511) // 512, 32), device=x.device, dtype=torch.float32
    )


@dataclass(frozen=True)
class HcPostPreNormConfig:
    d: int
    sinkhorn_iters: int = 20
    pdl: bool = False


class HcPostPreNormModule:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self.config = config

    def hc_post_pre_norm_out(
        self,
        x,
        residual,
        prev_post,
        prev_comb,
        weight,
        scale,
        base,
        norm_weight,
        mixed,
        output,
        post,
        comb,
        norm_eps=1e-6,
        hc_eps=1e-6,
        rms_eps=1e-6,
        *,
        scratch=None,
    ):
        if scratch is None:
            scratch = _allocate_scratch(x, self.config.d)
        self._binding.hc_post_pre_norm_out(
            x,
            residual,
            prev_post,
            prev_comb,
            weight,
            scale,
            base,
            norm_weight,
            mixed,
            output,
            post,
            comb,
            scratch,
            norm_eps,
            hc_eps,
            rms_eps,
            self.config.d,
            self._launcher,
        )
        return mixed, output, post, comb

    def hc_post_pre_norm(
        self,
        x,
        residual,
        prev_post,
        prev_comb,
        weight,
        scale,
        base,
        norm_weight,
        norm_eps=1e-6,
        hc_eps=1e-6,
        rms_eps=1e-6,
    ):
        prefix = x.shape[:-1]
        return self.hc_post_pre_norm_out(
            x,
            residual,
            prev_post,
            prev_comb,
            weight,
            scale,
            base,
            norm_weight,
            torch.empty_like(residual),
            torch.empty_like(x),
            torch.empty((*prefix, 4), device=x.device, dtype=torch.float32),
            torch.empty((*prefix, 4, 4), device=x.device, dtype=torch.float32),
            norm_eps,
            hc_eps,
            rms_eps,
        )


_JIT = SplitCudaJit(
    name="sm110a_hc_post_pre_norm",
    root=_ROOT,
    binding_sources=(_SOURCE / "hc_post_pre_norm_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "hc_post_pre_norm_kernel.cu",),
    kernel_dependencies=(_IMPL, _SOURCE / "hc_post_pre_norm_prefill.cuh"),
    kernel_symbol="sm110a_hc_post_pre_norm_launch",
    config_key=lambda c: (c.d, c.sinkhorn_iters, c.pdl),
    kernel_cflags=lambda c, _variant: (
        f"-DSM110A_PDL={int(c.pdl)}",
        f"-DSM110A_HC_PRE_NORM_D={c.d}",
        f"-DSM110A_HC_PRE_NORM_ITERS={c.sinkhorn_iters}",
    ),
    make_facade=HcPostPreNormModule,
)


def compile_hc_post_pre_norm(d, *, sinkhorn_iters=20, pdl=False):
    validate_pdl(pdl)
    if type(d) is not int or not 256 <= d <= 4096 or d % 32:
        raise ValueError("d must be a multiple of 32 in [256, 4096]")
    if type(sinkhorn_iters) is not int or not 1 <= sinkhorn_iters <= 100:
        raise ValueError("sinkhorn_iters must be an integer in [1, 100]")
    return _JIT.compile(HcPostPreNormConfig(d, sinkhorn_iters, pdl))


def hc_post_pre_norm(
    x,
    residual,
    prev_post,
    prev_comb,
    weight,
    scale,
    base,
    norm_weight,
    norm_eps=1e-6,
    hc_eps=1e-6,
    rms_eps=1e-6,
    *,
    sinkhorn_iters=20,
    pdl=False,
):
    return compile_hc_post_pre_norm(
        x.size(-1), sinkhorn_iters=sinkhorn_iters, pdl=pdl
    ).hc_post_pre_norm(
        x,
        residual,
        prev_post,
        prev_comb,
        weight,
        scale,
        base,
        norm_weight,
        norm_eps,
        hc_eps,
        rms_eps,
    )


def hc_post_pre_norm_out(
    x,
    residual,
    prev_post,
    prev_comb,
    weight,
    scale,
    base,
    norm_weight,
    mixed,
    output,
    post,
    comb,
    norm_eps=1e-6,
    hc_eps=1e-6,
    rms_eps=1e-6,
    *,
    sinkhorn_iters=20,
    pdl=False,
):
    return compile_hc_post_pre_norm(
        x.size(-1), sinkhorn_iters=sinkhorn_iters, pdl=pdl
    ).hc_post_pre_norm_out(
        x,
        residual,
        prev_post,
        prev_comb,
        weight,
        scale,
        base,
        norm_weight,
        mixed,
        output,
        post,
        comb,
        norm_eps,
        hc_eps,
        rms_eps,
    )


class HcPostPreNormRunner(FixedKernelRunner):
    """Allocation-free prepared launch; outputs must not alias inputs or each other."""

    def __init__(
        self,
        x,
        residual,
        prev_post,
        prev_comb,
        weight,
        scale,
        base,
        norm_weight,
        mixed,
        output,
        post,
        comb,
        norm_eps=1e-6,
        hc_eps=1e-6,
        rms_eps=1e-6,
        *,
        sinkhorn_iters=20,
        pdl=False,
    ):
        validate_pdl(pdl)
        scratch = _allocate_scratch(x, x.size(-1))
        super().__init__(
            x.device,
            "hc_post_pre_norm",
            (
                tuple(x.shape),
                str(x.dtype),
                norm_eps,
                hc_eps,
                rms_eps,
                sinkhorn_iters,
                pdl,
            ),
            lambda: compile_hc_post_pre_norm(
                x.size(-1), sinkhorn_iters=sinkhorn_iters, pdl=pdl
            ),
            lambda module: (
                lambda: module.hc_post_pre_norm_out(
                    x,
                    residual,
                    prev_post,
                    prev_comb,
                    weight,
                    scale,
                    base,
                    norm_weight,
                    mixed,
                    output,
                    post,
                    comb,
                    norm_eps,
                    hc_eps,
                    rms_eps,
                    scratch=scratch,
                )
            ),
        )
