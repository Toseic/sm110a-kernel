"""Weightless per-head RMS normalization, preserving Torch dtype boundaries."""

from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner
from .rmsnorm import _rmsnorm_config

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/rmsnorm/src"


class QkNormModule:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self.config = config

    def qk_norm_out(self, input, output, eps=1e-6):
        return self._binding.rmsnorm_out(
            input, None, output, eps, self.config.d, self._launcher
        )

    def qk_norm(self, input, eps=1e-6):
        return self.qk_norm_out(input, torch.empty_like(input), eps)


_JIT = SplitCudaJit(
    name="sm110a_qk_norm",
    root=_ROOT,
    binding_sources=(_SOURCE / "rmsnorm_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "rmsnorm_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_qk_norm_launch",
    config_key=lambda c: (c.d, c.warps_per_row, c.rows_per_cta, c.pdl),
    kernel_cflags=lambda c, _variant: (
        "-DSM110A_QK_NORM=1",
        f"-DSM110A_PDL={int(c.pdl)}",
        f"-DSM110A_RMSNORM_D={c.d}",
        f"-DSM110A_RMSNORM_WARPS={c.warps_per_row}",
        f"-DSM110A_RMSNORM_ROWS={c.rows_per_cta}",
    ),
    make_facade=QkNormModule,
)


def compile_qk_norm(d, *, warps_per_row=None, rows_per_cta=None, pdl=False):
    return _JIT.compile(_rmsnorm_config(d, warps_per_row, rows_per_cta, pdl))


def qk_norm(input, eps=1e-6, *, warps_per_row=None, rows_per_cta=None, pdl=False):
    """Normalize each last-axis row without a learned weight.

    Matches x * rsqrt(mean(x.square(), -1, keepdim=True) + eps), including
    input-dtype rounding after square, mean, epsilon addition and rsqrt.
    CUDA BF16/FP16/FP32 contiguous tensors are supported, with 1 <= D <= 32768.
    """
    return compile_qk_norm(
        input.size(-1), warps_per_row=warps_per_row, rows_per_cta=rows_per_cta, pdl=pdl
    ).qk_norm(input, eps)


def qk_norm_out(
    input, output, eps=1e-6, *, warps_per_row=None, rows_per_cta=None, pdl=False
):
    """Allocation-free variant; exact input/output aliasing is supported."""
    return compile_qk_norm(
        input.size(-1), warps_per_row=warps_per_row, rows_per_cta=rows_per_cta, pdl=pdl
    ).qk_norm_out(input, output, eps)


class QkNormRunner(FixedKernelRunner):
    def __init__(
        self,
        input,
        output,
        eps=1e-6,
        *,
        warps_per_row=None,
        rows_per_cta=None,
        pdl=False,
    ):
        config = _rmsnorm_config(input.size(-1), warps_per_row, rows_per_cta, pdl)
        super().__init__(
            input.device,
            "qk_norm",
            (tuple(input.shape), str(input.dtype), eps, config),
            lambda: _JIT.compile(config),
            lambda module: lambda: module.qk_norm_out(input, output, eps),
        )
