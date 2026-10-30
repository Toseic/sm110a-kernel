"""Row RMSNorm with a single global input read and register retention."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner
from .pdl import validate_pdl

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/rmsnorm/src"


@dataclass(frozen=True)
class RmsNormConfig:
    d: int
    warps_per_row: int
    rows_per_cta: int
    pdl: bool = False


class RmsNormModule:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self.config = config

    def rmsnorm_out(self, input, weight, output, eps=1e-6):
        if weight is None:
            raise ValueError(
                "RMSNorm requires weight; use qk_norm for weightless normalization"
            )
        return self._binding.rmsnorm_out(
            input, weight, output, eps, self.config.d, self._launcher
        )

    def rmsnorm(self, input, weight, eps=1e-6):
        return self.rmsnorm_out(input, weight, torch.empty_like(input), eps)


_JIT = SplitCudaJit(
    name="sm110a_rmsnorm",
    root=_ROOT,
    binding_sources=(_SOURCE / "rmsnorm_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "rmsnorm_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_rmsnorm_launch",
    config_key=lambda c: (c.d, c.warps_per_row, c.rows_per_cta, c.pdl),
    kernel_cflags=lambda c, _variant: (
        f"-DSM110A_PDL={int(c.pdl)}",
        f"-DSM110A_RMSNORM_D={c.d}",
        f"-DSM110A_RMSNORM_WARPS={c.warps_per_row}",
        f"-DSM110A_RMSNORM_ROWS={c.rows_per_cta}",
    ),
    make_facade=RmsNormModule,
)


def compile_rmsnorm(d, *, warps_per_row=None, rows_per_cta=None, pdl=False):
    """Compile a fixed D/config; no autotuning or device work at selection time.

    Supports D in [1, 32768], and at most 128 input values per thread.
    For vectorizable D, tensors must have 16-byte aligned base addresses.
    """
    return _JIT.compile(_rmsnorm_config(d, warps_per_row, rows_per_cta, pdl))


def _rmsnorm_config(d, warps_per_row=None, rows_per_cta=None, pdl=False):
    validate_pdl(pdl)
    if type(d) is not int or not 1 <= d <= 32768:
        raise ValueError("d must be an integer in [1, 32768]")
    if warps_per_row is None:
        warps_per_row = 1 if d <= 1024 else 4 if d <= 2048 else 8
    if type(warps_per_row) is not int or warps_per_row not in (1, 2, 4, 8):
        raise ValueError("warps_per_row must be 1, 2, 4 or 8")
    if rows_per_cta is None:
        rows_per_cta = 4 if d <= 8192 else 2 if d <= 16384 else 1
    if type(rows_per_cta) is not int or rows_per_cta not in (1, 2, 4, 8):
        raise ValueError("rows_per_cta must be 1, 2, 4 or 8")
    if warps_per_row * rows_per_cta > 32:
        raise ValueError("a CTA may contain at most 32 warps (1024 threads)")
    if d > warps_per_row * 32 * 128:
        raise ValueError("configuration exceeds 128 input values per thread")
    # Reserve room for addresses, reduction and epilogue temporaries even in
    # FP32, where inputs cannot be packed. Large CTAs must not force spills.
    values_per_thread = (
        ((d + warps_per_row * 32 - 1) // (warps_per_row * 32) + 7) // 8 * 8
    )
    if (values_per_thread + 32) * warps_per_row * 32 * rows_per_cta > 65536:
        raise ValueError("configuration exceeds the conservative register budget")
    return RmsNormConfig(d, warps_per_row, rows_per_cta, pdl)


def rmsnorm(
    input, weight, eps=1e-6, *, warps_per_row=None, rows_per_cta=None, pdl=False
):
    """Compute x * rsqrt(mean(x**2) + eps) * weight along the last axis.

    Input is contiguous CUDA BF16/FP16/FP32. Weight is [D], normally FP32
    as in DeepSeek-V4-Flash; input dtype weights are also supported.
    Reduction and both multiplies use FP32; output is cast once at the end.
    """
    return compile_rmsnorm(
        input.size(-1), warps_per_row=warps_per_row, rows_per_cta=rows_per_cta, pdl=pdl
    ).rmsnorm(input, weight, eps)


def rmsnorm_out(
    input, weight, output, eps=1e-6, *, warps_per_row=None, rows_per_cta=None, pdl=False
):
    """Allocation-free RMSNorm; exact input/output aliasing is supported."""
    return compile_rmsnorm(
        input.size(-1), warps_per_row=warps_per_row, rows_per_cta=rows_per_cta, pdl=pdl
    ).rmsnorm_out(input, weight, output, eps)


class RmsNormRunner(FixedKernelRunner):
    def __init__(
        self,
        input,
        weight,
        output,
        eps=1e-6,
        *,
        warps_per_row=None,
        rows_per_cta=None,
        pdl=False,
    ):
        validate_pdl(pdl)
        super().__init__(
            input.device,
            "rmsnorm",
            (
                tuple(input.shape),
                str(input.dtype),
                str(weight.dtype),
                eps,
                warps_per_row,
                rows_per_cta,
                pdl,
            ),
            lambda: compile_rmsnorm(
                input.size(-1),
                warps_per_row=warps_per_row,
                rows_per_cta=rows_per_cta,
                pdl=pdl,
            ),
            lambda module: lambda: module.rmsnorm_out(input, weight, output, eps),
        )
