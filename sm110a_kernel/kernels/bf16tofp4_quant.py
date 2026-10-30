"""BF16 to block-scaled E2M1/E8M0 (MXFP4-style) activation quantization."""

from __future__ import annotations

import ctypes
from pathlib import Path

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner
from .pdl import validate_pdl

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16tofp4_quant/src"
_INCLUDE_DIR = _ROOT / "include"


class Bf16ToFp4QuantModule:
    def __init__(self, binding, library, _config):
        self._binding = binding
        self._launcher = library.entry_address
        dequant_symbol = getattr(
            library.handle, "sm110a_bf16tofp4_quant_dequant_launch"
        )
        self._dequant_launcher = ctypes.cast(dequant_symbol, ctypes.c_void_p).value

    def quantize(self, input):
        return self._binding.bf16tofp4_quant(input, self._launcher)

    def quantize_out(self, input, packed, scales):
        return self._binding.bf16tofp4_quant_out(input, packed, scales, self._launcher)

    def quantize_dequant(self, input):
        return self._binding.bf16tofp4_quant_dequant(input, self._dequant_launcher)

    def quantize_dequant_out(self, input, output, scales):
        return self._binding.bf16tofp4_quant_dequant_out(
            input, output, scales, self._dequant_launcher
        )


_JIT = SplitCudaJit(
    name="sm110a_bf16tofp4_quant",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16tofp4_quant_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16tofp4_quant_kernel.cu",),
    kernel_dependencies=(_INCLUDE_DIR / "bf16tofp4_quant.h", _INCLUDE_DIR / "common.h"),
    kernel_symbol="sm110a_bf16tofp4_quant_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda pdl, _variant: (f"-DSM110A_PDL={int(pdl)}",),
    kernel_ldflags=("-lcuda",),
    make_facade=Bf16ToFp4QuantModule,
    include_paths=(_INCLUDE_DIR,),
)


def compile_bf16tofp4_quant(*, pdl=False) -> Bf16ToFp4QuantModule:
    validate_pdl(pdl)
    return _JIT.compile(pdl)


def bf16tofp4_quant(input, *, pdl=False):
    """Return packed E2M1 values and one E8M0 scale per contiguous 32-value block."""
    return compile_bf16tofp4_quant(pdl=pdl).quantize(input)


def bf16tofp4_quant_out(input, packed, scales, *, pdl=False):
    """Out variant of :func:`bf16tofp4_quant`."""
    return compile_bf16tofp4_quant(pdl=pdl).quantize_out(input, packed, scales)


def bf16tofp4_quant_dequant(input, *, pdl=False):
    """Fused quantize+dequantize: return the BF16 round trip of :func:`bf16tofp4_quant`."""
    return compile_bf16tofp4_quant(pdl=pdl).quantize_dequant(input)


def bf16tofp4_quant_dequant_out(input, output, scales, *, pdl=False):
    """Out variant of :func:`bf16tofp4_quant_dequant`."""
    return compile_bf16tofp4_quant(pdl=pdl).quantize_dequant_out(input, output, scales)


class Bf16ToFp4QuantRunner(FixedKernelRunner):
    def __init__(self, input, output, scales, *, dequant=False, pdl=False):
        validate_pdl(pdl)
        super().__init__(
            input.device,
            "bf16tofp4_quant",
            (tuple(input.shape), str(input.dtype), dequant, pdl),
            lambda: compile_bf16tofp4_quant(pdl=pdl),
            lambda module: (
                lambda: (
                    module.quantize_dequant_out(input, output, scales)
                    if dequant
                    else module.quantize_out(input, output, scales)
                )
            ),
        )
