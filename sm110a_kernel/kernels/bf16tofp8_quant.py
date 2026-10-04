"""BF16 to block-scaled E4M3/E8M0 activation quantization."""

from __future__ import annotations

from pathlib import Path

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16tofp8_quant/src"
_INCLUDE_DIR = _ROOT / "include"


class Bf16ToFp8QuantModule:
    def __init__(self, binding, library, _config):
        self._binding = binding
        self._launcher = library.entry_address

    def quantize(self, input):
        return self._binding.bf16tofp8_quant(input, self._launcher)

    def quantize_out(self, input, output, scales):
        return self._binding.bf16tofp8_quant_out(input, output, scales, self._launcher)


_JIT = SplitCudaJit(
    name="sm110a_bf16tofp8_quant",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16tofp8_quant_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16tofp8_quant_kernel.cu",),
    kernel_dependencies=(_INCLUDE_DIR / "bf16tofp8_quant.h", _INCLUDE_DIR / "common.h"),
    kernel_symbol="sm110a_bf16tofp8_quant_launch",
    config_key=lambda _config: 128,
    kernel_cflags=lambda _config, _variant: (),
    make_facade=Bf16ToFp8QuantModule,
    include_paths=(_INCLUDE_DIR,),
)


def compile_bf16tofp8_quant() -> Bf16ToFp8QuantModule:
    return _JIT.compile(128)


def bf16tofp8_quant(input):
    """Return E4M3 values and one E8M0 scale per contiguous 128-value block."""
    return compile_bf16tofp8_quant().quantize(input)


def bf16tofp8_quant_out(input, output, scales):
    """Out variant of :func:`bf16tofp8_quant`."""
    return compile_bf16tofp8_quant().quantize_out(input, output, scales)
