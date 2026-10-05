"""BF16 to packed-E2M1 (NVFP4) + blocked UE4M3 activation quantization."""

from __future__ import annotations

from pathlib import Path

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16tonvfp4_quant/src"
_INCLUDE_DIR = _ROOT / "include"


class Bf16ToNvfp4QuantModule:
    def __init__(self, binding, library, _config):
        self._binding = binding
        self._launcher = library.entry_address

    def quantize(self, input):
        return self._binding.bf16tonvfp4_quant(input, self._launcher)

    def quantize_out(self, input, packed, scales):
        return self._binding.bf16tonvfp4_quant_out(
            input, packed, scales, self._launcher)


_JIT = SplitCudaJit(
    name="sm110a_bf16tonvfp4_quant",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16tonvfp4_quant_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16tonvfp4_quant_kernel.cu",),
    kernel_dependencies=(_INCLUDE_DIR / "bf16tonvfp4_quant.h",
                         _INCLUDE_DIR / "common.h"),
    kernel_symbol="sm110a_bf16tonvfp4_quant_launch",
    config_key=lambda _config: 128,
    kernel_cflags=lambda _config, _variant: (),
    kernel_ldflags=("-lcuda",),
    make_facade=Bf16ToNvfp4QuantModule,
    include_paths=(_INCLUDE_DIR,),
)


def compile_bf16tonvfp4_quant() -> Bf16ToNvfp4QuantModule:
    return _JIT.compile(128)


def bf16tonvfp4_quant(input):
    """Return packed E2M1 pairs and blocked UE4M3 scales per 16-value block."""
    return compile_bf16tonvfp4_quant().quantize(input)


def bf16tonvfp4_quant_out(input, packed, scales):
    """Out variant of :func:`bf16tonvfp4_quant`."""
    return compile_bf16tonvfp4_quant().quantize_out(input, packed, scales)
