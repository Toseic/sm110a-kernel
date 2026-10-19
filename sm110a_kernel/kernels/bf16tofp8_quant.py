"""BF16 to block-scaled E4M3/E8M0 activation quantization."""

from __future__ import annotations

from pathlib import Path

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16tofp8_quant/src"
_INCLUDE_DIR = _ROOT / "include"


class Bf16ToFp8QuantModule:
    def __init__(self, binding, library, config):
        self.blocked, self.pdl = config
        self._binding = binding
        self._launcher = library.entry_address

    def quantize(self, input):
        return self._binding.bf16tofp8_quant(input, self._launcher, self.blocked)

    def quantize_out(self, input, output, scales):
        return self._binding.bf16tofp8_quant_out(
            input, output, scales, self._launcher, self.blocked
        )


_JIT = SplitCudaJit(
    name="sm110a_bf16tofp8_quant",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16tofp8_quant_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16tofp8_quant_kernel.cu",),
    kernel_dependencies=(
        _INCLUDE_DIR / "bf16tofp8_quant.h",
        _INCLUDE_DIR / "common.h",
        _INCLUDE_DIR / "blocked_quant.h",
    ),
    kernel_symbol="sm110a_bf16tofp8_quant_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda c, _variant: (
        f"-DSM110A_QUANT_BLOCKED={int(c[0])}",
        f"-DSM110A_QUANT_PDL={int(c[1])}",
    ),
    kernel_ldflags=("-lcuda",),
    make_facade=Bf16ToFp8QuantModule,
    include_paths=(_INCLUDE_DIR,),
)


def compile_bf16tofp8_quant(*, blocked=False, pdl=False) -> Bf16ToFp8QuantModule:
    if not isinstance(blocked, bool) or not isinstance(pdl, bool):
        raise ValueError("blocked and pdl must be bool")
    return _JIT.compile((blocked, pdl))


def bf16tofp8_quant(input):
    """Return E4M3 values and one E8M0 scale per contiguous 128-value block."""
    return compile_bf16tofp8_quant().quantize(input)


def bf16tofp8_quant_out(input, output, scales):
    """Out variant of :func:`bf16tofp8_quant`."""
    return compile_bf16tofp8_quant().quantize_out(input, output, scales)


class Bf16ToFp8QuantRunner(FixedKernelRunner):
    def __init__(self, input, output, scales, *, blocked=False, pdl=False):
        super().__init__(
            input.device,
            "bf16tofp8_quant",
            (tuple(input.shape), str(input.dtype), blocked, pdl),
            lambda: compile_bf16tofp8_quant(blocked=blocked, pdl=pdl),
            lambda module: lambda: module.quantize_out(input, output, scales),
        )
