"""Generalized BF16 to block-scaled E4M3 activation quantization.

Covers the variants ``bf16tofp8_quant`` does not: block sizes 64 and 128, and
either power-of-two E8M0 scales or plain FP32 scales.
"""

from __future__ import annotations

from pathlib import Path

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/fp8_act_quant/src"
_INCLUDE_DIR = _ROOT / "include"

# Must match the scale-mode constants in fp8_act_quant.h / the CUDA kernel.
SCALE_MODES = {"ue8m0": 0, "fp32": 1}
BLOCK_SIZES = (64, 128)


class Fp8ActQuantModule:
    def __init__(self, binding, library, config):
        block_size, scale, self.blocked, self.pdl = config
        if block_size not in BLOCK_SIZES:
            raise ValueError(f"unsupported block_size: {block_size}")
        if scale not in SCALE_MODES:
            raise ValueError(f"unsupported scale mode: {scale}")
        self._binding = binding
        self._launcher = library.entry_address
        self._block_size = int(block_size)
        self._scale_mode = SCALE_MODES[scale]

    def quantize(self, input):
        return self._binding.fp8_act_quant(
            input, self._block_size, self._scale_mode, self._launcher, self.blocked
        )

    def quantize_out(self, input, output, scales):
        return self._binding.fp8_act_quant_out(
            input,
            output,
            scales,
            self._block_size,
            self._scale_mode,
            self._launcher,
            self.blocked,
        )


_JIT = SplitCudaJit(
    name="sm110a_fp8_act_quant",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "fp8_act_quant_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "fp8_act_quant_kernel.cu",),
    kernel_dependencies=(
        _INCLUDE_DIR / "fp8_act_quant.h",
        _INCLUDE_DIR / "common.h",
        _INCLUDE_DIR / "blocked_quant.h",
    ),
    kernel_symbol="sm110a_fp8_act_quant_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda c, _variant: (
        f"-DSM110A_QUANT_BLOCKED={int(c[2])}",
        f"-DSM110A_QUANT_PDL={int(c[3])}",
    ),
    kernel_ldflags=("-lcuda",),
    make_facade=Fp8ActQuantModule,
    include_paths=(_INCLUDE_DIR,),
)


def compile_fp8_act_quant(
    block_size=128, scale="ue8m0", *, blocked=False, pdl=False
) -> Fp8ActQuantModule:
    if blocked and (block_size != 128 or scale != "ue8m0"):
        raise ValueError("blocked scales require block_size=128 and scale=ue8m0")
    if not isinstance(blocked, bool) or not isinstance(pdl, bool):
        raise ValueError("blocked and pdl must be bool")
    return _JIT.compile((block_size, scale, blocked, pdl))


def fp8_act_quant(input, block_size=128, scale="ue8m0"):
    """Return E4M3 values and one scale per contiguous ``block_size`` block.

    ``scale="ue8m0"`` stores power-of-two scales as ``torch.float8_e8m0fnu``;
    ``scale="fp32"`` stores ``max(amax, 1e-4) / 448`` as ``torch.float32``.
    """
    return compile_fp8_act_quant(block_size, scale).quantize(input)


def fp8_act_quant_out(input, output, scales, block_size=128, scale="ue8m0"):
    """Out variant of :func:`fp8_act_quant`."""
    return compile_fp8_act_quant(block_size, scale).quantize_out(input, output, scales)


class Fp8ActQuantRunner(FixedKernelRunner):
    def __init__(
        self,
        input,
        output,
        scales,
        block_size=128,
        scale="ue8m0",
        *,
        blocked=False,
        pdl=False,
    ):
        super().__init__(
            input.device,
            "fp8_act_quant",
            (tuple(input.shape), str(input.dtype), block_size, scale, blocked, pdl),
            lambda: compile_fp8_act_quant(block_size, scale, blocked=blocked, pdl=pdl),
            lambda module: lambda: module.quantize_out(input, output, scales),
        )
