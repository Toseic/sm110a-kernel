"""Fused main-attention compressor RMSNorm, RoPE, FP8 quantization and store."""

from dataclasses import dataclass
from pathlib import Path

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner
from .pdl import validate_pdl

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/compressor_store/src"


@dataclass(frozen=True)
class CompressorStoreConfig:
    scale: str = "ue8m0"
    pdl: bool = False


class CompressorStoreModule:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self.config = config

    def store_out(
        self,
        input,
        weight,
        frequencies,
        cache_values,
        cache_scales,
        cache_rope,
        position,
        eps=1e-6,
        packed=None,
    ):
        return self._binding.store_out(
            input,
            weight,
            frequencies,
            cache_values,
            cache_scales,
            cache_rope,
            position,
            eps,
            packed,
            self.config.scale == "ue8m0",
            self._launcher,
        )


_JIT = SplitCudaJit(
    name="sm110a_compressor_store",
    root=_ROOT,
    binding_sources=(_SOURCE / "compressor_store_torch_binding.cpp",),
    binding_dependencies=(_SOURCE / "compressor_store_params.h",),
    kernel_sources=(_SOURCE / "compressor_store_kernel.cu",),
    kernel_dependencies=(_SOURCE / "compressor_store_params.h",),
    kernel_symbol="sm110a_compressor_store_launch",
    config_key=lambda c: (c.scale, c.pdl),
    kernel_cflags=lambda c, _variant: (
        f"-DSM110A_STORE_E8M0={int(c.scale == 'ue8m0')}",
        f"-DSM110A_PDL={int(c.pdl)}",
    ),
    make_facade=CompressorStoreModule,
)


def compile_compressor_store(*, scale="ue8m0", pdl=False):
    validate_pdl(pdl)
    if scale not in ("ue8m0", "fp32"):
        raise ValueError("scale must be ue8m0 or fp32")
    return _JIT.compile(CompressorStoreConfig(scale, pdl))


def compressor_store_out(
    input,
    weight,
    frequencies,
    cache_values,
    cache_scales,
    cache_rope,
    position,
    eps=1e-6,
    *,
    packed=None,
    scale="ue8m0",
    pdl=False,
):
    """Normalize/rotate/quantize BF16 [B,S,512] directly into a packed KV cache.

    Uses FP32 [512] RMSNorm weights, complex64 [S,32] frequencies (strided and
    lazy conjugates supported), block-64 E4M3 non-RoPE and BF16 RoPE. Cache
    tensors are contiguous [max_batch,capacity,448/7/64]. Optional packed output
    tensors [B,S,448/7/64] receive a second copy for prefill attention. Outputs
    must not alias inputs or each other. Warm up before CUDA Graph capture.
    """
    return compile_compressor_store(scale=scale, pdl=pdl).store_out(
        input,
        weight,
        frequencies,
        cache_values,
        cache_scales,
        cache_rope,
        position,
        eps,
        packed,
    )


class CompressorStoreRunner(FixedKernelRunner):
    def __init__(
        self,
        input,
        weight,
        frequencies,
        cache_values,
        cache_scales,
        cache_rope,
        position,
        eps=1e-6,
        *,
        packed=None,
        scale="ue8m0",
        pdl=False,
    ):
        validate_pdl(pdl)
        if scale not in ("ue8m0", "fp32"):
            raise ValueError("scale must be ue8m0 or fp32")
        super().__init__(
            input.device,
            "compressor_store",
            (
                tuple(input.shape),
                tuple(cache_values.shape),
                tuple(frequencies.stride()),
                frequencies.is_conj(),
                position,
                eps,
                packed is not None,
                scale,
                pdl,
            ),
            lambda: compile_compressor_store(scale=scale, pdl=pdl),
            lambda module: (
                lambda: module.store_out(
                    input,
                    weight,
                    frequencies,
                    cache_values,
                    cache_scales,
                    cache_rope,
                    position,
                    eps,
                    packed,
                )
            ),
        )
