"""DSV4 gated pooling and incremental compressor state updates."""

from dataclasses import dataclass
from pathlib import Path

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner
from .pdl import validate_pdl

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/compressor_pool/src"


@dataclass(frozen=True)
class CompressorPoolConfig:
    ratio: int
    pdl: bool = False


class CompressorPoolModule:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self.config = config

    def pool_out(self, kv, score, ape, kv_state, score_state, output, start_pos=0):
        return self._binding.pool_out(
            kv,
            score,
            ape,
            kv_state,
            score_state,
            output,
            start_pos,
            self.config.ratio,
            self._launcher,
        )


_JIT = SplitCudaJit(
    name="sm110a_compressor_pool",
    root=_ROOT,
    binding_sources=(_SOURCE / "compressor_pool_torch_binding.cpp",),
    binding_dependencies=(_SOURCE / "compressor_pool_params.h",),
    kernel_sources=(_SOURCE / "compressor_pool_kernel.cu",),
    kernel_dependencies=(_SOURCE / "compressor_pool_params.h",),
    kernel_symbol="sm110a_compressor_pool_launch",
    config_key=lambda c: (c.ratio, c.pdl),
    kernel_cflags=lambda c, _variant: (
        f"-DSM110A_COMPRESS_RATIO={c.ratio}",
        f"-DSM110A_PDL={int(c.pdl)}",
    ),
    make_facade=CompressorPoolModule,
)


def compile_compressor_pool(ratio=4, *, pdl=False):
    validate_pdl(pdl)
    if type(ratio) is not int or ratio not in (4, 128):
        raise ValueError("ratio must be 4 or 128")
    return _JIT.compile(CompressorPoolConfig(ratio, pdl))


def compressor_pool_out(
    kv, score, ape, kv_state, score_state, output, start_pos=0, *, ratio=4, pdl=False
):
    """Pool BF16 [B,S,W] projections, updating FP32 compressor states in place.

    W=2*D for overlap ratio-4 and W=D for ratio-128. Inputs may be the
    identically strided halves of a joint projection. ape is FP32 [ratio,W];
    states are contiguous FP32 [max_batch,coeff*ratio,W], initialized by caller.
    start_pos=0 performs prefill; otherwise S must be one. Output is contiguous
    BF16 [B,S//ratio,D] for prefill, [B,1,D] on decode boundaries, [B,0,D]
    otherwise. Pooling uses FP32 and rounds to BF16 before downstream RMSNorm.
    State/output must not alias other tensors. Warm up before graph capture.
    """
    return compile_compressor_pool(ratio, pdl=pdl).pool_out(
        kv, score, ape, kv_state, score_state, output, start_pos
    )


class CompressorPoolRunner(FixedKernelRunner):
    def __init__(
        self,
        kv,
        score,
        ape,
        kv_state,
        score_state,
        output,
        start_pos=0,
        *,
        ratio=4,
        pdl=False,
    ):
        validate_pdl(pdl)
        if type(ratio) is not int or ratio not in (4, 128):
            raise ValueError("ratio must be 4 or 128")
        super().__init__(
            kv.device,
            "compressor_pool",
            (
                tuple(kv.shape),
                tuple(kv.stride()),
                tuple(kv_state.shape),
                start_pos,
                ratio,
                pdl,
            ),
            lambda: compile_compressor_pool(ratio, pdl=pdl),
            lambda module: (
                lambda: module.pool_out(
                    kv, score, ape, kv_state, score_state, output, start_pos
                )
            ),
        )
