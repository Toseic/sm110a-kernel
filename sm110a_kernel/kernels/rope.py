"""In-place adjacent-pair RoPE for strided DeepSeek-V4-Flash activations."""

from dataclasses import dataclass
from pathlib import Path

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner
from .pdl import validate_pdl

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/rope/src"


@dataclass(frozen=True)
class RopeConfig:
    d: int
    threads: int = 256
    pdl: bool = False


class RopeModule:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self.config = config

    def rope_(self, x, freqs_cis, inverse=False):
        return self._binding.rope_(x, freqs_cis, inverse, self.config.d, self._launcher)


_JIT = SplitCudaJit(
    name="sm110a_rope",
    root=_ROOT,
    binding_sources=(_SOURCE / "rope_torch_binding.cpp",),
    binding_dependencies=(_SOURCE / "rope_params.h",),
    kernel_sources=(_SOURCE / "rope_kernel.cu",),
    kernel_dependencies=(_SOURCE / "rope_params.h",),
    kernel_symbol="sm110a_rope_launch",
    config_key=lambda c: (c.d, c.threads, c.pdl),
    kernel_cflags=lambda c, _variant: (
        f"-DSM110A_ROPE_DIM={c.d}",
        f"-DSM110A_ROPE_THREADS={c.threads}",
        f"-DSM110A_PDL={int(c.pdl)}",
    ),
    make_facade=RopeModule,
)


def compile_rope(d=64, *, threads=256, pdl=False):
    """Compile a fixed rotary width; selection requires no GPU execution."""
    validate_pdl(pdl)
    if type(d) is not int or not 2 <= d <= 1024 or d % 2:
        raise ValueError("d must be an even integer in [2, 1024]")
    if type(threads) is not int or threads not in (64, 128, 256, 512):
        raise ValueError("threads must be 64, 128, 256 or 512")
    return _JIT.compile(RopeConfig(d, threads, pdl))


def rope_(x, freqs_cis, inverse=False, *, threads=256, pdl=False):
    """Rotate x in place and return x, matching the model apply_rotary_emb.

    x is CUDA BF16/FP16/FP32 [B,S,D] or [B,S,H,D] with unit last
    stride and nonoverlapping positive outer strides. freqs_cis is complex64
    [S,D/2]; sequence slicing and lazy conjugates are supported. Only the
    supplied view is modified, so pass x[..., -rope_dim:] for partial RoPE.
    Arithmetic is FP32 with a single final cast. No allocations or autotuning.
    Warm up eagerly before CUDA graph capture.
    """
    return compile_rope(x.size(-1), threads=threads, pdl=pdl).rope_(
        x, freqs_cis, inverse
    )


class RopeRunner(FixedKernelRunner):
    def __init__(self, x, freqs_cis, inverse=False, *, threads=256, pdl=False):
        validate_pdl(pdl)
        super().__init__(
            x.device,
            "rope",
            (
                tuple(x.shape),
                tuple(x.stride()),
                str(x.dtype),
                tuple(freqs_cis.stride()),
                freqs_cis.is_conj(),
                inverse,
                threads,
                pdl,
            ),
            lambda: compile_rope(x.size(-1), threads=threads, pdl=pdl),
            lambda module: lambda: module.rope_(x, freqs_cis, inverse),
        )
