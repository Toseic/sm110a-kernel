"""HC=4 post: FP32 gated expansion and residual mixing, one final cast."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import FixedKernelRunner
from .pdl import validate_pdl

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/hc_post/src"


@dataclass(frozen=True)
class HcPostConfig:
    d: int
    threads: int = 128
    vector_bytes: int = 16
    pdl: bool = False


class HcPostModule:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self.config = config

    def hc_post_out(self, x, residual, post, comb, output):
        return self._binding.hc_post_out(
            x, residual, post, comb, output, self.config.d, self._launcher
        )

    def hc_post(self, x, residual, post, comb):
        return self.hc_post_out(x, residual, post, comb, torch.empty_like(residual))


_JIT = SplitCudaJit(
    name="sm110a_hc_post",
    root=_ROOT,
    binding_sources=(_SOURCE / "hc_post_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "hc_post_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_hc_post_launch",
    config_key=lambda c: (c.d, c.threads, c.vector_bytes, c.pdl),
    kernel_cflags=lambda c, _variant: (
        f"-DSM110A_PDL={int(c.pdl)}",
        f"-DSM110A_HC_POST_D={c.d}",
        f"-DSM110A_HC_POST_THREADS={c.threads}",
        f"-DSM110A_HC_POST_VECTOR_BYTES={c.vector_bytes}",
    ),
    make_facade=HcPostModule,
)


def compile_hc_post(d, *, threads=128, vector_bytes=16, pdl=False):
    """Fixed configuration; aligned rows use 8/16-byte vectors, others scalar loads."""
    validate_pdl(pdl)
    if type(d) is not int or not 1 <= d <= 32768:
        raise ValueError("d must be an integer in [1, 32768]")
    if type(threads) is not int or threads not in (32, 64, 128, 256):
        raise ValueError("threads must be 32, 64, 128 or 256")
    if type(vector_bytes) is not int or vector_bytes not in (8, 16):
        raise ValueError("vector_bytes must be 8 or 16")
    return _JIT.compile(HcPostConfig(d, threads, vector_bytes, pdl))


def hc_post(x, residual, post, comb, *, threads=128, vector_bytes=16, pdl=False):
    return compile_hc_post(
        x.size(-1), threads=threads, vector_bytes=vector_bytes, pdl=pdl
    ).hc_post(x, residual, post, comb)


def hc_post_out(
    x, residual, post, comb, output, *, threads=128, vector_bytes=16, pdl=False
):
    """Allocation-free, graph-safe HC post. Output must not overlap inputs."""
    return compile_hc_post(
        x.size(-1), threads=threads, vector_bytes=vector_bytes, pdl=pdl
    ).hc_post_out(x, residual, post, comb, output)


class HcPostRunner(FixedKernelRunner):
    """Fixed kernel selection; warm up eagerly before CUDA Graph capture.

    x is [..., D], residual/output [..., 4, D], post [..., 4], comb [..., 4, 4].
    Data tensors share a CUDA device and BF16/FP16/FP32 dtype; gates are FP32.
    """

    def __init__(
        self,
        x,
        residual,
        post,
        comb,
        output,
        *,
        threads=128,
        vector_bytes=16,
        pdl=False,
    ):
        validate_pdl(pdl)
        super().__init__(
            x.device,
            "hc_post",
            (tuple(x.shape), str(x.dtype), threads, vector_bytes, pdl),
            lambda: compile_hc_post(
                x.size(-1), threads=threads, vector_bytes=vector_bytes, pdl=pdl
            ),
            lambda module: lambda: module.hc_post_out(x, residual, post, comb, output),
        )
