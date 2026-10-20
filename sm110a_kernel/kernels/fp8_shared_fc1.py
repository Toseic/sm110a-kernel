"""Dense FP8 shared-expert FC1 with a fused BF16/FP32 clamp/SwiGLU epilogue."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import BaseKernelRunner, PreparedKernel
from .fp8_act_quant import compile_fp8_act_quant
from .fp8_gemm import Fp8GemmConfig, to_blocked_fp8_scales

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "csrc/fp8_shared_fc1/src"
_INC = _ROOT / "include"


@dataclass(frozen=True)
class Fp8SharedFc1Config:
    tile_k: int = 128
    cluster_m: int = 1
    swiglu_limit: float = 0.0
    pdl: bool = False

    def validate(self):
        Fp8GemmConfig(
            tile_n=128, tile_k=self.tile_k, cluster_m=self.cluster_m, pdl=self.pdl
        ).validate()
        if not math.isfinite(self.swiglu_limit) or self.swiglu_limit < 0:
            raise ValueError("swiglu_limit must be finite and non-negative")


def _cflags(c, _variant):
    return tuple(
        f"-DSM110A_FP8_SHARED_FC1_{name}={value}"
        for name, value in (
            ("TILE_M", 128),
            ("TILE_N", 128),
            ("TILE_K", c.tile_k),
            ("CLUSTER_M", c.cluster_m),
            ("EPILOGUE_BLOCK_N", 32),
            ("SPLIT_K", 1),
            ("PDL", int(c.pdl)),
            ("SWIGLU_LIMIT", f"{float(c.swiglu_limit):.9e}f"),
        )
    )


class Fp8SharedFc1Module:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self._config = config

    def _args(self):
        c = self._config
        return self._launcher, 128, 128, c.tile_k, c.cluster_m, 1

    def fp8_shared_fc1(self, a, b, sfa, sfb):
        return self._binding.fp8_shared_fc1(a, b, sfa, sfb, *self._args())

    def fp8_shared_fc1_out(self, a, b, sfa, sfb, out):
        return self._binding.fp8_shared_fc1_out(a, b, sfa, sfb, out, *self._args())


_JIT = SplitCudaJit(
    name="sm110a_fp8_shared_fc1",
    root=_ROOT,
    binding_sources=(_SRC / "fp8_shared_fc1_torch_binding.cpp",),
    kernel_sources=(_SRC / "fp8_shared_fc1_kernel.cu",),
    kernel_dependencies=(_INC / "common.h", _INC / "pdl.h"),
    kernel_symbol="sm110a_fp8_shared_fc1_launch",
    config_key=lambda c: c,
    kernel_cflags=_cflags,
    make_facade=Fp8SharedFc1Module,
    include_paths=(_INC,),
)


def compile_fp8_shared_fc1(config: Fp8SharedFc1Config | None = None):
    config = config or Fp8SharedFc1Config()
    if not isinstance(config, Fp8SharedFc1Config):
        raise TypeError("config must be Fp8SharedFc1Config")
    config.validate()
    return _JIT.compile(config)


@cache
def _compiled(config):
    return compile_fp8_shared_fc1(config)


def fp8_shared_fc1(a, b, sfa, sfb, config=None):
    return _compiled(config or Fp8SharedFc1Config()).fp8_shared_fc1(a, b, sfa, sfb)


def fp8_shared_fc1_out(a, b, sfa, sfb, out, config=None):
    return _compiled(config or Fp8SharedFc1Config()).fp8_shared_fc1_out(
        a, b, sfa, sfb, out
    )


def pack_fp8_shared_fc1_weights(gate, up, gate_scales, up_scales):
    """Pack E4M3 [I,K] weights and logical E8M0 [I,K/32] scales.

    Interleave gate/up in groups of 64 rows, then block the scales for MMA.
    Static weights should be packed once before graph capture.
    """
    if (
        gate.ndim != 2
        or gate.shape != up.shape
        or gate.dtype != torch.float8_e4m3fn
        or up.dtype != gate.dtype
        or gate.device != up.device
    ):
        raise ValueError("gate/up must be E4M3 [I,K] on the same device")
    rows, k = gate.shape
    if rows <= 0 or k <= 0 or rows % 64 or k % 64:
        raise ValueError("I and K must be positive multiples of 64")
    for s in (gate_scales, up_scales):
        if (
            s.dtype != torch.float8_e8m0fnu
            or s.device != gate.device
            or tuple(s.shape) != (rows, k // 32)
        ):
            raise ValueError(
                "scales must be logical E8M0 [I,K/32] on the weight device"
            )

    def interleave(a, b):
        # Byte copies avoid requiring FP8 arithmetic/stack support.
        return torch.stack(
            (
                a.contiguous().view(torch.uint8).reshape(rows // 64, 64, -1),
                b.contiguous().view(torch.uint8).reshape(rows // 64, 64, -1),
            ),
            dim=1,
        ).reshape(2 * rows, -1)

    weight = interleave(gate, up).view(gate.dtype)
    scales = interleave(gate_scales, up_scales).view(gate_scales.dtype)
    return weight, to_blocked_fp8_scales(scales)


class Fp8SharedFc1Runner(BaseKernelRunner):
    namespace = "fp8_shared_fc1_v1"

    def __init__(self, a, b, sfa, sfb, out, *, input=None, swiglu_limit=0.0, pdl=False):
        super().__init__(a.device)
        self.a, self.b, self.sfa, self.sfb, self.out = a, b, sfa, sfb, out
        self.input, self.limit, self.pdl = input, float(swiglu_limit), pdl
        if pdl and input is None:
            raise ValueError("PDL requires the producer input")
        Fp8SharedFc1Config(swiglu_limit=self.limit, pdl=pdl).validate()
        self.graph_timing = input is not None

    @property
    def workload_key(self):
        return (
            self.a.size(0),
            self.b.size(0),
            self.a.size(1),
            self.limit,
            self.input is not None,
            self.pdl,
        )

    @staticmethod
    def format_candidate(c):
        return (
            f"128x128x{c.tile_k}/c{c.cluster_m}/limit{c.swiglu_limit}/pdl{int(c.pdl)}"
        )

    def compatible_candidates(self):
        return tuple(
            Fp8SharedFc1Config(tk, cm, self.limit, self.pdl)
            for cm in (1, 2)
            for tk in (64, 128)
            if self.a.size(0) % (128 * cm) == 0
            and self.a.size(1) % tk == 0
            and (not self.pdl or cm == 1)
        )

    def pre_jit(self, candidate):
        module = compile_fp8_shared_fc1(candidate)
        if self.input is not None:
            compile_fp8_act_quant(blocked=True, pdl=self.pdl)
        return module

    def prepare(self, candidate):
        module = self.pre_jit(candidate)
        quant = (
            compile_fp8_act_quant(blocked=True, pdl=self.pdl)
            if self.input is not None
            else None
        )

        def run():
            if quant is not None:
                quant.quantize_out(self.input, self.a, self.sfa)
            return module.fp8_shared_fc1_out(
                self.a, self.b, self.sfa, self.sfb, self.out
            )

        return PreparedKernel(candidate, run)
