"""BF16 FA4 MLA decoding: H=128, D_KV=512, D_rope=64."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16_fa4_mla/src"


@dataclass(frozen=True)
class Bf16Fa4MlaConfig:
    implementation: str = "1cta"

    def validate(self) -> None:
        if self.implementation not in _JITS:
            raise ValueError("implementation must be '1cta' or '2cta'")


class Bf16Fa4MlaModule:
    def __init__(self, binding, library, config: Bf16Fa4MlaConfig):
        self._binding = binding
        self._launcher = library.entry_address
        self._config = config

    def fa_mla(self, q, k, v, causal=False, scale=1.0):
        return self._binding.fa_mla(
            q, k, v, self._launcher, bool(causal), scale
        )

    def fa_mla_out(self, q, k, v, out, causal=False, scale=1.0):
        return self._binding.fa_mla_out(
            q, k, v, out, self._launcher, bool(causal), scale
        )


def _kernel_cflags(_config: Bf16Fa4MlaConfig, _variant) -> tuple[str, ...]:
    return ()


def _make_jit(name: str, source: str, symbol: str) -> SplitCudaJit:
    return SplitCudaJit(
        name=name,
        root=_ROOT,
        binding_sources=(_SOURCE_DIR / "bf16_fa4_mla_torch_binding.cpp",),
        kernel_sources=(_SOURCE_DIR / source,),
        kernel_dependencies=(),
        kernel_symbol=symbol,
        config_key=lambda config: config,
        kernel_cflags=_kernel_cflags,
        make_facade=Bf16Fa4MlaModule,
    )


_JITS = {
    "1cta": _make_jit(
        "sm110a_bf16_fa4_mla_1cta",
        "bf16_fa4_mla_kernel.cu",
        "sm110a_bf16_fa4_mla_launch",
    ),
    "2cta": _make_jit(
        "sm110a_bf16_fa4_mla_2cta",
        "bf16_fa4_mla_2cta_kernel.cu",
        "sm110a_bf16_fa4_mla_2cta_launch",
    ),
}


def compile_bf16_fa4_mla(
    config: Optional[Bf16Fa4MlaConfig] = None,
) -> Bf16Fa4MlaModule:
    config = config or Bf16Fa4MlaConfig()
    if not isinstance(config, Bf16Fa4MlaConfig):
        raise TypeError("config must be a Bf16Fa4MlaConfig")
    config.validate()
    return _JITS[config.implementation].compile(config)


def compile_bf16_fa4_mla_configs(
    configs: Iterable[Bf16Fa4MlaConfig], max_workers=None
):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Bf16Fa4MlaConfig):
            raise TypeError("configs must contain Bf16Fa4MlaConfig instances")
        config.validate()
    grouped = {implementation: [] for implementation in _JITS}
    for config in configs:
        grouped[config.implementation].append(config)
    modules = {}
    for implementation, selected in grouped.items():
        if selected:
            modules.update(zip(selected, _JITS[implementation].compile_many(selected, max_workers)))
    return [modules[config] for config in configs]


def get_bf16_fa4_mla_module(config: Optional[Bf16Fa4MlaConfig] = None):
    return compile_bf16_fa4_mla(config)


def _resolve_scale(q, scale):
    if scale is None:
        return 1.0 / math.sqrt(q.size(-1))
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return scale


def bf16_fa4_mla(q, k, v, *, causal=False, scale=None, implementation="1cta"):
    """Compute MLA decode from Q [B,128,1,576] and K [B,1,S,576].

    K stores concat(cKV, K_rope) row-wise; cKV is also V. ``v`` remains an
    ABI-compatible argument and must represent that cKV prefix. Returns
    [B,128,1,512]. Default scale is 1/sqrt(576)."""
    return compile_bf16_fa4_mla(Bf16Fa4MlaConfig(implementation)).fa_mla(
        q, k, v, bool(causal), _resolve_scale(q, scale)
    )


def bf16_fa4_mla_out(
    q, k, v, out, *, causal=False, scale=None, implementation="1cta"
):
    """Write MLA decoding output into contiguous out [B,128,1,512]."""
    return compile_bf16_fa4_mla(Bf16Fa4MlaConfig(implementation)).fa_mla_out(
        q, k, v, out, bool(causal), _resolve_scale(q, scale)
    )
