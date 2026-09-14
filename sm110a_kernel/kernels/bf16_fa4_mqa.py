"""Configuration-specialized BF16 FA4 MQA kernel.

The CUDA implementation uses the 16-warp tactic exclusively and always folds
32 adjacent query heads into each CTA's logical query-row dimension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16_fa4_mqa/src"


@dataclass(frozen=True)
class Bf16Fa4MqaConfig:
    def validate(self) -> None:
        pass


class Bf16Fa4MqaModule:
    def __init__(self, binding, library, config: Bf16Fa4MqaConfig):
        self._binding = binding
        self._launcher = library.entry_address
        self._config = config

    def fa_mqa(self, q, k, v, causal=False, scale=1.0):
        return self._binding.fa_mqa(
            q, k, v, self._launcher, bool(causal), scale
        )

    def fa_mqa_out(self, q, k, v, out, causal=False, scale=1.0):
        return self._binding.fa_mqa_out(
            q, k, v, out, self._launcher, bool(causal), scale
        )


def _kernel_cflags(config: Bf16Fa4MqaConfig, _variant) -> tuple[str, ...]:
    return ()


_JIT = SplitCudaJit(
    name="sm110a_bf16_fa4_mqa",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16_fa4_mqa_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16_fa4_mqa_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_bf16_fa4_mqa_launch",
    config_key=lambda config: config,
    kernel_cflags=_kernel_cflags,
    make_facade=Bf16Fa4MqaModule,
)


def compile_bf16_fa4_mqa(
    config: Optional[Bf16Fa4MqaConfig] = None,
) -> Bf16Fa4MqaModule:
    config = config or Bf16Fa4MqaConfig()
    if not isinstance(config, Bf16Fa4MqaConfig):
        raise TypeError("config must be a Bf16Fa4MqaConfig")
    config.validate()
    return _JIT.compile(config)


def compile_bf16_fa4_mqa_configs(
    configs: Iterable[Bf16Fa4MqaConfig], max_workers=None
):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Bf16Fa4MqaConfig):
            raise TypeError("configs must contain Bf16Fa4MqaConfig instances")
        config.validate()
    return _JIT.compile_many(configs, max_workers)


def get_bf16_fa4_mqa_module(config: Optional[Bf16Fa4MqaConfig] = None):
    return compile_bf16_fa4_mqa(config)


def _resolve_scale(q, scale):
    if scale is None:
        return 1.0 / math.sqrt(q.size(-1))
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return scale


def bf16_fa4_mqa(q, k, v, *, causal=False, scale=None):
    """Compute BF16 MQA with the fixed 32-head packed implementation."""
    return compile_bf16_fa4_mqa().fa_mqa(
        q, k, v, bool(causal), _resolve_scale(q, scale)
    )


def bf16_fa4_mqa_out(
    q, k, v, out, *, causal=False, scale=None
):
    """Write BF16 MQA output with the fixed 32-head packed implementation."""
    return compile_bf16_fa4_mqa().fa_mqa_out(
        q, k, v, out, bool(causal), _resolve_scale(q, scale)
    )
