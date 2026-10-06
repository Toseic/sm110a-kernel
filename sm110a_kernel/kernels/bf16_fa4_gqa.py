"""Configuration-specialized BF16 FA4 GQA kernel.

Two orthogonal axes are baked in at JIT compile time:

* ``tactic`` via ``-DSM110A_FA4_MHA_TACTIC`` selects the CTA schedule:
  - ``TACTIC_16WARP`` — 16 warps with dedicated softmax, O-rescale, and
    TMA-store warpgroups.
  - ``TACTIC_12WARP`` — 12 warps with one fused softmax/rescale/epilogue
    warpgroup per Q slot.
* ``pack_gqa`` via ``-DSM110A_FA4_MHA_PACK_GQA`` folds the four query heads
  of each K/V head into the logical query-row dimension of a single CTA
  (``True``), or assigns one query head per CTA (``False``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16_fa4/src"


class Bf16Fa4GqaTactic(IntEnum):
    """CTA schedules supported by the BF16 FA4 GQA kernel."""

    WARP_16 = 0
    WARP_12 = 1


# Short aliases keep the launch API convenient and preserve the initial
# tactic-facing public API.
TACTIC_16WARP = Bf16Fa4GqaTactic.WARP_16
TACTIC_12WARP = Bf16Fa4GqaTactic.WARP_12


@dataclass(frozen=True)
class Bf16Fa4GqaConfig:
    tactic: Bf16Fa4GqaTactic = TACTIC_16WARP
    pack_gqa: bool = False

    def __post_init__(self) -> None:
        try:
            tactic = Bf16Fa4GqaTactic(self.tactic)
        except (TypeError, ValueError) as exc:
            raise ValueError("tactic must be 0 (16-warp) or 1 (12-warp)") from exc
        object.__setattr__(self, "tactic", tactic)
        object.__setattr__(self, "pack_gqa", bool(self.pack_gqa))

    def validate(self) -> None:
        if not isinstance(self.tactic, Bf16Fa4GqaTactic):
            raise TypeError("tactic must be a Bf16Fa4GqaTactic")
        if not isinstance(self.pack_gqa, bool):
            raise TypeError("pack_gqa must be a bool")


class Bf16Fa4GqaModule:
    def __init__(self, binding, library, config: Bf16Fa4GqaConfig):
        self._binding, self._launcher, self._config = (
            binding,
            library.entry_address,
            config,
        )

    def fa_gqa(self, q, k, v, causal=False, scale=1.0):
        return self._binding.fa_gqa(q, k, v, self._launcher, bool(causal), scale)

    def fa_gqa_out(self, q, k, v, out, causal=False, scale=1.0):
        return self._binding.fa_gqa_out(
            q, k, v, out, self._launcher, bool(causal), scale
        )


def _kernel_cflags(config: Bf16Fa4GqaConfig, _variant) -> tuple[str, ...]:
    return (
        f"-DSM110A_FA4_MHA_TACTIC={int(config.tactic)}",
        f"-DSM110A_FA4_MHA_PACK_GQA={int(config.pack_gqa)}",
    )


_JIT = SplitCudaJit(
    name="sm110a_bf16_fa4_gqa",
    root=_ROOT,
    binding_sources=(_ROOT / "csrc/bf16_fa4_gqa/src/bf16_fa4_gqa_torch_binding.cpp",),
    kernel_sources=(_ROOT / "csrc/bf16_fa4_gqa/src/bf16_fa4_gqa_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_bf16_fa4_gqa_launch",
    config_key=lambda config: config,
    kernel_cflags=_kernel_cflags,
    make_facade=Bf16Fa4GqaModule,
)


def compile_bf16_fa4_gqa(config: Bf16Fa4GqaConfig | None = None) -> Bf16Fa4GqaModule:
    config = config or Bf16Fa4GqaConfig()
    if not isinstance(config, Bf16Fa4GqaConfig):
        raise TypeError("config must be a Bf16Fa4GqaConfig")
    config.validate()
    return _JIT.compile(config)


def compile_bf16_fa4_gqa_configs(configs: Iterable[Bf16Fa4GqaConfig], max_workers=None):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Bf16Fa4GqaConfig):
            raise TypeError("configs must contain Bf16Fa4GqaConfig instances")
        config.validate()
    return _JIT.compile_many(configs, max_workers)


def get_bf16_fa4_gqa_module(config: Bf16Fa4GqaConfig | None = None):
    return compile_bf16_fa4_gqa(config)


def _resolve_scale(q, scale):
    if scale is None:
        return 1.0 / math.sqrt(q.size(-1))
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return scale


def _config_for(
    tactic: int | Bf16Fa4GqaTactic | Bf16Fa4GqaConfig,
    pack_gqa: bool = False,
) -> Bf16Fa4GqaConfig:
    if isinstance(tactic, Bf16Fa4GqaConfig):
        return tactic
    return Bf16Fa4GqaConfig(tactic=tactic, pack_gqa=pack_gqa)


def bf16_fa4_gqa(
    q, k, v, *, causal=False, scale=None, tactic=TACTIC_16WARP, pack_gqa=False
):
    """Compute BF16 GQA for Q `[B,Hq,S,D]` and K/V `[B,Hkv,S,D]` tensors."""
    return compile_bf16_fa4_gqa(_config_for(tactic, pack_gqa)).fa_gqa(
        q, k, v, bool(causal), _resolve_scale(q, scale)
    )


def bf16_fa4_gqa_out(
    q, k, v, out, *, causal=False, scale=None, tactic=TACTIC_16WARP, pack_gqa=False
):
    """Write BF16 FA GQA to a caller-provided contiguous output."""
    return compile_bf16_fa4_gqa(_config_for(tactic, pack_gqa)).fa_gqa_out(
        q, k, v, out, bool(causal), _resolve_scale(q, scale)
    )
