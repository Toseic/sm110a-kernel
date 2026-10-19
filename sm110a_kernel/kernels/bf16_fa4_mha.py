"""Configuration-specialized BF16 FA4 MHA kernel.

The tactic selects the CTA schedule and is baked in at JIT compile time via
``-DSM110A_FA4_MHA_TACTIC``:

* ``TACTIC_16WARP`` — 16 warps with dedicated softmax, O-rescale, and
  TMA-store warpgroups.
* ``TACTIC_12WARP`` — 12 warps with one fused softmax/rescale/epilogue
  warpgroup per Q slot.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from ..jit_runtime import SplitCudaJit
from ..runner import BaseKernelRunner, PreparedKernel

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16_fa4/src"


class Bf16Fa4MhaTactic(IntEnum):
    """CTA schedules supported by the BF16 FA4 MHA kernel."""

    WARP_16 = 0
    WARP_12 = 1


# Short aliases keep the launch API convenient and preserve the initial
# tactic-facing public API.
TACTIC_16WARP = Bf16Fa4MhaTactic.WARP_16
TACTIC_12WARP = Bf16Fa4MhaTactic.WARP_12


@dataclass(frozen=True)
class Bf16Fa4MhaConfig:
    tactic: Bf16Fa4MhaTactic = TACTIC_16WARP

    def __post_init__(self) -> None:
        try:
            tactic = Bf16Fa4MhaTactic(self.tactic)
        except (TypeError, ValueError) as exc:
            raise ValueError("tactic must be 0 (16-warp) or 1 (12-warp)") from exc
        object.__setattr__(self, "tactic", tactic)

    def validate(self) -> None:
        if not isinstance(self.tactic, Bf16Fa4MhaTactic):
            raise TypeError("tactic must be a Bf16Fa4MhaTactic")


class Bf16Fa4MhaModule:
    def __init__(self, binding, library, config: Bf16Fa4MhaConfig):
        self._binding, self._launcher, self._config = (
            binding,
            library.entry_address,
            config,
        )

    def fa_mha(
        self,
        q,
        k,
        v,
        causal=False,
        scale=1.0,
        block_table=None,
        cu_seqlens_q=None,
        kv_seqlens=None,
        max_q_len=0,
        max_kv_len=0,
    ):
        return self._binding.fa_mha(
            q,
            k,
            v,
            self._launcher,
            bool(causal),
            scale,
            block_table,
            cu_seqlens_q,
            kv_seqlens,
            max_q_len,
            max_kv_len,
        )

    def fa_mha_out(
        self,
        q,
        k,
        v,
        out,
        causal=False,
        scale=1.0,
        block_table=None,
        cu_seqlens_q=None,
        kv_seqlens=None,
        max_q_len=0,
        max_kv_len=0,
    ):
        return self._binding.fa_mha_out(
            q,
            k,
            v,
            out,
            self._launcher,
            bool(causal),
            scale,
            block_table,
            cu_seqlens_q,
            kv_seqlens,
            max_q_len,
            max_kv_len,
        )


def _kernel_cflags(config: Bf16Fa4MhaConfig, _variant) -> tuple[str, ...]:
    return (f"-DSM110A_FA4_MHA_TACTIC={int(config.tactic)}",)


_JIT = SplitCudaJit(
    name="sm110a_bf16_fa4_mha",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16_fa4_mha_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16_fa4_mha_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_bf16_fa4_mha_launch",
    config_key=lambda config: config,
    kernel_cflags=_kernel_cflags,
    make_facade=Bf16Fa4MhaModule,
)


def compile_bf16_fa4_mha(config: Bf16Fa4MhaConfig | None = None) -> Bf16Fa4MhaModule:
    config = config or Bf16Fa4MhaConfig()
    if not isinstance(config, Bf16Fa4MhaConfig):
        raise TypeError("config must be a Bf16Fa4MhaConfig")
    config.validate()
    return _JIT.compile(config)


def compile_bf16_fa4_mha_configs(configs: Iterable[Bf16Fa4MhaConfig], max_workers=None):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Bf16Fa4MhaConfig):
            raise TypeError("configs must contain Bf16Fa4MhaConfig instances")
        config.validate()
    return _JIT.compile_many(configs, max_workers)


def get_bf16_fa4_mha_module(config: Bf16Fa4MhaConfig | None = None):
    return compile_bf16_fa4_mha(config)


def _resolve_scale(q, scale):
    if scale is None:
        return 1.0 / math.sqrt(q.size(-1))
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return scale


def _config_for(
    tactic: int | Bf16Fa4MhaTactic | Bf16Fa4MhaConfig,
) -> Bf16Fa4MhaConfig:
    if isinstance(tactic, Bf16Fa4MhaConfig):
        return tactic
    return Bf16Fa4MhaConfig(tactic=tactic)


def _require_paged(block_table, kv_seqlens, max_kv_len):
    if block_table is None or kv_seqlens is None or max_kv_len <= 0:
        raise ValueError(
            "paged KV requires block_table, kv_seqlens, and positive max_kv_len"
        )


def bf16_fa4_mha(
    q,
    k,
    v,
    *,
    causal=False,
    scale=None,
    tactic=TACTIC_16WARP,
    block_table=None,
    cu_seqlens_q=None,
    kv_seqlens=None,
    max_q_len=0,
    max_kv_len=0,
):
    """Compute MHA with K/V pages `[num_pages, H, page_size, D]` (page_size divides 128)."""
    _require_paged(block_table, kv_seqlens, max_kv_len)
    return compile_bf16_fa4_mha(_config_for(tactic)).fa_mha(
        q,
        k,
        v,
        bool(causal),
        _resolve_scale(q, scale),
        block_table,
        cu_seqlens_q,
        kv_seqlens,
        max_q_len,
        max_kv_len,
    )


def bf16_fa4_mha_out(
    q,
    k,
    v,
    out,
    *,
    causal=False,
    scale=None,
    tactic=TACTIC_16WARP,
    block_table=None,
    cu_seqlens_q=None,
    kv_seqlens=None,
    max_q_len=0,
    max_kv_len=0,
):
    """Write paged-KV BF16 FA MHA to a caller-provided output."""
    _require_paged(block_table, kv_seqlens, max_kv_len)
    return compile_bf16_fa4_mha(_config_for(tactic)).fa_mha_out(
        q,
        k,
        v,
        out,
        bool(causal),
        _resolve_scale(q, scale),
        block_table,
        cu_seqlens_q,
        kv_seqlens,
        max_q_len,
        max_kv_len,
    )


class Bf16Fa4MhaRunner(BaseKernelRunner):
    namespace = "bf16_fa4_mha_runner_v1"

    def __init__(
        self,
        q,
        k,
        v,
        out,
        *,
        causal=False,
        scale=None,
        block_table=None,
        cu_seqlens_q=None,
        kv_seqlens=None,
        max_q_len=0,
        max_kv_len=0,
    ):
        super().__init__(q.device)
        _require_paged(block_table, kv_seqlens, max_kv_len)
        self.q, self.k, self.v, self.out = q, k, v, out
        self.causal, self.scale = bool(causal), _resolve_scale(q, scale)
        self.block_table, self.cu_seqlens_q, self.kv_seqlens = (
            block_table,
            cu_seqlens_q,
            kv_seqlens,
        )
        self.max_q_len, self.max_kv_len = max_q_len, max_kv_len

    @property
    def workload_key(self):
        return (
            tuple(self.q.shape),
            tuple(self.k.shape),
            tuple(self.v.shape),
            tuple(self.block_table.shape),
            self.cu_seqlens_q is not None,
            self.max_q_len,
            self.max_kv_len,
            self.causal,
            self.scale,
        )

    def compatible_candidates(self):
        tactics = (
            (Bf16Fa4MhaTactic.WARP_12,)
            if self.q.ndim == 3
            else (Bf16Fa4MhaTactic.WARP_12, Bf16Fa4MhaTactic.WARP_16)
        )
        return tuple(Bf16Fa4MhaConfig(tactic=t) for t in tactics)

    def pre_jit(self, candidate):
        return compile_bf16_fa4_mha(candidate)

    def prepare(self, candidate):
        module = self.pre_jit(candidate)
        return PreparedKernel(
            candidate,
            lambda: module.fa_mha_out(
                self.q,
                self.k,
                self.v,
                self.out,
                self.causal,
                self.scale,
                self.block_table,
                self.cu_seqlens_q,
                self.kv_seqlens,
                self.max_q_len,
                self.max_kv_len,
            ),
        )
