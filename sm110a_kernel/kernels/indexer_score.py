"""Prefill indexer score: sum_h weight[q,h] * ReLU(Q[q,h] @ K[kv])."""

from pathlib import Path

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/indexer_score/src"


class IndexerScoreModule:
    def __init__(self, binding, library, _config):
        self._binding = binding
        self._launcher = library.entry_address

    def indexer_score_prefill(self, q, k, q_scales, k_scales, head_weight):
        return self._binding.indexer_score_prefill(
            q, k, q_scales, k_scales, head_weight, self._launcher
        )

    def indexer_score_prefill_out(self, q, k, q_scales, k_scales, head_weight, out):
        return self._binding.indexer_score_prefill_out(
            q, k, q_scales, k_scales, head_weight, out, self._launcher
        )


_JIT = SplitCudaJit(
    name="sm110a_indexer_score_prefill",
    root=_ROOT,
    binding_sources=(_SOURCE / "indexer_score_prefill_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "indexer_score_prefill_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_indexer_score_prefill_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda _config, _variant: (),
    make_facade=IndexerScoreModule,
)


def compile_indexer_score_prefill() -> IndexerScoreModule:
    return _JIT.compile(None)


def indexer_score_prefill(q, k, q_scales, k_scales, head_weight):
    return compile_indexer_score_prefill().indexer_score_prefill(
        q, k, q_scales, k_scales, head_weight
    )


def indexer_score_prefill_out(q, k, q_scales, k_scales, head_weight, out):
    return compile_indexer_score_prefill().indexer_score_prefill_out(
        q, k, q_scales, k_scales, head_weight, out
    )
