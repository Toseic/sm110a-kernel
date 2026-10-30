"""Exact unordered row top-k with CGA split-K selection."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import BaseKernelRunner, PreparedKernel
from .pdl import validate_pdl

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/topk/src"


def _check_scores(scores, k):
    if not scores.is_cuda or scores.dim() != 2 or not scores.is_contiguous():
        raise RuntimeError("scores must be contiguous CUDA [Qlen, KVlen]")
    if scores.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError("scores must be float32 or bfloat16")
    if k not in (512, min(512, scores.size(1))):
        raise RuntimeError("k must be 512 or min(512, KVlen)")


@dataclass(frozen=True)
class TopKConfig:
    split_k: int


class TopKModule:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._launcher = library.entry_address
        self._config = config
        self._split_k_cache = {}

    def select_split_k(self, scores, k=512, out=None, *, options=None):
        _check_scores(scores, k)
        if out is None:
            out = torch.empty(
                (scores.size(0), min(512, scores.size(1))),
                device=scores.device,
                dtype=torch.int32,
            )
        if scores.size(1) <= 512 or not scores.size(0):
            self._binding.topk_out(scores, out, k, 1, self._launcher)
            return 1
        runner = TopKRunner(
            scores,
            out,
            k,
            stages=self._config[0],
            copy_engine=self._config[1],
            pdl=self._config[2],
            _module=self,
        )
        key = runner.workload_key
        if options is None and key in self._split_k_cache:
            return self._split_k_cache[key]
        selection = runner.tune(options)
        winner = selection.candidate.split_k
        if options is None:
            self._split_k_cache[key] = winner
        return winner

    def topk(self, scores, k=512, *, split_k=None):
        if split_k is not None:
            return self._binding.topk(scores, k, split_k, self._launcher)
        _check_scores(scores, k)
        out = torch.empty(
            (scores.size(0), min(512, scores.size(1))),
            device=scores.device,
            dtype=torch.int32,
        )
        return self.topk_out(scores, out, k)

    def topk_out(self, scores, out, k=512, *, split_k=None):
        if split_k is None:
            _check_scores(scores, k)
            split_k = (
                1 if scores.size(1) <= 512 else self.select_split_k(scores, k, out)
            )
        return self._binding.topk_out(scores, out, k, split_k, self._launcher)


_JIT = SplitCudaJit(
    name="sm110a_topk_radix",
    root=_ROOT,
    binding_sources=(_SOURCE / "topk_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "topk_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_topk_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda config, _variant: (
        f"-DSM110A_PDL={int(config[2])}",
        f"-DSM110A_TOPK_STAGES={config[0]}",
        f"-DSM110A_TOPK_TMA={int(config[1] == 'tma')}",
    ),
    make_facade=TopKModule,
)


def compile_topk(*, stages=2, copy_engine="ldgsts", pdl=False) -> TopKModule:
    """Compile staging/copy choices and all four CGA split-K variants."""
    validate_pdl(pdl)
    if isinstance(stages, bool) or not isinstance(stages, int) or not 2 <= stages <= 8:
        raise ValueError("stages must be an integer in [2, 8]")
    if copy_engine not in ("tma", "ldgsts"):
        raise ValueError("copy_engine must be 'tma' or 'ldgsts'")
    return _JIT.compile((stages, copy_engine, pdl))


def topk(scores, k=512, *, split_k=None, pdl=False):
    """Return unordered int32 [Qlen, min(512, KVlen)] indices.

    k may be 512 (default) or min(512, KVlen). KVlen <= 512 returns all
    candidate indices in ascending order without reading scores or autotuning.
    Otherwise split_k=None autotunes 1/2/4/8. Ties choose smaller indices;
    NaNs rank above +inf. Masked -inf entries participate when necessary.
    """
    return compile_topk(pdl=pdl).topk(scores, k, split_k=split_k)


def topk_out(scores, out, k=512, *, split_k=None, pdl=False):
    """Reuse output; prewarm autotuning or fix split_k before graph capture."""
    return compile_topk(pdl=pdl).topk_out(scores, out, k, split_k=split_k)


class TopKRunner(BaseKernelRunner):
    namespace = "topk512_cga_stored_keys_v5"

    def __init__(
        self,
        scores,
        out,
        k=512,
        *,
        stages=2,
        copy_engine="ldgsts",
        pdl=False,
        _module=None,
    ):
        validate_pdl(pdl)
        super().__init__(scores.device)
        _check_scores(scores, k)
        self.scores, self.out, self.k = scores, out, k
        self.stages, self.copy_engine, self.module = stages, copy_engine, _module
        self.pdl = pdl

    @property
    def workload_key(self):
        return (
            self.scores.device.index,
            tuple(self.scores.shape),
            str(self.scores.dtype),
            self.k,
            self.scores.data_ptr() % 16,
            (self.stages, self.copy_engine, self.pdl),
        )

    def compatible_candidates(self):
        if self.scores.size(1) <= 512 or not self.scores.size(0):
            return (TopKConfig(1),)
        return tuple(
            TopKConfig(s)
            for s in (1, 2, 4, 8)
            if s <= (self.scores.size(1) + 1023) // 1024
        )

    def pre_jit(self, candidate):
        return self.module or compile_topk(
            stages=self.stages, copy_engine=self.copy_engine, pdl=self.pdl
        )

    def prepare(self, candidate):
        module = self.pre_jit(candidate)
        return PreparedKernel(
            candidate,
            lambda: module._binding.topk_out(
                self.scores, self.out, self.k, candidate.split_k, module._launcher
            ),
        )
