"""MXFP4 indexer score for prefill and batched decoding."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit, source_digest
from ..runner import BaseKernelRunner, PreparedKernel
from .pdl import validate_pdl

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "csrc/mxfp4_indexer_score/src"
_DECODE_TUNE_NAMESPACE = (
    "indexer_score_decode_splitkv_v2_"
    + source_digest((_SOURCE / "indexer_score_decode_kernel.cu",), _ROOT)[:12]
)


def pack_indexer_score_prefill_scales(scales):
    """Compact blocked E8M0 tiles into the 512-byte tcgen05 scale layout.

    Prepare Q and K scales before timing a prefill call. Each 1024-byte input
    tile becomes 512 bytes; the prefill API also accepts the original layout.
    """
    if scales.dtype != torch.float8_e8m0fnu or scales.numel() % 1024:
        raise ValueError("scales must be blocked E8M0 tiles of 1024 bytes")
    return (
        scales.contiguous()
        .view(torch.uint8)
        .reshape(-1, 2, 32, 4, 4)[..., :2]
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(torch.float8_e8m0fnu)
        .flatten()
    )


class IndexerScoreModule:
    def __init__(self, binding, library, _config):
        self.pdl = _config
        self._binding = binding
        self._launcher = library.entry_address
        self._cluster_m_cache = {}

    def select_cluster_m(
        self,
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        out=None,
        *,
        causal=True,
        block_table=None,
        cu_seqlens_q=None,
        kv_seqlens=None,
        max_q_len=None,
        max_kv_len=None,
        options=None,
    ):
        q_len, kv_len, _ = _prefill_shape(
            q, k, block_table, cu_seqlens_q, max_q_len, max_kv_len
        )
        if q_len <= 2:
            return 1
        if out is None:
            shape = (
                (q.size(0), q.size(1), kv_len) if q.ndim == 4 else (q.size(0), kv_len)
            )
            out = torch.empty(shape, device=q.device, dtype=torch.bfloat16)
        runner = IndexerScorePrefillRunner(
            q,
            k,
            q_scales,
            k_scales,
            head_weight,
            out,
            causal=causal,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            kv_seqlens=kv_seqlens,
            max_q_len=q_len,
            max_kv_len=kv_len,
            _module=self,
            pdl=self.pdl,
        )
        key = runner.workload_key
        if options is None and key in self._cluster_m_cache:
            return self._cluster_m_cache[key]
        selection = runner.tune(options)
        winner = selection.candidate.cluster_m
        if options is None:
            self._cluster_m_cache[key] = winner
        return winner

    def indexer_score_prefill(
        self,
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        cluster_m=None,
        *,
        causal=True,
        block_table=None,
        cu_seqlens_q=None,
        kv_seqlens=None,
        max_q_len=None,
        max_kv_len=None,
    ):
        q_len, kv_len, _ = _prefill_shape(
            q, k, block_table, cu_seqlens_q, max_q_len, max_kv_len
        )
        if cluster_m is None:
            cluster_m = self.select_cluster_m(
                q,
                k,
                q_scales,
                k_scales,
                head_weight,
                causal=causal,
                block_table=block_table,
                cu_seqlens_q=cu_seqlens_q,
                kv_seqlens=kv_seqlens,
                max_q_len=q_len,
                max_kv_len=kv_len,
            )
        return self._binding.indexer_score_prefill(
            q,
            k,
            q_scales,
            k_scales,
            head_weight,
            cluster_m,
            bool(causal),
            block_table,
            cu_seqlens_q,
            kv_seqlens,
            q_len,
            kv_len,
            self._launcher,
        )

    def indexer_score_prefill_out(
        self,
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        out,
        cluster_m=None,
        *,
        causal=True,
        block_table=None,
        cu_seqlens_q=None,
        kv_seqlens=None,
        max_q_len=None,
        max_kv_len=None,
    ):
        q_len, kv_len, _ = _prefill_shape(
            q, k, block_table, cu_seqlens_q, max_q_len, max_kv_len
        )
        if cluster_m is None:
            cluster_m = self.select_cluster_m(
                q,
                k,
                q_scales,
                k_scales,
                head_weight,
                out,
                causal=causal,
                block_table=block_table,
                cu_seqlens_q=cu_seqlens_q,
                kv_seqlens=kv_seqlens,
                max_q_len=q_len,
                max_kv_len=kv_len,
            )
        return self._binding.indexer_score_prefill_out(
            q,
            k,
            q_scales,
            k_scales,
            head_weight,
            out,
            cluster_m,
            bool(causal),
            block_table,
            cu_seqlens_q,
            kv_seqlens,
            q_len,
            kv_len,
            self._launcher,
        )


def _prefill_shape(q, k, block_table, cu_seqlens_q, max_q_len, max_kv_len):
    batch = (
        block_table.size(0)
        if block_table is not None
        else (k.size(0) if k.ndim == 3 else 1)
    )
    q_len = (
        max_q_len
        if cu_seqlens_q is not None
        else (q.size(1) if q.ndim == 4 else q.size(0))
    )
    kv_len = (
        max_kv_len
        if block_table is not None
        else (k.size(1) if k.ndim == 3 else k.size(0))
    )
    if q_len is None or kv_len is None:
        raise ValueError(
            "max_q_len is required for packed Q and max_kv_len for paged K"
        )
    return q_len, kv_len, batch


@dataclass(frozen=True)
class IndexerScorePrefillConfig:
    cluster_m: int


_PREFILL_TUNE_NAMESPACE = (
    "indexer_score_prefill_cluster_v1_"
    + source_digest((_SOURCE / "indexer_score_prefill_kernel.cu",), _ROOT)[:12]
)


_JIT = SplitCudaJit(
    name="sm110a_indexer_score_prefill",
    root=_ROOT,
    binding_sources=(_SOURCE / "indexer_score_prefill_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "indexer_score_prefill_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_indexer_score_prefill_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda pdl, _variant: (f"-DSM110A_PDL={int(pdl)}",),
    make_facade=IndexerScoreModule,
)


def compile_indexer_score_prefill(*, pdl=False) -> IndexerScoreModule:
    validate_pdl(pdl)
    return _JIT.compile(pdl)


def select_indexer_score_prefill_cluster_m(
    q,
    k,
    q_scales,
    k_scales,
    head_weight,
    out=None,
    *,
    causal=True,
    block_table=None,
    cu_seqlens_q=None,
    kv_seqlens=None,
    max_q_len=None,
    max_kv_len=None,
    options=None,
    pdl=False,
):
    return compile_indexer_score_prefill(pdl=pdl).select_cluster_m(
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        out,
        causal=causal,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        kv_seqlens=kv_seqlens,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        options=options,
    )


def indexer_score_prefill(
    q,
    k,
    q_scales,
    k_scales,
    head_weight,
    cluster_m=None,
    *,
    causal=True,
    block_table=None,
    cu_seqlens_q=None,
    kv_seqlens=None,
    max_q_len=None,
    max_kv_len=None,
    pdl=False,
):
    return compile_indexer_score_prefill(pdl=pdl).indexer_score_prefill(
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        cluster_m,
        causal=causal,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        kv_seqlens=kv_seqlens,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
    )


def indexer_score_prefill_out(
    q,
    k,
    q_scales,
    k_scales,
    head_weight,
    out,
    cluster_m=None,
    *,
    causal=True,
    block_table=None,
    cu_seqlens_q=None,
    kv_seqlens=None,
    max_q_len=None,
    max_kv_len=None,
    pdl=False,
):
    return compile_indexer_score_prefill(pdl=pdl).indexer_score_prefill_out(
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        out,
        cluster_m,
        causal=causal,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        kv_seqlens=kv_seqlens,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
    )


class IndexerScoreDecodeModule:
    def __init__(self, binding, library, _config):
        self.pdl = _config
        self._binding = binding
        self._launcher = library.entry_address
        self._split_kv_cache = {}

    def select_split_kv(
        self,
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        out=None,
        *,
        block_table=None,
        kv_seqlens=None,
        max_kv_len=None,
        options=None,
    ):
        kv_len = _decode_kv_len(k, block_table, max_kv_len)
        if out is None:
            out = torch.empty(
                (q.size(0), kv_len), device=q.device, dtype=torch.bfloat16
            )
        runner = IndexerScoreDecodeRunner(
            q,
            k,
            q_scales,
            k_scales,
            head_weight,
            out,
            block_table=block_table,
            kv_seqlens=kv_seqlens,
            max_kv_len=kv_len,
            _module=self,
            pdl=self.pdl,
        )
        key = runner.workload_key
        if options is None and key in self._split_kv_cache:
            return self._split_kv_cache[key]
        selection = runner.tune(options)
        winner = selection.candidate.split_kv
        if options is None:
            self._split_kv_cache[key] = winner
        return winner

    def indexer_score_decode(
        self,
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        split_kv=None,
        *,
        block_table=None,
        kv_seqlens=None,
        max_kv_len=None,
    ):
        kv_len = _decode_kv_len(k, block_table, max_kv_len)
        if split_kv is None:
            split_kv = self.select_split_kv(
                q,
                k,
                q_scales,
                k_scales,
                head_weight,
                block_table=block_table,
                kv_seqlens=kv_seqlens,
                max_kv_len=kv_len,
            )
        return self._binding.indexer_score_decode(
            q,
            k,
            q_scales,
            k_scales,
            head_weight,
            split_kv,
            block_table,
            kv_seqlens,
            kv_len,
            self._launcher,
        )

    def indexer_score_decode_out(
        self,
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        out,
        split_kv=None,
        *,
        block_table=None,
        kv_seqlens=None,
        max_kv_len=None,
    ):
        kv_len = _decode_kv_len(k, block_table, max_kv_len)
        if split_kv is None:
            split_kv = self.select_split_kv(
                q,
                k,
                q_scales,
                k_scales,
                head_weight,
                out,
                block_table=block_table,
                kv_seqlens=kv_seqlens,
                max_kv_len=kv_len,
            )
        return self._binding.indexer_score_decode_out(
            q,
            k,
            q_scales,
            k_scales,
            head_weight,
            out,
            split_kv,
            block_table,
            kv_seqlens,
            kv_len,
            self._launcher,
        )


def _decode_kv_len(k, block_table, max_kv_len):
    if block_table is not None:
        if max_kv_len is None:
            raise ValueError("max_kv_len is required for paged K")
        return max_kv_len
    return k.size(1)


@dataclass(frozen=True)
class IndexerScoreDecodeConfig:
    split_kv: int


_DECODE_JIT = SplitCudaJit(
    name="sm110a_indexer_score_decode",
    root=_ROOT,
    binding_sources=(_SOURCE / "indexer_score_decode_torch_binding.cpp",),
    kernel_sources=(_SOURCE / "indexer_score_decode_kernel.cu",),
    kernel_dependencies=(),
    kernel_symbol="sm110a_indexer_score_decode_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda pdl, _variant: (f"-DSM110A_PDL={int(pdl)}",),
    make_facade=IndexerScoreDecodeModule,
)


def compile_indexer_score_decode(*, pdl=False) -> IndexerScoreDecodeModule:
    validate_pdl(pdl)
    return _DECODE_JIT.compile(pdl)


def select_indexer_score_decode_split_kv(
    q,
    k,
    q_scales,
    k_scales,
    head_weight,
    out=None,
    *,
    block_table=None,
    kv_seqlens=None,
    max_kv_len=None,
    options=None,
    pdl=False,
):
    return compile_indexer_score_decode(pdl=pdl).select_split_kv(
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        out,
        block_table=block_table,
        kv_seqlens=kv_seqlens,
        max_kv_len=max_kv_len,
        options=options,
    )


def indexer_score_decode(
    q,
    k,
    q_scales,
    k_scales,
    head_weight,
    split_kv=None,
    *,
    block_table=None,
    kv_seqlens=None,
    max_kv_len=None,
    pdl=False,
):
    return compile_indexer_score_decode(pdl=pdl).indexer_score_decode(
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        split_kv,
        block_table=block_table,
        kv_seqlens=kv_seqlens,
        max_kv_len=max_kv_len,
    )


def indexer_score_decode_out(
    q,
    k,
    q_scales,
    k_scales,
    head_weight,
    out,
    split_kv=None,
    *,
    block_table=None,
    kv_seqlens=None,
    max_kv_len=None,
    pdl=False,
):
    return compile_indexer_score_decode(pdl=pdl).indexer_score_decode_out(
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        out,
        split_kv,
        block_table=block_table,
        kv_seqlens=kv_seqlens,
        max_kv_len=max_kv_len,
    )


class IndexerScorePrefillRunner(BaseKernelRunner):
    namespace = _PREFILL_TUNE_NAMESPACE

    def __init__(
        self,
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        out,
        *,
        causal=True,
        block_table=None,
        cu_seqlens_q=None,
        kv_seqlens=None,
        max_q_len=None,
        max_kv_len=None,
        _module=None,
        pdl=False,
    ):
        validate_pdl(pdl)
        super().__init__(q.device)
        self.q, self.k, self.q_scales, self.k_scales, self.head_weight, self.out = (
            q,
            k,
            q_scales,
            k_scales,
            head_weight,
            out,
        )
        self.causal, self.block_table, self.cu_seqlens_q, self.kv_seqlens = (
            causal,
            block_table,
            cu_seqlens_q,
            kv_seqlens,
        )
        self.q_len, self.kv_len, self.batch = _prefill_shape(
            q, k, block_table, cu_seqlens_q, max_q_len, max_kv_len
        )
        self.module = _module
        self.pdl = pdl

    @property
    def workload_key(self):
        return (
            self.device.index,
            self.pdl,
            self.batch,
            self.q_len,
            self.kv_len,
            self.k.size(1) if self.block_table is not None else 0,
            self.cu_seqlens_q is not None,
            self.kv_seqlens is not None,
            bool(self.causal),
        )

    def compatible_candidates(self):
        return tuple(
            IndexerScorePrefillConfig(c) for c in ((1,) if self.q_len <= 2 else (1, 2))
        )

    def pre_jit(self, candidate):
        return self.module or compile_indexer_score_prefill(pdl=self.pdl)

    def prepare(self, candidate):
        module = self.pre_jit(candidate)
        return PreparedKernel(
            candidate,
            lambda: module.indexer_score_prefill_out(
                self.q,
                self.k,
                self.q_scales,
                self.k_scales,
                self.head_weight,
                self.out,
                candidate.cluster_m,
                causal=self.causal,
                block_table=self.block_table,
                cu_seqlens_q=self.cu_seqlens_q,
                kv_seqlens=self.kv_seqlens,
                max_q_len=self.q_len,
                max_kv_len=self.kv_len,
            ),
        )


class IndexerScoreDecodeRunner(BaseKernelRunner):
    namespace = _DECODE_TUNE_NAMESPACE

    def __init__(
        self,
        q,
        k,
        q_scales,
        k_scales,
        head_weight,
        out,
        *,
        block_table=None,
        kv_seqlens=None,
        max_kv_len=None,
        _module=None,
        pdl=False,
    ):
        validate_pdl(pdl)
        super().__init__(q.device)
        self.q, self.k, self.q_scales, self.k_scales, self.head_weight, self.out = (
            q,
            k,
            q_scales,
            k_scales,
            head_weight,
            out,
        )
        self.block_table, self.kv_seqlens = block_table, kv_seqlens
        self.kv_len = _decode_kv_len(k, block_table, max_kv_len)
        self.module = _module
        self.pdl = pdl

    @property
    def workload_key(self):
        return (
            self.device.index,
            self.pdl,
            self.q.size(0),
            self.kv_len,
            self.k.size(1) if self.block_table is not None else 0,
            self.kv_seqlens is not None,
        )

    def compatible_candidates(self):
        return tuple(
            IndexerScoreDecodeConfig(s)
            for s in (1, 2, 4, 8)
            if s <= (self.kv_len + 127) // 128
        )

    def pre_jit(self, candidate):
        return self.module or compile_indexer_score_decode(pdl=self.pdl)

    def prepare(self, candidate):
        module = self.pre_jit(candidate)
        return PreparedKernel(
            candidate,
            lambda: module.indexer_score_decode_out(
                self.q,
                self.k,
                self.q_scales,
                self.k_scales,
                self.head_weight,
                self.out,
                candidate.split_kv,
                block_table=self.block_table,
                kv_seqlens=self.kv_seqlens,
                max_kv_len=self.kv_len,
            ),
        )
