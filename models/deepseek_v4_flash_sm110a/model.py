"""DeepSeek-V4-Flash runtime backed by this repository's SM110A kernels.

This module reuses the official readable model's module/state-dict layout while
swapping the operator backend for one forward call, so converted official
checkpoints load without another conversion step.
"""

# Remaining native-kernel support (including inherited reference operators):
# TODO(sm110a): fuse HC pre/projection, split-Sinkhorn, post mixing, and HC head.
# TODO(sm110a): implement FP32 compressor projections and fused ratio-4 overlap /
# ratio-128 pooling, incremental state updates, and compressed KV cache writes.
# TODO(sm110a): add RMSNorm, Q normalization, RoPE/inverse RoPE, and Hadamard
# rotation (currently an external extension or a PyTorch fallback).
# TODO(sm110a): fuse FP8 QAT quantize-dequantize and its inplace store.
# TODO(sm110a): connect existing MXFP4 indexer score prefill/decode kernels with
# packed Q/K scales and KV cache updates; adapt compressed causal masking and
# validate rounding against the current BF16/QAT score producer.
# TODO(sm110a): fuse FP32 MoE gate projection, scoring, hash/score selection,
# weight normalization, and metadata construction directly from expert IDs.
# TODO(sm110a): add deterministic reduction and PDL to the FC1/FC2 fused backend.
# TODO(sm110a): fuse shared-expert clamp/SwiGLU and shared/routed output addition.
# TODO(sm110a): fuse index generation, cache writes, padding, and layout
# conversions where useful; extend the LM-head GEMM to unaligned dimensions.
# TODO(sm110a): specialize small-M dense/MoE decode instead of padding tokens
# to 128 rows; cache wo_a's fixed grouped-layout scheduling metadata.
# TODO(sm110a): extend non-aligned GEMMs, non-E8M0 scales, other FP4 quantizer
# modes, sparse MLA head counts/dimensions, and multi-GPU fused MoE support.

from __future__ import annotations

import math
import sys
from collections import Counter, OrderedDict
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import torch
import torch.nn.functional as F

_OFFICIAL_INFERENCE_ROOT = (
    Path(__file__).resolve().parents[1] / "deepseek_v4_flash" / "inference"
)
if str(_OFFICIAL_INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_INFERENCE_ROOT))

_official = import_module("models.deepseek_v4_flash.inference.model")

from sm110a_kernel import (
    Bf16Fa4SparseMlaPrefillRunner,
    Bf16Fa4SparseMlaRunner,
    Bf16GemmRunner,
    Bf16GroupedMGemmRunner,
    Bf16ToFp4QuantRunner,
    Fp8ActQuantRunner,
    Fp8GemmRunner,
    Fp8Mxfp4GemmRunner,
    Mxfp4Fc1Fc2Runner,
    Mxfp4MoeRunner,
    TopKRunner,
    clear_kernel_runner_cache,
    compact_mxfp4_weight_scales,
    to_blocked_fp8_scales,
    to_blocked_fp8mxfp4_scales,
)

# Scoped to one model forward, like the official backend patches below.
_DECODE_ENABLED = False
_DECODE_PDL = False

ModelArgs = _official.ModelArgs
_OFFICIAL_FP4_ACT_QUANT = _official.fp4_act_quant

_STATS: Counter[str] = Counter()
_SCALE_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_SCALE_CACHE_BYTES = 0
_SCALE_CACHE_LIMIT = 256 * 2**20


def reset_backend_stats() -> None:
    _STATS.clear()


def backend_stats() -> dict[str, int]:
    return dict(sorted(_STATS.items()))


def clear_backend_cache() -> None:
    global _SCALE_CACHE_BYTES
    _SCALE_CACHE.clear()
    clear_kernel_runner_cache()
    _SCALE_CACHE_BYTES = 0


def _cache_scale(key: tuple, make) -> torch.Tensor:
    global _SCALE_CACHE_BYTES
    cached = _SCALE_CACHE.pop(key, None)
    if cached is not None:
        _SCALE_CACHE[key] = cached
        return cached
    value = make()
    size = value.numel() * value.element_size()
    while _SCALE_CACHE and _SCALE_CACHE_BYTES + size > _SCALE_CACHE_LIMIT:
        _, evicted = _SCALE_CACHE.popitem(last=False)
        _SCALE_CACHE_BYTES -= evicted.numel() * evicted.element_size()
    if size <= _SCALE_CACHE_LIMIT:
        _SCALE_CACHE[key] = value
        _SCALE_CACHE_BYTES += size
    return value


def _tensor_key(tensor: torch.Tensor, layout: str) -> tuple:
    return (
        layout,
        tensor.device.type,
        tensor.device.index,
        tensor.data_ptr(),
        tensor._version,
        tuple(tensor.shape),
    )


def _pad_quantized_rows(
    values: torch.Tensor, scales: torch.Tensor, multiple: int = 128
) -> tuple[torch.Tensor, torch.Tensor, int]:
    rows = values.size(0)
    padded_rows = math.ceil(rows / multiple) * multiple
    if padded_rows == rows:
        return values.contiguous(), scales.contiguous(), rows
    values = torch.cat(
        (values, values.new_zeros((padded_rows - rows, values.size(1)))), dim=0
    )
    if scales.dtype == torch.float8_e8m0fnu:
        # E8M0 byte 127 is exactly 1.0; the padded values are zero either way.
        padding = torch.full(
            (padded_rows - rows, scales.size(1)),
            127,
            device=scales.device,
            dtype=torch.uint8,
        ).view(torch.float8_e8m0fnu)
    else:
        padding = scales.new_ones((padded_rows - rows, scales.size(1)))
    return values.contiguous(), torch.cat((scales, padding), dim=0).contiguous(), rows


def _blocked_activation_scales(scales: torch.Tensor) -> torch.Tensor:
    if scales.dtype != torch.float8_e8m0fnu:
        raise TypeError("SM110A block-scaled GEMMs require E8M0 activation scales")
    expanded = scales.view(torch.uint8).repeat_interleave(4, dim=1)
    return to_blocked_fp8mxfp4_scales(expanded.view(torch.float8_e8m0fnu).contiguous())


def _blocked_fp8_weight_scales(scales: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    if scales.dtype != torch.float8_e8m0fnu:
        raise TypeError("SM110A FP8 GEMM requires E8M0 weight scales")

    def make() -> torch.Tensor:
        expected = (math.ceil(rows / 128), math.ceil(k / 128))
        if tuple(scales.shape) != expected:
            raise ValueError(
                f"official FP8 weight scales must have shape {expected}, got {tuple(scales.shape)}"
            )
        logical = scales.view(torch.uint8).repeat_interleave(128, 0)
        logical = logical[:rows].repeat_interleave(4, 1)[:, : k // 32]
        return to_blocked_fp8_scales(logical.view(torch.float8_e8m0fnu).contiguous())

    return _cache_scale(_tensor_key(scales, "fp8-weight"), make)


def _blocked_mxfp4_weight_scales(scales: torch.Tensor) -> torch.Tensor:
    if scales.dtype != torch.float8_e8m0fnu:
        raise TypeError("SM110A MXFP4 GEMM requires E8M0 weight scales")
    return _cache_scale(
        _tensor_key(scales, "mxfp4-weight"),
        lambda: to_blocked_fp8mxfp4_scales(scales.contiguous()),
    )


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    scale_mode = "ue8m0" if scale_dtype == torch.float8_e8m0fnu else "fp32"
    width = x.size(-1)
    flat = x.contiguous().view(-1, width)
    values = torch.empty_like(flat, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (flat.size(0), width // block_size),
        device=flat.device,
        dtype=torch.float8_e8m0fnu if scale_mode == "ue8m0" else torch.float32,
    )
    Fp8ActQuantRunner(flat, values, scales, block_size, scale_mode).run()
    values = values.view_as(x)
    scales = scales.view(*x.shape[:-1], width // block_size)
    _STATS["sm110a.fp8_act_quant"] += 1
    if not inplace:
        return values, scales

    # TODO(sm110a): replace the PyTorch dequant/copy with a fused FP8
    # quantize-dequantize kernel.  The quantization itself is already SM110A.
    restored = values.float() * scales.float().repeat_interleave(block_size, dim=-1)
    x.copy_(restored.to(x.dtype))
    _STATS["torch.fp8_dequant_inplace_todo"] += 1
    return x


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    if block_size != 32 or not inplace:
        # TODO(sm110a): add the remaining FP4 quantizer modes used by future
        # model variants.  DSV4-Flash only requests block-32 inplace QAT.
        _STATS["torch.fp4_quant_todo"] += 1
        return _OFFICIAL_FP4_ACT_QUANT(x, block_size, inplace)
    width = x.size(-1)
    flat = x.contiguous().view(-1, width)
    restored = torch.empty_like(flat)
    scales = torch.empty(
        (flat.size(0), width // 32), device=flat.device, dtype=torch.float8_e8m0fnu
    )
    Bf16ToFp4QuantRunner(flat, restored, scales, dequant=True).run()
    restored = restored.view_as(x)
    x.copy_(restored)
    _STATS["sm110a.bf16tofp4_quant_dequant"] += 1
    return x


def fp8_gemm_backend(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    k = a.size(-1)
    rows = a.numel() // k
    n = b.size(0)
    a2d, a_s2d, valid_rows = _pad_quantized_rows(
        a.reshape(rows, k), a_s.reshape(rows, -1)
    )
    if n % 128 or k % 128 or a_s2d.dtype != torch.float8_e8m0fnu:
        # TODO(sm110a): support non-E8M0 and non-128-aligned dense FP8 model
        # shapes.  This branch is not used by the DSV4-Flash benchmark config.
        _STATS["torch.fp8_linear_todo"] += 1
        lhs = a2d.float() * a_s2d.float().repeat_interleave(128, dim=1)
        weight_scale = b_s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
        result = lhs @ (b.float() * weight_scale[:n, :k]).t()
    else:
        out = torch.empty((a2d.size(0), n), device=a.device, dtype=torch.bfloat16)
        result = Fp8GemmRunner(
            a2d,
            b.contiguous(),
            _blocked_activation_scales(a_s2d),
            _blocked_fp8_weight_scales(b_s, n, k),
            out,
        ).run()
        _STATS["sm110a.fp8_gemm"] += 1
    return result[:valid_rows].view(*a.shape[:-1], n).to(torch.bfloat16)


def fp4_gemm_backend(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    k = a.size(-1)
    rows = a.numel() // k
    n = b.size(0)
    a2d, a_s2d, valid_rows = _pad_quantized_rows(
        a.reshape(rows, k), a_s.reshape(rows, -1)
    )
    if n % 128 or k % 128 or a_s2d.dtype != torch.float8_e8m0fnu:
        # TODO(sm110a): add odd-shape/FP32-scale W4A8 kernels.  Kept as a
        # clear functional fallback for configurations outside DSV4-Flash.
        _STATS["torch.mxfp4_linear_todo"] += 1
        result = _torch_fp4_linear(a2d, a_s2d, b, b_s)
    else:
        out = torch.empty((a2d.size(0), n), device=a.device, dtype=torch.bfloat16)
        result = Fp8Mxfp4GemmRunner(
            a2d,
            b.view(torch.uint8).contiguous(),
            _blocked_mxfp4_weight_scales(b_s),
            out,
            sfa=_blocked_activation_scales(a_s2d),
        ).run()
        _STATS["sm110a.fp8mxfp4_gemm"] += 1
    return result[:valid_rows].view(*a.shape[:-1], n).to(torch.bfloat16)


def _torch_fp4_linear(a, a_s, b, b_s):
    lhs = a.float() * a_s.float().repeat_interleave(128, dim=1)
    raw = b.view(torch.uint8)
    codes = torch.stack((raw & 0xF, raw >> 4), dim=-1).reshape(b.size(0), -1)
    magnitudes = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device=b.device,
        dtype=torch.float32,
    )
    weight = magnitudes[(codes & 7).long()]
    weight = torch.where((codes & 8) != 0, -weight, weight)
    weight *= b_s.float().repeat_interleave(32, dim=1)
    return (lhs @ weight.t()).to(torch.bfloat16)


def _decode_fp8_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    k, n = x.size(-1), weight.size(0)
    rows = x.numel() // k
    padded_rows = math.ceil(rows / 128) * 128
    a = x.reshape(rows, k).contiguous()
    if padded_rows != rows:
        a = torch.cat((a, a.new_zeros((padded_rows - rows, k))), dim=0)
    sfb = _blocked_fp8_weight_scales(weight.scale, n, k)
    b = weight.contiguous()
    aq = torch.empty(a.shape, device=x.device, dtype=torch.float8_e4m3fn)
    sfa = torch.empty(
        (padded_rows * k // 8,), device=x.device, dtype=torch.float8_e8m0fnu
    )
    out = torch.empty((padded_rows, n), device=x.device, dtype=torch.bfloat16)
    Fp8GemmRunner(aq, b, sfa, sfb, out, pdl=_DECODE_PDL, input=a).run()
    _STATS["sm110a.fp8_act_quant_blocked"] += 1
    _STATS["sm110a.fp8_gemm"] += 1
    return out[:rows].view(*x.shape[:-1], n)


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    assert bias is None
    if weight.dtype == torch.float4_e2m1fn_x2:
        values, scales = act_quant(
            x, _official.block_size, _official.scale_fmt, _official.scale_dtype
        )
        return fp4_gemm_backend(
            values, scales, weight, weight.scale, _official.scale_dtype
        )
    if weight.dtype == torch.float8_e4m3fn:
        if (
            _DECODE_ENABLED
            and x.dtype == torch.bfloat16
            and x.size(-1) % 128 == 0
            and weight.size(0) % 128 == 0
            and _official.scale_dtype == torch.float8_e8m0fnu
        ):
            return _decode_fp8_linear(x, weight)
        values, scales = act_quant(
            x, _official.block_size, _official.scale_fmt, _official.scale_dtype
        )
        return fp8_gemm_backend(
            values, scales, weight, weight.scale, _official.scale_dtype
        )
    if x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16:
        k = x.size(-1)
        rows = x.numel() // k
        n = weight.size(0)
        padded_rows = math.ceil(rows / 128) * 128
        a = x.reshape(rows, k).contiguous()
        if padded_rows != rows:
            a = torch.cat((a, a.new_zeros((padded_rows - rows, k))), dim=0)
        if n % 32:
            # TODO(sm110a): add an epilogue for BF16 projection widths that
            # are not a multiple of the current 32-column store atom.
            _STATS["torch.bf16_linear_todo"] += 1
            return F.linear(x, weight)
        out = torch.empty((padded_rows, n), device=x.device, dtype=torch.bfloat16)
        Bf16GemmRunner(a, weight.contiguous(), out).run()
        _STATS["sm110a.bf16_gemm"] += 1
        return out[:rows].view(*x.shape[:-1], n)

    # TODO(sm110a): RMS/HC/router projections are FP32 in the reference
    # checkpoint.  Replace these F.linear calls when an FP32-capable SM110A
    # kernel (or a fused surrounding operator) is available.
    _STATS["torch.fp32_linear_todo"] += 1
    return F.linear(x, weight)


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    b, seqlen, heads, dim = q.shape
    if heads == 64 and dim == 512:
        q_bhsd = q.transpose(1, 2).contiguous()
        out = torch.empty_like(q_bhsd)
        if seqlen == 1:
            Bf16Fa4SparseMlaRunner(
                q_bhsd,
                kv.contiguous(),
                topk_idxs[:, 0].contiguous(),
                attn_sink.contiguous(),
                out,
                scale=softmax_scale,
            ).run()
            _STATS["sm110a.sparse_mla_decode"] += 1
        else:
            Bf16Fa4SparseMlaPrefillRunner(
                q_bhsd,
                kv.contiguous(),
                topk_idxs.contiguous(),
                attn_sink.contiguous(),
                out,
                scale=softmax_scale,
            ).run()
            _STATS["sm110a.sparse_mla_prefill"] += 1
        return out.transpose(1, 2).contiguous()

    # TODO(sm110a): add sparse MLA families for other head counts/dimensions.
    _STATS["torch.sparse_attention_todo"] += 1
    return _torch_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)


def _torch_sparse_attn(q, kv, sink, indices, scale):
    safe = indices.clamp_min(0).long()
    batch = torch.arange(q.size(0), device=q.device)[:, None, None]
    selected = kv[batch, safe].float()
    scores = torch.einsum("bshd,bskd->bshk", q.float(), selected) * scale
    scores.masked_fill_(indices[:, :, None, :] < 0, -torch.inf)
    sink_scores = sink.view(1, 1, -1, 1).expand(*scores.shape[:-1], 1)
    probabilities = torch.cat((scores, sink_scores), dim=-1).softmax(-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probabilities, selected).to(q.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    # TODO(sm110a): replace this functional PyTorch implementation with a
    # native fused HC split + Sinkhorn kernel.
    _STATS["torch.hc_sinkhorn_todo"] += 1
    pre_logits, post_logits, comb_logits = torch.split(
        mixes, (hc_mult, hc_mult, hc_mult * hc_mult), dim=-1
    )
    pre = torch.sigmoid(pre_logits * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(post_logits * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = comb_logits * hc_scale[2] + hc_base[2 * hc_mult :]
    comb = comb.unflatten(-1, (hc_mult, hc_mult)).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


class Sm110aAttention(_official.Attention):
    """Reference attention flow with one layout-driven grouped wo_a GEMM."""

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        if ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        _official.apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = self.kv_norm(self.wkv(x))
        _official.apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant(kv[..., :-rd], 64, _official.scale_fmt, _official.scale_dtype, True)
        topk_idxs = _official.get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if ratio:
            offset = seqlen if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = _official.get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
            if ratio:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if ratio:
                self.compressor(x, start_pos)
            o = sparse_attn(
                q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale
            )
        _official.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
        return self.wo_b(self.project_output(o).flatten(2))

    def project_output(self, o: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = o.shape[:2]
        groups, rows = self.n_local_groups, bsz * seqlen
        # Group-major A: each contiguous segment of `rows` tokens selects one
        # weight matrix. The layout schedules all groups in a single GEMM.
        grouped = o.reshape(rows, groups, -1).transpose(0, 1).contiguous()
        a = grouped.view(groups * rows, -1)
        weights = self.wo_a.weight.view(groups, self.o_lora_rank, -1).contiguous()
        layout = torch.arange(groups, device=o.device, dtype=torch.int32)
        layout = layout.repeat_interleave(rows)
        out = torch.empty((a.size(0), self.o_lora_rank), device=o.device, dtype=o.dtype)
        Bf16GroupedMGemmRunner(a, weights, layout, out, dynamic_metadata=True).run()
        _STATS["sm110a.wo_a_grouped_m_gemm"] += 1
        return (
            out.view(groups, rows, self.o_lora_rank)
            .transpose(0, 1)
            .reshape(bsz, seqlen, groups, self.o_lora_rank)
        )


class Sm110aIndexer(_official.Indexer):
    """CSA indexer with SM110A row top-k after the reference score producer."""

    def forward(self, x, qr, start_pos: int, offset: int):
        bsz, seqlen, _ = x.shape
        ratio, rd = self.compress_ratio, self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        _official.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        q = _official.rotate_activation(q)
        fp4_act_quant(q, _official.fp4_block_size, True)
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        # TODO(sm110a): replace the reference QAT/BF16 score producer with
        # the native MXFP4 indexer score kernel and its packed KV cache.
        scores = torch.einsum(
            "bshd,btd->bsht", q, self.kv_cache[:bsz, : end_pos // ratio]
        )
        scores = (scores.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if _official.world_size > 1:
            torch.distributed.all_reduce(scores)
        if start_pos == 0:
            valid_groups = (
                torch.arange(1, seqlen + 1, device=x.device)[:, None] // ratio
            )
            mask = torch.arange(seqlen // ratio, device=x.device) >= valid_groups
            scores = scores.masked_fill(mask, -torch.inf)

        width = scores.size(-1)
        k = min(self.index_topk, width)
        indices = torch.empty(
            (bsz * seqlen, k), device=scores.device, dtype=torch.int32
        )
        TopKRunner(scores.contiguous().view(bsz * seqlen, width), indices, k).run()
        indices = indices.view(bsz, seqlen, k)
        _STATS["sm110a.indexer_topk"] += 1
        if start_pos == 0:
            indices = torch.where(indices >= valid_groups, -1, indices + offset)
        else:
            indices = indices + offset
        return indices


class Sm110aMoE(_official.MoE):
    """Official MoE state layout with separate or fused SM110A FC1/FC2."""

    def prepare_fused_weights(self) -> None:
        if getattr(self, "_sm110a_prepared", False):
            return
        if _official.world_size != 1:
            raise NotImplementedError(
                "TODO(sm110a): fused DSV4 MoE currently supports one GPU only"
            )
        if not self.experts or any(expert is None for expert in self.experts):
            raise RuntimeError("all single-GPU experts must be resident before packing")

        first = self.experts[0]
        experts = self.n_routed_experts
        inter_dim = first.w1.out_features
        dim = self.dim
        if self._sm110a_moe_backend == "fused" and (dim % 256 or inter_dim != 2048):
            raise ValueError("fused MoE requires dim divisible by 256, inter_dim=2048")
        device = first.w1.weight.device
        fp4 = torch.float4_e2m1fn_x2
        self._sm110a_swiglu_limit = first.swiglu_limit

        fc1_weight = torch.empty(
            (experts, 2 * inter_dim, dim // 2), device=device, dtype=fp4
        )
        fc2_weight = torch.empty(
            (experts, dim, inter_dim // 2), device=device, dtype=fp4
        )
        fc1_scale_parts = []
        fc2_scale_parts = []
        for expert_id, expert in enumerate(self.experts):
            gate = expert.w1.weight.view(inter_dim // 64, 64, dim // 2)
            up = expert.w3.weight.view(inter_dim // 64, 64, dim // 2)
            fc1_weight[expert_id].copy_(
                torch.stack((gate, up), dim=1).reshape(2 * inter_dim, dim // 2)
            )
            gate_scale = expert.w1.scale.view(inter_dim // 64, 64, dim // 32)
            up_scale = expert.w3.scale.view(inter_dim // 64, 64, dim // 32)
            fc1_logical_scale = torch.stack((gate_scale, up_scale), dim=1).reshape(
                2 * inter_dim, dim // 32
            )
            fc1_scale_parts.append(
                to_blocked_fp8mxfp4_scales(fc1_logical_scale.contiguous())
            )

            fc2_weight[expert_id].copy_(expert.w2.weight)
            fc2_scale_parts.append(
                to_blocked_fp8mxfp4_scales(expert.w2.scale.contiguous())
            )

        self.register_buffer("sm110a_fc1_weight", fc1_weight, persistent=False)
        self.register_buffer(
            "sm110a_fc1_scale", torch.cat(fc1_scale_parts), persistent=False
        )
        self.register_buffer("sm110a_fc2_weight", fc2_weight, persistent=False)
        self.register_buffer(
            "sm110a_fc2_scale", torch.cat(fc2_scale_parts), persistent=False
        )
        if self._sm110a_compact_weight_scales:
            # Static model weights are packed once; release padded scales.
            self.sm110a_fc1_scale = compact_mxfp4_weight_scales(self.sm110a_fc1_scale)
            self.sm110a_fc2_scale = compact_mxfp4_weight_scales(self.sm110a_fc2_scale)
        self._sm110a_inter_dim = inter_dim

        # The fused tensors now own the routed-expert weights.  Drop the
        # official per-expert modules instead of retaining a second 3+ GiB
        # copy per layer.
        self.experts = torch.nn.ModuleList()
        self._sm110a_prepared = True

    def _run_experts(self, x, weights, router, topk):
        out = torch.empty_like(x)
        semantics = dict(
            swiglu_limit=self._sm110a_swiglu_limit,
            compact_weight_scales=self._sm110a_compact_weight_scales,
            routing_weight_position=self._sm110a_weight_position,
            activation_bf16_rounding=self._sm110a_weight_position == "fc1",
        )
        if self._sm110a_moe_backend == "fused":
            aq = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            sfa = torch.empty(
                (x.numel() // 8,), device=x.device, dtype=torch.float8_e8m0fnu
            )
            routing_weights = weights.to(
                torch.float32
                if self._sm110a_weight_position == "fc1"
                else torch.bfloat16
            ).contiguous()
            runner = Mxfp4Fc1Fc2Runner(
                aq,
                self.sm110a_fc1_weight,
                sfa,
                self.sm110a_fc1_scale,
                self.sm110a_fc2_weight,
                self.sm110a_fc2_scale,
                out,
                router=router,
                routing_weights=routing_weights,
                topk=topk,
                input=x,
                dynamic_metadata=True,
                **semantics,
            )
        else:
            runner = Mxfp4MoeRunner(
                x,
                self.sm110a_fc1_weight,
                self.sm110a_fc1_scale,
                self.sm110a_fc2_weight,
                self.sm110a_fc2_scale,
                router,
                weights,
                topk,
                out,
                deterministic=self._sm110a_deterministic_fc2,
                pdl=_DECODE_ENABLED and self._sm110a_pdl,
                **semantics,
            )
        return runner.run()

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_sm110a_prepared", False):
            self.prepare_fused_weights()

        shape = x.shape
        x = x.view(-1, self.dim)
        tokens = x.size(0)
        weights, indices = self.gate(x, input_ids.flatten())
        topk = self.n_activated_experts
        padded_tokens = math.ceil(tokens / 128) * 128

        if padded_tokens != tokens:
            x_padded = torch.cat(
                (x, x.new_zeros((padded_tokens - tokens, self.dim))), dim=0
            )
            weights_padded = torch.cat(
                (
                    weights,
                    weights.new_zeros((padded_tokens - tokens, topk)),
                ),
                dim=0,
            )
            dummy_indices = torch.arange(
                topk, device=indices.device, dtype=indices.dtype
            ).expand(padded_tokens - tokens, -1)
            indices_padded = torch.cat((indices, dummy_indices), dim=0)
        else:
            x_padded, weights_padded, indices_padded = x, weights, indices

        # FC1 metadata performs its own top-k.  Synthetic descending values
        # preserve the exact order returned by the official gate, including
        # hash-routed layers. The selected compile-time weighting mode receives
        # the real routing weights without multiplying them a second time.
        router = torch.full(
            (padded_tokens, self.n_routed_experts),
            -torch.inf,
            device=x.device,
            dtype=torch.bfloat16,
        )
        rank_values = torch.arange(
            topk, 0, -1, device=x.device, dtype=torch.bfloat16
        ).expand(padded_tokens, -1)
        router.scatter_(1, indices_padded.long(), rank_values)

        out = self._run_experts(x_padded.contiguous(), weights_padded, router, topk)
        if self._sm110a_moe_backend == "fused":
            _STATS["sm110a.mxfp4_fc1fc2"] += 1
        else:
            _STATS["sm110a.mxfp4_fc1"] += 1
            _STATS["sm110a.mxfp4_fc2"] += 1

        # The shared expert is dense FP8 in the official checkpoint and goes
        # through the SM110A dense backend installed by the forward context.
        out = out[:tokens] + self.shared_experts(x)
        return out.to(x.dtype).view(shape)


_PATCHES = {
    "act_quant": act_quant,
    "fp4_act_quant": fp4_act_quant,
    "fp8_gemm": fp8_gemm_backend,
    "fp4_gemm": fp4_gemm_backend,
    "linear": linear,
    "sparse_attn": sparse_attn,
    "hc_split_sinkhorn": hc_split_sinkhorn,
}


@contextmanager
def _use_sm110a_backend():
    originals = {name: getattr(_official, name) for name in _PATCHES}
    try:
        for name, replacement in _PATCHES.items():
            setattr(_official, name, replacement)
        yield
    finally:
        for name, original in originals.items():
            setattr(_official, name, original)


class Sm110aParallelHead(_official.ParallelHead):
    """Keep checkpoint BF16 weights and write FP32 logits from the accumulator."""

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        # The reference projects only the last position, including during prefill.
        a = x[:, -1].contiguous()
        weight = self.weight
        rows, k = a.shape
        n = weight.size(0)
        if (
            a.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or not a.is_cuda
            or rows == 0
            or n % 32
            or k % 64
        ):
            _STATS["torch.lm_head_todo"] += 1
            return F.linear(a.float(), weight.float())

        tile_m = 64 if rows <= 64 else 128
        padded_rows = rows if rows <= tile_m else math.ceil(rows / tile_m) * tile_m
        if padded_rows != rows:
            a = torch.cat((a, a.new_zeros((padded_rows - rows, k))), dim=0)
        out = torch.empty((padded_rows, n), device=a.device, dtype=torch.float32)
        Bf16GemmRunner(a, weight.contiguous(), out).run()
        _STATS["sm110a.lm_head_bf16_gemm_fp32"] += 1
        return out[:rows]


class Transformer(_official.Transformer):
    """Checkpoint-compatible official model with SM110A operator dispatch."""

    def __init__(
        self,
        args: ModelArgs,
        *,
        routing_weight_position: str = "fc1",
        deterministic_fc2: bool | None = None,
        moe_backend: str = "unfused",
        compact_moe_scales: bool = True,
        pdl: bool = False,
    ):
        if routing_weight_position not in ("fc1", "fc2"):
            raise ValueError("routing_weight_position must be 'fc1' or 'fc2'")
        if not isinstance(compact_moe_scales, bool):
            raise ValueError("compact_moe_scales must be bool")
        if moe_backend not in ("unfused", "fused"):
            raise ValueError("moe_backend must be 'unfused' or 'fused'")
        if deterministic_fc2 is None:
            deterministic_fc2 = moe_backend == "unfused"
        if moe_backend == "fused" and deterministic_fc2:
            raise ValueError("fused MoE currently supports atomic reduction only")
        if moe_backend == "fused" and pdl:
            raise ValueError("fused MoE does not support PDL")
        super().__init__(args)
        # Official checkpoints store this parameter in BF16. Allocate in that
        # dtype before loading, retaining the original state-dict key and the
        # head object shared by MTP blocks. No per-token weight conversion.
        self.head.__class__ = Sm110aParallelHead
        self.head.weight = torch.nn.Parameter(
            torch.empty_like(self.head.weight, dtype=torch.bfloat16),
            requires_grad=self.head.weight.requires_grad,
        )
        self.compact_moe_scales = compact_moe_scales
        self.moe_backend = moe_backend
        self.pdl = pdl
        self.routing_weight_position = routing_weight_position
        self.deterministic_fc2 = deterministic_fc2
        for layer in self.layers:
            layer.ffn.__class__ = Sm110aMoE
        for layer in self.mtp:
            layer.ffn.__class__ = Sm110aMoE
        for layer in (*self.layers, *self.mtp):
            layer.attn.__class__ = Sm110aAttention
            layer.ffn._sm110a_compact_weight_scales = compact_moe_scales
            layer.ffn._sm110a_moe_backend = moe_backend
            layer.ffn._sm110a_pdl = pdl
            layer.ffn._sm110a_weight_position = routing_weight_position
            layer.ffn._sm110a_deterministic_fc2 = deterministic_fc2
            indexer = getattr(layer.attn, "indexer", None)
            if indexer is not None:
                indexer.__class__ = Sm110aIndexer
        self._sm110a_prepared = False

    def prepare_for_inference(self) -> None:
        if self._sm110a_prepared:
            return
        for layer in (*self.layers, *self.mtp):
            layer.ffn.prepare_fused_weights()
        self._sm110a_prepared = True

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        if not self._sm110a_prepared:
            self.prepare_for_inference()
        global _DECODE_ENABLED, _DECODE_PDL
        previous = _DECODE_ENABLED, _DECODE_PDL
        _DECODE_ENABLED = input_ids.size(1) == 1 and start_pos > 0
        _DECODE_PDL = self.pdl and _DECODE_ENABLED
        try:
            with _use_sm110a_backend():
                return super().forward(input_ids, start_pos)
        finally:
            _DECODE_ENABLED, _DECODE_PDL = previous
