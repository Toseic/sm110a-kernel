"""DeepSeek-V4-Flash runtime backed by this repository's SM110A kernels.

This module reuses the official readable model's module/state-dict layout while
swapping the operator backend for one forward call, so converted official
checkpoints load without another conversion step.
"""

# Remaining native-kernel support (including inherited reference operators):
# TODO(sm110a): tune HC pre/head for large-M prefill.
# TODO(sm110a): implement fused ratio-4 overlap /
# ratio-128 pooling, incremental state updates, and compressed KV cache writes.
# TODO(sm110a): add Hadamard rotation
# (currently an external extension or a PyTorch fallback).
# TODO(sm110a): tune FP8 KV sparse MLA long-prefill/full-selection append
# performance; current cold-L2 wins are workload dependent.
# TODO(sm110a): extend native FP8 KV sparse MLA beyond H=64/D=512,
# block-64 E4M3 non-RoPE + BF16 RoPE layouts (Torch handles other shapes).
# TODO(sm110a): fuse FP8 cache quantization, scale/RoPE stores and ring writes.
# TODO(sm110a): fuse indexer KV packing with compressor cache writes instead
# of repacking the valid BF16/QAT cache on every invocation.
# TODO(sm110a): fuse MoE gate projection and metadata construction
# (decode metadata accepts expert IDs directly).
# TODO(sm110a): add deterministic reduction and PDL to the FC1/FC2 fused backend.
# TODO(sm110a): fuse shared/routed output addition into the shared-expert FC2.
# TODO(sm110a): fuse index generation, cache writes, padding, and layout
# conversions where useful; extend the LM-head GEMM to unaligned dimensions.
# TODO(sm110a): specialize small-M dense/MoE decode instead of padding tokens
# to 128 rows.
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
    Fp8KvFa4SparseMlaRunner,
    Fp8Mxfp4GemmRunner,
    Fp8SharedFc1Runner,
    HcPostRunner,
    HcPostPreNormRunner,
    HcPreNormRunner,
    HcPreRunner,
    IndexerScoreDecodeRunner,
    IndexerScorePrefillRunner,
    Mxfp4Fc1Fc2Runner,
    Mxfp4MoeRunner,
    QkNormRunner,
    RmsNormRunner,
    RopeRunner,
    TopKRunner,
    clear_kernel_runner_cache,
    fused_gate,
    compact_mxfp4_weight_scales,
    pack_fp8_shared_fc1_weights,
    to_blocked_fp8_scales,
    to_blocked_fp8mxfp4_scales,
    to_blocked_mxfp4_scales,
)

# Scoped to one model forward, like the official backend patches below.
_DECODE_ENABLED = False
_DECODE_PDL = False
_DECODE_ZERO_PADDING = None

ModelArgs = _official.ModelArgs
_OFFICIAL_FP4_ACT_QUANT = _official.fp4_act_quant
_OFFICIAL_APPLY_ROTARY_EMB = _official.apply_rotary_emb

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


def apply_rotary_emb(x, freqs_cis, inverse=False):
    if (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.ndim in (3, 4)
        and 2 <= x.size(-1) <= 1024
        and x.size(-1) % 2 == 0
        and x.stride(-1) == 1
        and freqs_cis.device == x.device
        and freqs_cis.dtype == torch.complex64
        and freqs_cis.ndim == 2
        and tuple(freqs_cis.shape) == (x.size(1), x.size(-1) // 2)
        and freqs_cis.stride(-1) == 1
    ):
        RopeRunner(x, freqs_cis, inverse, pdl=_DECODE_PDL).run()
        _STATS["sm110a.rope_inverse" if inverse else "sm110a.rope"] += 1
        return x
    _STATS["torch.rope_fallback"] += 1
    return _OFFICIAL_APPLY_ROTARY_EMB(x, freqs_cis, inverse)


class Sm110aRMSNorm(_official.RMSNorm):
    """Native RMSNorm preserving the reference FP32 parameter and state dict."""

    def forward(self, x: torch.Tensor):
        if (
            not x.is_cuda
            or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or not 1 <= self.dim <= 32768
        ):
            _STATS["torch.rmsnorm_fallback"] += 1
            return super().forward(x)

        input = x.contiguous()
        weight = self.weight.contiguous()
        # Contiguous slices can still have unaligned storage offsets. Keep the
        # native kernel's vector alignment contract for these callers too.
        if input.data_ptr() % 16:
            input = input.clone()
        if weight.data_ptr() % 16:
            weight = weight.clone()
        output = torch.empty_like(input)
        RmsNormRunner(input, weight, output, self.eps, pdl=_DECODE_PDL).run()
        _STATS["sm110a.rmsnorm"] += 1
        return output


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
    Fp8ActQuantRunner(
        flat, values, scales, block_size, scale_mode, pdl=_DECODE_PDL
    ).run()
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
    Bf16ToFp4QuantRunner(flat, restored, scales, dequant=True, pdl=_DECODE_PDL).run()
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


def _bf16_projection(x: torch.Tensor, weight: torch.Tensor, *, output_fp32=False):
    """BF16 operands with an optional direct FP32 accumulator output."""
    a = x.to(torch.bfloat16).reshape(-1, x.size(-1)).contiguous()
    n, k = weight.shape
    output_dtype = torch.float32 if output_fp32 else torch.bfloat16
    if not a.is_cuda or n % 32 or k % 64:
        _STATS["torch.bf16_projection_fallback"] += 1
        if output_fp32:
            result = F.linear(a.float(), weight.float())
        else:
            result = F.linear(a, weight)
        return result.view(*x.shape[:-1], n)
    rows = a.size(0)
    tile_m = 64 if rows <= 64 else 128
    padded_rows = rows if rows <= tile_m else math.ceil(rows / tile_m) * tile_m
    if padded_rows != rows:
        a = torch.cat((a, a.new_zeros((padded_rows - rows, k))), dim=0)
    out = torch.empty((padded_rows, n), device=a.device, dtype=output_dtype)
    Bf16GemmRunner(a, weight.contiguous(), out, pdl=_DECODE_PDL).run()
    return out[:rows].view(*x.shape[:-1], n)


def _pack_fp8_kv(kv: torch.Tensor, rope_dim: int):
    values, scales = act_quant(
        kv[..., :-rope_dim], 64, _official.scale_fmt, _official.scale_dtype
    )
    return values, scales, kv[..., -rope_dim:]


class Fp8KvCache(torch.nn.Module):
    """Persistent mixed KV storage: E4M3 non-RoPE, block-64 scales, BF16 RoPE."""

    def __init__(self, shape, rope_dim, device, scale_dtype):
        super().__init__()
        self.rope_dim = rope_dim
        prefix, nope_dim = shape[:-1], shape[-1] - rope_dim
        if nope_dim <= 0 or nope_dim % 64:
            raise ValueError("FP8 KV cache requires a positive block-64 non-RoPE width")
        self.register_buffer(
            "kv_cache",
            torch.zeros(*prefix, nope_dim, device=device, dtype=torch.float8_e4m3fn),
            persistent=False,
        )
        scales = torch.empty(*prefix, nope_dim // 64, device=device, dtype=scale_dtype)
        # E8M0 has no zero. Unwritten entries must reconstruct as zero, so
        # initialize their scales to one (encoding 127), rather than NaN.
        if scale_dtype == torch.float8_e8m0fnu:
            scales.view(torch.uint8).fill_(127)
        else:
            scales.fill_(1)
        self.register_buffer("kv_cache_scale", scales, persistent=False)
        self.register_buffer(
            "kv_cache_rope",
            torch.zeros(*prefix, rope_dim, device=device, dtype=torch.bfloat16),
            persistent=False,
        )

    def tensors(self, batch_size):
        return (
            self.kv_cache[:batch_size],
            self.kv_cache_scale[:batch_size],
            self.kv_cache_rope[:batch_size],
        )

    def write(self, packed, position):
        batch_size, count = packed[0].shape[:2]
        for destination, source in zip(self.tensors(batch_size), packed):
            destination[:, position : position + count].copy_(source)
        _STATS["torch.fp8_kv_cache_store_todo"] += 1


class _Fp8KvCacheView:
    """Non-owning compressed-cache view; its owner stays registered on attention."""

    def __init__(self, cache, offset):
        self.cache, self.offset = cache, offset

    def write(self, kv, position):
        packed = _pack_fp8_kv(kv, self.cache.rope_dim)
        self.cache.write(packed, self.offset + position)
        return packed


def sparse_attn_fp8(q, packed, sink, indices, scale):
    values, scales, rope = packed
    if (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and q.shape[2:] == (64, 512)
        and values.dtype == torch.float8_e4m3fn
        and values.size(-1) == 448
        and rope.dtype == torch.bfloat16
        and rope.size(-1) == 64
        and scales.dtype in (torch.float8_e8m0fnu, torch.float32)
        and scales.size(-1) == 7
    ):
        query = q.transpose(1, 2).contiguous()
        out = torch.empty_like(query)
        Fp8KvFa4SparseMlaRunner(
            query,
            values.contiguous(),
            scales.contiguous(),
            rope.contiguous(),
            indices.contiguous(),
            sink.contiguous(),
            out,
            scale=scale,
            pdl=_DECODE_PDL,
        ).run()
        phase = "decode" if q.size(1) == 1 else "prefill"
        _STATS[f"sm110a.sparse_mla_fp8_kv_{phase}"] += 1
        return out.transpose(1, 2).contiguous()
    return _torch_sparse_attn_fp8(q, packed, sink, indices, scale)


def _torch_sparse_attn_fp8(q, packed, sink, indices, scale):
    """Torch FP8-cache fallback for unsupported native sparse MLA layouts.

    Only selected entries are decoded, in bounded query chunks. Q stays BF16;
    quantized non-RoPE values round to BF16 as in the previous QAT path, then
    scores, sink softmax and value accumulation use FP32.
    """
    values, scales, rope = packed
    batch = torch.arange(q.size(0), device=q.device)[:, None, None]
    out = torch.empty_like(q)
    chunk_size = max(1, 32 // q.size(0))
    for start in range(0, q.size(1), chunk_size):
        end = min(start + chunk_size, q.size(1))
        selected_indices = indices[:, start:end]
        safe = selected_indices.clamp_min(0).long()
        # Byte gathers also support FP8 formats without Torch index kernels.
        selected_values = values.view(torch.uint8)[batch, safe].view(values.dtype)
        if scales.dtype == torch.float8_e8m0fnu:
            selected_scales = scales.view(torch.uint8)[batch, safe].view(scales.dtype)
        else:
            selected_scales = scales[batch, safe]
        nope = (
            selected_values.float()
            * selected_scales.float().repeat_interleave(64, dim=-1)
        ).to(q.dtype)
        selected = torch.cat((nope, rope[batch, safe]), dim=-1).float()
        scores = (
            torch.einsum("bshd,bskd->bshk", q[:, start:end].float(), selected) * scale
        )
        scores.masked_fill_(selected_indices[:, :, None, :] < 0, -torch.inf)
        sink_scores = sink.view(1, 1, -1, 1).expand(*scores.shape[:-1], 1)
        probabilities = torch.cat((scores, sink_scores), dim=-1).softmax(-1)[..., :-1]
        out[:, start:end] = torch.einsum("bshk,bskd->bshd", probabilities, selected).to(
            q.dtype
        )
    _STATS["torch.sparse_mla_fp8_cache_todo"] += 1
    return out


class Sm110aCompressor(_official.Compressor):
    """Joint BF16 KV/score projection, retaining FP32 compression state."""

    def prepare_projection(self):
        self.register_buffer(
            "sm110a_kv_gate_weight",
            torch.cat((self.wkv.weight, self.wgate.weight), dim=0).contiguous(),
            persistent=False,
        )

    def project_kv_score(self, x: torch.Tensor):
        if not hasattr(self, "sm110a_kv_gate_weight"):
            self.prepare_projection()
        projected = _bf16_projection(x, self.sm110a_kv_gate_weight)
        _STATS["sm110a.compressor_bf16_gemm"] += 1
        return projected.chunk(2, dim=-1)

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.shape
        ratio, d, rd = self.compress_ratio, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        kv, score = self.project_kv_score(x)
        # Projection rounds to BF16; positional scoring and pooling use FP32.
        kv, score = kv.float(), score.float()
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if self.overlap else 0
            if self.overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = (
                    score[:, cutoff - ratio : cutoff] + self.ape
                )
            if remainder > 0:
                self.kv_state[:bsz, offset : offset + remainder] = kv[:, cutoff:]
                self.score_state[:bsz, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                kv, score = kv[:, :cutoff], score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if self.overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % ratio == 0
            score = score + self.ape[start_pos % ratio]
            offset = ratio if self.overlap else 0
            self.kv_state[:bsz, offset + start_pos % ratio] = kv.squeeze(1)
            self.score_state[:bsz, offset + start_pos % ratio] = score.squeeze(1)
            if should_compress:
                if self.overlap:
                    kv_state = torch.cat(
                        (
                            self.kv_state[:bsz, :ratio, :d],
                            self.kv_state[:bsz, ratio:, d:],
                        ),
                        dim=1,
                    )
                    score_state = torch.cat(
                        (
                            self.score_state[:bsz, :ratio, :d],
                            self.score_state[:bsz, ratio:, d:],
                        ),
                        dim=1,
                    )
                else:
                    kv_state, score_state = self.kv_state[:bsz], self.score_state[:bsz]
                kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                if self.overlap:
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        _official.apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if self.rotate:
            kv = _official.rotate_activation(kv)
            fp4_act_quant(kv, _official.fp4_block_size, True)
        else:
            # Main-attention compressors write quantized bytes/scales directly.
            return self.kv_cache.write(kv, 0 if start_pos == 0 else start_pos // ratio)
        if start_pos == 0:
            self.kv_cache[:bsz, : seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Sm110aGate(_official.Gate):
    """BF16 router operands with FP32 logits and reference routing semantics."""

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None = None):
        scores = _bf16_projection(x, self.weight, output_fp32=True)
        _STATS["sm110a.router_bf16_gemm_fp32"] += 1
        if (
            self.score_func not in ("softmax", "sigmoid")
            and scores.size(-1) <= 1024
            and self.topk <= 32
        ):
            hash_indices = self.tid2eid[input_ids].contiguous() if self.hash else None
            result = fused_gate(
                scores,
                self.topk,
                self.route_scale,
                bias=self.bias,
                hash_indices=hash_indices,
                pdl=_DECODE_PDL,
            )
            _STATS["sm110a.fused_gate"] += 1
            return result
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


def _zero_padding(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Reuse immutable padding owned by the current decoding model.

    Callers only read this tensor (normally as a cat input). Keep it separate
    from writable GEMM inputs and retain it across CUDA Graph captures.
    """
    cache = _DECODE_ZERO_PADDING if _DECODE_ENABLED else None
    if cache is None:
        return x.new_zeros(shape)
    key = (x.device, x.dtype, tuple(shape))
    if key not in cache:
        cache[key] = x.new_zeros(shape)
    return cache[key]


def _decode_fp8_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    k, n = x.size(-1), weight.size(0)
    rows = x.numel() // k
    padded_rows = math.ceil(rows / 128) * 128
    a = x.reshape(rows, k).contiguous()
    if padded_rows != rows:
        a = torch.cat((a, _zero_padding(a, (padded_rows - rows, k))), dim=0)
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
            a = torch.cat((a, _zero_padding(a, (padded_rows - rows, k))), dim=0)
        if n % 32:
            # TODO(sm110a): add an epilogue for BF16 projection widths that
            # are not a multiple of the current 32-column store atom.
            _STATS["torch.bf16_linear_todo"] += 1
            return F.linear(x, weight)
        out = torch.empty((padded_rows, n), device=x.device, dtype=torch.bfloat16)
        Bf16GemmRunner(a, weight.contiguous(), out, pdl=_DECODE_PDL).run()
        _STATS["sm110a.bf16_gemm"] += 1
        return out[:rows].view(*x.shape[:-1], n)

    # Other FP32 projections outside the specialized native modules retain
    # the reference computation.
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
    """FP8 KV-cache attention with a layout-driven grouped wo_a GEMM."""

    def prepare_kv_cache(self):
        old_cache = self.kv_cache
        self._wo_a_metadata_cache = {}
        del self.kv_cache
        self.fp8_cache = Fp8KvCache(
            old_cache.shape, self.rope_head_dim, old_cache.device, _official.scale_dtype
        )
        # Immutable index data belongs to the module so CUDA Graph replays
        # retain its storage, including after backend runner caches are cleared.
        self.register_buffer(
            "sm110a_group_ids",
            torch.arange(
                self.n_local_groups, device=old_cache.device, dtype=torch.int32
            ),
            persistent=False,
        )
        self.register_buffer(
            "sm110a_window_ring",
            torch.arange(
                self.window_size, device=old_cache.device, dtype=torch.int32
            ).repeat(2),
            persistent=False,
        )
        if self.compress_ratio and self.indexer is None:
            self.register_buffer(
                "sm110a_compress_ids",
                torch.arange(
                    self.window_size,
                    old_cache.size(1),
                    device=old_cache.device,
                    dtype=torch.int32,
                ),
                persistent=False,
            )

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        if ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = _Fp8KvCacheView(self.fp8_cache, win)
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        if (
            q.is_cuda
            and q.is_contiguous()
            and q.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and 1 <= self.head_dim <= 32768
            and q.data_ptr() % 16 == 0
        ):
            QkNormRunner(q, q, self.eps, pdl=_DECODE_PDL).run()
            _STATS["sm110a.qk_norm"] += 1
        else:
            q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
            _STATS["torch.qk_norm_fallback"] += 1
        _official.apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = self.kv_norm(self.wkv(x))
        _official.apply_rotary_emb(kv[..., -rd:], freqs_cis)
        packed = _pack_fp8_kv(kv, rd)
        if seqlen == 1 and start_pos >= win - 1:
            begin = start_pos % win + 1
            topk_idxs = (
                self.sm110a_window_ring[begin : begin + win]
                .unsqueeze(0)
                .expand(bsz, -1, -1)
            )
        else:
            topk_idxs = _official.get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if ratio:
            offset = seqlen if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            elif start_pos > 0:
                # Decode uses the fixed window-size cache offset. This is the
                # same int64 arange + offset, materialized once during setup.
                width = (start_pos + 1) // ratio
                compress_topk_idxs = (
                    self.sm110a_compress_ids[:width].unsqueeze(0).expand(bsz, -1, -1)
                )
            else:
                compress_topk_idxs = _official.get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        if start_pos == 0:
            if seqlen <= win:
                self.fp8_cache.write(packed, 0)
            else:
                cutoff = seqlen % win
                tail = tuple(t[:, -win:] for t in packed)
                self.fp8_cache.write(tuple(t[:, : win - cutoff] for t in tail), cutoff)
                self.fp8_cache.write(tuple(t[:, win - cutoff :] for t in tail), 0)
            if ratio:
                compressed = self.compressor(x, start_pos)
                if compressed is not None:
                    packed = tuple(
                        torch.cat((a, b), dim=1) for a, b in zip(packed, compressed)
                    )
        else:
            self.fp8_cache.write(packed, start_pos % win)
            if ratio:
                self.compressor(x, start_pos)
            packed = self.fp8_cache.tensors(bsz)
        o = sparse_attn_fp8(q, packed, self.attn_sink, topk_idxs, self.softmax_scale)
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
        layout = self.sm110a_group_ids
        if rows != 1:
            layout = layout.repeat_interleave(rows)
        out = torch.empty((a.size(0), self.o_lora_rank), device=o.device, dtype=o.dtype)
        Bf16GroupedMGemmRunner(
            a,
            weights,
            layout,
            out,
            dynamic_metadata=True,
            pdl=_DECODE_PDL,
            static_metadata_cache=self._wo_a_metadata_cache if rows == 1 else None,
        ).run()
        _STATS["sm110a.wo_a_grouped_m_gemm"] += 1
        return (
            out.view(groups, rows, self.o_lora_rank)
            .transpose(0, 1)
            .reshape(bsz, seqlen, groups, self.o_lora_rank)
        )


def _pack_indexer_fp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack [B, rows, 128], padding scale tiles independently per batch.

    Q uses 64 rows per query; K uses one row per compressed token. The score
    kernels address scales in independent 128-row tiles for each sequence.
    Values keep their logical row count, including odd query/KV lengths.
    """
    batch, rows, width = x.shape
    flat = x.contiguous().view(batch * rows, width)
    packed = torch.empty(
        (batch * rows, width // 2), device=x.device, dtype=torch.float4_e2m1fn_x2
    )
    scales = torch.empty(
        (batch * rows, width // 32), device=x.device, dtype=torch.float8_e8m0fnu
    )
    Bf16ToFp4QuantRunner(flat, packed, scales, pdl=_DECODE_PDL).run()
    padded_rows = math.ceil(rows / 128) * 128
    if padded_rows != rows:
        # E8M0 byte 127 represents 1.0. Padding is never a valid Q/K row.
        padded = torch.full(
            (batch, padded_rows, width // 32),
            127,
            device=x.device,
            dtype=torch.uint8,
        )
        padded[:, :rows].copy_(scales.view(torch.uint8).view(batch, rows, -1))
        scales = padded.view(torch.float8_e8m0fnu)
    blocked = to_blocked_mxfp4_scales(scales.reshape(batch * padded_rows, -1))
    return packed.view(batch, rows, width // 2), blocked


class Sm110aIndexer(_official.Indexer):
    """CSA indexer with native MXFP4 scoring and row top-k."""

    def _score(self, q: torch.Tensor, weights: torch.Tensor, width: int):
        bsz, seqlen, heads, dim = q.shape
        kv = self.kv_cache[:bsz, :width]
        if (
            q.is_cuda
            and q.dtype == kv.dtype == weights.dtype == torch.bfloat16
            and heads == 64
            and dim == 128
            and _official.world_size == 1
        ):
            # Q is packed directly from the rotated BF16 activation. The BF16
            # KV cache already contains block-32 FP4 QAT values; repacking it
            # also keeps external cache restores visible on every graph replay.
            pq, sq = _pack_indexer_fp4(q.reshape(bsz, seqlen * heads, dim))
            pk, sk = _pack_indexer_fp4(kv)
            scores = torch.empty(
                (bsz, seqlen, width), device=q.device, dtype=torch.bfloat16
            )
            if seqlen == 1:
                IndexerScoreDecodeRunner(
                    pq.view(bsz, heads, dim // 2),
                    pk,
                    sq,
                    sk,
                    weights.reshape(bsz, heads).contiguous(),
                    scores.view(bsz, width),
                    pdl=_DECODE_PDL,
                ).run()
                _STATS["sm110a.indexer_score_decode"] += 1
            else:
                # The kernel's ordinary token causal mask does not express
                # compressed visibility floor((query_position + 1) / ratio).
                # Compute the full matrix and apply that mask before top-k.
                IndexerScorePrefillRunner(
                    pq.view(bsz, seqlen, heads, dim // 2),
                    pk,
                    sq,
                    sk,
                    weights.contiguous(),
                    scores,
                    causal=False,
                    pdl=_DECODE_PDL,
                ).run()
                _STATS["sm110a.indexer_score_prefill"] += 1
            return scores

        fp4_act_quant(q, _official.fp4_block_size, True)
        scores = torch.einsum("bshd,btd->bsht", q, kv)
        scores = (scores.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if _official.world_size > 1:
            torch.distributed.all_reduce(scores)
        _STATS["torch.indexer_score_fallback"] += 1
        return scores

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
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        width = end_pos // ratio
        if width == 0:
            # A short prefill/decode still updates compressor state, but has
            # no completed compressed group to score or select.
            return torch.empty((bsz, seqlen, 0), device=x.device, dtype=torch.int32)
        scores = self._score(q, weights, width)
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
        TopKRunner(
            scores.contiguous().view(bsz * seqlen, width), indices, k, pdl=_DECODE_PDL
        ).run()
        indices = indices.view(bsz, seqlen, k)
        _STATS["sm110a.indexer_topk"] += 1
        if start_pos == 0:
            indices = torch.where(indices >= valid_groups, -1, indices + offset)
        else:
            indices = indices + offset
        return indices


class Sm110aSharedExpert(_official.Expert):
    """Checkpoint-compatible FP8 shared expert with a fused gate/up FC1."""

    def prepare_fused_weights(self) -> None:
        if getattr(self, "_sm110a_shared_prepared", False):
            return
        gate, up = self.w1.weight, self.w3.weight
        inter_dim, dim = gate.shape
        if (
            not gate.is_cuda
            or gate.dtype != torch.float8_e4m3fn
            or up.dtype != gate.dtype
            or up.device != gate.device
            or up.shape != gate.shape
            or dim % 128
            or inter_dim % 128
            or _official.block_size != 128
            or _official.scale_dtype != torch.float8_e8m0fnu
            or self.w1.scale is None
            or self.w3.scale is None
            or self.w1.scale.dtype != torch.float8_e8m0fnu
            or self.w3.scale.dtype != torch.float8_e8m0fnu
        ):
            return

        def expand(scales):
            expected = (inter_dim // 128, dim // 128)
            if tuple(scales.shape) != expected:
                raise ValueError(f"shared FP8 weight scales must have shape {expected}")
            return (
                scales.view(torch.uint8)
                .repeat_interleave(128, dim=0)
                .repeat_interleave(4, dim=1)
                .view(torch.float8_e8m0fnu)
            )

        weight, scales = pack_fp8_shared_fc1_weights(
            gate, up, expand(self.w1.scale), expand(self.w3.scale)
        )
        # Keep w1/w3 and their checkpoint keys. Packed execution buffers are
        # derived once after checkpoint loading, and are not saved in state_dict.
        self.register_buffer("sm110a_shared_fc1_weight", weight, persistent=False)
        self.register_buffer("sm110a_shared_fc1_scale", scales, persistent=False)
        self._sm110a_shared_prepared = True

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None):
        if (
            weights is not None
            or not x.is_cuda
            or x.dtype != torch.bfloat16
            or x.size(-1) != self.w1.in_features
        ):
            _STATS["torch.shared_expert_fallback"] += 1
            return super().forward(x, weights)
        self.prepare_fused_weights()
        if not getattr(self, "_sm110a_shared_prepared", False):
            _STATS["torch.shared_expert_fallback"] += 1
            return super().forward(x, weights)

        shape = x.shape
        dim, inter_dim = self.w1.in_features, self.w1.out_features
        flat = x.reshape(-1, dim).contiguous()
        tokens = flat.size(0)
        padded_tokens = math.ceil(tokens / 128) * 128
        if padded_tokens != tokens:
            flat = torch.cat((flat, _zero_padding(flat, (padded_tokens - tokens, dim))))
        aq = torch.empty(flat.shape, device=x.device, dtype=torch.float8_e4m3fn)
        sfa = torch.empty(
            padded_tokens * dim // 8, device=x.device, dtype=torch.float8_e8m0fnu
        )
        hidden = torch.empty(
            (padded_tokens, inter_dim), device=x.device, dtype=torch.bfloat16
        )
        Fp8SharedFc1Runner(
            aq,
            self.sm110a_shared_fc1_weight,
            sfa,
            self.sm110a_shared_fc1_scale,
            hidden,
            input=flat,
            swiglu_limit=self.swiglu_limit,
            pdl=_DECODE_PDL,
        ).run()
        _STATS["sm110a.fp8_shared_fc1"] += 1
        _STATS["sm110a.fp8_act_quant_blocked"] += 1
        # Leave the padded hidden rows in place so FC2 need not pad again.
        return self.w2(hidden)[:tokens].reshape(*shape[:-1], self.w2.out_features)


class Sm110aMoE(_official.MoE):
    """Official MoE state layout with separate or fused SM110A FC1/FC2."""

    def prepare_fused_weights(self) -> None:
        self.shared_experts.prepare_fused_weights()
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
        index_dtype = self.gate.tid2eid.dtype if self.gate.hash else torch.int64
        self.register_buffer(
            "sm110a_dummy_expert_ids",
            torch.arange(self.n_activated_experts, device=device, dtype=index_dtype),
            persistent=False,
        )
        self.register_buffer(
            "sm110a_route_ranks",
            torch.arange(
                self.n_activated_experts, 0, -1, device=device, dtype=torch.bfloat16
            ),
            persistent=False,
        )
        self._sm110a_direct_ids = (
            self._sm110a_moe_backend == "unfused"
            and self.n_routed_experts <= 1024
            and self.n_activated_experts <= 32
        )
        if self._sm110a_direct_ids and self.gate.hash:
            # Validate the immutable table once outside graph capture. Duplicate
            # IDs give scatter ambiguous overwrite semantics; retain the old
            # router path for such tables instead of changing their behavior.
            ids = self.gate.tid2eid
            ordered = ids.sort(dim=-1).values
            valid = ((ids >= 0) & (ids < self.n_routed_experts)).all()
            unique = (ordered[:, 1:] != ordered[:, :-1]).all()
            self._sm110a_direct_ids = bool((valid & unique).item())

        # The fused tensors now own the routed-expert weights.  Drop the
        # official per-expert modules instead of retaining a second 3+ GiB
        # copy per layer.
        self.experts = torch.nn.ModuleList()
        self._sm110a_prepared = True

    def _run_experts(self, x, weights, router, topk, *, expert_ids=None):
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
                expert_ids=expert_ids,
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
                (x, _zero_padding(x, (padded_tokens - tokens, self.dim))), dim=0
            )
            weights_padded = torch.cat(
                (
                    weights,
                    _zero_padding(weights, (padded_tokens - tokens, topk)),
                ),
                dim=0,
            )
            dummy_indices = self.sm110a_dummy_expert_ids.expand(
                padded_tokens - tokens, -1
            )
            indices_padded = torch.cat((indices, dummy_indices), dim=0)
        else:
            x_padded, weights_padded, indices_padded = x, weights, indices

        direct_ids = (
            _DECODE_ENABLED and self._sm110a_direct_ids and padded_tokens <= 1024
        )
        router = None
        if not direct_ids:
            # Compatibility path for prefill, fused MoE and hash tables with
            # repeated IDs. Synthetic scores retain the gate's rank order.
            router = torch.full(
                (padded_tokens, self.n_routed_experts),
                -torch.inf,
                device=x.device,
                dtype=torch.bfloat16,
            )
            router.scatter_(
                1,
                indices_padded.long(),
                self.sm110a_route_ranks.expand(padded_tokens, -1),
            )

        out = self._run_experts(
            x_padded.contiguous(),
            weights_padded,
            router,
            topk,
            expert_ids=indices_padded.contiguous() if direct_ids else None,
        )
        if self._sm110a_moe_backend == "fused":
            _STATS["sm110a.mxfp4_fc1fc2"] += 1
        else:
            _STATS["sm110a.mxfp4_fc1"] += 1
            _STATS["sm110a.mxfp4_fc2"] += 1

        # Shared FC1 fuses FP8 gate/up projection and the activation epilogue;
        # shared FC2 uses the native dense backend installed by this context.
        out = out[:tokens] + self.shared_experts(x)
        return out.to(x.dtype).view(shape)


_PATCHES = {
    "apply_rotary_emb": apply_rotary_emb,
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


def _can_fuse_hc_norm(x, hc_fn, hc_scale, hc_base, norm, *, head=False):
    # Preserve custom norm forward implementations and unsupported formats.
    return (
        type(norm) in (_official.RMSNorm, Sm110aRMSNorm)
        and x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.dim() == 4
        and x.size(-2) == 4
        and 1 <= x.size(-1) <= 4096
        and hc_fn.data_ptr() % 16 == 0
        and tuple(hc_fn.shape) == (4 if head else 24, 4 * x.size(-1))
        and tuple(hc_scale.shape) == (1 if head else 3,)
        and tuple(hc_base.shape) == (4 if head else 24,)
        and norm.dim == x.size(-1)
        and tuple(norm.weight.shape) == (x.size(-1),)
        and all(
            t.device == x.device and t.dtype == torch.float32 and t.is_contiguous()
            for t in (hc_fn, hc_scale, hc_base, norm.weight)
        )
    )


class Sm110aParallelHead(_official.ParallelHead):
    """Keep checkpoint BF16 weights and write FP32 logits from the accumulator."""

    def forward(self, x, hc_fn, hc_scale, hc_base, norm):
        if _can_fuse_hc_norm(x, hc_fn, hc_scale, hc_base, norm, head=True):
            output = torch.empty(
                (*x.shape[:-2], x.size(-1)), device=x.device, dtype=x.dtype
            )
            HcPreNormRunner(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                norm.weight,
                output,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
                rms_eps=norm.eps,
                head=True,
                pdl=_DECODE_PDL,
            ).run()
            _STATS["sm110a.hc_head_norm"] += 1
        else:
            _STATS["unfused.hc_head_norm"] += 1
            output = norm(self.hc_head(x, hc_fn, hc_scale, hc_base))
        logits = self.get_logits(output)
        if _official.world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(_official.world_size)]
            torch.distributed.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits

    def hc_head(self, x, hc_fn, hc_scale, hc_base):
        if not (
            x.is_cuda
            and x.is_contiguous()
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and x.size(-2) == 4
            and 1 <= x.size(-1) <= 4096
            and hc_fn.data_ptr() % 16 == 0
            and all(
                t.device == x.device and t.dtype == torch.float32 and t.is_contiguous()
                for t in (hc_fn, hc_scale, hc_base)
            )
        ):
            _STATS["torch.hc_head_fallback"] += 1
            return super().hc_head(x, hc_fn, hc_scale, hc_base)
        output = torch.empty(
            (*x.shape[:-2], x.size(-1)), device=x.device, dtype=x.dtype
        )
        HcPreRunner(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            output,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
            head=True,
        ).run()
        _STATS["sm110a.hc_head"] += 1
        return output

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
        Bf16GemmRunner(a, weight.contiguous(), out, pdl=_DECODE_PDL).run()
        _STATS["sm110a.lm_head_bf16_gemm_fp32"] += 1
        return out[:rows]


def _hc_pre(self, x, hc_fn, hc_scale, hc_base):
    if not (
        x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and self.hc_mult == 4
        and 1 <= x.size(-1) <= 4096
        and 1 <= self.hc_sinkhorn_iters <= 100
        and hc_fn.data_ptr() % 16 == 0
        and all(
            t.dtype == torch.float32 and t.is_contiguous()
            for t in (hc_fn, hc_scale, hc_base)
        )
    ):
        _STATS["torch.hc_pre_fallback"] += 1
        return _official.Block.hc_pre(self, x, hc_fn, hc_scale, hc_base)
    prefix = x.shape[:-2]
    y = torch.empty((*prefix, x.size(-1)), dtype=x.dtype, device=x.device)
    post = torch.empty((*prefix, 4), dtype=torch.float32, device=x.device)
    comb = torch.empty((*prefix, 4, 4), dtype=torch.float32, device=x.device)
    HcPreRunner(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        y,
        post,
        comb,
        self.norm_eps,
        self.hc_eps,
        sinkhorn_iters=self.hc_sinkhorn_iters,
    ).run()
    _STATS["sm110a.hc_pre"] += 1
    return y, post, comb


def _hc_pre_norm(self, x, hc_fn, hc_scale, hc_base, norm):
    if not (
        self.hc_mult == 4
        and 1 <= self.hc_sinkhorn_iters <= 100
        and _can_fuse_hc_norm(x, hc_fn, hc_scale, hc_base, norm)
    ):
        _STATS["unfused.hc_pre_norm"] += 1
        y, post, comb = self.hc_pre(x, hc_fn, hc_scale, hc_base)
        return norm(y), post, comb
    prefix = x.shape[:-2]
    y = torch.empty((*prefix, x.size(-1)), dtype=x.dtype, device=x.device)
    post = torch.empty((*prefix, 4), dtype=torch.float32, device=x.device)
    comb = torch.empty((*prefix, 4, 4), dtype=torch.float32, device=x.device)
    HcPreNormRunner(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        norm.weight,
        y,
        post,
        comb,
        norm_eps=self.norm_eps,
        hc_eps=self.hc_eps,
        rms_eps=norm.eps,
        sinkhorn_iters=self.hc_sinkhorn_iters,
        pdl=_DECODE_PDL,
    ).run()
    _STATS["sm110a.hc_pre_norm"] += 1
    return y, post, comb


def _hc_block_forward(self, x, start_pos, input_ids):
    residual = x
    x, post, comb = self.hc_pre_norm(
        x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self.attn_norm
    )
    x = self.attn(x, start_pos)
    # Fuse the measured decode shape and prefill regimes. The large-row kernel
    # reuses post source vectors across all four streams inside each CTA.
    if (
        self.hc_mult == 4
        and type(self).hc_post is _hc_post
        and type(self).hc_pre_norm is _hc_pre_norm
        and 1 <= self.hc_sinkhorn_iters <= 100
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and residual.dtype == x.dtype
        and residual.device == x.device
        and x.size(-1) == 4096
        and (x.numel() == 4096 or x.numel() >= 512 * 4096)
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and residual.data_ptr() % 16 == 0
        and _can_fuse_hc_norm(
            residual, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self.ffn_norm
        )
    ):
        mixed, normalized = torch.empty_like(residual), torch.empty_like(x)
        next_post, next_comb = torch.empty_like(post), torch.empty_like(comb)
        HcPostPreNormRunner(
            x, residual, post, comb, self.hc_ffn_fn, self.hc_ffn_scale,
            self.hc_ffn_base, self.ffn_norm.weight, mixed, normalized,
            next_post, next_comb, norm_eps=self.norm_eps, hc_eps=self.hc_eps,
            rms_eps=self.ffn_norm.eps, sinkhorn_iters=self.hc_sinkhorn_iters,
            pdl=_DECODE_PDL,
        ).run()
        _STATS["sm110a.hc_post_pre_norm"] += 1
        residual, x, post, comb = mixed, normalized, next_post, next_comb
    else:
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_pre_norm(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self.ffn_norm
        )
    x = self.ffn(x, input_ids)
    return self.hc_post(x, residual, post, comb)


def _hc_post(self, x, residual, post, comb):
    if not (
        self.hc_mult == 4
        and x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and 1 <= x.size(-1) <= 32768
        and residual.dtype == x.dtype
        and post.dtype == comb.dtype == torch.float32
        and all(
            t.is_contiguous() and t.device == x.device
            for t in (x, residual, post, comb)
        )
    ):
        _STATS["torch.hc_post_fallback"] += 1
        return _official.Block.hc_post(self, x, residual, post, comb)
    output = torch.empty_like(residual)
    HcPostRunner(x, residual, post, comb, output, pdl=_DECODE_PDL).run()
    _STATS["sm110a.hc_post"] += 1
    return output


class Sm110aBlock(_official.Block):
    hc_pre = _hc_pre
    hc_pre_norm = _hc_pre_norm
    hc_post = _hc_post
    forward = _hc_block_forward


class Sm110aMTPBlock(_official.MTPBlock):
    hc_pre = _hc_pre
    hc_pre_norm = _hc_pre_norm
    hc_post = _hc_post

    @torch.inference_mode()
    def forward(self, x, start_pos, input_ids):
        assert self.embed is not None and self.head is not None
        e = self.enorm(self.embed(input_ids))
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = _hc_block_forward(self, x, start_pos, input_ids)
        return self.head(
            x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm
        )


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
        pdl: bool = True,
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
            raise ValueError("fused MoE does not support PDL; pass pdl=False")
        super().__init__(args)
        for module in self.modules():
            if isinstance(module, _official.Compressor):
                module.__class__ = Sm110aCompressor
                for projection in (module.wkv, module.wgate):
                    projection.weight = torch.nn.Parameter(
                        projection.weight.to(torch.bfloat16),
                        requires_grad=projection.weight.requires_grad,
                    )
            elif isinstance(module, _official.Gate):
                module.__class__ = Sm110aGate
                module.weight = torch.nn.Parameter(
                    module.weight.to(torch.bfloat16),
                    requires_grad=module.weight.requires_grad,
                )
        for module in self.modules():
            if isinstance(module, _official.RMSNorm):
                module.__class__ = Sm110aRMSNorm
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
            layer.__class__ = Sm110aBlock
            layer.ffn.__class__ = Sm110aMoE
        for layer in self.mtp:
            layer.__class__ = Sm110aMTPBlock
            layer.ffn.__class__ = Sm110aMoE
        for layer in (*self.layers, *self.mtp):
            layer.ffn.shared_experts.__class__ = Sm110aSharedExpert
            layer.attn.__class__ = Sm110aAttention
            layer.attn.prepare_kv_cache()
            layer.ffn._sm110a_compact_weight_scales = compact_moe_scales
            layer.ffn._sm110a_moe_backend = moe_backend
            layer.ffn._sm110a_pdl = pdl
            layer.ffn._sm110a_weight_position = routing_weight_position
            layer.ffn._sm110a_deterministic_fc2 = deterministic_fc2
            indexer = getattr(layer.attn, "indexer", None)
            if indexer is not None:
                indexer.__class__ = Sm110aIndexer
        self._sm110a_prepared = False
        self._sm110a_zero_padding = {}

    def prepare_for_inference(self) -> None:
        if self._sm110a_prepared:
            return
        for module in self.modules():
            if isinstance(module, Sm110aCompressor):
                module.prepare_projection()
        for layer in (*self.layers, *self.mtp):
            layer.ffn.prepare_fused_weights()
        self._sm110a_prepared = True

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        if not self._sm110a_prepared:
            self.prepare_for_inference()
        global _DECODE_ENABLED, _DECODE_PDL, _DECODE_ZERO_PADDING
        previous = _DECODE_ENABLED, _DECODE_PDL, _DECODE_ZERO_PADDING
        _DECODE_ENABLED = input_ids.size(1) == 1 and start_pos > 0
        _DECODE_PDL = self.pdl and _DECODE_ENABLED
        _DECODE_ZERO_PADDING = self._sm110a_zero_padding
        try:
            with _use_sm110a_backend():
                return super().forward(input_ids, start_pos)
        finally:
            _DECODE_ENABLED, _DECODE_PDL, _DECODE_ZERO_PADDING = previous
