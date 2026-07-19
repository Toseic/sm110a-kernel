"""SM110A native NVFP4 block-scaled GEMM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/nvfp4_gemm/src"
_INCLUDE_DIR = _ROOT / "include"


@dataclass(frozen=True)
class Nvfp4GemmConfig:
    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 64
    cluster_m: int = 2
    epilogue_block_n: int = 32

    def validate(self) -> None:
        if self.tile_m != 128:
            raise ValueError("tile_m must be 128")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.tile_n not in (64, 128, 256):
            raise ValueError("tile_n must be 64, 128, or 256")
        if self.tile_k not in (64, 128, 256):
            raise ValueError("tile_k must be 64, 128, or 256")
        if self.epilogue_block_n not in (32, 64, 128) or self.tile_n % self.epilogue_block_n:
            raise ValueError("epilogue_block_n must be 32, 64, or 128")


def _kernel_cflags(config: Nvfp4GemmConfig, _variant) -> tuple[str, ...]:
    return tuple(
        f"-DSM110A_NVFP4_{name}={value}"
        for name, value in (
            ("TILE_M", config.tile_m), ("TILE_N", config.tile_n),
            ("TILE_K", config.tile_k), ("CLUSTER_M", config.cluster_m),
            ("EPILOGUE_BLOCK_N", config.epilogue_block_n),
        )
    )


class Nvfp4GemmModule:
    def __init__(self, binding, library, config: Nvfp4GemmConfig):
        self._binding = binding
        self._launcher = library.entry_address
        self._config = config

    def _args(self):
        c = self._config
        return self._launcher, c.tile_m, c.tile_n, c.tile_k, c.cluster_m

    def nvfp4_gemm(self, a, b, sfa, sfb):
        return self._binding.nvfp4_gemm(a, b, sfa, sfb, *self._args())

    def nvfp4_gemm_out(self, a, b, sfa, sfb, out):
        return self._binding.nvfp4_gemm_out(a, b, sfa, sfb, out, *self._args())


_JIT = SplitCudaJit(
    name="sm110a_nvfp4_gemm",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "nvfp4_gemm_split_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "nvfp4_gemm_kernel.cu",),
    kernel_dependencies=(
        _INCLUDE_DIR / "nvfp4_gemm.h",
        _INCLUDE_DIR / "common.h",
        _INCLUDE_DIR / "profiler.h",
    ),
    kernel_symbol="sm110a_nvfp4_gemm_launch",
    config_key=lambda config: config,
    kernel_cflags=_kernel_cflags,
    make_facade=Nvfp4GemmModule,
    include_paths=(_INCLUDE_DIR,),
)


def compile_nvfp4_gemm(config: Nvfp4GemmConfig | None = None) -> Nvfp4GemmModule:
    config = config or Nvfp4GemmConfig()
    if not isinstance(config, Nvfp4GemmConfig):
        raise TypeError("config must be an Nvfp4GemmConfig")
    config.validate()
    return _JIT.compile(config)


def compile_nvfp4_gemm_configs(
    configs: Iterable[Nvfp4GemmConfig], max_workers=None
) -> tuple[Nvfp4GemmModule, ...]:
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Nvfp4GemmConfig):
            raise TypeError("configs must contain Nvfp4GemmConfig instances")
        config.validate()
    return _JIT.compile_many(configs, max_workers)


def to_blocked_nvfp4_scales(scales):
    """Convert logical [rows, K/16] scales to Blackwell SWIZZLE_32_4_4."""
    import torch

    if scales.ndim != 2 or scales.dtype != torch.float8_e4m3fn:
        raise TypeError("scales must be a 2D torch.float8_e4m3fn tensor")
    rows, cols = scales.shape
    if rows % 128 or cols % 4:
        raise ValueError("scale shape must be divisible by [128, 4]")
    blocks = scales.contiguous().view(rows // 128, 128, cols // 4, 4)
    blocks = blocks.permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


def from_blocked_nvfp4_scales(scales, rows: int, k: int):
    """Inverse of :func:`to_blocked_nvfp4_scales`, mainly for references."""
    cols = k // 16
    tiles = scales.view(rows // 128, cols // 4, 32, 4, 4)
    blocks = tiles.transpose(2, 3).reshape(rows // 128, cols // 4, 128, 4)
    return blocks.permute(0, 2, 1, 3).reshape(rows, cols).contiguous()


def _pack_e2m1(values):
    """Pack an in-place, clamped FP32 tensor into E2M1 pairs.

    ``values`` is a quantization temporary, so consuming it here avoids two
    otherwise full-sized FP32 allocations for ``clamp`` and ``abs``.
    """
    import torch

    midpoints = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
                             device=values.device, dtype=torch.float32)
    negative = values < 0
    values.abs_()
    # E2M1 has only eight magnitude bins.  Request int32 indices explicitly:
    # bucketize otherwise materializes an int64 tensor for every input value.
    code = torch.bucketize(values, midpoints, out_int32=True).to(torch.uint8)
    code |= negative.to(torch.uint8) << 3
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def quantize_nvfp4_rows(matrix):
    """Quantize contiguous [rows,K] data to packed E2M1 plus blocked UE4M3."""
    import torch

    if matrix.ndim != 2 or matrix.size(0) % 128 or matrix.size(1) % 64:
        raise ValueError("matrix shape must be divisible by [128, 64]")
    x = matrix.float().contiguous()
    blocks = x.view(x.size(0), x.size(1) // 16, 16)
    scale = (blocks.abs().amax(dim=-1) / 6.0).to(torch.float8_e4m3fn)
    scale_f = scale.float()
    scale_f.masked_fill_(scale_f == 0, 1.0)
    normalized = blocks / scale_f.unsqueeze(-1)
    # Neither x nor its blocks view is needed after normalization.  Releasing
    # them before packing is important for large CUDA-resident matrices.
    del blocks, x, scale_f
    normalized.clamp_(-6.0, 6.0)
    packed = _pack_e2m1(normalized.reshape(matrix.shape))
    return packed, to_blocked_nvfp4_scales(scale)


def quantize_nvfp4_gemm(a, b):
    """Quantize mathematical A[M,K] and B[K,N] for :func:`nvfp4_gemm`."""
    if a.ndim != 2 or b.ndim != 2 or a.size(1) != b.size(0):
        raise ValueError("expected compatible 2D GEMM operands")
    aq, sfa = quantize_nvfp4_rows(a)
    bq, sfb = quantize_nvfp4_rows(b.t().contiguous())
    return aq, bq, sfa, sfb


def nvfp4_gemm(a, b, sfa, sfb, config: Nvfp4GemmConfig | None = None):
    return compile_nvfp4_gemm(config).nvfp4_gemm(a, b, sfa, sfb)


def nvfp4_gemm_out(a, b, sfa, sfb, out, config: Nvfp4GemmConfig | None = None):
    return compile_nvfp4_gemm(config).nvfp4_gemm_out(a, b, sfa, sfb, out)
