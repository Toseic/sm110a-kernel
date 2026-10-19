"""SM110A native MXFP4 block-scaled GEMM."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import BaseKernelRunner, PreparedKernel

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/mxfp4_gemm/src"
_INCLUDE_DIR = _ROOT / "include"


@dataclass(frozen=True)
class Mxfp4GemmConfig:
    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 64
    cluster_m: int = 2
    epilogue_block_n: int = 32
    split_k: int = 1
    use_prefetch: bool = False

    def validate(self) -> None:
        if self.tile_m != 128:
            # mxf4 has an M=128 instruction shape.  Small-M workloads
            # use the one-CTA M=128 tile; unlike BF16, it has no M=64 form.
            raise ValueError("tile_m must be 128 for MXFP4 mxf4")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.split_k not in (1, 2, 4, 8) or self.cluster_m * self.split_k > 8:
            raise ValueError(
                "split_k must be 1, 2, 4, or 8 and cluster_m * split_k <= 8"
            )
        if self.tile_n not in (64, 128, 256):
            raise ValueError("tile_n must be 64, 128, or 256")
        if self.tile_k not in (64, 128, 256):
            raise ValueError("tile_k must be 64, 128, or 256")
        if (
            self.epilogue_block_n not in (32, 64, 128)
            or self.tile_n % self.epilogue_block_n
        ):
            raise ValueError("epilogue_block_n must be 32, 64, or 128")
        # Keep this in sync with KernelConfig.  Split-K reserves one FP32
        # epilogue block for DSMEM partial reduction, and every configuration
        # must still leave room for two mainloop stages.
        per_stage_payload = (
            self.tile_m * 128
            + (self.tile_n // self.cluster_m) * 128
            # Each K=64 slab is stored as one 512-byte 128x4 scale tile.
            # MXFP4 consumes two E8M0 values and leaves the other two bytes
            # as layout padding, so its shared-memory footprint matches NVFP4.
            + self.tile_m * (self.tile_k // 16)
            + (self.tile_k // 64) * ((self.tile_n + 127) // 128) * 512
        )
        per_stage_smem = (per_stage_payload + 1023) & ~1023
        epilogue_smem = 2 * self.tile_m * self.epilogue_block_n * 2
        reduction_smem = (
            self.tile_m * self.epilogue_block_n * 4 if self.split_k > 1 else 0
        )
        if (232448 - 1024 - 256 - epilogue_smem - reduction_smem) // per_stage_smem < 2:
            raise ValueError(
                "configuration leaves room for fewer than two mainloop stages"
            )
        if not isinstance(self.use_prefetch, bool):
            raise TypeError("use_prefetch must be a bool")


def _kernel_cflags(config: Mxfp4GemmConfig, _variant) -> tuple[str, ...]:
    return tuple(
        f"-DSM110A_MXFP4_{name}={value}"
        for name, value in (
            ("TILE_M", config.tile_m),
            ("TILE_N", config.tile_n),
            ("TILE_K", config.tile_k),
            ("CLUSTER_M", config.cluster_m),
            ("SPLIT_K", config.split_k),
            ("EPILOGUE_BLOCK_N", config.epilogue_block_n),
            ("USE_TMA_PREFETCH", int(config.use_prefetch)),
        )
    )


class Mxfp4GemmModule:
    def __init__(self, binding, library, config: Mxfp4GemmConfig):
        self._binding = binding
        self._launcher = library.entry_address
        self._config = config

    def _args(self):
        c = self._config
        return self._launcher, c.tile_m, c.tile_n, c.tile_k, c.cluster_m, c.split_k

    def mxfp4_gemm(self, a, b, sfa, sfb):
        return self._binding.mxfp4_gemm(a, b, sfa, sfb, *self._args())

    def mxfp4_gemm_out(self, a, b, sfa, sfb, out):
        return self._binding.mxfp4_gemm_out(a, b, sfa, sfb, out, *self._args())


def _default_config(a, b) -> Mxfp4GemmConfig:
    """Pick compatible one-CTA/narrow tiles for the smallest legal shapes."""
    if getattr(a, "ndim", None) != 2 or getattr(b, "ndim", None) != 2:
        return Mxfp4GemmConfig()
    k = a.size(1) * 2
    return Mxfp4GemmConfig(
        cluster_m=1 if a.size(0) % 256 else 2,
        tile_n=256 if b.size(0) % 256 == 0 else 128,
        tile_k=256 if k % 256 == 0 else (128 if k % 128 == 0 else 64),
    )


_JIT = SplitCudaJit(
    name="sm110a_mxfp4_gemm",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "mxfp4_gemm_split_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "mxfp4_gemm_kernel.cu",),
    kernel_dependencies=(
        _INCLUDE_DIR / "mxfp4_gemm.h",
        _INCLUDE_DIR / "common.h",
    ),
    kernel_symbol="sm110a_mxfp4_gemm_launch",
    config_key=lambda config: config,
    kernel_cflags=_kernel_cflags,
    make_facade=Mxfp4GemmModule,
    include_paths=(_INCLUDE_DIR,),
)


def compile_mxfp4_gemm(config: Mxfp4GemmConfig | None = None) -> Mxfp4GemmModule:
    config = config or Mxfp4GemmConfig()
    if not isinstance(config, Mxfp4GemmConfig):
        raise TypeError("config must be an Mxfp4GemmConfig")
    config.validate()
    return _JIT.compile(config)


def compile_mxfp4_gemm_configs(
    configs: Iterable[Mxfp4GemmConfig], max_workers=None
) -> tuple[Mxfp4GemmModule, ...]:
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Mxfp4GemmConfig):
            raise TypeError("configs must contain Mxfp4GemmConfig instances")
        config.validate()
    return _JIT.compile_many(configs, max_workers)


def to_blocked_mxfp4_scales(scales):
    """Pack logical E8M0 ``[rows, K/32]`` scales for the MXFP4 kernel.

    ``tcgen05.cp.32x128b.warpx4`` transfers one 512-byte 128x4 tile per
    K=64 slab.  MXFP4 consumes the first two columns (one E8M0 scale for
    each 32-element block); columns 2 and 3 are zero padding required by the
    hardware copy layout.
    """
    import torch

    if scales.ndim != 2 or scales.dtype != torch.float8_e8m0fnu:
        raise TypeError("scales must be a 2D torch.float8_e8m0fnu tensor")
    rows, cols = scales.shape
    if rows % 128 or cols % 2:
        raise ValueError("scale shape must be divisible by [128, 2]")
    # K/32 logical columns come in pairs per K=64 MMA.  Materialize the two
    # unused bytes per row before applying the standard 128x4 swizzle.
    logical = scales.contiguous().view(rows, cols // 2, 2)
    padded = torch.zeros((rows, cols // 2, 4), device=scales.device, dtype=torch.uint8)
    padded[..., :2] = logical.view(torch.uint8)
    blocks = padded.view(rows // 128, 128, cols // 2, 4)
    blocks = blocks.permute(0, 2, 1, 3)
    return (
        blocks.reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1)
        .view(torch.float8_e8m0fnu)
        .contiguous()
    )


def from_blocked_mxfp4_scales(scales, rows: int, k: int):
    """Inverse of :func:`to_blocked_mxfp4_scales`, mainly for references."""
    import torch

    if scales.dtype != torch.float8_e8m0fnu or rows % 128 or k % 64:
        raise ValueError(
            "expected E8M0 MXFP4 scales with rows % 128 == 0 and K % 64 == 0"
        )
    k64_tiles = k // 64
    tiles = scales.view(rows // 128, k64_tiles, 32, 4, 4)
    blocks = (
        tiles.view(torch.uint8).transpose(2, 3).reshape(rows // 128, k64_tiles, 128, 4)
    )
    logical = blocks.permute(0, 2, 1, 3)[..., :2].reshape(rows, k // 32)
    return logical.view(torch.float8_e8m0fnu).contiguous()


def _pack_e2m1(values):
    """Pack an in-place, clamped FP32 tensor into E2M1 pairs.

    ``values`` is a quantization temporary, so consuming it here avoids two
    otherwise full-sized FP32 allocations for ``clamp`` and ``abs``.
    """
    import torch

    midpoints = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
        device=values.device,
        dtype=torch.float32,
    )
    negative = values < 0
    values.abs_()
    # E2M1 has only eight magnitude bins.  Request int32 indices explicitly:
    # bucketize otherwise materializes an int64 tensor for every input value.
    code = torch.bucketize(values, midpoints, out_int32=True).to(torch.uint8)
    code |= negative.to(torch.uint8) << 3
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def quantize_mxfp4_rows(matrix):
    """Quantize [rows,K] to E2M1 plus E8M0 block-32 scales.

    The scale is ``2**(floor(log2(amax)) - 2)``: a power-of-two MX scale
    which keeps E2M1's largest finite value (6) in range.
    """
    import torch

    if matrix.ndim != 2 or matrix.size(0) % 128 or matrix.size(1) % 64:
        raise ValueError("matrix shape must be divisible by [128, 64]")
    x = matrix.float().contiguous()
    blocks = x.view(x.size(0), x.size(1) // 32, 32)
    amax = blocks.abs().amax(dim=-1).contiguous()
    # E8M0's byte is a biased exponent.  The zero encoding is 2**-127,
    # which makes all-zero blocks remain zero after FP4 packing.
    exponent = (amax.view(torch.int32) >> 23) & 0xFF
    scale = (exponent - 2).clamp_(0, 255).to(torch.uint8).view(torch.float8_e8m0fnu)
    scale_f = (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    normalized = blocks / scale_f.unsqueeze(-1)
    # Neither x nor its blocks view is needed after normalization.  Releasing
    # them before packing is important for large CUDA-resident matrices.
    del blocks, x, scale_f
    normalized.clamp_(-6.0, 6.0)
    packed = _pack_e2m1(normalized.reshape(matrix.shape))
    return packed, to_blocked_mxfp4_scales(scale)


def quantize_mxfp4_gemm(a, b):
    """Quantize mathematical A[M,K] and B[K,N] for :func:`mxfp4_gemm`."""
    if a.ndim != 2 or b.ndim != 2 or a.size(1) != b.size(0):
        raise ValueError("expected compatible 2D GEMM operands")
    aq, sfa = quantize_mxfp4_rows(a)
    bq, sfb = quantize_mxfp4_rows(b.t().contiguous())
    return aq, bq, sfa, sfb


def dequantize_mxfp4_rows(packed, scales, rows: int, k: int):
    """Decode a packed MXFP4 row matrix for testing and reference paths."""
    import torch

    if (
        packed.ndim != 2
        or packed.shape != (rows, k // 2)
        or packed.dtype != torch.float4_e2m1fn_x2
    ):
        raise ValueError(
            "packed must have shape [rows, K/2] and dtype float4_e2m1fn_x2"
        )
    logical_scales = from_blocked_mxfp4_scales(scales, rows, k)
    scale_f = (logical_scales.view(torch.uint8).to(torch.int32) << 23).view(
        torch.float32
    )
    codes = packed.view(torch.uint8)
    codes = torch.stack((codes & 0xF, codes >> 4), dim=-1).reshape(rows, k)
    magnitudes = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device=packed.device,
        dtype=torch.float32,
    )
    values = magnitudes[(codes & 0x7).to(torch.long)]
    values = torch.where((codes & 0x8) != 0, -values, values)
    return (values.view(rows, k // 32, 32) * scale_f.unsqueeze(-1)).reshape(rows, k)


def mxfp4_gemm_reference(a, b, sfa, sfb):
    """Reference MXFP4 GEMM by dequantizing E2M1/E8M0 operands in FP32."""
    import torch

    m, k2 = a.shape
    n = b.shape[0]
    k = k2 * 2
    if b.shape != (n, k2):
        raise ValueError("A and transposed-B packed K dimensions must match")
    return (
        dequantize_mxfp4_rows(a, sfa, m, k) @ dequantize_mxfp4_rows(b, sfb, n, k).t()
    ).to(torch.bfloat16)


def mxfp4_gemm(a, b, sfa, sfb, config: Mxfp4GemmConfig | None = None):
    return compile_mxfp4_gemm(config or _default_config(a, b)).mxfp4_gemm(
        a, b, sfa, sfb
    )


def mxfp4_gemm_out(a, b, sfa, sfb, out, config: Mxfp4GemmConfig | None = None):
    return compile_mxfp4_gemm(config or _default_config(a, b)).mxfp4_gemm_out(
        a, b, sfa, sfb, out
    )


MXFP4_GEMM_CONFIGS = tuple(
    Mxfp4GemmConfig(
        tile_n=tile_n,
        tile_k=tile_k,
        cluster_m=cluster_m,
        epilogue_block_n=epilogue_block_n,
        split_k=split_k,
        use_prefetch=use_prefetch,
    )
    for cluster_m in (1, 2)
    for split_k in (1, 2, 4, 8)
    for tile_n in (64, 128, 256)
    for tile_k in (64, 128, 256)
    for epilogue_block_n in (32, 64, 128)
    for use_prefetch in (False, True)
    if tile_n % epilogue_block_n == 0 and cluster_m * split_k <= 8
    if not (
        # Mirror Mxfp4GemmConfig.validate without constructing invalid
        # candidates during module import.
        (
            232448
            - 1024
            - 256
            - 2 * 128 * epilogue_block_n * 2
            - (128 * epilogue_block_n * 4 if split_k > 1 else 0)
        )
        // (
            (
                128 * 128
                + (tile_n // cluster_m) * 128
                +
                # MXFP4 stores each K=64 pair in a padded 128x4 scale tile.
                128 * (tile_k // 16)
                + (tile_k // 64) * ((tile_n + 127) // 128) * 512
                + 1023
            )
            & ~1023
        )
        < 2
    )
)


def compatible_candidates(m, n, k):
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    return tuple(
        config
        for config in MXFP4_GEMM_CONFIGS
        if m % (config.tile_m * config.cluster_m) == 0
        and n % config.tile_n == 0
        and n % 128 == 0
        and k % (config.tile_k * config.split_k) == 0
        # Match CuTE's pruning: rolling L2 prefetch is a tactic for large K
        # or an incomplete CTA wave, rather than a universally profitable
        # instruction stream.
        and (
            not config.use_prefetch
            or k >= 8192
            or 0.5
            < (
                math.ceil(m / (config.tile_m * config.cluster_m))
                * math.ceil(n / config.tile_n)
                / sm_count
            )
            < 1.0
        )
    )


def format_candidate(config):
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/c{config.cluster_m}/e{config.epilogue_block_n}/sk{config.split_k}"
        f"/{'prefetch' if config.use_prefetch else 'noprefetch'}"
    )


class Mxfp4GemmRunner(BaseKernelRunner):
    namespace = "mxfp4_gemm_splitk_v1"
    format_candidate = staticmethod(format_candidate)

    def __init__(self, a, b, sfa, sfb, out):
        super().__init__(a.device)
        self.a, self.b, self.sfa, self.sfb, self.out = a, b, sfa, sfb, out
        self.m, self.n, self.k = a.size(0), b.size(0), a.size(1) * 2

    @property
    def workload_key(self):
        return (self.m, self.n, self.k)

    def compatible_candidates(self):
        return compatible_candidates(self.m, self.n, self.k)

    def pre_jit(self, candidate):
        return compile_mxfp4_gemm(candidate)

    def prepare(self, candidate):
        module = self.pre_jit(candidate)
        return PreparedKernel(
            candidate,
            lambda: module.mxfp4_gemm_out(self.a, self.b, self.sfa, self.sfb, self.out),
        )
