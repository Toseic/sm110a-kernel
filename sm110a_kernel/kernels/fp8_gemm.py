"""SM110A E4M3 x E4M3 GEMM with UE8M0 K=32 block scales."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "csrc/fp8_gemm/src"
_INC = _ROOT / "include"

@dataclass(frozen=True)
class Fp8GemmConfig:
    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 128
    cluster_m: int = 2
    epilogue_block_n: int = 32
    split_k: int = 1
    use_prefetch: bool = False
    def validate(self) -> None:
        if self.tile_m != 128 or self.tile_n not in (64, 128, 256) or self.tile_k not in (64, 128):
            raise ValueError("FP8 GEMM requires tile_m=128, tile_n={64,128,256}, tile_k={64,128}")
        if self.cluster_m not in (1, 2) or self.split_k not in (1, 2, 4, 8) or self.cluster_m * self.split_k > 8:
            raise ValueError("cluster_m must be 1/2 and cluster_m * split_k <= 8")
        if self.epilogue_block_n not in (32, 64, 128) or self.tile_n % self.epilogue_block_n:
            raise ValueError("epilogue_block_n must divide tile_n and be 32, 64, or 128")
        payload = self.tile_m * 128 + (self.tile_n // self.cluster_m) * 128 + self.tile_m * (self.tile_k // 8) + (self.tile_k // 32) * ((self.tile_n + 127) // 128) * 512
        stage = (payload + 1023) & ~1023
        epilogue = 2 * self.tile_m * self.epilogue_block_n * 2
        reduction = self.tile_m * self.epilogue_block_n * 4 if self.split_k > 1 else 0
        if (232448 - 1024 - 256 - epilogue - reduction) // stage < 2:
            raise ValueError("configuration leaves room for fewer than two mainloop stages")

def _cflags(c: Fp8GemmConfig, _variant) -> tuple[str, ...]:
    return tuple(f"-DSM110A_FP8_{name}={value}" for name, value in (("TILE_M",c.tile_m),("TILE_N",c.tile_n),("TILE_K",c.tile_k),("CLUSTER_M",c.cluster_m),("SPLIT_K",c.split_k),("EPILOGUE_BLOCK_N",c.epilogue_block_n),("USE_TMA_PREFETCH",int(c.use_prefetch))))

class Fp8GemmModule:
    def __init__(self, binding, library, config): self._binding, self._launcher, self._config = binding, library.entry_address, config
    def _args(self):
        c=self._config; return self._launcher,c.tile_m,c.tile_n,c.tile_k,c.cluster_m,c.split_k
    def fp8_gemm(self, a, b, sfa, sfb): return self._binding.fp8_gemm(a,b,sfa,sfb,*self._args())
    def fp8_gemm_out(self, a, b, sfa, sfb, out): return self._binding.fp8_gemm_out(a,b,sfa,sfb,out,*self._args())

_JIT = SplitCudaJit(name="sm110a_fp8_gemm", root=_ROOT,
    binding_sources=(_SRC / "fp8_gemm_split_torch_binding.cpp",), kernel_sources=(_SRC / "fp8_gemm_kernel.cu",),
    kernel_dependencies=(_INC / "fp8_gemm.h", _INC / "common.h"), kernel_symbol="sm110a_fp8_gemm_launch",
    config_key=lambda c:c, kernel_cflags=_cflags, make_facade=Fp8GemmModule, include_paths=(_INC,))

def compile_fp8_gemm(config: Fp8GemmConfig | None = None) -> Fp8GemmModule:
    config = config or Fp8GemmConfig()
    if not isinstance(config, Fp8GemmConfig): raise TypeError("config must be Fp8GemmConfig")
    config.validate(); return _JIT.compile(config)
def compile_fp8_gemm_configs(configs: Iterable[Fp8GemmConfig], max_workers=None):
    configs=tuple(configs)
    for c in configs:
        if not isinstance(c,Fp8GemmConfig): raise TypeError("configs must contain Fp8GemmConfig")
        c.validate()
    return _JIT.compile_many(configs,max_workers)

def to_blocked_fp8_scales(scales):
    """Pack logical E8M0 ``[rows,K/32]`` scales into 128x4x4 hardware layout."""
    import torch
    if scales.ndim != 2 or scales.dtype != torch.float8_e8m0fnu: raise TypeError("scales must be 2D torch.float8_e8m0fnu")
    rows, cols = scales.shape
    if rows % 128: raise ValueError("scale rows must be divisible by 128")
    padded=torch.zeros((rows,cols,4),device=scales.device,dtype=torch.uint8)
    padded[...,0]=scales.contiguous().view(torch.uint8)
    blocks=padded.view(rows//128,128,cols,4).permute(0,2,1,3)
    return blocks.reshape(-1,4,32,4).transpose(1,2).reshape(-1).view(torch.float8_e8m0fnu).contiguous()

def from_blocked_fp8_scales(scales, rows:int, k:int):
    import torch
    if scales.dtype != torch.float8_e8m0fnu or rows % 128 or k % 32: raise ValueError("expected E8M0 scales with rows % 128 == 0 and K % 32 == 0")
    tiles=scales.view(rows//128,k//32,32,4,4)
    blocks=tiles.view(torch.uint8).transpose(2,3).reshape(rows//128,k//32,128,4)
    return blocks.permute(0,2,1,3)[...,0].reshape(rows,k//32).view(torch.float8_e8m0fnu).contiguous()

def quantize_fp8_rows(matrix):
    """Quantize [rows,K] to E4M3 values and K=32 UE8M0 scales.

    Each 32-element block uses the block amax exponent as its power-of-two
    scale, so the normalized amax lands in [1, 2) and survives the E4M3 range.
    """
    import torch
    if matrix.ndim != 2 or matrix.size(0)%128 or matrix.size(1)%32: raise ValueError("matrix shape must be divisible by [128,32]")
    x=matrix.float().contiguous(); blocks=x.view(x.size(0),x.size(1)//32,32); amax=blocks.abs().amax(-1).contiguous()
    exponent=((amax.view(torch.int32)>>23)&0xff); scale=exponent.to(torch.uint8).view(torch.float8_e8m0fnu)
    scale_f=(scale.view(torch.uint8).to(torch.int32)<<23).view(torch.float32)
    codes=(blocks/scale_f.unsqueeze(-1)).to(torch.float8_e4m3fn).view(x.shape).contiguous()
    return codes, to_blocked_fp8_scales(scale)

def quantize_fp8_gemm(a, b):
    """Quantize BF16/FP32 A[M,K], B[K,N] for :func:`fp8_gemm`."""
    if a.ndim != 2 or b.ndim != 2 or a.size(1)!=b.size(0): raise ValueError("expected compatible 2D operands")
    b_t = b.t().contiguous()
    a_codes, sfa = quantize_fp8_rows(a)
    b_codes, sfb = quantize_fp8_rows(b_t)
    return a_codes, b_codes, sfa, sfb

def dequantize_fp8_rows(codes, sfb, rows:int, k:int):
    import torch
    if codes.dtype != torch.float8_e4m3fn or tuple(codes.shape) != (rows,k): raise ValueError("codes must be float8_e4m3fn [rows,K]")
    scale=(from_blocked_fp8_scales(sfb,rows,k).view(torch.uint8).to(torch.int32)<<23).view(torch.float32)
    return (codes.float().view(rows,k//32,32)*scale.unsqueeze(-1)).reshape(rows,k)

def fp8_gemm_reference(a,b,sfa,sfb):
    import torch
    return (dequantize_fp8_rows(a,sfa,a.size(0),a.size(1)) @ dequantize_fp8_rows(b,sfb,b.size(0),b.size(1)).t()).to(torch.bfloat16)

def _default(a,b): return Fp8GemmConfig(cluster_m=1 if a.size(0)%256 else 2,tile_n=256 if b.size(0)%256==0 else 128,tile_k=128 if a.size(1)%128==0 else 64)


@lru_cache(maxsize=None)
def _compiled_facade(config: Fp8GemmConfig) -> Fp8GemmModule:
    """Resolve a convenience-API configuration once per process.

    ``SplitCudaJit.compile`` deliberately checks environment-selected variants
    on every call.  That is appropriate for explicit compilation, but calling
    it between tiny kernel launches leaves visible host-side gaps in the CUDA
    stream.  The convenience APIs are a steady-state execution path, so hold
    their already-resolved facade here.  Call ``compile_fp8_gemm``
    explicitly when selecting a new JIT/environment variant at runtime.
    """
    return compile_fp8_gemm(config)


def fp8_gemm(a,b,sfa,sfb,config=None):
    return _compiled_facade(config or _default(a,b)).fp8_gemm(a,b,sfa,sfb)


def fp8_gemm_out(a,b,sfa,sfb,out,config=None):
    return _compiled_facade(config or _default(a,b)).fp8_gemm_out(a,b,sfa,sfb,out)
