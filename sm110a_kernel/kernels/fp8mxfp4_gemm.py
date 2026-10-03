"""SM110A E4M3 activation x E2M1/E8M0(K=32) MXFP4 GEMM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "csrc/fp8mxfp4_gemm/src"
_INC = _ROOT / "include"

@dataclass(frozen=True)
class Fp8Mxfp4GemmConfig:
    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 128
    cluster_m: int = 2
    epilogue_block_n: int = 32
    split_k: int = 1
    use_prefetch: bool = False
    compact_b_gmem: bool = True
    def validate(self) -> None:
        if self.tile_m != 128 or self.tile_n not in (64, 128, 256) or self.tile_k not in (64, 128):
            raise ValueError("FP8xMXFP4 requires tile_m=128, tile_n={64,128,256}, tile_k={64,128}")
        if self.compact_b_gmem and self.tile_k != 128:
            raise ValueError("compact_b_gmem uses TMA b4x16_p64 and requires tile_k=128")
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

def _cflags(c: Fp8Mxfp4GemmConfig, _variant) -> tuple[str, ...]:
    return tuple(f"-DSM110A_FP8MXFP4_{name}={value}" for name, value in (("TILE_M",c.tile_m),("TILE_N",c.tile_n),("TILE_K",c.tile_k),("CLUSTER_M",c.cluster_m),("SPLIT_K",c.split_k),("EPILOGUE_BLOCK_N",c.epilogue_block_n),("USE_TMA_PREFETCH",int(c.use_prefetch)),("COMPACT_B_GMEM",int(c.compact_b_gmem))))

class Fp8Mxfp4GemmModule:
    def __init__(self, binding, library, config): self._binding, self._launcher, self._config, self._identity = binding, library.entry_address, config, {}
    def _args(self):
        c=self._config; return self._launcher,c.tile_m,c.tile_n,c.tile_k,c.cluster_m,c.split_k,int(c.compact_b_gmem)
    def _sfa(self, a):
        import torch
        key = (a.device, a.size(0), a.size(1))
        if key not in self._identity:
            logical = torch.full((a.size(0), a.size(1) // 32), 127,
                                 device=a.device, dtype=torch.uint8).view(torch.float8_e8m0fnu)
            self._identity[key] = to_blocked_fp8mxfp4_scales(logical)
        return self._identity[key]
    def fp8mxfp4_gemm(self, a, b, sfb): return self._binding.fp8mxfp4_gemm(a,b,self._sfa(a),sfb,*self._args())
    def fp8mxfp4_gemm_out(self, a, b, sfb, out): return self._binding.fp8mxfp4_gemm_out(a,b,self._sfa(a),sfb,out,*self._args())

_JIT = SplitCudaJit(name="sm110a_fp8mxfp4_gemm", root=_ROOT,
    binding_sources=(_SRC / "fp8mxfp4_gemm_split_torch_binding.cpp",), kernel_sources=(_SRC / "fp8mxfp4_gemm_kernel.cu",),
    kernel_dependencies=(_INC / "fp8mxfp4_gemm.h", _INC / "common.h"), kernel_symbol="sm110a_fp8mxfp4_gemm_launch",
    config_key=lambda c:c, kernel_cflags=_cflags, make_facade=Fp8Mxfp4GemmModule, include_paths=(_INC,))

def compile_fp8mxfp4_gemm(config: Fp8Mxfp4GemmConfig | None = None) -> Fp8Mxfp4GemmModule:
    config = config or Fp8Mxfp4GemmConfig()
    if not isinstance(config, Fp8Mxfp4GemmConfig): raise TypeError("config must be Fp8Mxfp4GemmConfig")
    config.validate(); return _JIT.compile(config)
def compile_fp8mxfp4_gemm_configs(configs: Iterable[Fp8Mxfp4GemmConfig], max_workers=None):
    configs=tuple(configs)
    for c in configs:
        if not isinstance(c,Fp8Mxfp4GemmConfig): raise TypeError("configs must contain Fp8Mxfp4GemmConfig")
        c.validate()
    return _JIT.compile_many(configs,max_workers)

def to_blocked_fp8mxfp4_scales(scales):
    """Pack logical E8M0 ``[N,K/32]`` scales into 128x4x4 hardware layout."""
    import torch
    if scales.ndim != 2 or scales.dtype != torch.float8_e8m0fnu: raise TypeError("scales must be 2D torch.float8_e8m0fnu")
    rows, cols = scales.shape
    if rows % 128: raise ValueError("scale rows must be divisible by 128")
    padded=torch.zeros((rows,cols,4),device=scales.device,dtype=torch.uint8)
    padded[...,0]=scales.contiguous().view(torch.uint8)
    blocks=padded.view(rows//128,128,cols,4).permute(0,2,1,3)
    return blocks.reshape(-1,4,32,4).transpose(1,2).reshape(-1).view(torch.float8_e8m0fnu).contiguous()

def from_blocked_fp8mxfp4_scales(scales, rows:int, k:int):
    import torch
    if scales.dtype != torch.float8_e8m0fnu or rows % 128 or k % 32: raise ValueError("expected E8M0 scales with rows % 128 == 0 and K % 32 == 0")
    tiles=scales.view(rows//128,k//32,32,4,4)
    blocks=tiles.view(torch.uint8).transpose(2,3).reshape(rows//128,k//32,128,4)
    return blocks.permute(0,2,1,3)[...,0].reshape(rows,k//32).view(torch.float8_e8m0fnu).contiguous()

def quantize_fp8mxfp4_weights(matrix, compact_b_gmem: bool | None = None):
    """Quantize [N,K] weights to padded or compact E2M1 bytes and K=32 E8M0 scales.

    By default this returns native two-FP4-per-byte ``[N,K/2]`` storage when
    K is divisible by 128; TMA expands it directly to ``b4x16_p64`` SMEM.
    K=64 falls back to padded storage because that TMA format requires a
    128-element box.
    """
    import torch
    if matrix.ndim != 2 or matrix.size(0)%128 or matrix.size(1)%32: raise ValueError("weight shape must be divisible by [128,32]")
    if compact_b_gmem is None:
        compact_b_gmem = matrix.size(1) % 128 == 0
    if compact_b_gmem and matrix.size(1) % 128:
        raise ValueError("compact_b_gmem requires K divisible by 128")
    x=matrix.float().contiguous(); blocks=x.view(x.size(0),x.size(1)//32,32); amax=blocks.abs().amax(-1).contiguous()
    exponent=((amax.view(torch.int32)>>23)&0xff); scale=(exponent-2).clamp_(0,255).to(torch.uint8).view(torch.float8_e8m0fnu)
    scale_f=(scale.view(torch.uint8).to(torch.int32)<<23).view(torch.float32); normalized=(blocks/scale_f.unsqueeze(-1)).clamp_(-6,6)
    mid=torch.tensor((.25,.75,1.25,1.75,2.5,3.5,5.),device=x.device); neg=normalized<0
    codes=torch.bucketize(normalized.abs(),mid,out_int32=True).to(torch.uint8) | (neg.to(torch.uint8)<<3)
    pairs = codes.reshape(matrix.size(0), matrix.size(1) // 16, 16)
    packed = pairs[..., 0::2] | (pairs[..., 1::2] << 4)
    if compact_b_gmem:
        return packed.reshape(matrix.size(0), matrix.size(1) // 2).contiguous(), to_blocked_fp8mxfp4_scales(scale)
    # mxf8f6f4 consumes 16 values as eight bytes followed by 64 bits of
    # padding in shared memory.  The padded-GMEM baseline carries it verbatim.
    storage = torch.zeros((matrix.size(0), matrix.size(1) // 16, 16),
                          device=matrix.device, dtype=torch.uint8)
    storage[..., :8] = packed
    return storage.reshape(matrix.shape).contiguous(), to_blocked_fp8mxfp4_scales(scale)

def quantize_fp8mxfp4_gemm(a, b, compact_b_gmem: bool | None = None):
    """Quantize BF16/FP32 A[M,K], B[K,N] for :func:`fp8mxfp4_gemm`."""
    if a.ndim != 2 or b.ndim != 2 or a.size(1)!=b.size(0): raise ValueError("expected compatible 2D operands")
    return a.to(dtype=__import__('torch').float8_e4m3fn).contiguous(), *quantize_fp8mxfp4_weights(b.t().contiguous(), compact_b_gmem)

def dequantize_fp8mxfp4_weights(packed, sfb, rows:int, k:int):
    import torch
    if packed.dtype != torch.uint8 or packed.shape not in ((rows,k), (rows,k//2)): raise ValueError("packed weights must be uint8 [rows,K] padded or [rows,K/2] compact")
    scale=(from_blocked_fp8mxfp4_scales(sfb,rows,k).view(torch.uint8).to(torch.int32)<<23).view(torch.float32)
    storage=packed.view(rows,k//16,16)[...,:8] if packed.size(1) == k else packed.view(rows,k//16,8)
    code=torch.stack((storage & 0xF, storage >> 4), dim=-1).reshape(rows,k)
    mag=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=packed.device)
    values=mag[(code&7).long()]; values=torch.where((code&8)!=0,-values,values)
    return (values.view(rows,k//32,32)*scale.unsqueeze(-1)).reshape(rows,k)
def fp8mxfp4_gemm_reference(a,b,sfb): return (a.float() @ dequantize_fp8mxfp4_weights(b,sfb,b.size(0),a.size(1)).t()).to(__import__('torch').bfloat16)
def _default(a,b): return Fp8Mxfp4GemmConfig(cluster_m=1 if a.size(0)%256 else 2,tile_n=256 if b.size(0)%256==0 else 128,tile_k=128 if a.size(1)%128==0 else 64,compact_b_gmem=b.size(1)==a.size(1)//2)
def fp8mxfp4_gemm(a,b,sfb,config=None): return compile_fp8mxfp4_gemm(config or _default(a,b)).fp8mxfp4_gemm(a,b,sfb)
def fp8mxfp4_gemm_out(a,b,sfb,out,config=None): return compile_fp8mxfp4_gemm(config or _default(a,b)).fp8mxfp4_gemm_out(a,b,sfb,out)
