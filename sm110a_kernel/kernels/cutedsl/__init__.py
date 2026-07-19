"""Standalone CuTE DSL kernel runners."""

# Apply before importing the TensorRT-LLM CuTE DSL kernels below: those modules
# instantiate CUTLASS's block-scaled MMA wrappers during JIT compilation.
from .thor_compat import apply_thor_mma_compat_patch

apply_thor_mma_compat_patch()

from .bf16_gemm import (
    CuteDslBf16GemmConfig,
    CuteDslBf16GemmRunner,
    cute_dsl_bf16_gemm_out,
)
from .nvfp4_gemm import (
    CuteDslNvfp4GemmConfig,
    CuteDslNvfp4GemmRunner,
    cute_dsl_nvfp4_gemm_out,
)

__all__ = [
    "CuteDslBf16GemmConfig",
    "CuteDslBf16GemmRunner",
    "cute_dsl_bf16_gemm_out",
    "CuteDslNvfp4GemmConfig",
    "CuteDslNvfp4GemmRunner",
    "cute_dsl_nvfp4_gemm_out",
]
