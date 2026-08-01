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
from .nvfp4_fc1 import (
    CuteDslNvfp4Fc1Config,
    CuteDslNvfp4Fc1Runner,
    cute_dsl_nvfp4_fc1_out,
)
from .nvfp4_gemm import (
    CuteDslNvfp4GemmConfig,
    CuteDslNvfp4GemmRunner,
    cute_dsl_nvfp4_gemm_out,
)
from .nvfp4_grouped_m_gemm import (
    CuteDslNvfp4GroupedMGemmConfig,
    CuteDslNvfp4GroupedMGemmRunner,
    cute_dsl_nvfp4_grouped_m_gemm_out,
)

__all__ = [
    "CuteDslBf16GemmConfig",
    "CuteDslBf16GemmRunner",
    "cute_dsl_bf16_gemm_out",
    "CuteDslNvfp4Fc1Config",
    "CuteDslNvfp4Fc1Runner",
    "cute_dsl_nvfp4_fc1_out",
    "CuteDslNvfp4GemmConfig",
    "CuteDslNvfp4GemmRunner",
    "cute_dsl_nvfp4_gemm_out",
    "CuteDslNvfp4GroupedMGemmConfig",
    "CuteDslNvfp4GroupedMGemmRunner",
    "cute_dsl_nvfp4_grouped_m_gemm_out",
]
