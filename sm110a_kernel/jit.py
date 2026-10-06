"""Backward-compatible imports for the kernel-owned JIT modules.

New kernel-specific logic belongs in :mod:`sm110a_kernel.kernels`; this
module remains as a stable import path for existing callers.
"""

from .kernels.bf16_gemm import (
    Bf16GemmConfig,
    bf16_gemm,
    bf16_gemm_out,
    compile_bf16_gemm,
    compile_bf16_gemm_configs,
    get_bf16_gemm_module,
)
from .kernels.fc1_block_interleaved_swiglu import (
    Fc1BlockInterleavedSwiGLUConfig,
    Fc1Metadata,
    compile_fc1_block_interleaved_swiglu,
    compile_fc1_block_interleaved_swiglu_configs,
    fc1_block_interleaved_swiglu,
    fc1_block_interleaved_swiglu_out,
    get_fc1_block_interleaved_swiglu_module,
    prepare_fc1_metadata,
)
from .kernels.fp8mxfp4_gemm import (
    Fp8Mxfp4GemmConfig,
    compile_fp8mxfp4_gemm,
    compile_fp8mxfp4_gemm_configs,
    fp8mxfp4_gemm,
    fp8mxfp4_gemm_out,
)
from .kernels.grouped_m_gemm import (
    GroupedMConfig,
    GroupedMMetadata,
    bf16_grouped_m_gemm,
    bf16_grouped_m_gemm_out,
    compile_bf16_grouped_m_gemm,
    compile_bf16_grouped_m_gemm_configs,
    get_bf16_grouped_m_gemm_module,
    prepare_grouped_m_metadata,
)
from .kernels.mxfp4_gemm import (
    Mxfp4GemmConfig,
    compile_mxfp4_gemm,
    compile_mxfp4_gemm_configs,
    mxfp4_gemm,
    mxfp4_gemm_out,
)
from .kernels.nvfp4_gemm import (
    Nvfp4GemmConfig,
    compile_nvfp4_gemm,
    compile_nvfp4_gemm_configs,
    nvfp4_gemm,
    nvfp4_gemm_out,
)
from .kernels.nvfp4_grouped_m_gemm import (
    Nvfp4GroupedMConfig,
    compile_nvfp4_grouped_m_gemm,
    compile_nvfp4_grouped_m_gemm_configs,
    nvfp4_grouped_m_gemm,
    nvfp4_grouped_m_gemm_out,
    prepare_nvfp4_grouped_m_metadata,
)

__all__ = [
    "Bf16GemmConfig",
    "Fp8Mxfp4GemmConfig",
    "Fc1BlockInterleavedSwiGLUConfig",
    "Fc1Metadata",
    "GroupedMConfig",
    "GroupedMMetadata",
    "Mxfp4GemmConfig",
    "Nvfp4GemmConfig",
    "Nvfp4GroupedMConfig",
    "bf16_gemm",
    "bf16_gemm_out",
    "bf16_grouped_m_gemm",
    "bf16_grouped_m_gemm_out",
    "compile_bf16_gemm",
    "compile_bf16_gemm_configs",
    "compile_bf16_grouped_m_gemm",
    "compile_bf16_grouped_m_gemm_configs",
    "compile_fc1_block_interleaved_swiglu",
    "compile_fc1_block_interleaved_swiglu_configs",
    "compile_fp8mxfp4_gemm",
    "compile_fp8mxfp4_gemm_configs",
    "compile_mxfp4_gemm",
    "compile_mxfp4_gemm_configs",
    "compile_nvfp4_gemm",
    "compile_nvfp4_gemm_configs",
    "compile_nvfp4_grouped_m_gemm",
    "compile_nvfp4_grouped_m_gemm_configs",
    "fc1_block_interleaved_swiglu",
    "fc1_block_interleaved_swiglu_out",
    "fp8mxfp4_gemm",
    "fp8mxfp4_gemm_out",
    "get_bf16_gemm_module",
    "get_bf16_grouped_m_gemm_module",
    "get_fc1_block_interleaved_swiglu_module",
    "mxfp4_gemm",
    "mxfp4_gemm_out",
    "nvfp4_gemm",
    "nvfp4_gemm_out",
    "nvfp4_grouped_m_gemm",
    "nvfp4_grouped_m_gemm_out",
    "prepare_fc1_metadata",
    "prepare_grouped_m_metadata",
    "prepare_nvfp4_grouped_m_metadata",
]
