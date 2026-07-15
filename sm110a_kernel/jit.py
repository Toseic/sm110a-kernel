"""Backward-compatible imports for the kernel-owned JIT modules.

New kernel-specific logic belongs in :mod:`sm110a_kernel.kernels`; this
module remains as a stable import path for existing callers.
"""

from .kernels.bf16_gemm import (
    Bf16GemmConfig, bf16_gemm, bf16_gemm_out, compile_bf16_gemm,
    get_bf16_gemm_module,
)
from .kernels.fc1_block_interleaved_swiglu import (
    Fc1BlockInterleavedSwiGLUConfig,
    Fc1Metadata,
    compile_fc1_block_interleaved_swiglu,
    compile_fc1_block_interleaved_swiglu_configs,
    get_fc1_block_interleaved_swiglu_module,
    fc1_block_interleaved_swiglu,
    fc1_block_interleaved_swiglu_out,
    prepare_fc1_metadata,
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

__all__ = [name for name in globals() if not name.startswith("_")]
