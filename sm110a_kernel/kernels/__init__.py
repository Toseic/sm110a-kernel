"""Kernel-owned Python entry points.

Each module in this package owns the configuration, JIT recipe and public
facade for one CUDA kernel family.  The package root only re-exports these
entry points for backwards compatibility.
"""

from .bf16_gemm import (
    Bf16GemmConfig,
    bf16_gemm,
    bf16_gemm_out,
    compile_bf16_gemm,
    get_bf16_gemm_module,
)
from .fc1_block_interleaved_swiglu import (
    Fc1BlockInterleavedSwiGLUConfig,
    Fc1Metadata,
    compile_fc1_block_interleaved_swiglu,
    compile_fc1_block_interleaved_swiglu_configs,
    fc1_block_interleaved_swiglu,
    fc1_block_interleaved_swiglu_out,
    get_fc1_block_interleaved_swiglu_module,
    prepare_fc1_metadata,
)
from .fc2_finalize import (
    Fc2FinalizeConfig,
    compile_fc2_finalize,
    compile_fc2_finalize_configs,
    fc2_finalize,
    fc2_finalize_out,
    fc2_finalize_with_routing_out,
    get_fc2_finalize_module,
)
from .grouped_m_gemm import (
    GroupedMConfig,
    compile_bf16_grouped_m_gemm,
    compile_bf16_grouped_m_gemm_configs,
    bf16_grouped_m_gemm,
    bf16_grouped_m_gemm_out,
    get_bf16_grouped_m_gemm_module,
)
from .moe_routing import pack_routing_metadata

__all__ = [name for name in globals() if not name.startswith("_")]
