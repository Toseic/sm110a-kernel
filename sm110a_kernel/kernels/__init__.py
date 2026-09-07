"""Kernel-owned Python entry points.

Each module in this package owns the configuration, JIT recipe and public
facade for one CUDA kernel family.  The package root only re-exports these
entry points for backwards compatibility.
"""

from .bf16_gemm import (
    Bf16GemmConfig,
    bf16_gemm,
    bf16_gemm_out,
    bf16_gemm_nt_out,
    compile_bf16_gemm,
    compile_bf16_gemm_configs,
    get_bf16_gemm_module,
)
from .bf16_fa4_mha import (
    bf16_fa4_mha,
    bf16_fa4_mha_out,
    get_bf16_fa4_mha_module,
)
from .bf16_fa4_gqa import (
    Bf16Fa4GqaConfig,
    Bf16Fa4GqaTactic,
    bf16_fa4_gqa,
    bf16_fa4_gqa_out,
    compile_bf16_fa4_gqa,
    compile_bf16_fa4_gqa_configs,
    get_bf16_fa4_gqa_module,
)
from .nvfp4_gemm import (
    Nvfp4GemmConfig,
    compile_nvfp4_gemm,
    compile_nvfp4_gemm_configs,
    nvfp4_gemm,
    nvfp4_gemm_out,
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
from .nvfp4_grouped_m_gemm import (
    Nvfp4GroupedMConfig, compile_nvfp4_grouped_m_gemm,
    compile_nvfp4_grouped_m_gemm_configs,
    prepare_nvfp4_grouped_m_metadata, nvfp4_grouped_m_gemm,
    nvfp4_grouped_m_gemm_out,
)
from .nvfp4_fc1 import (
    Nvfp4Fc1Config, Nvfp4Fc1Metadata, compile_nvfp4_fc1, compile_nvfp4_fc1_configs,
    prepare_nvfp4_fc1_metadata, nvfp4_fc1, nvfp4_fc1_out,
)
from .moe_routing import pack_routing_metadata

__all__ = [name for name in globals() if not name.startswith("_")]
