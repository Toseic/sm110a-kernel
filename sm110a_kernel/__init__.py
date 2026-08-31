"""Public API for the SM110A kernel collection."""

from .profiler import (
    PROFILER_WARPS_PER_BLOCK,
    make_profiler_buffer,
    profiler_events,
    write_trace,
)
from .kernels.moe_routing import pack_routing_metadata
from .kernels.bf16_gemm import (
    Bf16GemmConfig,
    bf16_gemm,
    bf16_gemm_out,
    bf16_gemm_nt_out,
    compile_bf16_gemm,
    compile_bf16_gemm_configs,
    get_bf16_gemm_module,
)
from .kernels.bf16_fa4_mha import (
    bf16_fa4_mha,
    bf16_fa4_mha_out,
    get_bf16_fa4_mha_module,
)
from .kernels.nvfp4_gemm import (
    Nvfp4GemmConfig,
    compile_nvfp4_gemm,
    compile_nvfp4_gemm_configs,
    from_blocked_nvfp4_scales,
    nvfp4_gemm,
    nvfp4_gemm_out,
    quantize_nvfp4_gemm,
    quantize_nvfp4_rows,
    to_blocked_nvfp4_scales,
)
from .kernels.cublaslt import (
    cublaslt_gemm,
    cublaslt_gemm_out,
    cublaslt_gemm_out_batched,
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
from .kernels.fc2_finalize import (
    Fc2FinalizeConfig,
    compile_fc2_finalize,
    compile_fc2_finalize_configs,
    fc2_finalize,
    fc2_finalize_out,
    fc2_finalize_with_routing_out,
    get_fc2_finalize_module,
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
from .kernels.nvfp4_grouped_m_gemm import (
    Nvfp4GroupedMConfig, compile_nvfp4_grouped_m_gemm,
    compile_nvfp4_grouped_m_gemm_configs,
    prepare_nvfp4_grouped_m_metadata, nvfp4_grouped_m_gemm,
    nvfp4_grouped_m_gemm_out,
)
from .kernels.nvfp4_fc1 import (
    Nvfp4Fc1Config, Nvfp4Fc1Metadata, compile_nvfp4_fc1,
    compile_nvfp4_fc1_configs, prepare_nvfp4_fc1_metadata, nvfp4_fc1,
    nvfp4_fc1_out,
)
from .kernels.nvfp4_fc2 import (
    Nvfp4Fc2Config, Nvfp4Fc2Metadata, compile_nvfp4_fc2,
    compile_nvfp4_fc2_configs, prepare_nvfp4_fc2_metadata, nvfp4_fc2,
    nvfp4_fc2_out,
)
_RUNNER_EXPORTS = {"AutotuneRunner", "PreparedKernel", "Selection", "TuneOptions"}
_CUTEDSL_EXPORTS = {
    "CuteDslBf16GemmConfig",
    "CuteDslBf16GemmRunner",
    "cute_dsl_bf16_gemm_out",
    "CuteDslNvfp4Fc1Config",
    "CuteDslNvfp4Fc1Runner",
    "cute_dsl_nvfp4_fc1_out",
    "CuteDslNvfp4Fc2Config",
    "CuteDslNvfp4Fc2Runner",
    "cute_dsl_nvfp4_fc2_out",
    "CuteDslNvfp4GemmConfig",
    "CuteDslNvfp4GemmRunner",
    "cute_dsl_nvfp4_gemm_out",
    "CuteDslNvfp4GroupedMGemmConfig",
    "CuteDslNvfp4GroupedMGemmRunner",
    "cute_dsl_nvfp4_grouped_m_gemm_out",
}


def __getattr__(name):
    if name in _RUNNER_EXPORTS:
        from . import runner
        return getattr(runner, name)
    if name in _CUTEDSL_EXPORTS:
        from .kernels import cutedsl
        return getattr(cutedsl, name)
    raise AttributeError(name)


__all__ = [name for name in globals() if not name.startswith("_")] + sorted(
    _RUNNER_EXPORTS | _CUTEDSL_EXPORTS
)
