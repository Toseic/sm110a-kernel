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
    compile_bf16_gemm,
    get_bf16_gemm_module,
)
from .kernels.nvfp4_gemm import (
    Nvfp4GemmConfig,
    compile_nvfp4_gemm,
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
_RUNNER_EXPORTS = {"AutotuneRunner", "PreparedKernel", "Selection", "TuneOptions"}


def __getattr__(name):
    if name in _RUNNER_EXPORTS:
        from . import runner
        return getattr(runner, name)
    raise AttributeError(name)


__all__ = [name for name in globals() if not name.startswith("_")] + sorted(_RUNNER_EXPORTS)
