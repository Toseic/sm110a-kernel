"""Public API for the SM110A kernel collection."""

from .green_context import (
    GreenContext,
    create_green_context,
    create_green_contexts,
    sm_resource_info,
    split_green_contexts,
)
from .kernels.bf16_fa4_gqa import (
    Bf16Fa4GqaConfig,
    Bf16Fa4GqaTactic,
    bf16_fa4_gqa,
    bf16_fa4_gqa_out,
    compile_bf16_fa4_gqa,
    compile_bf16_fa4_gqa_configs,
    get_bf16_fa4_gqa_module,
)
from .kernels.bf16_fa4_mha import (
    TACTIC_12WARP,
    TACTIC_16WARP,
    Bf16Fa4MhaConfig,
    Bf16Fa4MhaTactic,
    bf16_fa4_mha,
    bf16_fa4_mha_out,
    compile_bf16_fa4_mha,
    compile_bf16_fa4_mha_configs,
    get_bf16_fa4_mha_module,
)
from .kernels.bf16_fa4_mla import (
    Bf16Fa4MlaConfig,
    bf16_fa4_mla,
    bf16_fa4_mla_out,
    compile_bf16_fa4_mla,
    compile_bf16_fa4_mla_configs,
    get_bf16_fa4_mla_module,
)
from .kernels.bf16_fa4_mqa import (
    Bf16Fa4MqaConfig,
    bf16_fa4_mqa,
    bf16_fa4_mqa_out,
    compile_bf16_fa4_mqa,
    compile_bf16_fa4_mqa_configs,
    get_bf16_fa4_mqa_module,
)
from .kernels.bf16_fa4_sparse_mla import (
    bf16_fa4_sparse_mla,
    bf16_fa4_sparse_mla_out,
    bf16_fa4_sparse_mla_prefill,
    bf16_fa4_sparse_mla_prefill_out,
    compile_bf16_fa4_sparse_mla,
    compile_bf16_fa4_sparse_mla_prefill,
)
from .kernels.bf16_gemm import (
    Bf16GemmConfig,
    bf16_gemm,
    bf16_gemm_nt_out,
    bf16_gemm_out,
    compile_bf16_gemm,
    compile_bf16_gemm_configs,
    get_bf16_gemm_module,
)
from .kernels.bf16_gemm_serving import (
    Bf16GemmPersistentServingQueue,
    Bf16GemmServingConfig,
    bf16_gemm_serving,
    bf16_gemm_serving_nt_out,
    bf16_gemm_serving_out,
    compile_bf16_gemm_serving,
    compile_bf16_gemm_serving_configs,
    get_bf16_gemm_serving_module,
)
from .kernels.bf16tofp4_quant import (
    Bf16ToFp4QuantModule,
    bf16tofp4_quant,
    bf16tofp4_quant_dequant,
    bf16tofp4_quant_dequant_out,
    bf16tofp4_quant_out,
    compile_bf16tofp4_quant,
)
from .kernels.bf16tofp8_quant import (
    Bf16ToFp8QuantModule,
    bf16tofp8_quant,
    bf16tofp8_quant_out,
    compile_bf16tofp8_quant,
)
from .kernels.bf16tonvfp4_quant import (
    Bf16ToNvfp4QuantModule,
    bf16tonvfp4_quant,
    bf16tonvfp4_quant_out,
    compile_bf16tonvfp4_quant,
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
    fc1_block_interleaved_swiglu,
    fc1_block_interleaved_swiglu_out,
    get_fc1_block_interleaved_swiglu_module,
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
from .kernels.fp8_act_quant import (
    Fp8ActQuantModule,
    compile_fp8_act_quant,
    fp8_act_quant,
    fp8_act_quant_out,
)
from .kernels.fp8_gemm import (
    Fp8GemmConfig,
    compile_fp8_gemm,
    compile_fp8_gemm_configs,
    dequantize_fp8_rows,
    fp8_gemm,
    fp8_gemm_out,
    fp8_gemm_reference,
    from_blocked_fp8_scales,
    quantize_fp8_gemm,
    quantize_fp8_rows,
    to_blocked_fp8_scales,
)
from .kernels.fp8mxfp4_gemm import (
    Fp8Mxfp4GemmConfig,
    compile_fp8mxfp4_gemm,
    compile_fp8mxfp4_gemm_configs,
    fp8mxfp4_gemm,
    fp8mxfp4_gemm_out,
    fp8mxfp4_gemm_reference,
    from_blocked_fp8mxfp4_scales,
    quantize_fp8mxfp4_gemm,
    quantize_fp8mxfp4_weights,
    to_blocked_fp8mxfp4_scales,
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
from .kernels.indexer_score import (
    compile_indexer_score_decode,
    compile_indexer_score_prefill,
    indexer_score_decode,
    indexer_score_decode_out,
    indexer_score_prefill,
    indexer_score_prefill_out,
    select_indexer_score_decode_split_kv,
)
from .kernels.moe_routing import pack_routing_metadata
from .kernels.mxfp4_fc1 import (
    Mxfp4Fc1Config,
    Mxfp4Fc1Metadata,
    compile_mxfp4_fc1,
    compile_mxfp4_fc1_configs,
    mxfp4_fc1,
    mxfp4_fc1_out,
    prepare_mxfp4_fc1_metadata,
    quantize_mxfp4_fc1_weights,
)
from .kernels.mxfp4_fc2 import (
    Mxfp4Fc2Config,
    Mxfp4Fc2Metadata,
    compile_mxfp4_fc2,
    compile_mxfp4_fc2_configs,
    mxfp4_fc2,
    mxfp4_fc2_out,
    prepare_mxfp4_fc2_metadata,
    to_blocked_mxfp4_sfa,
)
from .kernels.mxfp4_gemm import (
    Mxfp4GemmConfig,
    compile_mxfp4_gemm,
    compile_mxfp4_gemm_configs,
    dequantize_mxfp4_rows,
    from_blocked_mxfp4_scales,
    mxfp4_gemm,
    mxfp4_gemm_out,
    mxfp4_gemm_reference,
    quantize_mxfp4_gemm,
    quantize_mxfp4_rows,
    to_blocked_mxfp4_scales,
)
from .kernels.nccl_lsa import NcclLsaAllReduce, nccl_unique_id
from .kernels.nvfp4_fc1 import (
    Nvfp4Fc1Config,
    Nvfp4Fc1Metadata,
    compile_nvfp4_fc1,
    compile_nvfp4_fc1_configs,
    nvfp4_fc1,
    nvfp4_fc1_out,
    prepare_nvfp4_fc1_metadata,
)
from .kernels.nvfp4_fc1fc2 import (
    Nvfp4Fc1Fc2Config,
    compile_nvfp4_fc1fc2,
    compile_nvfp4_fc1fc2_configs,
    nvfp4_fc1fc2,
    nvfp4_fc1fc2_out,
    nvfp4_fc1fc2_routed_out,
)
from .kernels.nvfp4_fc2 import (
    Nvfp4Fc2Config,
    Nvfp4Fc2Metadata,
    compile_nvfp4_fc2,
    compile_nvfp4_fc2_configs,
    nvfp4_fc2,
    nvfp4_fc2_out,
    prepare_nvfp4_fc2_metadata,
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
from .kernels.nvfp4_grouped_m_gemm import (
    Nvfp4GroupedMConfig,
    compile_nvfp4_grouped_m_gemm,
    compile_nvfp4_grouped_m_gemm_configs,
    nvfp4_grouped_m_gemm,
    nvfp4_grouped_m_gemm_out,
    prepare_nvfp4_grouped_m_metadata,
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
