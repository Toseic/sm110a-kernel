"""Kernel-owned Python entry points.

Each module in this package owns the configuration, JIT recipe and public
facade for one CUDA kernel family.  The package root only re-exports these
entry points for backwards compatibility.
"""

from .bf16_fa4_gqa import (
    Bf16Fa4GqaConfig,
    Bf16Fa4GqaRunner,
    Bf16Fa4GqaTactic,
    bf16_fa4_gqa,
    bf16_fa4_gqa_out,
    compile_bf16_fa4_gqa,
    compile_bf16_fa4_gqa_configs,
    get_bf16_fa4_gqa_module,
)
from .bf16_fa4_mha import (
    Bf16Fa4MhaRunner,
    bf16_fa4_mha,
    bf16_fa4_mha_out,
    get_bf16_fa4_mha_module,
)
from .bf16_fa4_mla import (
    Bf16Fa4MlaConfig,
    Bf16Fa4MlaRunner,
    bf16_fa4_mla,
    bf16_fa4_mla_out,
    compile_bf16_fa4_mla,
    compile_bf16_fa4_mla_configs,
    get_bf16_fa4_mla_module,
)
from .bf16_fa4_mqa import (
    Bf16Fa4MqaConfig,
    Bf16Fa4MqaRunner,
    bf16_fa4_mqa,
    bf16_fa4_mqa_out,
    compile_bf16_fa4_mqa,
    compile_bf16_fa4_mqa_configs,
    get_bf16_fa4_mqa_module,
)
from .bf16_fa4_sparse_mla import (
    Bf16Fa4SparseMlaPrefillRunner,
    Bf16Fa4SparseMlaRunner,
    bf16_fa4_sparse_mla,
    bf16_fa4_sparse_mla_out,
    bf16_fa4_sparse_mla_prefill,
    bf16_fa4_sparse_mla_prefill_out,
    compile_bf16_fa4_sparse_mla,
    compile_bf16_fa4_sparse_mla_prefill,
)
from .bf16_gemm import (
    Bf16GemmConfig,
    Bf16GemmRunner,
    bf16_gemm,
    bf16_gemm_nt_out,
    bf16_gemm_out,
    compile_bf16_gemm,
    compile_bf16_gemm_configs,
    get_bf16_gemm_module,
)
from .bf16_gemm_serving import (
    Bf16GemmPersistentServingQueue,
    Bf16GemmServingConfig,
    Bf16GemmServingRunner,
    bf16_gemm_serving,
    bf16_gemm_serving_nt_out,
    bf16_gemm_serving_out,
    compile_bf16_gemm_serving,
    compile_bf16_gemm_serving_configs,
    get_bf16_gemm_serving_module,
)
from .bf16tofp4_quant import (
    Bf16ToFp4QuantModule,
    Bf16ToFp4QuantRunner,
    bf16tofp4_quant,
    bf16tofp4_quant_dequant,
    bf16tofp4_quant_dequant_out,
    bf16tofp4_quant_out,
    compile_bf16tofp4_quant,
)
from .bf16tofp8_quant import (
    Bf16ToFp8QuantModule,
    Bf16ToFp8QuantRunner,
    bf16tofp8_quant,
    bf16tofp8_quant_out,
    compile_bf16tofp8_quant,
)
from .bf16tonvfp4_quant import (
    Bf16ToNvfp4QuantModule,
    Bf16ToNvfp4QuantRunner,
    bf16tonvfp4_quant,
    bf16tonvfp4_quant_out,
    compile_bf16tonvfp4_quant,
)
from .fc1_block_interleaved_swiglu import (
    Fc1BlockInterleavedSwiGLUConfig,
    Fc1BlockInterleavedSwiGLURunner,
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
    Fc2FinalizeRunner,
    compile_fc2_finalize,
    compile_fc2_finalize_configs,
    fc2_finalize,
    fc2_finalize_out,
    fc2_finalize_with_routing_out,
    get_fc2_finalize_module,
)
from .fp8_act_quant import (
    Fp8ActQuantModule,
    Fp8ActQuantRunner,
    compile_fp8_act_quant,
    fp8_act_quant,
    fp8_act_quant_out,
)
from .fp8_gemm import (
    Fp8GemmRunner,
)
from .fp8_kv_fa4_sparse_mla import (
    Fp8KvFa4SparseMlaRunner,
    compile_fp8_kv_fa4_sparse_mla,
    fp8_kv_fa4_sparse_mla,
    fp8_kv_fa4_sparse_mla_out,
)
from .fp8_shared_fc1 import (
    Fp8SharedFc1Config,
    Fp8SharedFc1Runner,
    compile_fp8_shared_fc1,
    fp8_shared_fc1,
    fp8_shared_fc1_out,
    pack_fp8_shared_fc1_weights,
)
from .fp8mxfp4_gemm import (
    Fp8Mxfp4GemmConfig,
    Fp8Mxfp4GemmRunner,
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
from .fused_gate import FusedGateConfig, compile_fused_gate, fused_gate
from .grouped_m_gemm import (
    Bf16GroupedMGemmRunner,
    GroupedMConfig,
    bf16_grouped_m_gemm,
    bf16_grouped_m_gemm_out,
    compile_bf16_grouped_m_gemm,
    compile_bf16_grouped_m_gemm_configs,
    get_bf16_grouped_m_gemm_module,
)
from .hc_post import HcPostConfig, HcPostRunner, compile_hc_post, hc_post, hc_post_out
from .hc_post_pre_norm import (
    HcPostPreNormConfig,
    HcPostPreNormRunner,
    compile_hc_post_pre_norm,
    hc_post_pre_norm,
    hc_post_pre_norm_out,
)
from .hc_pre import HcPreConfig, HcPreRunner, compile_hc_pre, hc_pre, hc_pre_out
from .hc_pre_norm import (
    HcPreNormConfig,
    HcPreNormRunner,
    compile_hc_pre_norm,
    hc_pre_norm,
    hc_pre_norm_out,
)
from .moe_routing import pack_routing_metadata
from .mxfp4_fc1 import (
    Mxfp4Fc1Config,
    Mxfp4Fc1Metadata,
    Mxfp4Fc1Runner,
    compile_mxfp4_fc1,
    compile_mxfp4_fc1_configs,
    mxfp4_fc1,
    mxfp4_fc1_out,
    prepare_mxfp4_fc1_metadata,
    quantize_mxfp4_fc1_weights,
)
from .mxfp4_fc1fc2 import (
    Mxfp4Fc1Fc2Runner,
    Mxfp4MoeRunner,
    Mxfp4StandaloneConfig,
    Mxfp4StandaloneRunner,
)
from .mxfp4_fc2 import (
    Mxfp4Fc2Config,
    Mxfp4Fc2Metadata,
    Mxfp4Fc2Runner,
    compile_mxfp4_fc2,
    compile_mxfp4_fc2_configs,
    mxfp4_fc2,
    mxfp4_fc2_out,
    prepare_mxfp4_fc2_metadata,
    to_blocked_mxfp4_sfa,
)
from .mxfp4_gemm import (
    Mxfp4GemmConfig,
    Mxfp4GemmRunner,
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
from .mxfp4_indexer_score import IndexerScoreDecodeRunner, IndexerScorePrefillRunner
from .nccl_lsa import NcclLsaAllReduce, NcclLsaRunner, nccl_unique_id
from .nvfp4_fc1 import (
    Nvfp4Fc1Config,
    Nvfp4Fc1Metadata,
    Nvfp4Fc1Runner,
    compile_nvfp4_fc1,
    compile_nvfp4_fc1_configs,
    nvfp4_fc1,
    nvfp4_fc1_out,
    prepare_nvfp4_fc1_metadata,
)
from .nvfp4_fc1fc2 import (
    Nvfp4Fc1Fc2Config,
    Nvfp4Fc1Fc2Runner,
    compile_nvfp4_fc1fc2,
    compile_nvfp4_fc1fc2_configs,
    nvfp4_fc1fc2,
    nvfp4_fc1fc2_out,
)
from .nvfp4_fc2 import Nvfp4Fc2Runner
from .nvfp4_gemm import (
    Nvfp4GemmConfig,
    Nvfp4GemmRunner,
    compile_nvfp4_gemm,
    compile_nvfp4_gemm_configs,
    nvfp4_gemm,
    nvfp4_gemm_out,
)
from .nvfp4_grouped_m_gemm import (
    Nvfp4GroupedMConfig,
    Nvfp4GroupedMGemmRunner,
    compile_nvfp4_grouped_m_gemm,
    compile_nvfp4_grouped_m_gemm_configs,
    nvfp4_grouped_m_gemm,
    nvfp4_grouped_m_gemm_out,
    prepare_nvfp4_grouped_m_metadata,
)
from .rmsnorm import (
    RmsNormConfig,
    RmsNormRunner,
    compile_rmsnorm,
    rmsnorm,
    rmsnorm_out,
)
from .topk import TopKRunner

__all__ = [name for name in globals() if not name.startswith("_")]
