#pragma once

#include "common.h"

// Fixed-shape grouped FC1 -> SwiGLU/NVFP4 -> FC2 implementation used to bring
// up the two-CTA time-switched pipeline. The initial entry supports one
// expert, H=6144, I=2048, M divisible by 512, and a compile-time configurable
// SMEM streaming buffer. Small-M specializations use a reusable global
// workspace to distribute FC1-I and FC2-H across resident clusters. The
// optional routed variant gathers token-major A/SFA through source_rows and
// reduces routing-weighted FC2 tiles directly into token-major output.
void launch_nvfp4_fc1fc2(
    const uint8_t* a, const uint8_t* w1, const uint8_t* sfa,
    const uint8_t* sfw1, const uint8_t* w2, const uint8_t* sfw2,
    const int32_t* source_rows, const int32_t* grouped_assignments,
    const int32_t* tasks, const int32_t* state,
    const __nv_bfloat16* routing_weights, __nv_bfloat16* output,
    uint8_t* workspace, int tokens, int m, int experts, int topk,
    int row_task_count, int max_tasks, cudaStream_t stream);
