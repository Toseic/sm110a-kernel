#pragma once

#include <cstdint>

#include "common.h"

// FC1 produces rows grouped by expert. When its optional routing metadata is
// supplied, FC2 reuses that grouping instead of reconstructing it from scores.
void launch_fc2_finalize(
    const __nv_bfloat16* d_fc1,
    const __nv_bfloat16* d_b,
    const __nv_bfloat16* d_routing_weights,
    __nv_bfloat16* d_out,
    int tokens,
    int experts,
    int topk,
    int h,
    int i,
    const int32_t* d_routing_metadata,
    const int32_t* d_tasks,
    int32_t* d_task_state,
    int max_tasks,
    cudaStream_t stream);
