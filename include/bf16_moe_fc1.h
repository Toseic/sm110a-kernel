#pragma once

#include "common.h"

void launch_fc1_block_interleaved_swiglu(
    const __nv_bfloat16* d_a,
    const __nv_bfloat16* d_b_interleaved,
    const __nv_bfloat16* d_router_scores,
    __nv_bfloat16* d_out,
    int tokens,
    int experts,
    int topk,
    int n,
    int k,
    cudaStream_t stream);
