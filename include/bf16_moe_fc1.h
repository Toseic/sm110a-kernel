#pragma once

#include "common.h"

void launch_fc1_block_interleaved_swiglu(
    const __nv_bfloat16* d_a,
    const __nv_bfloat16* d_b_interleaved,
    const int32_t* d_grouped_layout,
    __nv_bfloat16* d_out,
    int m,
    int experts,
    int n,
    int k,
    cudaStream_t stream);
