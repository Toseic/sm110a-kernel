#pragma once

#include "common.h"

// A and B contain two packed E2M1 values per byte. A is [M, K/2] and B is
// [experts, N, K/2]. SFA and SFB use Blackwell SWIZZLE_32_4_4 blocked UE4M3
// layout, independently blocked over A rows and each expert's B rows.
void launch_nvfp4_grouped_m_gemm_main(
    const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
    const uint8_t* sfb, const int32_t* tasks, int32_t* state,
    __nv_bfloat16* c, int m, int experts, int n, int k, int max_tasks,
    cudaStream_t stream);
