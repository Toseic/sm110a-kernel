#pragma once

#include "common.h"

void launch_bf16_grouped_m_gemm(const __nv_bfloat16* d_a,
                                const __nv_bfloat16* d_b,
                                const int32_t* d_grouped_layout,
                                __nv_bfloat16* d_c, int m, int experts,
                                int n, int k, bool swap_ab,
                                cudaStream_t stream);
