#pragma once

#include "common.h"

void launch_bf16_gemm(const __nv_bfloat16* d_a, const __nv_bfloat16* d_b,
                      __nv_bfloat16* d_c, int m, int n, int k,
                      bool swap_ab, cudaStream_t stream,
                      int max_active_clusters = 0);

void launch_bf16_gemm_profile(const __nv_bfloat16* d_a, const __nv_bfloat16* d_b,
                              __nv_bfloat16* d_c, int m, int n, int k,
                              int64_t* profiler, int num_entries,
                              bool swap_ab, cudaStream_t stream,
                              int max_active_clusters = 0);
