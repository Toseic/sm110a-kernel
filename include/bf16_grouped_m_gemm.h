#pragma once

#include "common.h"

// Main execution path. `tasks` is an [max_tasks, 5] int32 queue prepared by
// the metadata module; state is {task_count, next_cluster_tile}.
void launch_bf16_grouped_m_gemm_main(const __nv_bfloat16* d_a,
                                     const __nv_bfloat16* d_b,
                                     const int32_t* tasks, int32_t* state,
                                     __nv_bfloat16* d_c, int m, int experts,
                                     int n, int k, int max_tasks, bool swap_ab,
                                     cudaStream_t stream);
