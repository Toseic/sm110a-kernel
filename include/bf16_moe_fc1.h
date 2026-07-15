#pragma once

#include "common.h"

// FC1 main consumes preparation outputs only. `tasks` is [max_tasks, 5]
// int32 and state is {task_count, reserved}; routing is managed separately.
void launch_fc1_main(const __nv_bfloat16* d_a, const __nv_bfloat16* d_b_interleaved,
                     const int32_t* source_rows, const int32_t* tasks,
                     int32_t* state, __nv_bfloat16* d_out, int m, int experts,
                     int n, int k, int max_tasks, cudaStream_t stream);
