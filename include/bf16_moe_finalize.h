#pragma once

#include "common.h"

// Final reduction is intentionally separate from routed/grouped GEMM setup so
// benchmarks can time and tune it independently.
void launch_moe_finalize(const __nv_bfloat16* grouped_fc2,
                         const int32_t* assignment_to_grouped,
                         const __nv_bfloat16* assignment_weights,
                         __nv_bfloat16* out,
                         int tokens, int topk, int h,
                         cudaStream_t stream);
