#pragma once

#include "common.h"

// A is the expert-grouped FC1 activation in packed NVFP4 form [M, K/2],
// B contains one NVFP4 matrix per expert [experts, N, K/2], and C is the
// token-major, routing-weighted FC2 result [tokens, N]. The token-major
// reduction is fused into the grouped-GEMM epilogue, so no workspace is
// required and C is accumulated in place via cp.reduce.async.bulk.
//
// grouped_assignments maps each grouped output row to its expanded assignment
// id (assignment = token * topk + rank); the epilogue uses it to recover both
// the output row and the router scale.
void launch_nvfp4_fc2_main(
    const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
    const uint8_t* sfb, const int32_t* grouped_assignments,
    const int32_t* tasks, int32_t* state, __nv_bfloat16* c,
    const __nv_bfloat16* routing_weights, int tokens,
    int m, int experts, int n, int k, int topk, int max_tasks, int max_active_clusters,
    cudaStream_t stream);
