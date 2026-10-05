#pragma once

#include "common.h"

// A is the expert-grouped FC1 activation quantized to FP8 E4M3 [M, K] with
// blocked UE8M0 scales (one 512-byte SWIZZLE_32_4_4 tile per 128 physical
// rows and K=32 step; the live scale byte sits in column 0, matching the
// fp8mxfp4 GEMM layout). B contains one MXFP4 matrix per expert
// [experts, N, K/2] (packed E2M1, low nibble = even element) with blocked
// UE8M0 K=32 scales in the same 512-byte tile format. C is the token-major,
// routing-weighted FC2 result [tokens, N] in BF16. The token-major reduction
// is fused into the grouped-GEMM epilogue, so no workspace is required and C
// is accumulated in place via cp.reduce.async.bulk.
//
// grouped_assignments maps each grouped output row to its expanded assignment
// id (assignment = token * topk + rank); the epilogue uses it to recover both
// the output row and the router scale.
void launch_mxfp4_fc2_main(
    const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
    const uint8_t* sfb, const int32_t* grouped_assignments,
    const int32_t* tasks, int32_t* state, __nv_bfloat16* c,
    const __nv_bfloat16* routing_weights, int tokens,
    int m, int experts, int n, int k, int topk, int max_tasks, int max_active_clusters,
    cudaStream_t stream);
