#pragma once

#include "common.h"

// MXFP4 (W4A8) routed FC1 with fused token permutation and block-interleaved
// SwiGLU. A is the unpermuted FP8 E4M3 token matrix [tokens, K]; B is the
// interleaved packed-E2M1 gate/up weight [experts, 2N, K/2] (physical layout
// [gate:64, up:64] blocks). SFA/SFB are blocked UE8M0 scales: one 512-byte
// SWIZZLE_32_4_4 tile per (128 rows, K=32 step) with the live byte in
// column 0 (the fp8mxfp4 GEMM layout). One scale per contiguous 128 A
// elements is legal; the consumer replicates it across the four K=32
// sub-blocks it covers.
//
// C is the quantized expert-grouped FC2 input [M, N] in FP8 E4M3 and SFC its
// blocked UE8M0 scale tensor: each 128-output-element block shares one scale,
// replicated across the four K=32 scale tiles it covers, so SFC feeds the
// MXFP4 FC2 kernel directly. swiglu_limit clamps gate from above and up from
// both sides before the SwiGLU product (0 disables the clamp).
// source_rows maps each grouped output row back to its source token row.
// The WeightInFc1 template specialization multiplies FP32 routing_weights
// indexed through grouped_assignments before quantization. Bf16Rounding
// rounds gate/up, then the weighted hidden, to match model activation nodes.
void launch_mxfp4_fc1_main(
    const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
    const uint8_t* sfb, const int32_t* source_rows, const int32_t* tasks,
    int32_t* state, uint8_t* c, uint8_t* sfc, int tokens, int m, int experts,
    int n, int k, float swiglu_limit, int max_tasks, int max_active_clusters,
    cudaStream_t stream, const int32_t* grouped_assignments,
    const float* routing_weights);
