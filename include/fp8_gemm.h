#pragma once

#include "common.h"

// A is E4M3 [M,K] and B is the transposed E4M3 weight [N,K]. SFA and SFB hold
// one UE8M0 scale per 32 contiguous K values in the 128x4x4 blocked layout
// consumed by tcgen05.mma.kind::mxf8f6f4.block_scale.scale_vec::1X.
void launch_fp8_gemm(const uint8_t* a, const uint8_t* b,
                     const uint8_t* sfa, const uint8_t* sfb,
                     __nv_bfloat16* c, int m, int n, int k,
                     cudaStream_t stream);
