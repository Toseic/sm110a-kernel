#pragma once

#include "common.h"

// A and B contain two packed E2M1 values per byte.  A is physically
// [M, K/2], while B is physically [N, K/2] (the transpose of mathematical
// B). SFA and SFB contain E8M0 block-32 scales in the Blackwell 32x4x4
// blocked layout; each K=64 slab has two live bytes and two padding bytes.
void launch_mxfp4_gemm(const uint8_t* a, const uint8_t* b,
                       const uint8_t* sfa, const uint8_t* sfb,
                       __nv_bfloat16* c, int m, int n, int k,
                       cudaStream_t stream);
