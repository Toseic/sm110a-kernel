#pragma once

#include "common.h"

// A is raw E4M3 [M,K]. B is transposed MXFP4 [N,K] in the byte-padded E2M1
// format required by tcgen05.kind::mxf8f6f4: each 16-K group holds eight
// packed E2M1 bytes followed by eight padding bytes. SFB contains K=32 E8M0 scales in 128x4x4 blocked layout;
// one live byte and three padding bytes form each 128-row scale tile. The
// The public facade supplies E8M0=1 identity scales for the unscaled E4M3 activation.
void launch_fp8mxfp4_gemm(const uint8_t* a, const uint8_t* b,
                       const uint8_t* sfa, const uint8_t* sfb,
                       __nv_bfloat16* c, int m, int n, int k,
                       cudaStream_t stream);
