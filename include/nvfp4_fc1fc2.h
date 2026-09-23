#pragma once

#include "common.h"

// Fixed-shape grouped FC1 -> SwiGLU/NVFP4 -> FC2 implementation used to bring
// up the two-CTA time-switched pipeline. The initial entry supports one
// expert, H=6144, I=2048, M divisible by 512, and a complete-I SMEM buffer.
void launch_nvfp4_fc1fc2(
    const uint8_t* a, const uint8_t* w1, const uint8_t* sfa,
    const uint8_t* sfw1, const uint8_t* w2, const uint8_t* sfw2,
    __nv_bfloat16* output, int m, cudaStream_t stream);
