#pragma once

#include "common.h"

// Quantize contiguous BF16 rows to packed E2M1 (MXFP4-style, two 4-bit codes
// per byte) with one E8M0 scale per contiguous 32 values. The dequant variant
// instead writes the round-tripped BF16 values.
void launch_bf16tofp4_quant(const __nv_bfloat16* input, uint8_t* output,
                            uint8_t* scales, int tokens, int d,
                            cudaStream_t stream);
void launch_bf16tofp4_quant_dequant(const __nv_bfloat16* input,
                                    __nv_bfloat16* output, uint8_t* scales,
                                    int tokens, int d, cudaStream_t stream);
