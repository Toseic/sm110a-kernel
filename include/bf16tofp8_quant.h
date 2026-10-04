#pragma once

#include "common.h"

// Quantize contiguous BF16 rows to E4M3FN with one E8M0 scale per 128 values.
void launch_bf16tofp8_quant(const __nv_bfloat16* input, uint8_t* output,
                            uint8_t* scales, int tokens, int d,
                            cudaStream_t stream);
