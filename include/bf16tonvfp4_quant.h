#pragma once

#include "common.h"

// Quantize contiguous BF16 rows to packed E2M1 (NVFP4) plus blocked UE4M3
// block-16 scales, bit-compatible with quantize_nvfp4_rows and directly
// consumable by the NVFP4 GEMM kernels.  ``tokens`` must be a multiple of 128
// and ``d`` a multiple of 64 (the scale blocks are written directly in the
// Blackwell SWIZZLE_32_4_4 blocked layout).
void launch_bf16tonvfp4_quant(const __nv_bfloat16* input, uint8_t* packed,
                              uint8_t* scales, int tokens, int d,
                              cudaStream_t stream);
