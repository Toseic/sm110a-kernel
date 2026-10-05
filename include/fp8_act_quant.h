#pragma once

#include "common.h"

// Generalized block-scaled FP8 activation quantization of contiguous BF16
// rows to E4M3FN.  ``block_size`` selects one scale per contiguous block and
// must be 64 or 128.  ``scale_mode`` selects the stored scale format:
//   0 (kScaleModeUe8m0): power-of-two ceil rounding, one uint8 E8M0 byte.
//   1 (kScaleModeFp32):  plain amax * (1/448) after a 1e-4 amax floor, one
//                        IEEE float.
// Pass nullptr for the scale pointer that is not in use.
void launch_fp8_act_quant(const __nv_bfloat16* input, uint8_t* output,
                          uint8_t* ue8m0_scales, float* fp32_scales,
                          int tokens, int d, int block_size, int scale_mode,
                          cudaStream_t stream);
