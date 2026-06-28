#pragma once

#include "common.h"

void launch_bf16_gemm(const __nv_bfloat16* d_a, const __nv_bfloat16* d_b,
                      __nv_bfloat16* d_c, int m, int n, int k,
                      cudaStream_t stream);
