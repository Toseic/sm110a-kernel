#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#ifndef SM110A_QUANT_BLOCKED
#define SM110A_QUANT_BLOCKED 0
#endif
#ifndef SM110A_QUANT_PDL
#define SM110A_QUANT_PDL 0
#endif

__device__ __forceinline__ void store_blocked_quant_scale(
    uint8_t* scales, int row, int block, int blocks, uint8_t value) {
    const size_t base = (static_cast<size_t>(row / 128) * (blocks * 4) + block * 4) * 512
                      + (row % 32) * 16 + ((row % 128) / 32) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        *reinterpret_cast<uint32_t*>(scales + base + i * 512) = value;
}
