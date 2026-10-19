#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// One warp expands compact E8M0 bytes in place to the MMA scale layout.
// Keep all source bytes in registers before overwriting the shared tile.
template <int Steps>
__device__ __forceinline__ void unpack_mxfp4_weight_scales(uint8_t *scales) {
    const int lane = threadIdx.x & 31;
    uint32_t packed[Steps];
#pragma unroll
    for (int step = 0; step < Steps; ++step)
        packed[step] = reinterpret_cast<const uint32_t *>(scales)[step * 32 + lane];
    __syncwarp();
#pragma unroll
    for (int step = 0; step < Steps; ++step) {
        const uint32_t raw = packed[step];
        reinterpret_cast<uint4 *>(scales)[step * 32 + lane] =
            uint4{raw & 255U, (raw >> 8) & 255U, (raw >> 16) & 255U, raw >> 24};
    }
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
