#include <cassert>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>

#include "bf16tofp8_quant.h"

namespace {

// TileLang-compatible work decomposition: 32 rows by one 128-value K block.
// Eight threads cooperate on one row; each contributes a contiguous 16-value
// segment.  Thus every 128-thread CTA owns 32 scales and 4096 FP8 values.
constexpr int kThreads = 128;
constexpr int kRowsPerCta = 32;
constexpr int kThreadsPerRow = 8;
constexpr int kValuesPerThread = 16;
constexpr int kBlockSize = 128;
constexpr float kE4M3Max = 448.0f;

union SharedStorage {
    __nv_bfloat16 input[kRowsPerCta * kBlockSize];
    uint8_t output[kRowsPerCta * kBlockSize];
};

__device__ __forceinline__ float reduce_row_max(
    float value) {
#pragma unroll
    for (int offset = kThreadsPerRow / 2; offset != 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset, kThreadsPerRow));
    }
    return value;
}

__device__ __forceinline__ uint32_t fp8x4(
    float x0,
    float x1,
    float x2,
    float x3,
    float scale) {
    const float4 scaled = make_float4(x0 / scale, x1 / scale, x2 / scale, x3 / scale);
    return __nv_fp8x4_e4m3(scaled).__x;
}

__global__ __launch_bounds__(kThreads)
void bf16tofp8_quant_kernel(const __nv_bfloat16* __restrict__ input,
                            uint8_t* __restrict__ output,
                            uint8_t* __restrict__ scales, int tokens, int d) {
    alignas(16) __shared__ SharedStorage storage;
    const int tid = threadIdx.x;
    const int row_group = tid / kThreadsPerRow; // [0, 15]
    const int row_lane = tid % kThreadsPerRow;  // 16 K-values per lane
    const int row0 = blockIdx.x * kRowsPerCta + row_group;
    const int row1 = row0 + kRowsPerCta / 2;
    const int k_block = blockIdx.y;
    const int k_offset = k_block * kBlockSize + row_lane * kValuesPerThread;
    const bool row0_valid = row0 < tokens;
    const bool row1_valid = row1 < tokens;

    // Four 16-byte vector loads per thread: two 16-BF16 fragments, one for
    // each of its rows.  Invalid tail rows are explicitly zero-filled.
    uint4 input_vectors[4];
    if (row0_valid) {
        const uint4 *src =
            reinterpret_cast<const uint4 *>(input + static_cast<size_t>(row0) * d + k_offset);
        input_vectors[0] = src[0];
        input_vectors[1] = src[1];
    } else {
        input_vectors[0] = make_uint4(0, 0, 0, 0);
        input_vectors[1] = make_uint4(0, 0, 0, 0);
    }
    if (row1_valid) {
        const uint4 *src =
            reinterpret_cast<const uint4 *>(input + static_cast<size_t>(row1) * d + k_offset);
        input_vectors[2] = src[0];
        input_vectors[3] = src[1];
    } else {
        input_vectors[2] = make_uint4(0, 0, 0, 0);
        input_vectors[3] = make_uint4(0, 0, 0, 0);
    }

    auto *input_u4 = reinterpret_cast<uint4 *>(storage.input);
    constexpr int kU4PerRow = kBlockSize * sizeof(__nv_bfloat16) / sizeof(uint4);
    const int row_u4_offset = row_lane * kValuesPerThread * sizeof(__nv_bfloat16) / sizeof(uint4);
    input_u4[row_group * kU4PerRow + row_u4_offset] = input_vectors[0];
    input_u4[row_group * kU4PerRow + row_u4_offset + 1] = input_vectors[1];
    input_u4[(row_group + kRowsPerCta / 2) * kU4PerRow + row_u4_offset] = input_vectors[2];
    input_u4[(row_group + kRowsPerCta / 2) * kU4PerRow + row_u4_offset + 1] = input_vectors[3];
    __syncthreads();

    // Reload the two fragments into registers before reusing the 8KB shared
    // allocation as the 4KB FP8 output staging buffer.
    __nv_bfloat16 values[2][kValuesPerThread];
    const auto *input_fragment0 =
        storage.input + row_group * kBlockSize + row_lane * kValuesPerThread;
    const auto *input_fragment1 = input_fragment0 + (kRowsPerCta / 2) * kBlockSize;
    reinterpret_cast<uint4 *>(values[0])[0] = reinterpret_cast<const uint4 *>(input_fragment0)[0];
    reinterpret_cast<uint4 *>(values[0])[1] = reinterpret_cast<const uint4 *>(input_fragment0)[1];
    reinterpret_cast<uint4 *>(values[1])[0] = reinterpret_cast<const uint4 *>(input_fragment1)[0];
    reinterpret_cast<uint4 *>(values[1])[1] = reinterpret_cast<const uint4 *>(input_fragment1)[1];

    float amax0 = 0.0f;
    float amax1 = 0.0f;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
        amax0 = fmaxf(amax0, fabsf(__bfloat162float(values[0][i])));
        amax1 = fmaxf(amax1, fabsf(__bfloat162float(values[1][i])));
    }
    amax0 = reduce_row_max(amax0);
    amax1 = reduce_row_max(amax1);

    float scale0 = 1.0f;
    float scale1 = 1.0f;
    if (row_lane == 0) {
        const __nv_fp8_e8m0 e8m0_0(amax0 / kE4M3Max);
        const __nv_fp8_e8m0 e8m0_1(amax1 / kE4M3Max);
        if (row0_valid) {
            scales[static_cast<size_t>(row0) * (d / kBlockSize) + k_block] = e8m0_0.__x;
        }
        if (row1_valid) {
            scales[static_cast<size_t>(row1) * (d / kBlockSize) + k_block] = e8m0_1.__x;
        }
        scale0 = ldexpf(1.0f, static_cast<int>(e8m0_0.__x) - 127);
        scale1 = ldexpf(1.0f, static_cast<int>(e8m0_1.__x) - 127);
    }
    scale0 = __shfl_sync(0xffffffff, scale0, 0, kThreadsPerRow);
    scale1 = __shfl_sync(0xffffffff, scale1, 0, kThreadsPerRow);

    // The FP8 staging layout is row-major, so each 8-thread row group writes
    // and then globally stores a contiguous 128-byte row tile in 16-byte chunks.
    auto *output_u4 = reinterpret_cast<uint4 *>(storage.output);
    constexpr int kOutputU4PerRow = kBlockSize / sizeof(uint4);
    const int output_u4_offset = row_lane;
    uint4 packed0, packed1;
    packed0.x = fp8x4(__bfloat162float(values[0][0]),
        __bfloat162float(values[0][1]),
        __bfloat162float(values[0][2]),
        __bfloat162float(values[0][3]),
        scale0);
    packed0.y = fp8x4(__bfloat162float(values[0][4]),
        __bfloat162float(values[0][5]),
        __bfloat162float(values[0][6]),
        __bfloat162float(values[0][7]),
        scale0);
    packed0.z = fp8x4(__bfloat162float(values[0][8]),
        __bfloat162float(values[0][9]),
        __bfloat162float(values[0][10]),
        __bfloat162float(values[0][11]),
        scale0);
    packed0.w = fp8x4(__bfloat162float(values[0][12]),
        __bfloat162float(values[0][13]),
        __bfloat162float(values[0][14]),
        __bfloat162float(values[0][15]),
        scale0);
    packed1.x = fp8x4(__bfloat162float(values[1][0]),
        __bfloat162float(values[1][1]),
        __bfloat162float(values[1][2]),
        __bfloat162float(values[1][3]),
        scale1);
    packed1.y = fp8x4(__bfloat162float(values[1][4]),
        __bfloat162float(values[1][5]),
        __bfloat162float(values[1][6]),
        __bfloat162float(values[1][7]),
        scale1);
    packed1.z = fp8x4(__bfloat162float(values[1][8]),
        __bfloat162float(values[1][9]),
        __bfloat162float(values[1][10]),
        __bfloat162float(values[1][11]),
        scale1);
    packed1.w = fp8x4(__bfloat162float(values[1][12]),
        __bfloat162float(values[1][13]),
        __bfloat162float(values[1][14]),
        __bfloat162float(values[1][15]),
        scale1);
    output_u4[row_group * kOutputU4PerRow + output_u4_offset] = packed0;
    output_u4[(row_group + kRowsPerCta / 2) * kOutputU4PerRow + output_u4_offset] = packed1;
    __syncthreads();

    if (row0_valid) {
        uint4 *dst = reinterpret_cast<uint4 *>(output + static_cast<size_t>(row0) * d +
                                               k_block * kBlockSize + row_lane * kValuesPerThread);
        dst[0] = output_u4[row_group * kOutputU4PerRow + output_u4_offset];
    }
    if (row1_valid) {
        uint4 *dst = reinterpret_cast<uint4 *>(output + static_cast<size_t>(row1) * d +
                                               k_block * kBlockSize + row_lane * kValuesPerThread);
        dst[0] = output_u4[(row_group + kRowsPerCta / 2) * kOutputU4PerRow + output_u4_offset];
    }
}

} // namespace

void launch_bf16tofp8_quant(
    const __nv_bfloat16 *input,
    uint8_t *output,
    uint8_t *scales,
    int tokens,
    int d,
    cudaStream_t stream) {
    const dim3 grid((tokens + kRowsPerCta - 1) / kRowsPerCta, d / kBlockSize);
    bf16tofp8_quant_kernel<<<grid, kThreads, 0, stream>>>(input, output, scales, tokens, d);
}

extern "C" void sm110a_bf16tofp8_quant_launch(
    const __nv_bfloat16 *input,
    uint8_t *output,
    uint8_t *scales,
    int tokens,
    int d,
    cudaStream_t stream) {
    launch_bf16tofp8_quant(input, output, scales, tokens, d, stream);
}
