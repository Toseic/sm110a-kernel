#include "../../../include/pdl.h"
#include "compressor_store_params.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sm110a::compressor_store {
constexpr int D = 512, NOPE = 448, ROPE = 64, SCALES = 7;
__device__ float warp_sum(
    float v) {
#pragma unroll
    for (int delta = 16; delta; delta /= 2)
        v += __shfl_down_sync(0xffffffff, v, delta);
    return v;
}
// One warp per row, four rows per CTA. Retain the existing D=512 RMSNorm
// vector/reduction order and BF16 rounding before RoPE and quantization.
__global__ void store(
    Params p) {
    sm110a::pdl_entry();
    const int lane = threadIdx.x % 32;
    const int64_t row = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (row >= p.rows)
        return;
    const int64_t sequence = row % p.sequence;
    const int64_t destination = (row / p.sequence) * p.capacity + p.position + sequence;
    const auto *input = static_cast<const __nv_bfloat16 *>(p.input) + row * D;
    float values[2][8];
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int col = (i * 32 + lane) * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            values[i][j] = __bfloat162float(input[col + j]);
            sum = fmaf(values[i][j], values[i][j], sum);
        }
    }
    sum = warp_sum(sum);
    const float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / float(D) + p.eps);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int col = (i * 32 + lane) * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            values[i][j] = __bfloat162float(
                __float2bfloat16_rn(__fmul_rn(__fmul_rn(values[i][j], inv), p.weight[col + j])));
        }
        if (col >= NOPE) {
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                const float2 f = static_cast<const float2 *>(
                    p.frequencies)[sequence * p.frequency_stride + (col - NOPE + j) / 2];
                const float sine = p.conjugate ? -f.y : f.y;
                const float re = values[i][j], im = values[i][j + 1];
                const __nv_bfloat16 a =
                    __float2bfloat16_rn(__fsub_rn(__fmul_rn(re, f.x), __fmul_rn(im, sine)));
                const __nv_bfloat16 b =
                    __float2bfloat16_rn(__fadd_rn(__fmul_rn(re, sine), __fmul_rn(im, f.x)));
                static_cast<__nv_bfloat16 *>(p.cache_rope)[destination * ROPE + col - NOPE + j] = a;
                static_cast<__nv_bfloat16 *>(
                    p.cache_rope)[destination * ROPE + col - NOPE + j + 1] = b;
                if (p.packed_rope) {
                    static_cast<__nv_bfloat16 *>(p.packed_rope)[row * ROPE + col - NOPE + j] = a;
                    static_cast<__nv_bfloat16 *>(p.packed_rope)[row * ROPE + col - NOPE + j + 1] =
                        b;
                }
            }
        }
        // All lanes participate in shuffles, including the final RoPE subgroup.
        float amax = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            amax = fmaxf(amax, fabsf(values[i][j]));
#pragma unroll
        for (int delta = 4; delta; delta /= 2)
            amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, delta, 8));
        float multiplier = 1;
        const int block = i * 4 + lane / 8;
        if (lane % 8 == 0 && block < SCALES) {
#if SM110A_STORE_E8M0
            const uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
            const int exponent = int((bits >> 23) & 0xff) + ((bits & 0x7fffff) != 0);
            const uint8_t byte = min(254, max(0, exponent));
            static_cast<uint8_t *>(p.cache_scales)[destination * SCALES + block] = byte;
            if (p.packed_scales)
                static_cast<uint8_t *>(p.packed_scales)[row * SCALES + block] = byte;
            multiplier = ldexpf(1.0f, 127 - int(byte));
#else
            const float scale = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);
            static_cast<float *>(p.cache_scales)[destination * SCALES + block] = scale;
            if (p.packed_scales)
                static_cast<float *>(p.packed_scales)[row * SCALES + block] = scale;
            multiplier = 1.0f / scale;
#endif
        }
        multiplier = __shfl_sync(0xffffffff, multiplier, 0, 8);
        if (col < NOPE) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
#if SM110A_STORE_E8M0
                const float scaled = __bfloat162float(
                    __hmul(__float2bfloat16_rn(values[i][j]), __float2bfloat16_rn(multiplier)));
#else
                const float scaled = fminf(fmaxf(values[i][j] * multiplier, -448.0f), 448.0f);
#endif
                const uint8_t byte = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
                static_cast<uint8_t *>(p.cache_values)[destination * NOPE + col + j] = byte;
                if (p.packed_values)
                    static_cast<uint8_t *>(p.packed_values)[row * NOPE + col + j] = byte;
            }
        }
    }
}
} // namespace sm110a::compressor_store
extern "C" void sm110a_compressor_store_launch(
    sm110a::compressor_store::Params p,
    cudaStream_t stream) {
    if (!p.rows)
        return;
    sm110a::launch_kernel(sm110a::compressor_store::store, (p.rows + 3) / 4, 128, 0, stream, p);
}
