#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef SM110A_HC_PRE_D
#define SM110A_HC_PRE_D 4096
#endif
#ifndef SM110A_HC_PRE_ITERS
#define SM110A_HC_PRE_ITERS 20
#endif

namespace {
constexpr int D = SM110A_HC_PRE_D;
constexpr int K = 4 * D;
constexpr int Threads = 256;
constexpr int Warps = Threads / 32;

__device__ __forceinline__ float as_float(
    float x) {
    return x;
}
__device__ __forceinline__ float as_float(
    __half x) {
    return __half2float(x);
}
__device__ __forceinline__ float as_float(
    __nv_bfloat16 x) {
    return __bfloat162float(x);
}
template <typename T> __device__ T from_float(float x);
template <>
__device__ float from_float<float>(
    float x) {
    return x;
}
template <>
__device__ __half from_float<__half>(
    float x) {
    return __float2half_rn(x);
}
template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(
    float x) {
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float warp_sum(
    float x) {
#pragma unroll
    for (int offset = 16; offset; offset /= 2)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return x;
}

// One CTA owns every output of one token. Only the original-dtype input is
// staged; FP32 conversions feed both reductions immediately and then die.
template <typename T>
__global__ __launch_bounds__(Threads)
void hc_pre_kernel(const T* __restrict__ input, const float* __restrict__ weight,
                   const float* __restrict__ scale, const float* __restrict__ base,
                   T* __restrict__ output, float* __restrict__ post,
                   float* __restrict__ comb, float norm_eps, float hc_eps) {
    extern __shared__ __align__(16) unsigned char storage[];
    T *retained = reinterpret_cast<T *>(storage);
    __shared__ float partial[25][Warps];
    __shared__ float pre[4];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int64_t row = blockIdx.x;
    float dots[24] = {};
    float square_sum = 0.f;
    // Keep K loop rolled: unrolling the whole projection bloats the binary.
#pragma unroll 1
    for (int k = tid * 4; k < K; k += Threads * 4) {
        float x[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            T packed = input[row * K + k + i];
            retained[k + i] = packed;
            x[i] = as_float(packed);
            square_sum = fmaf(x[i], x[i], square_sum);
        }
#pragma unroll
        for (int j = 0; j < 24; ++j) {
            float4 w = *reinterpret_cast<const float4 *>(weight + j * K + k);
            dots[j] = fmaf(x[0], w.x, dots[j]);
            dots[j] = fmaf(x[1], w.y, dots[j]);
            dots[j] = fmaf(x[2], w.z, dots[j]);
            dots[j] = fmaf(x[3], w.w, dots[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < 24; ++j) {
        float value = warp_sum(dots[j]);
        if (lane == 0)
            partial[j][warp] = value;
    }
    square_sum = warp_sum(square_sum);
    if (lane == 0)
        partial[24][warp] = square_sum;
    __syncthreads();

    if (warp == 0) {
        // Each lane owns one projection (lane 24 owns the square sum).
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < Warps; ++w)
            sum += partial[lane < 25 ? lane : 24][w];
        float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 24) / float(K) + norm_eps);
        float mix = sum * inv;
        if (lane < 4)
            pre[lane] = 1.f / (1.f + expf(-(__fmul_rn(mix, scale[0]) + base[lane]))) + hc_eps;
        if (lane >= 4 && lane < 8)
            post[row * 4 + lane - 4] = 2.f / (1.f + expf(-(__fmul_rn(mix, scale[1]) + base[lane])));

        // Sixteen lanes hold a 4x4 matrix, row-major. Every lane executes
        // every shuffle, including inactive lanes, with a full warp mask.
        float c = __shfl_sync(0xffffffff, mix, lane < 16 ? lane + 8 : 8);
        c = __fmul_rn(c, scale[2]) + base[8 + (lane % 16)];
        float max_row = fmaxf(c, __shfl_xor_sync(0xffffffff, c, 1));
        max_row = fmaxf(max_row, __shfl_xor_sync(0xffffffff, max_row, 2));
        c = expf(c - max_row);
        float denom = c + __shfl_xor_sync(0xffffffff, c, 1);
        denom += __shfl_xor_sync(0xffffffff, denom, 2);
        c = c / denom + hc_eps;
        denom = c + __shfl_xor_sync(0xffffffff, c, 4);
        denom += __shfl_xor_sync(0xffffffff, denom, 8);
        c /= denom + hc_eps;
#pragma unroll 1
        for (int i = 1; i < SM110A_HC_PRE_ITERS; ++i) {
            denom = c + __shfl_xor_sync(0xffffffff, c, 1);
            denom += __shfl_xor_sync(0xffffffff, denom, 2);
            c /= denom + hc_eps;
            denom = c + __shfl_xor_sync(0xffffffff, c, 4);
            denom += __shfl_xor_sync(0xffffffff, denom, 8);
            c /= denom + hc_eps;
        }
        if (lane < 16)
            comb[row * 16 + lane] = c;
    }
    __syncthreads();
    for (int d = tid; d < D; d += Threads) {
        // Match the reference's separately rounded products before summation.
        float y = __fmul_rn(pre[0], as_float(retained[d]));
#pragma unroll
        for (int h = 1; h < 4; ++h)
            y += __fmul_rn(pre[h], as_float(retained[h * D + d]));
        output[row * D + d] = from_float<T>(y);
    }
}

template <typename T>
cudaError_t launch(
    const void *input,
    const float *weight,
    const float *scale,
    const float *base,
    void *output,
    float *post,
    float *comb,
    int64_t rows,
    float norm_eps,
    float hc_eps,
    cudaStream_t stream) {
    constexpr int bytes = K * sizeof(T);
    if constexpr (bytes > 48 * 1024 - 25 * Warps * sizeof(float) - 16) {
        auto error = cudaFuncSetAttribute(hc_pre_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            bytes);
        if (error != cudaSuccess)
            return error;
    }
    hc_pre_kernel<T>
        <<<static_cast<unsigned int>(rows), Threads, bytes, stream>>>(static_cast<const T *>(input),
            weight,
            scale,
            base,
            static_cast<T *>(output),
            post,
            comb,
            norm_eps,
            hc_eps);
    return cudaGetLastError();
}
} // namespace

extern "C" cudaError_t sm110a_hc_pre_launch(
    const void *input,
    const float *weight,
    const float *scale,
    const float *base,
    void *output,
    float *post,
    float *comb,
    int64_t rows,
    int dtype,
    float norm_eps,
    float hc_eps,
    cudaStream_t stream) {
    if (!rows)
        return cudaSuccess;
    if (dtype == 0)
        return launch<__nv_bfloat16>(input,
            weight,
            scale,
            base,
            output,
            post,
            comb,
            rows,
            norm_eps,
            hc_eps,
            stream);
    if (dtype == 1)
        return launch<
            __half>(input, weight, scale, base, output, post, comb, rows, norm_eps, hc_eps, stream);
    return launch<
        float>(input, weight, scale, base, output, post, comb, rows, norm_eps, hc_eps, stream);
}
