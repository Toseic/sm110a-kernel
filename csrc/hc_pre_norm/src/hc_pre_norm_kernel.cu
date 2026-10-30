#include "hc_pre_norm_impl.cuh"

namespace sm110a::hc_pre_norm {
// One CTA owns every output of one token. Only the original-dtype input is
// staged; FP32 conversions feed both reductions immediately and then die.
template <typename T>
__global__ __launch_bounds__(Threads)
void hc_pre_norm_kernel(const T* __restrict__ input, const float* __restrict__ weight,
                   const float* __restrict__ scale, const float* __restrict__ base,
                   T* __restrict__ output, float* __restrict__ post,
                   float* __restrict__ comb, const float* norm_weight, float rms_eps, float norm_eps, float hc_eps) {
    sm110a::pdl_entry();
    extern __shared__ __align__(16) unsigned char storage[];
    T *retained = reinterpret_cast<T *>(storage);
    __shared__ float partial[Projections + 1][Warps];
    __shared__ float pre[4];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int64_t row = blockIdx.x;
    float dots[Projections] = {};
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
        for (int j = 0; j < Projections; ++j) {
            float4 w = *reinterpret_cast<const float4 *>(weight + j * K + k);
            dots[j] = fmaf(x[0], w.x, dots[j]);
            dots[j] = fmaf(x[1], w.y, dots[j]);
            dots[j] = fmaf(x[2], w.z, dots[j]);
            dots[j] = fmaf(x[3], w.w, dots[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < Projections; ++j) {
        float value = warp_sum(dots[j]);
        if (lane == 0)
            partial[j][warp] = value;
    }
    square_sum = warp_sum(square_sum);
    if (lane == 0)
        partial[Projections][warp] = square_sum;
    __syncthreads();

    if (warp == 0) {
        // Each lane owns one projection; the next lane owns the square sum.
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < Warps; ++w)
            sum += partial[lane < Projections + 1 ? lane : Projections][w];
        float inv = rsqrtf(__shfl_sync(0xffffffff, sum, Projections) / float(K) + norm_eps);
        float mix = sum * inv;
        if (lane < 4)
            pre[lane] = 1.f / (1.f + expf(-(__fmul_rn(mix, scale[0]) + base[lane]))) + hc_eps;
        if constexpr (!Head) {
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
            for (int i = 1; i < SM110A_HC_PRE_NORM_ITERS; ++i) {
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
    }
    __syncthreads();
    collapse_norm(retained, pre, norm_weight, output, row, rms_eps);
}

template <int ChunkK>
cudaError_t launch_tc(const void* input, const float* weight, const float* scale,
                      const float* base, void* output, float* post, float* comb,
                      float* scratch, int64_t rows, const float* norm_weight,
                      float rms_eps, float norm_eps, float hc_eps, cudaStream_t stream) {
    constexpr int Splits = (K + ChunkK - 1) / ChunkK;
    constexpr int bytes_tc = 2 * (TileM + 3 * 24) * TileK * 2;
    if constexpr (bytes_tc > 48 * 1024) {
        auto err = cudaFuncSetAttribute(hc_pre_norm_projection_tc<ChunkK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            bytes_tc);
        if (err != cudaSuccess)
            return err;
    }
    sm110a::launch_kernel(hc_pre_norm_projection_tc<ChunkK>, dim3((rows + TileM - 1) / TileM, Splits), 544, bytes_tc, stream,
        static_cast<const __nv_bfloat16 *>(input),
        weight,
        scratch,
        rows);
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;
    sm110a::launch_kernel(hc_pre_norm_finish_tc<ChunkK>, rows, Threads, K * 2, stream,
        static_cast<const __nv_bfloat16 *>(input),
        scale,
        base,
        static_cast<__nv_bfloat16 *>(output),
        post,
        comb,
        scratch,
        norm_weight, rms_eps, norm_eps,
        hc_eps);
    return cudaGetLastError();
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
    const float* norm_weight, float rms_eps, float norm_eps,
    float hc_eps,
    cudaStream_t stream) {
    constexpr int bytes = K * sizeof(T);
    if constexpr (bytes > 48 * 1024 - 25 * Warps * sizeof(float) - 16) {
        auto error = cudaFuncSetAttribute(hc_pre_norm_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            bytes);
        if (error != cudaSuccess)
            return error;
    }
    sm110a::launch_kernel(hc_pre_norm_kernel<T>, static_cast<unsigned int>(rows), Threads, bytes, stream, static_cast<const T *>(input),
            weight,
            scale,
            base,
            static_cast<T *>(output),
            post,
            comb,
            norm_weight, rms_eps, norm_eps,
            hc_eps);
    return cudaGetLastError();
}
} // namespace sm110a::hc_pre_norm

using namespace sm110a::hc_pre_norm;

extern "C" cudaError_t sm110a_hc_pre_norm_launch(
    const void *input,
    const float *weight,
    const float *scale,
    const float *base,
    void *output,
    float *post,
    float *comb,
    float *scratch,
    int64_t rows,
    int dtype,
    const float* norm_weight, float rms_eps, float norm_eps,
    float hc_eps,
    cudaStream_t stream) {
    if (!rows)
        return cudaSuccess;
    if (!Head && dtype == 0 && D >= 256 && D % 2 == 0 && (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(output) % 16 == 0)) {
        // Larger chunks amortize projection setup and halve scratch traffic.
        // Small token counts retain more split-K parallelism. Scratch remains
        // sized for the 512-element variant so either launch can reuse it.
        if constexpr (D == 4096) {
            if (rows >= 256)
                return launch_tc<1024>(input, weight, scale, base, output, post, comb,
                    scratch, rows, norm_weight, rms_eps, norm_eps, hc_eps, stream);
        }
        return launch_tc<512>(input, weight, scale, base, output, post, comb,
            scratch, rows, norm_weight, rms_eps, norm_eps, hc_eps, stream);
    }
    if (dtype == 0)
        return launch<__nv_bfloat16>(input,
            weight,
            scale,
            base,
            output,
            post,
            comb,
            rows,
            norm_weight, rms_eps, norm_eps,
            hc_eps,
            stream);
    if (dtype == 1)
        return launch<
            __half>(input, weight, scale, base, output, post, comb, rows, norm_weight, rms_eps, norm_eps, hc_eps, stream);
    return launch<
        float>(input, weight, scale, base, output, post, comb, rows, norm_weight, rms_eps, norm_eps, hc_eps, stream);
}
