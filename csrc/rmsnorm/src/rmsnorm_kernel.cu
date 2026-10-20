#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Standalone defaults also let clangd analyze this file without JIT flags.
#ifndef SM110A_RMSNORM_D
#define SM110A_RMSNORM_D 4096
#endif
#ifndef SM110A_RMSNORM_WARPS
#define SM110A_RMSNORM_WARPS 4
#endif
#ifndef SM110A_RMSNORM_ROWS
#define SM110A_RMSNORM_ROWS 1
#endif

namespace {
__device__ __forceinline__ float to_float(
    float x) {
    return x;
}
__device__ __forceinline__ float to_float(
    __half x) {
    return __half2float(x);
}
__device__ __forceinline__ float to_float(
    __nv_bfloat16 x) {
    return __bfloat162float(x);
}
template <typename T> __device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ float from_float<float>(
    float x) {
    return x;
}
template <>
__device__ __forceinline__ __half from_float<__half>(
    float x) {
    return __float2half_rn(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(
    float x) {
    return __float2bfloat16_rn(x);
}

constexpr int kD = SM110A_RMSNORM_D;
constexpr int kWarps = SM110A_RMSNORM_WARPS;
constexpr int kRows = SM110A_RMSNORM_ROWS;
constexpr int kThreadsPerRow = 32 * kWarps;

__device__ __forceinline__ uint4 load_vector(
    const void *address) {
    uint4 value;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(address)
        : "memory");
    return value;
}

__device__ __forceinline__ float warp_sum(
    float value) {
#pragma unroll
    for (int delta = 16; delta; delta /= 2)
        value += __shfl_down_sync(0xffffffff, value, delta);
    return value;
}

template <typename T, typename W>
__global__ __launch_bounds__(kThreadsPerRow * kRows)
void rmsnorm_kernel(const T* input, const W* weight, T* output,
                    int64_t rows, float eps) {
    constexpr int kVector = 16 / sizeof(T);
    constexpr int kChunks = (kD + kThreadsPerRow * kVector - 1) / (kThreadsPerRow * kVector);
    const int row_in_cta = threadIdx.x / kThreadsPerRow;
    const int tid = threadIdx.x % kThreadsPerRow;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int64_t row = int64_t(blockIdx.x) * kRows + row_in_cta;
    const bool active = row < rows;
    // Fully unrolled and statically indexed: each input value stays in a
    // register through the reduction. No input staging or rereading.
    union Vector {
        uint4 packed;
        T elements[kVector];
    };
    Vector values[kChunks];
    float scalar_values[kChunks][kVector];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
        const int col = (i * kThreadsPerRow + tid) * kVector;
        if constexpr (kD % kVector == 0) {
            if (active && col < kD)
                values[i].packed = load_vector(input + row * kD + col);
#pragma unroll
            for (int j = 0; j < kVector; ++j) {
                float x = active && col < kD ? to_float(values[i].elements[j]) : 0.f;
                sum = fmaf(x, x, sum);
            }
        } else {
#pragma unroll
            for (int j = 0; j < kVector; ++j) {
                float x = active && col + j < kD ? to_float(input[row * kD + col + j]) : 0.f;
                scalar_values[i][j] = x;
                sum = fmaf(x, x, sum);
            }
        }
    }
    sum = warp_sum(sum);
    float inv;
    if constexpr (kWarps == 1) {
        inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / float(kD) + eps);
    } else {
        __shared__ float partials[kRows][kWarps];
        __shared__ float inverse[kRows];
        if (lane == 0)
            partials[row_in_cta][warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = lane < kWarps ? partials[row_in_cta][lane] : 0.f;
            sum = warp_sum(sum);
            if (lane == 0)
                inverse[row_in_cta] = rsqrtf(sum / float(kD) + eps);
        }
        __syncthreads();
        inv = inverse[row_in_cta];
    }
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
        const int col = (i * kThreadsPerRow + tid) * kVector;
        if constexpr (kD % kVector == 0) {
            if (active && col < kD) {
                constexpr int kWeightVectors = kVector * sizeof(W) / sizeof(uint4);
                union {
                    uint4 packed[kWeightVectors];
                    W elements[kVector];
                } w;
                Vector y;
#pragma unroll
                for (int v = 0; v < kWeightVectors; ++v)
                    w.packed[v] = reinterpret_cast<const uint4 *>(weight + col)[v];
                Vector retained;
                retained.packed = values[i].packed;
                // An opaque register identity prevents the compiler from
                // retaining unpacked per-element conversions across reduction.
                asm volatile(""
                    : "+r"(retained.packed.x),
                    "+r"(retained.packed.y),
                    "+r"(retained.packed.z),
                    "+r"(retained.packed.w));
#pragma unroll
                for (int j = 0; j < kVector; ++j)
                    y.elements[j] = from_float<T>(
                        (to_float(retained.elements[j]) * inv) * to_float(w.elements[j]));
                *reinterpret_cast<uint4 *>(output + row * kD + col) = y.packed;
            }
        } else {
#pragma unroll
            for (int j = 0; j < kVector; ++j)
                if (active && col + j < kD)
                    output[row * kD + col + j] =
                        from_float<T>((scalar_values[i][j] * inv) * to_float(weight[col + j]));
        }
    }
}
} // namespace

extern "C" void sm110a_rmsnorm_launch(
    const void *input,
    const void *weight,
    void *output,
    int64_t rows,
    int dtype,
    int fp32_weight,
    float eps,
    cudaStream_t stream) {
    if (!rows)
        return;
    const int blocks = (rows + kRows - 1) / kRows;
#define LAUNCH(T, W)                                                                               \
    rmsnorm_kernel<T, W>                                                                           \
        <<<blocks, kThreadsPerRow * kRows, 0, stream>>>(static_cast<const T *>(input),             \
            static_cast<const W *>(weight),                                                        \
            static_cast<T *>(output),                                                              \
            rows,                                                                                  \
            eps)
    if (dtype == 0) {
        if (fp32_weight) {
            LAUNCH(__nv_bfloat16, float);
        } else {
            LAUNCH(__nv_bfloat16, __nv_bfloat16);
        }
    } else if (dtype == 1) {
        if (fp32_weight) {
            LAUNCH(__half, float);
        } else {
            LAUNCH(__half, __half);
        }
    } else {
        LAUNCH(float, float);
    }
#undef LAUNCH
}
