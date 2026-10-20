#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

#ifndef SM110A_HC_POST_D
#define SM110A_HC_POST_D 4096
#endif
#ifndef SM110A_HC_POST_THREADS
#define SM110A_HC_POST_THREADS 128
#endif

#ifndef SM110A_HC_POST_VECTOR_BYTES
#define SM110A_HC_POST_VECTOR_BYTES 16
#endif

namespace {
constexpr int kBytes = SM110A_HC_POST_VECTOR_BYTES;
using Packed = std::conditional_t<kBytes == 16, uint4, uint2>;
constexpr int kD = SM110A_HC_POST_D;
constexpr int kThreads = SM110A_HC_POST_THREADS;

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
template <typename T> __device__ __forceinline__ T cast(float x);
template <>
__device__ __forceinline__ float cast<float>(
    float x) {
    return x;
}
template <>
__device__ __forceinline__ __half cast<__half>(
    float x) {
    return __float2half_rn(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 cast<__nv_bfloat16>(
    float x) {
    return __float2bfloat16_rn(x);
}

template <int Bytes = kBytes, typename P = Packed>
__device__ __forceinline__ P load_vector(
    const void *ptr) {
    P v;
    if constexpr (Bytes == 16)
        asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
            : "l"(ptr)
            : "memory");
    else
        asm volatile("ld.global.cs.v2.u32 {%0,%1}, [%2];"
            : "=r"(v.x), "=r"(v.y)
            : "l"(ptr)
            : "memory");
    return v;
}
template <int Bytes = kBytes, typename P = Packed>
__device__ __forceinline__ void store_vector(
    void *ptr,
    P v) {
    if constexpr (Bytes == 16)
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr),
            "r"(v.x),
            "r"(v.y),
            "r"(v.z),
            "r"(v.w)
            : "memory");
    else
        asm volatile("st.global.cs.v2.u32 [%0], {%1,%2};" ::"l"(ptr), "r"(v.x), "r"(v.y)
            : "memory");
}

// A thread owns a vector of D columns and all four output streams. Each
// residual/input vector is loaded once and reused for every output stream.
template <typename T, bool Aligned>
__global__ __launch_bounds__(kThreads)
void hc_post_kernel(const T* __restrict__ x, const T* __restrict__ residual,
                    const float* __restrict__ post, const float* __restrict__ comb,
                    T* __restrict__ output) {
    constexpr int V = Aligned ? kBytes / sizeof(T) : 1;
    constexpr int Tiles = (kD + kThreads * V - 1) / (kThreads * V);
    const int64_t row = blockIdx.x / Tiles;
    const int col = (blockIdx.x % Tiles * kThreads + threadIdx.x) * V;
    if (col >= kD)
        return;
    union Vector {
        Packed packed;
        T values[V];
    } xv, rv[4], yv[4];
    if constexpr (Aligned) {
        xv.packed = load_vector(x + row * kD + col);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            rv[i].packed = load_vector(residual + (row * 4 + i) * kD + col);
    } else {
        xv.values[0] = x[row * kD + col];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            rv[i].values[0] = residual[(row * 4 + i) * kD + col];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float p = post[row * 4 + j];
        float c0 = comb[row * 16 + j];
        float c1 = comb[row * 16 + 4 + j];
        float c2 = comb[row * 16 + 8 + j];
        float c3 = comb[row * 16 + 12 + j];
#pragma unroll
        for (int v = 0; v < V; ++v) {
            // Match the reference's FP32 multiply/reduce/add rounding boundaries.
            // Explicit intrinsics prevent contraction into FMAs.
            float s = __fmul_rn(c0, as_float(rv[0].values[v]));
            s = __fadd_rn(s, __fmul_rn(c1, as_float(rv[1].values[v])));
            s = __fadd_rn(s, __fmul_rn(c2, as_float(rv[2].values[v])));
            s = __fadd_rn(s, __fmul_rn(c3, as_float(rv[3].values[v])));
            yv[j].values[v] = cast<T>(__fadd_rn(__fmul_rn(p, as_float(xv.values[v])), s));
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if constexpr (Aligned)
            store_vector(output + (row * 4 + j) * kD + col, yv[j].packed);
        else
            output[(row * 4 + j) * kD + col] = yv[j].values[0];
    }
}
} // namespace

extern "C" void sm110a_hc_post_launch(
    const void *x,
    const void *residual,
    const float *post,
    const float *comb,
    void *output,
    int64_t rows,
    int dtype,
    cudaStream_t stream) {
    if (!rows)
        return;
    const bool aligned = ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(residual) |
                              reinterpret_cast<uintptr_t>(output)) &
                             (kBytes - 1)) == 0;
#define LAUNCH(T)                                                                                  \
    do {                                                                                           \
        constexpr int V = kBytes / sizeof(T);                                                      \
        if (kD % V == 0 && aligned)                                                                \
            hc_post_kernel<T, true>                                                                \
                <<<rows *((kD + kThreads * V - 1) / (kThreads * V)), kThreads, 0, stream>>>(       \
                    static_cast<const T *>(x),                                                     \
                    static_cast<const T *>(residual),                                              \
                    post,                                                                          \
                    comb,                                                                          \
                    static_cast<T *>(output));                                                     \
        else                                                                                       \
            hc_post_kernel<T, false>                                                               \
                <<<rows *((kD + kThreads - 1) / kThreads), kThreads, 0, stream>>>(                 \
                    static_cast<const T *>(x),                                                     \
                    static_cast<const T *>(residual),                                              \
                    post,                                                                          \
                    comb,                                                                          \
                    static_cast<T *>(output));                                                     \
    } while (0)
    if (dtype == 0) {
        LAUNCH(__nv_bfloat16);
    } else if (dtype == 1) {
        LAUNCH(__half);
    } else {
        LAUNCH(float);
    }
#undef LAUNCH
}
