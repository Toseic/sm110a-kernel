#include "../../../include/pdl.h"
#include "rope_params.h"
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

#ifndef SM110A_ROPE_DIM
#define SM110A_ROPE_DIM 64
#endif
#ifndef SM110A_ROPE_THREADS
#define SM110A_ROPE_THREADS 256
#endif
#ifndef SM110A_ROPE_VECTOR_PAIRS
#define SM110A_ROPE_VECTOR_PAIRS 4
#endif

namespace sm110a::rope {
constexpr int kPairs = SM110A_ROPE_DIM / 2;
constexpr int kThreads = SM110A_ROPE_THREADS;
constexpr int kVectorPairs = SM110A_ROPE_VECTOR_PAIRS;

__device__ float to_float(
    float x) {
    return x;
}
__device__ float to_float(
    __half x) {
    return __half2float(x);
}
__device__ float to_float(
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

template <typename T, bool Packed>
__global__ void rope_rows_kernel(
    T *x,
    const float2 *frequencies,
    Layout layout,
    bool inverse) {
    sm110a::pdl_entry();
    const int i = blockIdx.x * kThreads + threadIdx.x;
    const int head = i / kPairs;
    const int pair = i % kPairs;
    if (head >= layout.heads)
        return;
    T *value = x + int64_t(blockIdx.z) * layout.batch_stride +
               int64_t(blockIdx.y) * layout.sequence_stride + int64_t(head) * layout.head_stride +
               2 * pair;
    using Bits = std::conditional_t<sizeof(T) == 2, uint32_t, uint2>;
    union Pair {
        Bits bits;
        T elements[2];
    } input, output;
    if constexpr (Packed)
        input.bits = *reinterpret_cast<const Bits *>(value);
    else {
        input.elements[0] = value[0];
        input.elements[1] = value[1];
    }
    const float re = to_float(input.elements[0]);
    const float im = to_float(input.elements[1]);
    const float2 f = frequencies[int64_t(blockIdx.y) * layout.frequency_stride + pair];
    const float sine = inverse ? -f.y : f.y;
    output.elements[0] = from_float<T>(__fsub_rn(__fmul_rn(re, f.x), __fmul_rn(im, sine)));
    output.elements[1] = from_float<T>(__fadd_rn(__fmul_rn(re, sine), __fmul_rn(im, f.x)));
    if constexpr (Packed)
        *reinterpret_cast<Bits *>(value) = output.bits;
    else {
        value[0] = output.elements[0];
        value[1] = output.elements[1];
    }
}

template <typename T>
__global__ void rope_vector_kernel(
    T *x,
    const float2 *frequencies,
    Layout layout,
    bool inverse) {
    sm110a::pdl_entry();
    constexpr int groups = kPairs / kVectorPairs;
    const int i = blockIdx.x * kThreads + threadIdx.x;
    const int head = i / groups;
    const int pair = (i % groups) * kVectorPairs;
    if (head >= layout.heads)
        return;
    T *value = x + int64_t(blockIdx.z) * layout.batch_stride +
               int64_t(blockIdx.y) * layout.sequence_stride + int64_t(head) * layout.head_stride +
               2 * pair;
    constexpr int bytes = sizeof(T) * 2 * kVectorPairs;
    using Bits =
        std::conditional_t<(bytes <= 4), uint32_t, std::conditional_t<(bytes <= 8), uint2, uint4>>;
    constexpr int chunks = bytes / sizeof(Bits);
    union Values {
        Bits bits[chunks];
        T elements[2 * kVectorPairs];
    } input, output;
#pragma unroll
    for (int j = 0; j < chunks; ++j)
        input.bits[j] = reinterpret_cast<const Bits *>(value)[j];
    float2 f[kVectorPairs];
#pragma unroll
    for (int j = 0; j < kVectorPairs; ++j)
        f[j] = frequencies[int64_t(blockIdx.y) * layout.frequency_stride + pair + j];
#pragma unroll
    for (int j = 0; j < kVectorPairs; ++j) {
        const float re = to_float(input.elements[2 * j]);
        const float im = to_float(input.elements[2 * j + 1]);
        const float sine = inverse ? -f[j].y : f[j].y;
        output.elements[2 * j] =
            from_float<T>(__fsub_rn(__fmul_rn(re, f[j].x), __fmul_rn(im, sine)));
        output.elements[2 * j + 1] =
            from_float<T>(__fadd_rn(__fmul_rn(re, sine), __fmul_rn(im, f[j].x)));
    }
#pragma unroll
    for (int j = 0; j < chunks; ++j)
        reinterpret_cast<Bits *>(value)[j] = output.bits[j];
}

template <typename T>
__global__ void rope_kernel(
    T *x,
    const float2 *frequencies,
    Layout layout,
    bool inverse) {
    sm110a::pdl_entry();
    for (int64_t i = int64_t(blockIdx.x) * kThreads + threadIdx.x; i < layout.pairs;
        i += int64_t(gridDim.x) * kThreads) {
        const int pair = i % kPairs;
        const int64_t row = i / kPairs;
        const int64_t head = row % layout.heads;
        const int64_t token = row / layout.heads % layout.sequence;
        const int64_t batch = row / layout.heads / layout.sequence;
        T *value = x + batch * layout.batch_stride + token * layout.sequence_stride +
                   head * layout.head_stride + 2 * pair;
        const float re = to_float(value[0]);
        const float im = to_float(value[1]);
        const float2 f = frequencies[token * layout.frequency_stride + pair];
        const float sine = inverse ? -f.y : f.y;
        // Complex multiply in FP32 with separate rounding before the final cast.
        value[0] = from_float<T>(__fsub_rn(__fmul_rn(re, f.x), __fmul_rn(im, sine)));
        value[1] = from_float<T>(__fadd_rn(__fmul_rn(re, sine), __fmul_rn(im, f.x)));
    }
}
} // namespace sm110a::rope

extern "C" void sm110a_rope_launch(
    void *x,
    const void *frequencies,
    sm110a::rope::Layout layout,
    int dtype,
    bool inverse,
    cudaStream_t stream) {
    using namespace sm110a::rope;
    if (!layout.pairs)
        return;
    const int64_t batch = layout.pairs / kPairs / layout.heads / layout.sequence;
    if (layout.sequence <= 65535 && batch <= 65535 &&
        layout.heads * kPairs <= 0x7fffffff - kThreads + 1) {
        const int alignment = std::min(16, (dtype == 2 ? 8 : 4) * kVectorPairs);
        const int elements = alignment / (dtype == 2 ? 4 : 2);
        const bool vectorized =
            kPairs % kVectorPairs == 0 && reinterpret_cast<uintptr_t>(x) % alignment == 0 &&
            layout.batch_stride % elements == 0 && layout.sequence_stride % elements == 0 &&
            layout.head_stride % elements == 0;
        if constexpr (kPairs % kVectorPairs == 0) {
            if (vectorized) {
                const dim3 vector_grid(
                    (layout.heads * (kPairs / kVectorPairs) + kThreads - 1) / kThreads,
                    layout.sequence,
                    batch);
#define LAUNCH_VECTOR(T)                                                                           \
    sm110a::launch_kernel(rope_vector_kernel<T>,                                                   \
        vector_grid,                                                                               \
        kThreads,                                                                                  \
        0,                                                                                         \
        stream,                                                                                    \
        static_cast<T *>(x),                                                                       \
        static_cast<const float2 *>(frequencies),                                                  \
        layout,                                                                                    \
        inverse)
                if (dtype == 0) {
                    LAUNCH_VECTOR(__nv_bfloat16);
                } else if (dtype == 1) {
                    LAUNCH_VECTOR(__half);
                } else {
                    LAUNCH_VECTOR(float);
                }
#undef LAUNCH_VECTOR
                return;
            }
        }
        const dim3 grid((layout.heads * kPairs + kThreads - 1) / kThreads, layout.sequence, batch);
#define LAUNCH_ROWS_VARIANT(T, P)                                                                  \
    sm110a::launch_kernel(rope_rows_kernel<T, P>,                                                  \
        grid,                                                                                      \
        kThreads,                                                                                  \
        0,                                                                                         \
        stream,                                                                                    \
        static_cast<T *>(x),                                                                       \
        static_cast<const float2 *>(frequencies),                                                  \
        layout,                                                                                    \
        inverse)
        const bool packed = reinterpret_cast<uintptr_t>(x) % (dtype == 2 ? 8 : 4) == 0 &&
                            layout.batch_stride % 2 == 0 && layout.sequence_stride % 2 == 0 &&
                            layout.head_stride % 2 == 0;
#define LAUNCH_ROWS(T)                                                                             \
    if (packed) {                                                                                  \
        LAUNCH_ROWS_VARIANT(T, true);                                                              \
    } else {                                                                                       \
        LAUNCH_ROWS_VARIANT(T, false);                                                             \
    }
        if (dtype == 0) {
            LAUNCH_ROWS(__nv_bfloat16);
        } else if (dtype == 1) {
            LAUNCH_ROWS(__half);
        } else {
            LAUNCH_ROWS(float);
        }
#undef LAUNCH_ROWS
#undef LAUNCH_ROWS_VARIANT
        return;
    }
    const int blocks = int(std::min<int64_t>((layout.pairs + kThreads - 1) / kThreads, 65535));
#define LAUNCH(T)                                                                                  \
    sm110a::launch_kernel(rope_kernel<T>,                                                          \
        blocks,                                                                                    \
        kThreads,                                                                                  \
        0,                                                                                         \
        stream,                                                                                    \
        static_cast<T *>(x),                                                                       \
        static_cast<const float2 *>(frequencies),                                                  \
        layout,                                                                                    \
        inverse)
    if (dtype == 0) {
        LAUNCH(__nv_bfloat16);
    } else if (dtype == 1) {
        LAUNCH(__half);
    } else {
        LAUNCH(float);
    }
#undef LAUNCH
}
