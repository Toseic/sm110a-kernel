#include "../../../include/pdl.h"
#include <cstdint>
#include <cuda_runtime.h>

#ifndef SM110A_GATE_EXPERTS
#define SM110A_GATE_EXPERTS 256
#endif
#ifndef SM110A_GATE_TOPK
#define SM110A_GATE_TOPK 6
#endif

namespace sm110a::gate {
constexpr int E = SM110A_GATE_EXPERTS;
constexpr int K = SM110A_GATE_TOPK;
constexpr int V = (E + 31) / 32;
constexpr int kSmallRowThreads = E <= 32 ? 32 : E <= 64 ? 64 : E <= 128 ? 128 : 256;

__device__ __forceinline__ float activation(
    float x) {
    return sqrtf(x > 20.f ? x : log1pf(expf(x)));
}
// An unsigned key orders all finite floats and infinities; canonicalize NaNs
// above +Inf, and both signed zeros to +0. Zero is reserved for removed IDs.
__device__ __forceinline__ uint32_t score_key(
    float x) {
    uint32_t bits = __float_as_uint(x == 0.f ? 0.f : x);
    uint32_t key = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    return isnan(x) ? 0xffffffffu : key;
}

__device__ __forceinline__ uint32_t warp_max(
    uint32_t x) {
    uint32_t out;
    asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(out) : "r"(x));
    return out;
}

__device__ __forceinline__ int warp_min(
    int x) {
    int out;
    asm("redux.sync.min.s32 %0, %1, 0xffffffff;" : "=r"(out) : "r"(x));
    return out;
}

template <typename Index, bool Hash>
__global__ void fused_gate_kernel(
    const float *logits,
    const float *bias,
    const Index *hash,
    float *weights,
    Index *indices,
    int64_t rows,
    float scale) {
    sm110a::pdl_entry();
    const int lane = threadIdx.x % 32;
    const int64_t row = Hash ? int64_t(blockIdx.x) * 4 + threadIdx.x / 32 : blockIdx.x;
    if (row >= rows)
        return;
    float selected = 0.f;
    if constexpr (Hash) {
        if (lane < K) {
            Index id = hash[row * K + lane];
            indices[row * K + lane] = id;
            // Hash indices are device data; reject invalid IDs without an OOB read.
            selected = id >= 0 && id < E ? activation(logits[row * E + id]) : nanf("");
        }
    } else {
        // Cooperatively evaluate expensive activations before the selecting warp
        // retains selection keys in registers. Decode rows no longer serialize E/32
        // calls to libdevice log1p/sqrt in a single warp.
        __shared__ float staged_values[E];
        __shared__ uint32_t staged_keys[E];
        for (int id = threadIdx.x; id < E; id += blockDim.x) {
            float value = activation(logits[row * E + id]);
            staged_values[id] = value;
            staged_keys[id] = score_key(value + (bias ? bias[id] : 0.f));
        }
        __syncthreads();
        if (threadIdx.x >= 32)
            return;
        uint32_t keys[V];
#pragma unroll
        for (int i = 0; i < V; ++i) {
            int id = lane + i * 32;
            keys[i] = id < E ? staged_keys[id] : 0;
        }
#pragma unroll
        for (int k = 0; k < K; ++k) {
            uint32_t best = 0;
#pragma unroll
            for (int i = 0; i < V; ++i)
                best = max(best, keys[i]);
            best = warp_max(best);
            int id = 0x7fffffff;
#pragma unroll
            for (int i = 0; i < V; ++i)
                id = min(id, keys[i] == best ? lane + i * 32 : 0x7fffffff);
            id = warp_min(id);
#pragma unroll
            for (int i = 0; i < V; ++i)
                if (lane + i * 32 == id)
                    keys[i] = 0;
            // Every lane requests the same address: one shared-memory broadcast
            // avoids retaining V scores per lane and shuffling the selected one.
            float value = staged_values[id];
            if (lane == k) {
                selected = value;
                indices[row * K + k] = id;
            }
        }
    }
    float sum = selected;
#pragma unroll
    for (int delta = 16; delta; delta /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, delta);
    sum = __shfl_sync(0xffffffff, sum, 0);
    if (lane < K)
        weights[row * K + lane] = __fmul_rn(__fdiv_rn(selected, sum), scale);
}
} // namespace sm110a::gate

extern "C" void sm110a_fused_gate_launch(
    const float *logits,
    const float *bias,
    const void *hash,
    float *weights,
    void *indices,
    int64_t rows,
    bool int32_indices,
    float scale,
    cudaStream_t stream) {
    if (!rows)
        return;
    using namespace sm110a::gate;
    // Small decode batches benefit from one activation per lane at E=256.
    // Larger batches use fewer warps per CTA to improve resident CTA count.
    const int topk_threads = rows <= 32 ? kSmallRowThreads : (E <= 128 ? kSmallRowThreads : 128);
#define LAUNCH(T, H)                                                                               \
    sm110a::launch_kernel(fused_gate_kernel<T, H>,                                                 \
        dim3(H ? (rows + 3) / 4 : rows),                                                           \
        dim3(H ? 128 : topk_threads),                                                              \
        0,                                                                                         \
        stream,                                                                                    \
        logits,                                                                                    \
        bias,                                                                                      \
        static_cast<const T *>(hash),                                                              \
        weights,                                                                                   \
        static_cast<T *>(indices),                                                                 \
        rows,                                                                                      \
        scale)
    if (hash) {
        if (int32_indices) {
            LAUNCH(int32_t, true);
        } else {
            LAUNCH(int64_t, true);
        }
    } else {
        LAUNCH(int64_t, false);
    }
#undef LAUNCH
}
