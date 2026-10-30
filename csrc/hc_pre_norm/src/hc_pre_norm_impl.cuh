#pragma once

// Shared device implementation for HC pre-norm and fused HC post-pre-norm.
// The standalone SIMT/head kernels and host dispatch stay in the .cu file.
#include "../../../include/pdl.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef SM110A_HC_PRE_NORM_D
#define SM110A_HC_PRE_NORM_D 4096
#endif
#ifndef SM110A_HC_PRE_NORM_ITERS
#define SM110A_HC_PRE_NORM_ITERS 20
#endif

#ifndef SM110A_HC_PRE_NORM_HEAD
#define SM110A_HC_PRE_NORM_HEAD 0
#endif

namespace sm110a::hc_pre_norm {
constexpr bool Head = SM110A_HC_PRE_NORM_HEAD;
constexpr int Projections = Head ? 4 : 24;
constexpr int D = SM110A_HC_PRE_NORM_D;
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

// Preserve the collapse cast before RMSNorm, matching two separate kernels.
template <typename T>
__device__ void collapse_norm(const T* retained, const float* pre,
                             const float* norm_weight, T* output,
                             int64_t row, float rms_eps) {
    __shared__ float sums[Warps];
    constexpr int Vec = D % 8 == 0 && sizeof(T) == 2 ? 8 : 1;
    constexpr int Tiles = (D + Threads * Vec - 1) / (Threads * Vec);
    float values[Tiles * Vec];
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < Tiles; ++i) {
        int d = (threadIdx.x + i * Threads) * Vec;
        float y[Vec] = {};
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            if (d < D) {
                if constexpr (Vec == 8) {
                    uint4 packed = *reinterpret_cast<const uint4*>(retained + h * D + d);
                    const T* v = reinterpret_cast<const T*>(&packed);
#pragma unroll
                    for (int e = 0; e < Vec; ++e) {
                        float product = __fmul_rn(pre[h], as_float(v[e]));
                        y[e] = h == 0 ? product : y[e] + product;
                    }
                } else {
                    float product = __fmul_rn(pre[h], as_float(retained[h * D + d]));
                    y[0] = h == 0 ? product : y[0] + product;
                }
            }
        }
#pragma unroll
        for (int e = 0; e < Vec; ++e) {
            float v = as_float(from_float<T>(y[e]));
            values[i * Vec + e] = v;
            sq = fmaf(v, v, sq);
        }
    }
    sq = warp_sum(sq);
    if (threadIdx.x % 32 == 0) sums[threadIdx.x / 32] = sq;
    __syncthreads();
    if (threadIdx.x < 32) {
        sq = warp_sum(threadIdx.x < Warps ? sums[threadIdx.x] : 0.f);
        if (threadIdx.x == 0) sums[0] = rsqrtf(sq / D + rms_eps);
    }
    __syncthreads();
    float inv = sums[0];
#pragma unroll
    for (int i = 0; i < Tiles; ++i) {
        int d = (threadIdx.x + i * Threads) * Vec;
        if (d < D) {
            if constexpr (Vec == 8) {
                uint4 packed;
                float weights[8];
#pragma unroll
                for (int w = 0; w < 2; ++w) {
                    float4 v;
                    const float* address = norm_weight + d + w * 4;
                    // A contiguous FP32 weight may have a nonzero storage offset.
                    if (reinterpret_cast<uintptr_t>(norm_weight) % 16 == 0) {
                        asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                            : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                            : "l"(address) : "memory");
                    } else {
                        v = make_float4(address[0], address[1], address[2], address[3]);
                    }
                    weights[w * 4] = v.x;
                    weights[w * 4 + 1] = v.y;
                    weights[w * 4 + 2] = v.z;
                    weights[w * 4 + 3] = v.w;
                }
                T* v = reinterpret_cast<T*>(&packed);
#pragma unroll
                for (int e = 0; e < Vec; ++e)
                    v[e] = from_float<T>((values[i * Vec + e] * inv) * weights[e]);
                *reinterpret_cast<uint4*>(output + row * D + d) = packed;
            } else {
                output[row * D + d] = from_float<T>((values[i] * inv) * norm_weight[d]);
            }
        }
    }
}

// Instruction N is exactly 24; the two
// 32-column TMEM regions hold high and residual accumulators separately.
constexpr int TileK = 128;
constexpr int ProducerThreads = 512;
constexpr int TileM = 128;
// K-major SW128: each 64-element slab contains all M/N rows.
__device__ __forceinline__ int swizzle_index(
    int row,
    int k,
    int rows) {
    int byte = ((k / 64) * rows * 64 + row * 64 + k % 64) * 2;
    return (byte ^ ((byte >> 3) & 0x70)) / 2;
}
__device__ __forceinline__ uint64_t matrix_desc(
    const void *ptr) {
    uint64_t addr = __cvta_generic_to_shared(ptr);
    return ((addr & 0x3ffff) >> 4) | (uint64_t(1024 / 16) << 32) | (1ULL << 46) | (2ULL << 61);
}
__device__ __forceinline__ void mma_tcgen(
    unsigned dst,
    uint64_t a,
    uint64_t b,
    int accumulate) {
    constexpr unsigned desc = (1U << 4) | (1U << 7) | (1U << 10) | (3U << 17) | (8U << 24);
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; "
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }" ::"r"(dst),
        "l"(a),
        "l"(b),
        "r"(desc),
        "r"(accumulate)
        : "memory");
}
__device__ __forceinline__ uint4 load_projection_vector(
    const __nv_bfloat16* input, int64_t row, int k) {
    uint4 packed = {};
    if (k + 7 < K)
        packed = *reinterpret_cast<const uint4*>(input + row * K + k);
    else {
        auto* values = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
        for (int e = 0; e < 8; ++e)
            if (k + e < K)
                values[e] = input[row * K + k + e];
    }
    return packed;
}

template <int ChunkK, typename Input = const __nv_bfloat16*, int ProducerCount = 512>
__global__ __launch_bounds__(ProducerCount + 32)
void hc_pre_norm_projection_tc(Input input, const float* weight,
                          float* scratch, int64_t rows) {
    sm110a::pdl_entry();
    constexpr int ProducerThreads = ProducerCount;
    constexpr int StageCount = ChunkK / TileK;
    constexpr int Splits = (K + ChunkK - 1) / ChunkK;
    extern __shared__ __align__(1024) __nv_bfloat16 storage_tc[];
    constexpr int StageElements = (TileM + 3 * 24) * TileK;
    // A completion barrier per shared buffer prevents producers from
    // overwriting operands still in use by asynchronous MMA.
    __shared__ __align__(8) uint64_t barriers[2];
    __shared__ float norms[TileM];
    __shared__ unsigned tmem;
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    int r0 = blockIdx.x * TileM;
    if (tid < TileM)
        norms[tid] = 0.f;
    if (tid < 2) {
        unsigned ba = __cvta_generic_to_shared(barriers + tid);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(ba));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        unsigned ta = __cvta_generic_to_shared(&tmem);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;" ::"r"(ta)
            : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
#pragma unroll 1
    for (int stage = 0; stage < StageCount; ++stage) {
        int buf = stage % 2;
        unsigned ba = __cvta_generic_to_shared(barriers + buf);
        auto *x = storage_tc + buf * StageElements;
        auto *w = reinterpret_cast<__nv_bfloat16(*)[24 * TileK]>(x + TileM * TileK);
        int k0 = blockIdx.y * ChunkK + stage * TileK;
        if (stage >= 2) {
            int phase = ((stage / 2) - 1) % 2;
            asm volatile(
                "{ .reg .pred p; WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, "
                "[%0], %1; @!p bra WAIT; }" ::"r"(ba),
                "r"(phase)
                : "memory");
        }
        // Producer warps advance to the next buffer while the dedicated
        // consumer warp issues MMA for the current buffer.
        if (tid < ProducerThreads) {
#pragma unroll 4
            for (int v = tid; v < TileM * TileK / 8; v += ProducerThreads) {
                int r = v / (TileK / 8), k = (v % (TileK / 8)) * 8;
                if (r0 + r >= rows) {
                    // Padding is immutable across K stages. Initialize each
                    // shared buffer on its first use, then leave it in place.
                    if (stage < 2)
                        *reinterpret_cast<uint4 *>(x + swizzle_index(r, k, TileM)) =
                            make_uint4(0, 0, 0, 0);
                    continue;
                }
                uint4 packed = load_projection_vector(input, int64_t(r0) + r, k0 + k);
                *reinterpret_cast<uint4 *>(x + swizzle_index(r, k, TileM)) = packed;
                float sq = 0.f;
                const auto *vals = reinterpret_cast<const __nv_bfloat16 *>(&packed);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    float value = as_float(vals[e]);
                    sq = fmaf(value, value, sq);
                }
#pragma unroll
                for (int offset = TileK / 8 / 2; offset; offset /= 2)
                    // One row occupies exactly half a warp; the other half
                    // may be padding and has skipped this reduction.
                    sq += __shfl_xor_sync(0xffffu << (lane & 16), sq, offset);
                if (v % (TileK / 8) == 0)
                    norms[r] += sq;
            }
#pragma unroll
            for (int v = tid; v < 24 * TileK / 4; v += ProducerThreads) {
                int n = v / (TileK / 4), k = (v % (TileK / 4)) * 4;
                float4 packed = k0 + k < K
                                    ? *reinterpret_cast<const float4 *>(weight + n * K + k0 + k)
                                    : make_float4(0, 0, 0, 0);
                float vals[4] = {packed.x, packed.y, packed.z, packed.w};
                int idx = swizzle_index(n, k, 24);
#pragma unroll
                // Successive residuals preserve the FP32 weight interface;
                // no persistent weight packing or stale-parameter cache.
                for (int c = 0; c < 3; ++c) {
                    uint2 bits;
                    auto *part = reinterpret_cast<__nv_bfloat16 *>(&bits);
                    auto *pairs = reinterpret_cast<__nv_bfloat162 *>(&bits);
                    pairs[0] = __floats2bfloat162_rn(vals[0], vals[1]);
                    pairs[1] = __floats2bfloat162_rn(vals[2], vals[3]);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        vals[e] -= as_float(part[e]);
                    }
                    *reinterpret_cast<uint2 *>(w[c] + idx) = bits;
                }
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (tid == ProducerThreads) {
#pragma unroll
            for (int k = 0; k < TileK; k += 16) {
                int ao = ((k / 64) * TileM * 64 + k % 64) * 2;
                int bo = ((k / 64) * 24 * 64 + k % 64) * 2;
#pragma unroll
                for (int c = 0; c < 3; ++c)
                    mma_tcgen(tmem + (c == 0 ? 0 : 32),
                        matrix_desc(reinterpret_cast<char *>(x) + ao),
                        matrix_desc(reinterpret_cast<char *>(w[c]) + bo),
                        stage != 0 || k != 0 || c == 2);
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
                    ba)
                : "memory");
        }
    }
    unsigned ba = __cvta_generic_to_shared(barriers + (StageCount - 1) % 2);
    int phase = ((StageCount - 1) / 2) % 2;
    asm volatile("{ .reg .pred p; WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, "
                 "[%0], %1; @!p bra WAIT; }" ::"r"(ba),
        "r"(phase)
        : "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    __syncthreads();
    if (tid < TileM && r0 + tid < rows)
        scratch[((int64_t(r0) + tid) * Splits + blockIdx.y) * 32 + 24] = norms[tid];
    // TMEM loads are warp-collective, but padding-only warps have no output.
    if (warp < 4 && r0 + warp * 32 < rows)
        for (int n = 0; n < 24; n += 8) {
            float v[8];
            unsigned addr = tmem + (warp * 32 << 16) + n;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(v[0]),
                "=f"(v[1]),
                "=f"(v[2]),
                "=f"(v[3]),
                "=f"(v[4]),
                "=f"(v[5]),
                "=f"(v[6]),
                "=f"(v[7])
                : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            float lo[8];
            addr += 32;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(lo[0]),
                "=f"(lo[1]),
                "=f"(lo[2]),
                "=f"(lo[3]),
                "=f"(lo[4]),
                "=f"(lo[5]),
                "=f"(lo[6]),
                "=f"(lo[7])
                : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
            for (int j = 0; j < 8; ++j)
                v[j] += lo[j];
            int r = r0 + warp * 32 + lane;
            if (r < rows) {
                float *dst = scratch + (int64_t(r) * Splits + blockIdx.y) * 32 + n;
                *reinterpret_cast<float4 *>(dst) = make_float4(v[0], v[1], v[2], v[3]);
                *reinterpret_cast<float4 *>(dst + 4) = make_float4(v[4], v[5], v[6], v[7]);
            }
        }
    __syncthreads();
    if (warp == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" ::"r"(tmem)
            : "memory");
}
template <int ChunkK>
__global__ __launch_bounds__(Threads)
void hc_pre_norm_finish_tc(const __nv_bfloat16* input, const float* scale, const float* base,
                      __nv_bfloat16* output, float* post, float* comb,
                      const float* scratch, const float* norm_weight, float rms_eps, float norm_eps, float hc_eps) {
    sm110a::pdl_entry();
    constexpr int Splits = (K + ChunkK - 1) / ChunkK;
    extern __shared__ __align__(16) __nv_bfloat16 retained[];
    __shared__ float pre[4];
    __shared__ float partial[Warps][32];
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    int64_t row = blockIdx.x;
    float part = 0.f;
#pragma unroll
    for (int split = warp; split < Splits; split += Warps)
        part += scratch[(row * Splits + split) * 32 + (lane < 25 ? lane : 24)];
    if (lane < 25)
        partial[warp][lane] = part;
    __syncthreads();
    if (warp != 0) {
        for (int k = (tid - 32) * 8; k < K; k += (Threads - 32) * 8) {
            unsigned dst = __cvta_generic_to_shared(retained + k);
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(input + row * K + k)
                : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    if (warp == 0) {
        // Each lane owns one projection (lane 24 owns the square sum).
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < Warps; ++w)
            sum += partial[w][lane < 25 ? lane : 24];
        float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 24) / float(K) + norm_eps);
        float mix = sum * inv;
        if (lane < 4)
            pre[lane] = 1.f / (1.f + __expf(-(__fmul_rn(mix, scale[0]) + base[lane]))) + hc_eps;
        if (lane >= 4 && lane < 8)
            post[row * 4 + lane - 4] =
                2.f / (1.f + __expf(-(__fmul_rn(mix, scale[1]) + base[lane])));

        // Sixteen lanes hold a 4x4 matrix, row-major. Every lane executes
        // every shuffle, including inactive lanes, with a full warp mask.
        float c = __shfl_sync(0xffffffff, mix, lane < 16 ? lane + 8 : 8);
        c = __fmul_rn(c, scale[2]) + base[8 + (lane % 16)];
        float max_row = fmaxf(c, __shfl_xor_sync(0xffffffff, c, 1));
        max_row = fmaxf(max_row, __shfl_xor_sync(0xffffffff, max_row, 2));
        c = __expf(c - max_row);
        float denom = c + __shfl_xor_sync(0xffffffff, c, 1);
        denom += __shfl_xor_sync(0xffffffff, denom, 2);
        c = __fdividef(c, denom) + hc_eps;
        denom = c + __shfl_xor_sync(0xffffffff, c, 4);
        denom += __shfl_xor_sync(0xffffffff, denom, 8);
        c = __fdividef(c, denom + hc_eps);
#pragma unroll 1
        for (int i = 1; i < SM110A_HC_PRE_NORM_ITERS; ++i) {
            denom = c + __shfl_xor_sync(0xffffffff, c, 1);
            denom += __shfl_xor_sync(0xffffffff, denom, 2);
            c = __fdividef(c, denom + hc_eps);
            denom = c + __shfl_xor_sync(0xffffffff, c, 4);
            denom += __shfl_xor_sync(0xffffffff, denom, 8);
            c = __fdividef(c, denom + hc_eps);
        }
        if (lane < 16)
            comb[row * 16 + lane] = c;
    }
    __syncthreads();
    collapse_norm(retained, pre, norm_weight, output, row, rms_eps);
}

} // namespace sm110a::hc_pre_norm
