#pragma once

#include "../../hc_pre_norm/src/hc_pre_norm_impl.cuh"

namespace sm110a::hc_post_pre_norm {
using namespace sm110a::hc_pre_norm;

// Four independent split-K accumulators share the post source vectors. M=64
// leaves room for all four A tiles and two B slots in 100 KiB shared memory.
// The M=64 non-warp-specialized MMA layout uses lanes 0..15 of each TMEM warp.
constexpr int GroupRows = 64;
constexpr int GroupProducers = 256;
constexpr int GroupA = GroupRows * TileK;
constexpr int GroupW = 3 * 24 * TileK;
constexpr int GroupBytes = (4 * GroupA + 2 * GroupW) * sizeof(__nv_bfloat16);

__device__ __forceinline__ void grouped_mma(
    unsigned dst,
    uint64_t a,
    uint64_t b,
    int accumulate) {
    constexpr unsigned desc = (1U << 4) | (1U << 7) | (1U << 10) | (3U << 17) | (4U << 24);
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; "
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }" ::"r"(dst),
        "l"(a),
        "l"(b),
        "r"(desc),
        "r"(accumulate)
        : "memory");
}

__device__ __forceinline__ void grouped_wait(
    unsigned barrier,
    int phase) {
    asm volatile("{ .reg .pred p; WAIT: "
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1; "
                 "@!p bra WAIT; }" ::"r"(barrier),
        "r"(phase)
        : "memory");
}

template <int ChunkK, typename Input>
__global__ __launch_bounds__(GroupProducers + 32)
void hc_post_pre_norm_projection_grouped(Input input, const float* weight,
    float* scratch, int64_t rows) {
    sm110a::pdl_entry();
    static_assert(D % ChunkK == 0);
    constexpr int Segments = D / ChunkK;
    constexpr int Splits = 4 * Segments;
    constexpr int Stages = ChunkK / TileK;
    extern __shared__ __align__(1024) __nv_bfloat16 storage[];
    __shared__ __align__(8) uint64_t barriers[2];
    __shared__ float norms[4][GroupRows];
    __shared__ unsigned tmem;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int r0 = blockIdx.x * GroupRows;
    if (tid < 4 * GroupRows)
        norms[tid / GroupRows][tid % GroupRows] = 0.f;
    if (tid < 2) {
        unsigned address = __cvta_generic_to_shared(barriers + tid);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        unsigned address = __cvta_generic_to_shared(&tmem);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" ::"r"(address)
            : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
#pragma unroll 1
    for (int stage = 0; stage < Stages; ++stage) {
        // The last commit of each stage is slot 1's second arrival. Wait before
        // overwriting the four A tiles, which are shared by all four MMAs.
        if (stage)
            grouped_wait(__cvta_generic_to_shared(barriers + 1), 1);
        const int d0 = blockIdx.y * ChunkK + stage * TileK;
        if (tid < GroupProducers) {
#pragma unroll 1
            for (int v = tid; v < GroupRows * TileK / 8; v += GroupProducers) {
                const int r = v / (TileK / 8), k = (v % (TileK / 8)) * 8;
                const int64_t row = int64_t(r0) + r;
                if (row >= rows) {
                    if (stage == 0) {
#pragma unroll
                        for (int h = 0; h < 4; ++h)
                            *reinterpret_cast<uint4 *>(
                                storage + h * GroupA + swizzle_index(r, k, GroupRows)) =
                                make_uint4(0, 0, 0, 0);
                    }
                    continue;
                }
                const int d = d0 + k;
                uint4 xv = *reinterpret_cast<const uint4 *>(input.x + row * D + d);
                uint4 rv[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    rv[i] =
                        *reinterpret_cast<const uint4 *>(input.residual + (row * 4 + i) * D + d);
                // Source vectors stay live across all outputs. Never reload
                // them for a different HC stream or publish partial FP32 data.
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const float p = input.post[row * 4 + h];
                    const float c0 = input.comb[row * 16 + h];
                    const float c1 = input.comb[row * 16 + 4 + h];
                    const float c2 = input.comb[row * 16 + 8 + h];
                    const float c3 = input.comb[row * 16 + 12 + h];
                    uint4 packed;
                    auto *out = reinterpret_cast<__nv_bfloat16 *>(&packed);
                    float sq = 0.f;
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        float s = __fmul_rn(c0,
                            as_float(reinterpret_cast<const __nv_bfloat16 *>(&rv[0])[e]));
                        s = __fadd_rn(s,
                            __fmul_rn(c1,
                                as_float(reinterpret_cast<const __nv_bfloat16 *>(&rv[1])[e])));
                        s = __fadd_rn(s,
                            __fmul_rn(c2,
                                as_float(reinterpret_cast<const __nv_bfloat16 *>(&rv[2])[e])));
                        s = __fadd_rn(s,
                            __fmul_rn(c3,
                                as_float(reinterpret_cast<const __nv_bfloat16 *>(&rv[3])[e])));
                        out[e] = __float2bfloat16_rn(__fadd_rn(
                            __fmul_rn(p, as_float(reinterpret_cast<const __nv_bfloat16 *>(&xv)[e])),
                            s));
                        float value = as_float(out[e]);
                        sq = fmaf(value, value, sq);
                    }
                    *reinterpret_cast<uint4 *>(
                        storage + h * GroupA + swizzle_index(r, k, GroupRows)) = packed;
                    *reinterpret_cast<uint4 *>(input.mixed + (row * 4 + h) * D + d) = packed;
#pragma unroll
                    for (int offset = TileK / 8 / 2; offset; offset /= 2)
                        sq += __shfl_xor_sync(0xffffu << (lane & 16), sq, offset);
                    if (v % (TileK / 8) == 0)
                        norms[h][r] += sq;
                }
            }
        }
#pragma unroll 1
        for (int h = 0; h < 4; ++h) {
            const int seq = stage * 4 + h;
            unsigned barrier = __cvta_generic_to_shared(barriers + h % 2);
            if (seq >= 2)
                grouped_wait(barrier, ((seq / 2) - 1) % 2);
            auto *w = storage + 4 * GroupA + (h % 2) * GroupW;
            const int k0 = h * D + d0;
            if (tid < GroupProducers) {
#pragma unroll
                for (int v = tid; v < 24 * TileK / 4; v += GroupProducers) {
                    const int n = v / (TileK / 4), k = (v % (TileK / 4)) * 4;
                    float4 packed = *reinterpret_cast<const float4 *>(weight + n * K + k0 + k);
                    float vals[4] = {packed.x, packed.y, packed.z, packed.w};
                    const int index = swizzle_index(n, k, 24);
#pragma unroll
                    for (int c = 0; c < 3; ++c) {
                        uint2 bits;
                        auto *pairs = reinterpret_cast<__nv_bfloat162 *>(&bits);
                        pairs[0] = __floats2bfloat162_rn(vals[0], vals[1]);
                        pairs[1] = __floats2bfloat162_rn(vals[2], vals[3]);
                        auto *part = reinterpret_cast<__nv_bfloat16 *>(&bits);
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            vals[e] -= as_float(part[e]);
                        *reinterpret_cast<uint2 *>(w + c * 24 * TileK + index) = bits;
                    }
                }
            }
            __syncthreads();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            if (tid == GroupProducers) {
#pragma unroll
                for (int k = 0; k < TileK; k += 16) {
                    int ao = ((k / 64) * GroupRows * 64 + k % 64) * 2;
                    int bo = ((k / 64) * 24 * 64 + k % 64) * 2;
#pragma unroll
                    for (int c = 0; c < 3; ++c)
                        grouped_mma(tmem + h * 64 + (c == 0 ? 0 : 32),
                            matrix_desc(reinterpret_cast<char *>(storage + h * GroupA) + ao),
                            matrix_desc(reinterpret_cast<char *>(w + c * 24 * TileK) + bo),
                            stage != 0 || k != 0 || c == 2);
                }
                asm volatile(
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::
                        "r"(barrier)
                    : "memory");
            }
        }
    }
    grouped_wait(__cvta_generic_to_shared(barriers + 1), 1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    __syncthreads();
    if (tid < 4 * GroupRows) {
        const int h = tid / GroupRows, r = r0 + tid % GroupRows;
        if (r < rows)
            scratch[(int64_t(r) * Splits + h * Segments + blockIdx.y) * 32 + 24] =
                norms[h][tid % GroupRows];
    }
    if (warp < 4 && r0 + warp * 16 < rows) {
#pragma unroll 1
        for (int h = 0; h < 4; ++h) {
            for (int n = 0; n < 24; n += 8) {
                float hi[8], lo[8];
                unsigned addr = tmem + h * 64 + (warp * 32 << 16) + n;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=f"(hi[0]),
                    "=f"(hi[1]),
                    "=f"(hi[2]),
                    "=f"(hi[3]),
                    "=f"(hi[4]),
                    "=f"(hi[5]),
                    "=f"(hi[6]),
                    "=f"(hi[7])
                    : "r"(addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                addr += 32;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
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
                int r = r0 + warp * 16 + lane;
                if (lane < 16 && r < rows) {
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        hi[j] += lo[j];
                    float *dst =
                        scratch + (int64_t(r) * Splits + h * Segments + blockIdx.y) * 32 + n;
                    *reinterpret_cast<float4 *>(dst) = make_float4(hi[0], hi[1], hi[2], hi[3]);
                    *reinterpret_cast<float4 *>(dst + 4) = make_float4(hi[4], hi[5], hi[6], hi[7]);
                }
            }
        }
    }
    __syncthreads();
    if (warp == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tmem)
            : "memory");
}
} // namespace sm110a::hc_post_pre_norm
