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

// One fixed decode configuration. Instruction N is exactly 24; the two
// 32-column TMEM regions hold high and residual accumulators separately.
constexpr int TileK = 128;
constexpr int ChunkK = 512;
constexpr int StageCount = ChunkK / TileK;
constexpr int ProducerThreads = 512;
constexpr int TileM = 128;
constexpr int Splits = (K + ChunkK - 1) / ChunkK;
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
__global__ __launch_bounds__(544)
void hc_pre_projection_tc(const __nv_bfloat16* input, const float* weight,
                          float* scratch, int64_t rows) {
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
                uint4 packed = {};
                if (r0 + r < rows && k0 + k + 7 < K)
                    packed =
                        *reinterpret_cast<const uint4 *>(input + (int64_t(r0) + r) * K + k0 + k);
                else if (r0 + r < rows) {
                    auto *vals = reinterpret_cast<__nv_bfloat16 *>(&packed);
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        if (k0 + k + e < K)
                            vals[e] = input[(int64_t(r0) + r) * K + k0 + k + e];
                }
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
                    sq += __shfl_xor_sync(0xffffffff, sq, offset);
                if (v % (TileK / 8) == 0 && r0 + r < rows)
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
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        part[e] = __float2bfloat16_rn(vals[e]);
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
    if (warp < 4)
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
__global__ __launch_bounds__(Threads)
void hc_pre_finish_tc(const __nv_bfloat16* input, const float* scale, const float* base,
                      __nv_bfloat16* output, float* post, float* comb,
                      const float* scratch, float norm_eps, float hc_eps) {
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
        for (int i = 1; i < SM110A_HC_PRE_ITERS; ++i) {
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
    if constexpr (D % 8 == 0) {
        const float gates[4] = {pre[0], pre[1], pre[2], pre[3]};
        for (int d = tid * 8; d < D; d += Threads * 8) {
            float y[8] = {};
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                uint4 bits = *reinterpret_cast<const uint4 *>(retained + h * D + d);
                const auto *vals = reinterpret_cast<const __nv_bfloat16 *>(&bits);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    float product = __fmul_rn(gates[h], as_float(vals[e]));
                    if (h == 0)
                        y[e] = product;
                    else
                        y[e] += product;
                }
            }
            uint4 bits;
            auto *vals = reinterpret_cast<__nv_bfloat16 *>(&bits);
#pragma unroll
            for (int e = 0; e < 8; ++e)
                vals[e] = from_float<__nv_bfloat16>(y[e]);
            *reinterpret_cast<uint4 *>(output + row * D + d) = bits;
        }
    } else {
        for (int d = tid; d < D; d += Threads) {
            float y = __fmul_rn(pre[0], as_float(retained[d]));
#pragma unroll
            for (int h = 1; h < 4; ++h)
                y += __fmul_rn(pre[h], as_float(retained[h * D + d]));
            output[row * D + d] = from_float<__nv_bfloat16>(y);
        }
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
    float *scratch,
    int64_t rows,
    int dtype,
    float norm_eps,
    float hc_eps,
    cudaStream_t stream) {
    if (!rows)
        return cudaSuccess;
    if (dtype == 0 && D >= 256 && D % 2 == 0 && (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(output) % 16 == 0)) {
        constexpr int bytes_tc = 2 * (TileM + 3 * 24) * TileK * 2;
        if constexpr (bytes_tc > 48 * 1024) {
            auto err = cudaFuncSetAttribute(hc_pre_projection_tc,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                bytes_tc);
            if (err != cudaSuccess)
                return err;
        }
        hc_pre_projection_tc<<<dim3((rows + TileM - 1) / TileM, Splits), 544, bytes_tc, stream>>>(
            static_cast<const __nv_bfloat16 *>(input),
            weight,
            scratch,
            rows);
        auto err = cudaGetLastError();
        if (err != cudaSuccess)
            return err;
        hc_pre_finish_tc<<<rows, Threads, K * 2, stream>>>(
            static_cast<const __nv_bfloat16 *>(input),
            scale,
            base,
            static_cast<__nv_bfloat16 *>(output),
            post,
            comb,
            scratch,
            norm_eps,
            hc_eps);
        return cudaGetLastError();
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
            norm_eps,
            hc_eps,
            stream);
    if (dtype == 1)
        return launch<
            __half>(input, weight, scale, base, output, post, comb, rows, norm_eps, hc_eps, stream);
    return launch<
        float>(input, weight, scale, base, output, post, comb, rows, norm_eps, hc_eps, stream);
}
