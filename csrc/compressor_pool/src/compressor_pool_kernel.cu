#include "../../../include/pdl.h"
#include "compressor_pool_params.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sm110a::compressor_pool {
constexpr int R = SM110A_COMPRESS_RATIO;
constexpr int WARP = 32;
constexpr int CHANNELS = R == 4 ? 128 : 32;
constexpr int WINDOW = R == 4 ? 8 : 128;
constexpr int LOCAL = R == 4 ? 8 : 32;

// All CTAs own disjoint channel ranges, including both halves of overlap state.
__device__ void update_state(
    Params p,
    int b,
    int d,
    int group,
    int groups) {
    const int width = (R == 4 ? 2 : 1) * p.d;
    const auto *kv = static_cast<const __nv_bfloat16 *>(p.kv) + b * p.input_batch_stride;
    const auto *score = static_cast<const __nv_bfloat16 *>(p.score) + b * p.input_batch_stride;
    float *vs = p.kv_state + int64_t(b) * (R == 4 ? 2 * R : R) * width;
    float *ss = p.score_state + int64_t(b) * (R == 4 ? 2 * R : R) * width;
    if (p.start == 0) {
        if (group != groups)
            return;
        const int cutoff = groups * R;
        for (int h = 0; h < (R == 4 ? 2 : 1); ++h) {
            const int c = d + h * p.d;
            if constexpr (R == 4) {
                if (groups) {
                    for (int t = 0; t < R; ++t) {
                        const int64_t src = int64_t(cutoff - R + t) * p.input_row_stride + c;
                        vs[t * width + c] = __bfloat162float(kv[src]);
                        ss[t * width + c] =
                            __fadd_rn(__bfloat162float(score[src]), p.ape[t * width + c]);
                    }
                }
            }
            for (int t = R == 128 ? threadIdx.x / 32 : 0; t < p.sequence - cutoff;
                t += R == 128 ? 4 : 1) {
                const int64_t src = int64_t(cutoff + t) * p.input_row_stride + c;
                const int dst = ((R == 4 ? R : 0) + t) * width + c;
                vs[dst] = __bfloat162float(kv[src]);
                ss[dst] = __fadd_rn(__bfloat162float(score[src]), p.ape[t * width + c]);
            }
        }
    } else {
        const int slot = p.start % R;
        for (int h = 0; h < (R == 4 ? 2 : 1); ++h) {
            const int c = d + h * p.d;
            const int dst = ((R == 4 ? R : 0) + slot) * width + c;
            vs[dst] = __bfloat162float(kv[c]);
            ss[dst] = __fadd_rn(__bfloat162float(score[c]), p.ape[slot * width + c]);
        }
    }
}

__device__ void load(
    Params p,
    int b,
    int group,
    int d,
    int t,
    float &v,
    float &s) {
    const int width = (R == 4 ? 2 : 1) * p.d;
    const int token = R == 4 ? (group - 1) * R + t : group * R + t;
    const int c = d + (R == 4 && t >= R ? p.d : 0);
    if (p.start == 0) {
        if (token < 0) {
            v = 0;
            s = -CUDART_INF_F;
            return;
        }
        const int64_t src = b * p.input_batch_stride + int64_t(token) * p.input_row_stride + c;
        v = __bfloat162float(static_cast<const __nv_bfloat16 *>(p.kv)[src]);
        s = __fadd_rn(__bfloat162float(static_cast<const __nv_bfloat16 *>(p.score)[src]),
            p.ape[(t % R) * width + c]);
    } else if (t == WINDOW - 1) {
        // Consume the new token directly; no dependency on a state-store owner.
        const int64_t src = b * p.input_batch_stride + c;
        v = __bfloat162float(static_cast<const __nv_bfloat16 *>(p.kv)[src]);
        s = __fadd_rn(__bfloat162float(static_cast<const __nv_bfloat16 *>(p.score)[src]),
            p.ape[(R - 1) * width + c]);
    } else {
        const int64_t src = int64_t(b) * (R == 4 ? 2 * R : R) * width + t * width + c;
        v = p.kv_state[src];
        s = p.score_state[src];
    }
}

__global__ void pool(
    Params p) {
    sm110a::pdl_entry();
    const int b = blockIdx.z;
    const int group = blockIdx.y;
    const int groups = p.start == 0 ? p.sequence / R : ((p.start + 1) % R == 0);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int d = blockIdx.x * CHANNELS + (R == 4 ? threadIdx.x : lane);
    const bool valid = d < p.d;
    // A separate prefill CTA initializes state using projection input only.
    // Pooling CTAs do not read that state, so no inter-CTA synchronization is needed.
    if (valid && (R == 4 || warp == 0 || p.start == 0))
        update_state(p, b, d, group, groups);
    if (group >= groups)
        return;
    if constexpr (R == 4) {
        if (!valid)
            return;
    }
    float values[LOCAL], scores[LOCAL];
    float maximum = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < LOCAL; ++i) {
        const int t = R == 4 ? i : warp * LOCAL + i;
        values[i] = 0;
        scores[i] = -CUDART_INF_F;
        if (valid)
            load(p, b, group, d, t, values[i], scores[i]);
        maximum = fmaxf(maximum, scores[i]);
    }
    __shared__ float scratch[4][32];
    if constexpr (R == 128) {
        scratch[warp][lane] = maximum;
        __syncthreads();
#pragma unroll
        for (int w = 0; w < 4; ++w)
            maximum = fmaxf(maximum, scratch[w][lane]);
        __syncthreads();
    }
    float denominator = 0;
#pragma unroll
    for (int i = 0; i < LOCAL; ++i) {
        scores[i] = valid ? expf(scores[i] - maximum) : 0;
        denominator = __fadd_rn(denominator, scores[i]);
    }
    if constexpr (R == 128) {
        scratch[warp][lane] = denominator;
        __syncthreads();
        denominator = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w)
            denominator = __fadd_rn(denominator, scratch[w][lane]);
        __syncthreads();
    }
    float result = 0;
#pragma unroll
    for (int i = 0; i < LOCAL; ++i) {
        const float probability = valid ? __fdividef(scores[i], denominator) : 0;
        result = __fadd_rn(result, __fmul_rn(values[i], probability));
    }
    if constexpr (R == 128) {
        scratch[warp][lane] = result;
        __syncthreads();
        result = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w)
            result = __fadd_rn(result, scratch[w][lane]);
    }
    if (valid && (R == 4 || warp == 0)) {
        static_cast<__nv_bfloat16 *>(p.output)[(int64_t(b) * groups + group) * p.d + d] =
            __float2bfloat16_rn(result);
        if constexpr (R == 4) {
            if (p.start != 0) {
                // This thread owns both halves of its channel, so the new slot
                // written above is visible before copying the complete window.
                const int width = 2 * p.d;
                const int64_t base = int64_t(b) * 2 * R * width;
                for (int t = 0; t < R; ++t)
                    for (int h = 0; h < 2; ++h) {
                        const int dst = t * width + d + h * p.d;
                        p.kv_state[base + dst] = p.kv_state[base + R * width + dst];
                        p.score_state[base + dst] = p.score_state[base + R * width + dst];
                    }
            }
        }
    }
}
} // namespace sm110a::compressor_pool
extern "C" void sm110a_compressor_pool_launch(
    sm110a::compressor_pool::Params p,
    cudaStream_t stream) {
    using namespace sm110a::compressor_pool;
    if (!p.batch)
        return;
    const int groups = p.start == 0 ? p.sequence / R : ((p.start + 1) % R == 0);
    const dim3 grid((p.d + CHANNELS - 1) / CHANNELS, p.start == 0 ? groups + 1 : 1, p.batch);
    sm110a::launch_kernel(pool, grid, 128, 0, stream, p);
}
