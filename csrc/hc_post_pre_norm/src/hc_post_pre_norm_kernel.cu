#include "../../hc_pre_norm/src/hc_pre_norm_impl.cuh"
#include "hc_post_pre_norm_prefill.cuh"

namespace sm110a::hc_post_pre_norm {
using namespace sm110a::hc_pre_norm;

struct PostInput {
    const __nv_bfloat16 *x;
    const __nv_bfloat16 *residual;
    const float *post;
    const float *comb;
    __nv_bfloat16 *mixed;
};

// Called by the projection's producers via argument-dependent lookup. Split-K
// partitions disjoint mixed-output vectors, so each vector is written once.
// Finish consumes this BF16 tensor after the projection grid completes. It is
// also returned to the caller for the next sublayer's residual connection.
__device__ __forceinline__ uint4 load_projection_vector(
    PostInput input,
    int64_t row,
    int k) {
    uint4 result = {};
    if (k >= K)
        return result;
    const int h = k / D, col = k % D;
    uint4 xv = *reinterpret_cast<const uint4 *>(input.x + row * D + col);
    uint4 rv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        rv[i] = *reinterpret_cast<const uint4 *>(input.residual + (row * 4 + i) * D + col);
    const float p = input.post[row * 4 + h];
    const float c0 = input.comb[row * 16 + h];
    const float c1 = input.comb[row * 16 + 4 + h];
    const float c2 = input.comb[row * 16 + 8 + h];
    const float c3 = input.comb[row * 16 + 12 + h];
    auto *out = reinterpret_cast<__nv_bfloat16 *>(&result);
    const auto *x = reinterpret_cast<const __nv_bfloat16 *>(&xv);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        float s = __fmul_rn(c0, as_float(reinterpret_cast<const __nv_bfloat16 *>(&rv[0])[e]));
        s = __fadd_rn(s,
            __fmul_rn(c1, as_float(reinterpret_cast<const __nv_bfloat16 *>(&rv[1])[e])));
        s = __fadd_rn(s,
            __fmul_rn(c2, as_float(reinterpret_cast<const __nv_bfloat16 *>(&rv[2])[e])));
        s = __fadd_rn(s,
            __fmul_rn(c3, as_float(reinterpret_cast<const __nv_bfloat16 *>(&rv[3])[e])));
        out[e] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(p, as_float(x[e])), s));
    }
    *reinterpret_cast<uint4 *>(input.mixed + row * K + k) = result;
    return result;
}

template <int ChunkK>
cudaError_t launch(
    PostInput input,
    const float *weight,
    const float *scale,
    const float *base,
    const float *norm_weight,
    __nv_bfloat16 *output,
    float *post,
    float *comb,
    float *scratch,
    int64_t rows,
    float norm_eps,
    float hc_eps,
    float rms_eps,
    cudaStream_t stream) {
    constexpr int Splits = (K + ChunkK - 1) / ChunkK;
    constexpr int bytes = 2 * (TileM + 3 * 24) * TileK * 2;
    constexpr int Producers = 384;
    auto kernel = hc_pre_norm_projection_tc<ChunkK, PostInput, Producers>;
    int grid_rows = (rows + TileM - 1) / TileM;
    int grid_splits = Splits;
    int threads = Producers + 32;
    int shared_bytes = bytes;
    if constexpr (D == 4096) {
        if (rows >= 512) {
            shared_bytes = GroupBytes;
            kernel = hc_post_pre_norm_projection_grouped<ChunkK, PostInput>;
            grid_rows = (rows + GroupRows - 1) / GroupRows;
            grid_splits = D / ChunkK;
            threads = GroupProducers + 32;
        }
    }
    auto err =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    if (err != cudaSuccess)
        return err;
    sm110a::launch_kernel(kernel,
        dim3(grid_rows, grid_splits),
        threads,
        shared_bytes,
        stream,
        input,
        weight,
        scratch,
        rows);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;
    sm110a::launch_kernel(hc_pre_norm_finish_tc<ChunkK>,
        rows,
        Threads,
        K * 2,
        stream,
        static_cast<const __nv_bfloat16 *>(input.mixed),
        scale,
        base,
        output,
        post,
        comb,
        static_cast<const float *>(scratch),
        norm_weight,
        rms_eps,
        norm_eps,
        hc_eps);
    return cudaGetLastError();
}
} // namespace sm110a::hc_post_pre_norm

extern "C" cudaError_t sm110a_hc_post_pre_norm_launch(
    const void *x,
    const void *residual,
    const float *prev_post,
    const float *prev_comb,
    const float *weight,
    const float *scale,
    const float *base,
    const float *norm_weight,
    void *mixed,
    void *output,
    float *post,
    float *comb,
    float *scratch,
    int64_t rows,
    float norm_eps,
    float hc_eps,
    float rms_eps,
    cudaStream_t stream) {
    using namespace sm110a::hc_post_pre_norm;
    if (!rows)
        return cudaSuccess;
    PostInput input{static_cast<const __nv_bfloat16 *>(x),
        static_cast<const __nv_bfloat16 *>(residual),
        prev_post,
        prev_comb,
        static_cast<__nv_bfloat16 *>(mixed)};
    if constexpr (D == 4096) {
        if (rows >= 256)
            return launch<1024>(input,
                weight,
                scale,
                base,
                norm_weight,
                static_cast<__nv_bfloat16 *>(output),
                post,
                comb,
                scratch,
                rows,
                norm_eps,
                hc_eps,
                rms_eps,
                stream);
    }
    return launch<512>(input,
        weight,
        scale,
        base,
        norm_weight,
        static_cast<__nv_bfloat16 *>(output),
        post,
        comb,
        scratch,
        rows,
        norm_eps,
        hc_eps,
        rms_eps,
        stream);
}
