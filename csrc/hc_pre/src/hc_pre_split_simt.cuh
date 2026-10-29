// One-token BF16 split-K SIMT projection with last-CTA finalization.
// Scratch uses compact ceil(K/1024) rows within the public ceil(K/512) allocation.
template <typename T>
__global__ __launch_bounds__(Threads)
void hc_pre_split_simt(const T* __restrict__ input, const float* __restrict__ weight,
                   const float* __restrict__ scale, const float* __restrict__ base,
                   T* __restrict__ output, float* __restrict__ post,
                   float* __restrict__ comb, float* scratch, float norm_eps, float hc_eps) {
    extern __shared__ __align__(16) unsigned char storage[];
    T *retained = reinterpret_cast<T *>(storage);
    __shared__ float partial[Projections + 1][Warps];
    __shared__ float pre[4];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int64_t row = blockIdx.x;
    constexpr int SplitsSIMT = (K + 1023) / 1024;
    float dots[Projections] = {};
    float square_sum = 0.f;
    // Keep K loop rolled: unrolling the whole projection bloats the binary.
#pragma unroll 1
    for (int k = blockIdx.y * 1024 + tid * 4; k < K && k < (blockIdx.y + 1) * 1024; k += Threads * 4) {
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

    if (warp == 0 && lane < Projections + 1) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < Warps; ++w) v += partial[lane][w];
        scratch[(row * SplitsSIMT + blockIdx.y) * 32 + lane] = v;
        __threadfence();
    }
    __syncthreads();
    __shared__ int last;
    auto* counter = reinterpret_cast<unsigned*>(scratch + row * SplitsSIMT * 32 + 31);
    if (tid == 0) last = atomicAdd(counter, 1u) == SplitsSIMT - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    float reduced = 0.f;
    for (int split = warp; split < SplitsSIMT; split += Warps)
        reduced += scratch[(row * SplitsSIMT + split) * 32 + (lane < 25 ? lane : 24)];
    if (lane < 25) partial[lane][warp] = reduced;
    for (int k = tid * 8; k < K; k += Threads * 8) {
        unsigned dst = __cvta_generic_to_shared(retained + k);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(input + row*K+k) : "memory");
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
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
            pre[lane] = 1.f / (1.f + __expf(-(__fmul_rn(mix, scale[0]) + base[lane]))) + hc_eps;
        if constexpr (!Head) {
            if (lane >= 4 && lane < 8)
                post[row * 4 + lane - 4] = 2.f / (1.f + __expf(-(__fmul_rn(mix, scale[1]) + base[lane])));

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
    __syncthreads();
    if (tid == 0) atomicExch(counter, 0u);

}

