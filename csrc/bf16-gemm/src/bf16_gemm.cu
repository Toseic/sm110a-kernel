#include <cudaTypedefs.h> // required for CUtensorMap
#include <cuda_bf16.h>

#include "bf16_gemm.h"

namespace {

__global__ void bf16_gemm_kernel(
    const __nv_bfloat16 *a,
    const __nv_bfloat16 *b,
    __nv_bfloat16 *c,
    int m,
    int n,
    int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }

    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        const float av = __bfloat162float(a[row * k + kk]);
        const float bv = __bfloat162float(b[kk * n + col]);
        acc += av * bv;
    }
    c[row * n + col] = __float2bfloat16(acc);
}

} // namespace

void launch_bf16_gemm(
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b,
    __nv_bfloat16 *d_c,
    int m,
    int n,
    int k) {
    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    bf16_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
}

float benchmark_bf16_gemm(
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b,
    __nv_bfloat16 *d_c,
    int m,
    int n,
    int k,
    int warmup,
    int iters) {
    for (int i = 0; i < warmup; ++i) {
        launch_bf16_gemm(d_a, d_b, d_c, m, n, k);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch_bf16_gemm(d_a, d_b, d_c, m, n, k);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(iters);
}
