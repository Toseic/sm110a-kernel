#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
constexpr int Threads = 256;
__device__ float as_float(
    float x) {
    return x;
}
__device__ float as_float(
    __nv_bfloat16 x) {
    return __bfloat162float(x);
}
// Monotonic float key: NaNs rank above +inf; signed zeros compare equal.
__device__ uint32_t key(
    float x) {
    if (isnan(x))
        return 0xffffffffu;
    if (x == 0.f)
        x = 0.f;
    uint32_t bits = __float_as_uint(x);
    return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}
template <typename T>
__global__ void topk_naive(
    const T *scores,
    int32_t *out,
    int n,
    int k) {
    using Reduce = cub::BlockReduce<int, Threads>;
    using Scan = cub::BlockScan<int, Threads>;
    __shared__ typename Reduce::TempStorage reduce;
    __shared__ typename Scan::TempStorage scan;
    __shared__ uint32_t prefix, mask;
    __shared__ int remaining, above, emitted, equal_seen;
    const T *row = scores + int64_t(blockIdx.x) * n;
    int32_t *result = out + int64_t(blockIdx.x) * k;
    if (threadIdx.x == 0) {
        prefix = 0;
        mask = 0;
        remaining = k;
    }
    __syncthreads();
    // Find the kth largest key, one radix bit per full-row pass.
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t flag = 1u << bit;
        int count = 0;
        for (int64_t i = threadIdx.x; i < n; i += Threads) {
            uint32_t v = key(as_float(row[i]));
            count += (v & mask) == prefix && (v & flag) != 0;
        }
        int total = Reduce(reduce).Sum(count);
        __syncthreads();
        if (threadIdx.x == 0) {
            if (total >= remaining)
                prefix |= flag;
            else
                remaining -= total;
            mask |= flag;
        }
        __syncthreads();
    }
    int count = 0;
    for (int64_t i = threadIdx.x; i < n; i += Threads)
        count += key(as_float(row[i])) > prefix;
    int total = Reduce(reduce).Sum(count);
    __syncthreads();
    if (threadIdx.x == 0) {
        above = total;
        emitted = 0;
        equal_seen = 0;
    }
    __syncthreads();
    // Emit in input-index order; threshold ties select the lowest indices.
    for (int64_t base = 0; base < n; base += Threads) {
        int64_t i = base + threadIdx.x;
        uint32_t v = i < n ? key(as_float(row[i])) : 0;
        int eq = i < n && v == prefix;
        int eq_rank, eq_total;
        Scan(scan).ExclusiveSum(eq, eq_rank, eq_total);
        __syncthreads();
        int take = i < n && (v > prefix || (eq && equal_seen + eq_rank < k - above));
        int rank, tile_total;
        Scan(scan).ExclusiveSum(take, rank, tile_total);
        __syncthreads();
        if (take)
            result[emitted + rank] = int32_t(i);
        __syncthreads();
        if (threadIdx.x == 0) {
            emitted += tile_total;
            equal_seen += eq_total;
        }
        __syncthreads();
    }
}
} // namespace
extern "C" void sm110a_topk_launch(
    const void *scores,
    int32_t *out,
    int rows,
    int n,
    int k,
    int dtype,
    cudaStream_t stream) {
    if (!rows || !k)
        return;
    if (dtype == 0)
        topk_naive<<<rows, Threads, 0, stream>>>(static_cast<const float *>(scores), out, n, k);
    else
        topk_naive<<<rows, Threads, 0, stream>>>(static_cast<const __nv_bfloat16 *>(scores),
            out,
            n,
            k);
}
