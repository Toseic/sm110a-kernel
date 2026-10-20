#include <cmath>
#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

extern "C" void sm110a_nvfp4_fc1_tasks_from_ranges(
    const int32_t *, const int32_t *, int32_t *, int32_t *, int, int,
    int, int, int, cudaStream_t);

namespace sm110a::nvfp4_fc1_routing {
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
#ifndef SM110A_NVFP4_FC1_CLUSTER_N
#define SM110A_NVFP4_FC1_CLUSTER_N 1
#endif

__global__ void topk_router_kernel(
    const __nv_bfloat16 *scores, int tokens, int experts, int topk,
    int *selected, int *counts, int *offsets) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ char smem[];
    auto *reduce_scores = reinterpret_cast<float *>(smem);
    auto *reduce_experts = reinterpret_cast<int *>(reduce_scores + blockDim.x);

    for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
        for (int rank = 0; rank < topk; ++rank) {
            float best_score = -INFINITY;
            int best_expert = -1;
            for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
                bool chosen = false;
                for (int previous = 0; previous < rank; ++previous)
                    chosen |= selected[static_cast<size_t>(token) * topk + previous] == expert;
                if (chosen) continue;
                float score = __bfloat162float(
                    scores[static_cast<size_t>(token) * experts + expert]);
                if (isnan(score)) score = -INFINITY;
                if (best_expert < 0 || score > best_score ||
                    (score == best_score && expert < best_expert)) {
                    best_score = score;
                    best_expert = expert;
                }
            }
            reduce_scores[threadIdx.x] = best_score;
            reduce_experts[threadIdx.x] = best_expert;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride; stride >>= 1) {
                if (threadIdx.x < stride) {
                    const float other_score = reduce_scores[threadIdx.x + stride];
                    const int other_expert = reduce_experts[threadIdx.x + stride];
                    const float own_score = reduce_scores[threadIdx.x];
                    const int own_expert = reduce_experts[threadIdx.x];
                    if (other_expert >= 0 &&
                        (own_expert < 0 || other_score > own_score ||
                            (other_score == own_score && other_expert < own_expert))) {
                        reduce_scores[threadIdx.x] = other_score;
                        reduce_experts[threadIdx.x] = other_expert;
                    }
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                const int expert = reduce_experts[0];
                selected[static_cast<size_t>(token) * topk + rank] = expert;
                atomicAdd(counts + expert, 1);
            }
            __syncthreads();
        }
    }
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int offset = 0;
        for (int expert = 0; expert < experts; ++expert) {
            offsets[expert] = offset;
            offset += counts[expert];
        }
    }
    grid.sync();
}

__global__ void build_grouped_rows(
    const int *selected, const int *offsets, int *source_rows,
    int *grouped_assignments, int m, int topk) {
    constexpr int kWarps = kThreads / kWarpSize;
    const int expert = blockIdx.x;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    __shared__ int warp_offsets[kWarps];
    __shared__ int rows_written;
    __shared__ int block_matches;
    if (threadIdx.x == 0) rows_written = offsets[expert];
    __syncthreads();
    for (int base = 0; base < m; base += kThreads) {
        const int assignment = base + threadIdx.x;
        const bool match = assignment < m && selected[assignment] == expert;
        const uint32_t mask = __ballot_sync(0xffffffffu, match);
        const uint32_t lower = (uint32_t{1} << lane) - 1;
        const int prefix = __popc(mask & lower);
        if (lane == 0) warp_offsets[warp] = __popc(mask);
        __syncthreads();
        if (threadIdx.x == 0) {
            int running = 0;
            for (int i = 0; i < kWarps; ++i) {
                const int count = warp_offsets[i];
                warp_offsets[i] = running;
                running += count;
            }
            block_matches = running;
        }
        __syncthreads();
        if (match) {
            const int grouped = rows_written + warp_offsets[warp] + prefix;
            source_rows[grouped] = assignment / topk;
            grouped_assignments[grouped] = assignment;
        }
        __syncthreads();
        if (threadIdx.x == 0) rows_written += block_matches;
        __syncthreads();
    }
}

__global__ void materialize_layout(
    const int *offsets, const int *counts, int experts, int *layout) {
    const int expert = blockIdx.x;
    if (expert >= experts) return;
    for (int row = offsets[expert] + threadIdx.x;
         row < offsets[expert] + counts[expert]; row += blockDim.x)
        layout[row] = expert;
}

__global__ void invert_assignments(
    const int *grouped_assignments, int m, int *assignment_to_grouped) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) assignment_to_grouped[grouped_assignments[row]] = row;
}
} // namespace sm110a::nvfp4_fc1_routing

using namespace sm110a::nvfp4_fc1_routing;

extern "C" void sm110a_nvfp4_fc1_routing_prepare(
    const __nv_bfloat16 *router, int tokens, int experts, int topk,
    int32_t *routing, int32_t *source_rows, int32_t *selected,
    int32_t *tasks, int32_t *state, int n, int tile_m, int tile_n,
    int cluster_m,
    cudaStream_t stream) {
    if (!router || !routing || !source_rows || !selected || !tasks || !state || tokens <= 0 ||
        experts <= 0 || topk <= 0 || topk > experts) return;
    const int m = tokens * topk;
    auto *counts = routing;
    auto *offsets = counts + experts;
    auto *layout = offsets + experts;
    auto *grouped_assignments = layout + m;
    auto *assignment_to_grouped = grouped_assignments + m;
    cudaMemsetAsync(counts, 0, experts * sizeof(int32_t), stream);

    int device = 0, sm_count = 0, blocks_per_sm = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    const size_t smem_bytes = 2 * kThreads * sizeof(int);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, topk_router_kernel, kThreads, smem_bytes);
    int blocks = blocks_per_sm * sm_count;
    if (blocks > tokens) blocks = tokens;
    void *args[] = {&router, &tokens, &experts, &topk, &selected, &counts, &offsets};
    if (blocks > 0)
        cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(topk_router_kernel),
            dim3(blocks), dim3(kThreads), args, smem_bytes, stream);
    build_grouped_rows<<<experts, kThreads, 0, stream>>>(
        selected, offsets, source_rows, grouped_assignments, m, topk);
    materialize_layout<<<experts, kThreads, 0, stream>>>(offsets, counts, experts, layout);
    invert_assignments<<<(m + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        grouped_assignments, m, assignment_to_grouped);
    sm110a_nvfp4_fc1_tasks_from_ranges(
        offsets, counts, tasks, state, experts, n, tile_m,
        tile_n * SM110A_NVFP4_FC1_CLUSTER_N, cluster_m, stream);
}
