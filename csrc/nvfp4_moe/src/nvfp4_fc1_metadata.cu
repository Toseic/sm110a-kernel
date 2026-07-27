#include <cstdint>
#include <cuda_runtime.h>

namespace {
constexpr int kThreads = 256;
#ifndef SM110A_NVFP4_FC1_MACRO_SWIZZLE_N_MAJOR
#define SM110A_NVFP4_FC1_MACRO_SWIZZLE_N_MAJOR 1
#endif
constexpr int kMacroSwizzleM = 8;
constexpr int kMacroSwizzleN = 8;

struct ExpertTileTask {
    int row_start, row_count, tile_row, expert, tile_n;
};

__global__ void collect_expert_metadata(
    const int32_t *layout,
    int m,
    int experts,
    int32_t *offsets,
    int32_t *counts) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m)
        return;
    const int expert = layout[row];
    if (expert < 0 || expert >= experts)
        return;
    atomicMin(offsets + expert, row);
    atomicAdd(counts + expert, 1);
}

__global__ void build_expert_tile_tasks(
    const int32_t *offsets,
    const int32_t *counts,
    int experts,
    int grid_n,
    int task_tile_rows,
    ExpertTileTask *tasks,
    int32_t *state) {
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= experts) return;
    const int rows = counts[expert];
    if (!rows) return;
    const int start = offsets[expert];
    const int m_tasks = (rows + task_tile_rows - 1) / task_tile_rows;
    int next = atomicAdd(state, m_tasks * grid_n);
        for (int group_row = 0; group_row < rows; group_row += kMacroSwizzleM * task_tile_rows) {
            const int group_end = min(group_row + kMacroSwizzleM * task_tile_rows, rows);
            for (int macro_n = 0; macro_n < grid_n; macro_n += kMacroSwizzleN) {
                const int band_end = min(macro_n + kMacroSwizzleN, grid_n);
#if SM110A_NVFP4_FC1_MACRO_SWIZZLE_N_MAJOR
                // Reuse the same expert B/SFB tile over adjacent M tiles.
                for (int tile_n = macro_n; tile_n < band_end; ++tile_n)
                    for (int tile_row = group_row; tile_row < group_end;
                         tile_row += task_tile_rows)
                        tasks[next++] = {start, rows, tile_row, expert, tile_n};
#else
                for (int tile_row = group_row; tile_row < group_end; tile_row += task_tile_rows)
                    for (int tile_n = macro_n; tile_n < band_end; ++tile_n)
                        tasks[next++] = {start, rows, tile_row, expert, tile_n};
#endif
            }
        }
}
} // namespace

extern "C" void sm110a_nvfp4_fc1_tasks_prepare(
    const int32_t *layout,
    int32_t *offsets,
    int32_t *counts,
    int32_t *tasks,
    int32_t *state,
    int m,
    int experts,
    int n,
    int tile_m,
    int tile_n,
    int cluster_m,
    bool swap_ab,
    cudaStream_t stream) {
    if (!layout || !offsets || !counts || !tasks || !state || m <= 0 || experts <= 0)
        return;
    const int task_tile_rows = swap_ab ? tile_n : cluster_m * tile_m;
    const int logical_n_tile = swap_ab ? cluster_m * tile_m : tile_n;
    if (n % logical_n_tile)
        return;
    cudaMemsetAsync(offsets, 0x7f, experts * sizeof(int32_t), stream);
    cudaMemsetAsync(counts, 0, experts * sizeof(int32_t), stream);
    cudaMemsetAsync(state, 0, 2 * sizeof(int32_t), stream);
    collect_expert_metadata<<<(m + kThreads - 1) / kThreads, kThreads, 0, stream>>>(layout,
        m,
        experts,
        offsets,
        counts);
    build_expert_tile_tasks<<<(experts + kThreads - 1) / kThreads, kThreads, 0, stream>>>(offsets,
        counts,
        experts,
        n / logical_n_tile,
        task_tile_rows,
        reinterpret_cast<ExpertTileTask *>(tasks),
        state);
}

extern "C" void sm110a_nvfp4_fc1_tasks_from_ranges(
    const int32_t *offsets, const int32_t *counts, int32_t *tasks,
    int32_t *state, int experts, int n, int tile_m, int tile_n,
    int cluster_m, cudaStream_t stream) {
    if (!offsets || !counts || !tasks || !state || experts <= 0 || n <= 0 ||
        tile_n <= 0 || n % tile_n) return;
    cudaMemsetAsync(state, 0, 2 * sizeof(int32_t), stream);
    build_expert_tile_tasks<<<(experts + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(offsets, counts, experts, n / tile_n,
        cluster_m * tile_m, reinterpret_cast<ExpertTileTask *>(tasks), state);
}
