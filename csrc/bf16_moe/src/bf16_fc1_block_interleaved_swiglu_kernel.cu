// Design goal for this kernel:
// 1. Compute fused FC1 + SwiGLU for each output element as:
//      gate = silu(acc_gate + bias_gate);
//      up   = acc_up + bias_up;
//      out  = gate * up;
// 2. Organize gate and up weights in a block-interleaved layout so the kernel
//    can consume both branches efficiently within the same GEMM pipeline.
// 3. Fuse the SwiGLU epilogue directly into the kernel, avoiding a separate
//    post-processing pass for bias add, SiLU, and elementwise multiply.
// 4. Keep the mainloop aligned with the grouped GEMM M implementation, and
//    build the fused gate/up path on top of that same scheduling structure.
// 5. Accept unexpanded token activations plus router scores, select top-k
//    experts, and gather token rows directly in the GEMM A-load pipeline.

#include <cmath>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdlib>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bf16_moe_fc1.h"

#ifndef SM110A_FC1_TILE_M
#define SM110A_FC1_TILE_M 128
#endif
#ifndef SM110A_FC1_TILE_N
#define SM110A_FC1_TILE_N 128
#endif
#ifndef SM110A_FC1_TILE_K
#define SM110A_FC1_TILE_K 128
#endif
#ifndef SM110A_FC1_CLUSTER_M
#define SM110A_FC1_CLUSTER_M 1
#endif
#ifndef SM110A_FC1_CLUSTER_N
#define SM110A_FC1_CLUSTER_N 1
#endif
#ifndef SM110A_FC1_DIRECT_GMEM
#define SM110A_FC1_DIRECT_GMEM 0
#endif

namespace sm110a::bf16_fc1_block_interleaved_swiglu {

constexpr int kMmaK = 16;
constexpr int kWarpSize = 32;
constexpr int kEpilogueWarpCount = 4;
constexpr int kProducerWarpCount = 4;
constexpr int kProducerWarpBase = kEpilogueWarpCount;
constexpr int kMmaWarp = kProducerWarpBase + kProducerWarpCount;
constexpr int kWarpCount = kMmaWarp + 1;
constexpr int kBlockThreads = kWarpSize * kWarpCount;
constexpr int kAProducerThreadCount = kProducerWarpCount * kWarpSize;
constexpr int kTmemStageCount = 2;
constexpr int kPersistentTileCount = 2;
constexpr int kMacroSwizzleM = 8;
// A four-wide physical-B band (12 MiB at N=256, K=6144) leaves room in L2
// for the concurrently reused routed-A tiles.  Larger bands made B-TMA
// completion visible to the consumer at the 32k-token persistent workload.
constexpr int kMacroSwizzleN = 4;
constexpr int kInterleaveBlock = 32;
constexpr int kRoutingThreads = 256;

namespace cg = cooperative_groups;

template <int kTileM, int kTileN, int kTileK, int kClusterM, bool kDirectGmem = false>
struct KernelConfig {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert(kTileM == 4 * kWarpSize);
    static_assert(kTileN >= 8 * kClusterM);
    static_assert(kTileN <= 256);
    static_assert(kTileN % (8 * kClusterM) == 0);
    static_assert(kTileN % kClusterM == 0);
    static_assert(kTileK % 64 == 0);
    static_assert(kTileK % kMmaK == 0);

    static constexpr int kClusterTileM = kClusterM * kTileM;
    static constexpr int kASmemBytes = kTileM * kTileK * sizeof(nv_bfloat16);
    static constexpr int kBSmemBytes = (kTileN / kClusterM) * kTileK * sizeof(nv_bfloat16);
    static constexpr int kPerStageSmemBytes = kASmemBytes + kBSmemBytes;
    static constexpr int kMaxSmemBytes = 227 * 1024;
    // Double-buffer two 32-column output blocks so TMEM -> SMEM can run ahead
    // of the SMEM -> TMA writeback. The direct path keeps the old footprint.
    static constexpr int kEpilogueTileN = 32;
    static constexpr int kEpilogueStageCount = 2;
    static constexpr int kEpilogueStageBytes = kTileM * kEpilogueTileN * sizeof(nv_bfloat16);
    static constexpr int kEpilogueSmemBytes =
        kDirectGmem ? 0 : kEpilogueStageCount * kEpilogueStageBytes;
    static constexpr int kStageCount =
        (kMaxSmemBytes - kEpilogueSmemBytes) / kPerStageSmemBytes;
    static_assert(kStageCount >= 2);
    static_assert(kStageCount * kPerStageSmemBytes + kEpilogueSmemBytes <= kMaxSmemBytes);
    static constexpr int kMainloopSmemBytes = kStageCount * kPerStageSmemBytes;
    static constexpr int kDynamicSmemBytes = kMainloopSmemBytes + kEpilogueSmemBytes;
    static constexpr int kTmemColumnsNeeded = kTileN * kTmemStageCount;
    static constexpr int kTmemAllocationColumns = kTmemColumnsNeeded <= 32    ? 32
                                                  : kTmemColumnsNeeded <= 64  ? 64
                                                  : kTmemColumnsNeeded <= 128 ? 128
                                                  : kTmemColumnsNeeded <= 256 ? 256
                                                                              : 512;
    static_assert(kTmemColumnsNeeded <= 512);
    static constexpr int kStageSmemBytes = kPerStageSmemBytes;
};

struct ExpertTileTask {
    int row_start;
    int row_count;
    int tile_row;
    int expert;
    int tile_n;
};

__device__ inline float silu(
    float x) {
    return x / (1.0f + expf(-x));
}
__device__ inline float router_score_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__global__ void topk_router_kernel(
    const __nv_bfloat16 *router_scores,
    int tokens,
    int experts,
    int topk,
    int *selected_experts,
    int *expert_counts,
    int *expert_offsets) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ char routing_smem[];
    auto *reduce_scores = reinterpret_cast<float *>(routing_smem);
    auto *reduce_experts = reinterpret_cast<int *>(reduce_scores + blockDim.x);

    for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
        for (int rank = 0; rank < topk; ++rank) {
            float best_score = -INFINITY;
            int best_expert = -1;
            for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
                bool already_chosen = false;
                for (int previous = 0; previous < rank; ++previous) {
                    already_chosen |=
                        selected_experts[static_cast<size_t>(token) * topk + previous] == expert;
                }
                if (already_chosen) {
                    continue;
                }
                float score = router_score_to_float(
                    router_scores[static_cast<size_t>(token) * experts + expert]);
                if (isnan(score)) {
                    score = -INFINITY;
                }
                if (best_expert < 0 || score > best_score ||
                    (score == best_score && (best_expert < 0 || expert < best_expert))) {
                    best_score = score;
                    best_expert = expert;
                }
            }
            reduce_scores[threadIdx.x] = best_score;
            reduce_experts[threadIdx.x] = best_expert;
            __syncthreads();

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    const float other_score = reduce_scores[threadIdx.x + stride];
                    const int other_expert = reduce_experts[threadIdx.x + stride];
                    const float own_score = reduce_scores[threadIdx.x];
                    const int own_expert = reduce_experts[threadIdx.x];
                    if (other_expert >= 0 &&
                        (own_expert < 0 || other_score > own_score ||
                            (other_score == own_score &&
                                (own_expert < 0 || other_expert < own_expert)))) {
                        reduce_scores[threadIdx.x] = other_score;
                        reduce_experts[threadIdx.x] = other_expert;
                    }
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                const int chosen = reduce_experts[0];
                selected_experts[static_cast<size_t>(token) * topk + rank] = chosen;
                atomicAdd(expert_counts + chosen, 1);
            }
            __syncthreads();
        }
    }

    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int offset = 0;
        for (int expert = 0; expert < experts; ++expert) {
            expert_offsets[expert] = offset;
            offset += expert_counts[expert];
        }
    }
    grid.sync();
}

__global__ void build_grouped_row_map_kernel(
    const int *__restrict__ selected_experts,
    const int *__restrict__ expert_offsets,
    int *__restrict__ source_rows,
    int *__restrict__ grouped_assignments,
    int m,
    int topk) {
    constexpr int kWarpsPerBlock = kRoutingThreads / kWarpSize;
    const int expert = blockIdx.x;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    __shared__ int warp_offsets[kWarpsPerBlock];
    __shared__ int rows_written;
    __shared__ int block_matches;

    if (threadIdx.x == 0) {
        rows_written = expert_offsets[expert];
    }
    __syncthreads();

    // Stable block-wide compaction: tiles and lanes are visited in increasing
    // assignment order, preserving (token, top-k rank) within each expert.
    for (int base = 0; base < m; base += kRoutingThreads) {
        const int assignment = base + threadIdx.x;
        const bool matches = assignment < m && selected_experts[assignment] == expert;
        const uint32_t match_mask = __ballot_sync(0xFFFFFFFFu, matches);
        const uint32_t lower_lane_mask = (uint32_t{1} << lane) - 1;
        const int warp_prefix = __popc(match_mask & lower_lane_mask);
        if (lane == 0) {
            warp_offsets[warp] = __popc(match_mask);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            int running = 0;
            for (int i = 0; i < kWarpsPerBlock; ++i) {
                const int count = warp_offsets[i];
                warp_offsets[i] = running;
                running += count;
            }
            block_matches = running;
        }
        __syncthreads();

        if (matches) {
            const int grouped_row = rows_written + warp_offsets[warp] + warp_prefix;
            source_rows[grouped_row] = assignment / topk;
            if (grouped_assignments != nullptr)
                grouped_assignments[grouped_row] = assignment;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            rows_written += block_matches;
        }
        __syncthreads();
    }
}

__global__ void materialize_grouped_layout_kernel(
    const int *expert_offsets, const int *expert_counts, int experts, int *grouped_layout) {
    const int expert = blockIdx.x;
    if (expert >= experts)
        return;
    for (int row = expert_offsets[expert] + threadIdx.x;
         row < expert_offsets[expert] + expert_counts[expert]; row += blockDim.x)
        grouped_layout[row] = expert;
}

__global__ void invert_grouped_assignments_kernel(
    const int *grouped_assignments, int m, int *assignment_to_grouped) {
    const int grouped_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (grouped_row < m)
        assignment_to_grouped[grouped_assignments[grouped_row]] = grouped_row;
}

template <int kTaskTileRows>
__global__ void build_expert_tile_tasks(
    const int *expert_offsets,
    const int *expert_counts,
    int experts,
    int grid_n,
    ExpertTileTask *tasks,
    int *task_count) {
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= experts) {
        return;
    }
    const int row_count = expert_counts[expert];
    if (row_count == 0) {
        return;
    }
    const int row_start = expert_offsets[expert];
    const int m_tasks = (row_count + kTaskTileRows - 1) / kTaskTileRows;
    // Reserve one contiguous queue span per expert, then traverse a compact
    // M-group/N-band working set.  Keeping tile_n in the task makes the device
    // kernel consume this locality-preserving order without remapping it.
    int next_task = atomicAdd(task_count, m_tasks * grid_n);
    for (int group_row = 0; group_row < row_count; group_row += kMacroSwizzleM * kTaskTileRows) {
        const int group_row_end = group_row + kMacroSwizzleM * kTaskTileRows < row_count
                                      ? group_row + kMacroSwizzleM * kTaskTileRows
                                      : row_count;
        for (int macro_n = 0; macro_n < grid_n; macro_n += kMacroSwizzleN) {
            const int band_end =
                macro_n + kMacroSwizzleN < grid_n ? macro_n + kMacroSwizzleN : grid_n;
            for (int tile_row = group_row; tile_row < group_row_end; tile_row += kTaskTileRows) {
                for (int tile_n = macro_n; tile_n < band_end; ++tile_n) {
                    tasks[next_task++] = {row_start, row_count, tile_row, expert, tile_n};
                }
            }
        }
    }
}

__device__ inline uint32_t elect_sync() {
    uint32_t elected = 0;
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "elect.sync _|p, %1;\n\t"
                 "@p mov.s32 %0, 1;\n\t"
                 "}"
        : "+r"(elected)
        : "r"(0xFFFFFFFF));
    return elected;
}

__device__ inline constexpr uint64_t matrix_desc_encode(
    uint64_t value) {
    return (value & 0x3FFFFULL) >> 4ULL;
}

__device__ inline uint64_t make_k_major_128b_desc(
    int smem_addr) {
    constexpr int kStrideByteOffset = 8 * 64 * sizeof(nv_bfloat16);
    return matrix_desc_encode(smem_addr) | (matrix_desc_encode(kStrideByteOffset) << 32ULL) |
           (1ULL << 46ULL) | (2ULL << 61ULL);
}

// True asynchronous A gather: GMEM --cp.async--> SMEM.  Each producer
// thread associates its preceding copies with the stage A-full mbarrier.
__device__ __forceinline__ void cp_async_a_16b(
    int dst_smem,
    const void *src_gmem,
    bool valid) {
    const uint32_t ignore_src = valid ? 0u : 1u;
    asm volatile("{\n\t"
                 ".reg .pred p_ignore_src;\n\t"
                 "setp.ne.u32 p_ignore_src, %2, 0;\n\t"
                 "cp.async.ca.shared::cta.global [%0], [%1], 16, p_ignore_src;\n\t"
                 "}\n"
        :
        : "r"(dst_smem), "l"(src_gmem), "r"(ignore_src)
        : "memory");
}

__device__ __forceinline__ void cp_async_a_mbarrier_arrive_noinc(
    int mbar_addr) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"
        :
        : "r"(mbar_addr)
        : "memory");
}

template <int kTileM, int kTileK>
__device__ __forceinline__ void load_gather_a_tile(
    const nv_bfloat16 *__restrict__ a,
    const int *__restrict__ task_source_rows,
    int valid_rows,
    int tile_k,
    int k,
    char *__restrict__ a_smem) {
    constexpr int kElementsPerVector = sizeof(uint4) / sizeof(nv_bfloat16);
    constexpr int kVectorsPerKSlab = 64 / kElementsPerVector;
    constexpr int kKSlabCount = kTileK / 64;
    constexpr int kRowsPerIssue = kWarpSize / kVectorsPerKSlab;
    constexpr int kRowsPerProducerWarp = kTileM / kProducerWarpCount;
    constexpr int kRowGroupCount = kRowsPerProducerWarp / kRowsPerIssue;

    static_assert(kElementsPerVector == 8);
    static_assert(kVectorsPerKSlab == 8);
    static_assert(kTileK % 64 == 0);
    static_assert(kTileM % kProducerWarpCount == 0);
    static_assert(kRowsPerProducerWarp % kRowsPerIssue == 0);
    static_assert(kRowsPerProducerWarp == kRowGroupCount * kRowsPerIssue);

    const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
    const int producer_rank = warp_id - kProducerWarpBase;
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int row_in_group = lane / kVectorsPerKSlab;
    const int vector_in_slab = lane % kVectorsPerKSlab;
    const int producer_row_begin = producer_rank * kRowsPerProducerWarp;

#pragma unroll
    for (int k_slab = 0; k_slab < kKSlabCount; ++k_slab) {
#pragma unroll
        for (int row_group = 0; row_group < kRowGroupCount; ++row_group) {
            const int row = producer_row_begin + row_group * kRowsPerIssue + row_in_group;
            const int logical_vector =
                k_slab * kTileM * kVectorsPerKSlab + row * kVectorsPerKSlab + vector_in_slab;
            const int swizzled_vector =
                (logical_vector & ~7) + ((logical_vector & 7) ^ ((logical_vector >> 3) & 7));
            const int dst_smem = static_cast<int>(__cvta_generic_to_shared(
                a_smem + static_cast<size_t>(swizzled_vector) * sizeof(uint4)));
            const bool valid = row < valid_rows;
            const nv_bfloat16 *src_gmem = a;
            if (valid) {
                const int source_row = task_source_rows[row];
                const size_t source_element =
                    static_cast<size_t>(source_row) * static_cast<size_t>(k) +
                    static_cast<size_t>(tile_k) * static_cast<size_t>(kTileK) +
                    static_cast<size_t>(k_slab) * 64 +
                    static_cast<size_t>(vector_in_slab) * kElementsPerVector;
                src_gmem = a + source_element;
            }
            cp_async_a_16b(dst_smem, src_gmem, valid);
        }
    }
}

__device__ inline void mbarrier_init(
    int mbar_addr,
    int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr), "r"(count));
}

// Analysis-only split: these helpers deliberately keep identical PTX while
// giving SourceCounters a distinct line-info location for each pipeline edge.
// Keep their operands, ordering, and suspend interval in sync.
__device__ inline void mbarrier_wait_mma_reuse(
    int mbar_addr,
    int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred done;\n\t"
                 "WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE;\n\t"
                 "bra.uni WAIT;\n\t"
                 "COMPLETE:\n\t"
                 "}" ::"r"(mbar_addr),
        "r"(phase),
        "r"(kSuspendTicks)
        : "memory");
}

__device__ inline void mbarrier_wait_producer_a_ready(
    int mbar_addr,
    int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred done;\n\t"
                 "WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE;\n\t"
                 "bra.uni WAIT;\n\t"
                 "COMPLETE:\n\t"
                 "}" :: "r"(mbar_addr),
        "r"(phase),
        "r"(kSuspendTicks)
        : "memory");
}

__device__ inline void mbarrier_wait_forward_a_ready(
    int mbar_addr,
    int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred done;\n\t"
                 "WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE;\n\t"
                 "bra.uni WAIT;\n\t"
                 "COMPLETE:\n\t"
                 "}" :: "r"(mbar_addr),
        "r"(phase),
        "r"(kSuspendTicks)
        : "memory");
}

__device__ inline void mbarrier_wait_epilogue_slot(
    int mbar_addr,
    int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred done;\n\t"
                 "WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE;\n\t"
                 "bra.uni WAIT;\n\t"
                 "COMPLETE:\n\t"
                 "}" :: "r"(mbar_addr),
        "r"(phase),
        "r"(kSuspendTicks)
        : "memory");
}

__device__ inline void mbarrier_wait_consumer_a_ready(
    int mbar_addr,
    int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred done;\n\t"
                 "WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE;\n\t"
                 "bra.uni WAIT;\n\t"
                 "COMPLETE:\n\t"
                 "}" :: "r"(mbar_addr),
        "r"(phase),
        "r"(kSuspendTicks)
        : "memory");
}

__device__ inline void mbarrier_wait_consumer_cluster_a_ready(
    int mbar_addr,
    int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred done;\n\t"
                 "WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE;\n\t"
                 "bra.uni WAIT;\n\t"
                 "COMPLETE:\n\t"
                 "}" :: "r"(mbar_addr),
        "r"(phase),
        "r"(kSuspendTicks)
        : "memory");
}

__device__ inline void mbarrier_wait_tma_b_ready(
    int mbar_addr,
    int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred done;\n\t"
                 "WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE;\n\t"
                 "bra.uni WAIT;\n\t"
                 "COMPLETE:\n\t"
                 "}" :: "r"(mbar_addr),
        "r"(phase),
        "r"(kSuspendTicks)
        : "memory");
}

__device__ inline void mbarrier_wait_mainloop_output(
    int mbar_addr,
    int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred done;\n\t"
                 "WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE;\n\t"
                 "bra.uni WAIT;\n\t"
                 "COMPLETE:\n\t"
                 "}" :: "r"(mbar_addr),
        "r"(phase),
        "r"(kSuspendTicks)
        : "memory");
}

__device__ inline void clc_mbarrier_wait(
    int mbar_addr,
    int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred done;\n\t"
                 "WAIT_CLC:\n\t"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE_CLC;\n\t"
                 "bra.uni WAIT_CLC;\n\t"
                 "COMPLETE_CLC:\n\t"
                 "}" :: "r"(mbar_addr),
        "r"(phase),
        "r"(kSuspendTicks)
        : "memory");
}

// Arrive on an mbarrier hosted by another CTA in this cluster.  This is used
// to aggregate the independently completed per-CTA A gathers before CTA 0
// issues a cta_group::2 MMA.
__device__ inline void mbarrier_arrive_on_cluster_cta(
    int local_mbar_addr,
    int target_cta_rank) {
    int remote_mbar_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote_mbar_addr)
        : "r"(local_mbar_addr), "r"(target_cta_rank));
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 "
                 "_, [%0];"
        :
        : "r"(remote_mbar_addr)
        : "memory");
}

__device__ inline int map_shared_to_cluster_cta(
    int local_smem_addr,
    int target_cta_rank) {
    int remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote_addr)
        : "r"(local_smem_addr), "r"(target_cta_rank));
    return remote_addr;
}

template <int kBytes>
__device__ inline void bulk_copy_a_stage_to_cluster_cta(
    int src_smem,
    int dst_smem,
    int remote_mbar_addr,
    int target_cta_rank) {
    static_assert(kBytes % 16 == 0);
    const int remote_dst = map_shared_to_cluster_cta(dst_smem, target_cta_rank);
    const int remote_mbar = map_shared_to_cluster_cta(remote_mbar_addr, target_cta_rank);
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
                 "_, [%0], %1;" ::"r"(remote_mbar),
        "r"(kBytes)
        : "memory");
    asm volatile("cp.async.bulk.shared::cluster.shared::cta."
                 "mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];" ::"r"(remote_dst),
        "r"(src_smem),
        "r"(kBytes),
        "r"(remote_mbar)
        : "memory");
}

template <int kClusterM>
__device__ inline void tma_load_4d(
    int dst_smem,
    const void *tensor_map,
    int x,
    int y,
    int z,
    int w,
    int mbar_addr) {
    if constexpr (kClusterM == 1) {
        asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global."
                     "mbarrier::complete_tx::bytes.cta_group::1 "
                     "[%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst_smem),
            "l"(tensor_map),
            "r"(x),
            "r"(y),
            "r"(z),
            "r"(w),
            "r"(mbar_addr)
            : "memory");
    } else {
        asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global."
                     "mbarrier::complete_tx::bytes.cta_group::2 "
                     "[%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst_smem),
            "l"(tensor_map),
            "r"(x),
            "r"(y),
            "r"(z),
            "r"(w),
            "r"(mbar_addr)
            : "memory");
    }
}

__device__ inline void tma_store_3d(
    int src_smem,
    const void *tensor_map,
    int x,
    int y,
    int z) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
                 "[%0, {%1, %2, %3}], [%4];" ::"l"(tensor_map),
        "r"(x),
        "r"(y),
        "r"(z),
        "r"(src_smem)
        : "memory");
}

__device__ inline void tma_store_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ inline void prefetch_tensor_map(
    const void *tensor_map) {
    const uint64_t ptr = reinterpret_cast<uint64_t>(tensor_map);
    asm volatile("prefetch.tensormap [%0];" ::"l"(ptr) : "memory");
}

template <int kClusterM>
__device__ inline void tcgen05_mma_bf16(
    int tmem_addr,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t instruction_desc,
    int accumulate) {
    if constexpr (kClusterM == 1) {
        asm volatile("{\n\t"
                     ".reg .pred input_d;\n\t"
                     "setp.ne.b32 input_d, %4, 0;\n\t"
                     "tcgen05.mma.cta_group::1.kind::f16 "
                     "[%0], %1, %2, %3, input_d;\n\t"
                     "}" ::"r"(tmem_addr),
            "l"(a_desc),
            "l"(b_desc),
            "r"(instruction_desc),
            "r"(accumulate));
    } else {
        asm volatile("{\n\t"
                     ".reg .pred input_d;\n\t"
                     "setp.ne.b32 input_d, %4, 0;\n\t"
                     "tcgen05.mma.cta_group::2.kind::f16 "
                     "[%0], %1, %2, %3, input_d;\n\t"
                     "}" ::"r"(tmem_addr),
            "l"(a_desc),
            "l"(b_desc),
            "r"(instruction_desc),
            "r"(accumulate));
    }
}

template <int kClusterM>
__device__ inline void tcgen05_commit(
    int mbar_addr,
    int cta_rank) {
    if constexpr (kClusterM == 1) {
        asm volatile("tcgen05.commit.cta_group::1."
                     "mbarrier::arrive::one.shared::cluster.b64 "
                     "[%0];" ::"r"(mbar_addr)
            : "memory");
    } else {
        const uint16_t pair_mask = static_cast<uint16_t>(0x3u << (cta_rank & ~1));
        asm volatile("tcgen05.commit.cta_group::2."
                     "mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
                     "[%0], %1;" ::"r"(mbar_addr),
            "h"(pair_mask)
            : "memory");
    }
}

template <int kTileM, int kTileK>
inline bool init_a_tensor_map(
    CUtensorMap *tensor_map,
    const nv_bfloat16 *ptr,
    uint64_t m,
    uint64_t k) {
    constexpr uint32_t rank = 4;
    uint64_t global_dim[rank] = {64, m, k / 64, 1};
    uint64_t global_stride[rank - 1] = {k * sizeof(nv_bfloat16), 128, m * k * sizeof(nv_bfloat16)};
    uint32_t box_dim[rank] = {64, kTileM, kTileK / 64, 1};
    uint32_t element_stride[rank] = {1, 1, 1, 1};

    return cuTensorMapEncodeTiled(tensor_map,
               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
               rank,
               const_cast<nv_bfloat16 *>(ptr),
               global_dim,
               global_stride,
               box_dim,
               element_stride,
               CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_128B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <int kTileN, int kTileK, int kClusterM>
inline bool init_b_tensor_map(
    CUtensorMap *tensor_map,
    const nv_bfloat16 *ptr,
    uint64_t experts,
    uint64_t n,
    uint64_t k) {
    constexpr uint32_t rank = 4;
    uint64_t global_dim[rank] = {64, n, k / 64, experts};
    uint64_t global_stride[rank - 1] = {k * sizeof(nv_bfloat16), 128, n * k * sizeof(nv_bfloat16)};
    uint32_t box_dim[rank] = {64, kTileN / kClusterM, kTileK / 64, 1};
    uint32_t element_stride[rank] = {1, 1, 1, 1};

    return cuTensorMapEncodeTiled(tensor_map,
               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
               rank,
               const_cast<nv_bfloat16 *>(ptr),
               global_dim,
               global_stride,
               box_dim,
               element_stride,
               CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_128B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <int kTileM, int kEpilogueTileN>
inline bool init_c_tensor_map(
    CUtensorMap *tensor_map,
    nv_bfloat16 *ptr,
    uint64_t m,
    uint64_t n) {
    constexpr uint32_t rank = 3;
    uint64_t global_dim[rank] = {n, m, 1};
    uint64_t global_stride[rank - 1] = {n * sizeof(nv_bfloat16), m * n * sizeof(nv_bfloat16)};
    uint32_t box_dim[rank] = {kEpilogueTileN, kTileM, 1};
    uint32_t element_stride[rank] = {1, 1, 1};

    return cuTensorMapEncodeTiled(tensor_map,
               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
               rank,
               ptr,
               global_dim,
               global_stride,
               box_dim,
               element_stride,
               CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_64B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <int kTileM, int kTileN, int kTileK, int kClusterM, int kClusterN, bool kDirectGmem>
__global__ __cluster_dims__(kClusterM, kClusterN, 1)
__launch_bounds__(kBlockThreads) void fc1_block_interleaved_swiglu_kernel(
    const nv_bfloat16 *__restrict__ a,
    const int *__restrict__ source_rows,
    const __grid_constant__ CUtensorMap b_tensor_map,
    const __grid_constant__ CUtensorMap c_tensor_map,
    const ExpertTileTask *tasks,
    const int *task_count,
    nv_bfloat16 *out,
    int n,
    int k) {
    using Config = KernelConfig<kTileM, kTileN, kTileK, kClusterM, kDirectGmem>;
    static_assert(kClusterN == 1 || kClusterN == 2);
    static_assert(kClusterN == 1 || kClusterM == 2);
    constexpr int kClusterTileM = Config::kClusterTileM;
    constexpr int kStageCount = Config::kStageCount;
    constexpr int kASmemBytes = Config::kASmemBytes;
    constexpr int kBSmemBytes = Config::kBSmemBytes;
    constexpr int kStageSmemBytes = Config::kStageSmemBytes;
    constexpr int kMainloopSmemBytes = Config::kMainloopSmemBytes;
    constexpr int kEpilogueTileN = Config::kEpilogueTileN;
    constexpr int kEpilogueStageCount = Config::kEpilogueStageCount;
    constexpr int kEpilogueStageBytes = Config::kEpilogueStageBytes;
    constexpr int kMmaStepsPerStage = kTileK / kMmaK;
    constexpr int kTmemAllocationColumns = Config::kTmemAllocationColumns;
    constexpr int kLogicalTileN = kTileN / 2;
    static_assert(kTileN % 2 == 0);
    static_assert(kTileN % (2 * kInterleaveBlock) == 0);
    static_assert(kLogicalTileN % kEpilogueTileN == 0);

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int m_rank = cta_rank % kClusterM;
    const int n_rank = cta_rank / kClusterM;

    const int worker_id = blockIdx.x / kClusterM;
    const int cluster_tiles = *task_count;
    constexpr int kWorkerTileCount = kPersistentTileCount;
    const int static_first_cluster_bid = worker_id * kWorkerTileCount;

    __shared__ int tmem_base[1];
    __shared__ int next_work_smem[2];
    __shared__ __align__(16) uint4 clc_response[1];
    __shared__ __align__(8) uint64_t clc_mbar[1];
    __shared__ int task_source_rows[kTileM];
    __shared__ __align__(8) uint64_t a_mbars[kStageCount];
    // Hosted by CTA 0.  For ClusterM=2, each CTA arrives after its local
    // A-full barrier completes, allowing CTA 0 to wait for both A tiles.
    __shared__ __align__(8) uint64_t a_cluster_mbars[kStageCount];
    __shared__ __align__(8) uint64_t tma_mbars[kStageCount];
    __shared__ __align__(8) uint64_t mma_mbars[kStageCount];
    __shared__ __align__(8) uint64_t mainloop_mbars[kTmemStageCount];
    __shared__ __align__(8) uint64_t epilogue_mbars[kTmemStageCount];

    extern __shared__ __align__(1024) char dynamic_smem[];
    const int mainloop_smem = static_cast<int>(__cvta_generic_to_shared(dynamic_smem));
    const int base_a_mbar_addr = static_cast<int>(__cvta_generic_to_shared(a_mbars));
    const int base_a_cluster_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(a_cluster_mbars));
    const int base_tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(tma_mbars));
    const int base_mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mma_mbars));
    const int base_mainloop_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mainloop_mbars));
    const int base_epilogue_mbar_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_mbars));

    if (tid == 0) {
        static_assert(kAProducerThreadCount == 128);
        for (int stage = 0; stage < kStageCount; ++stage) {
            mbarrier_init(base_a_mbar_addr + stage * sizeof(uint64_t),
                n_rank == 0 ? kAProducerThreadCount : 1);
            if constexpr (kClusterM > 1) {
                if (m_rank == 0) {
                    mbarrier_init(base_a_cluster_mbar_addr + stage * sizeof(uint64_t), kClusterM);
                }
            }
            mbarrier_init(base_tma_mbar_addr + stage * sizeof(uint64_t), kClusterM);
            int mma_arrival_count = 1;
            if constexpr (kClusterN == 2) {
                mma_arrival_count = n_rank == 0 ? 2 : 1;
            }
            mbarrier_init(
                base_mma_mbar_addr + stage * sizeof(uint64_t), mma_arrival_count);
        }
        for (int stage = 0; stage < kTmemStageCount; ++stage) {
            mbarrier_init(base_mainloop_mbar_addr + stage * sizeof(uint64_t), 1);
            mbarrier_init(base_epilogue_mbar_addr + stage * sizeof(uint64_t), 4 * kClusterM);
        }
        mbarrier_init(static_cast<int>(__cvta_generic_to_shared(clc_mbar)), 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp_id == kMmaWarp) {
        const int tmem_base_smem = static_cast<int>(__cvta_generic_to_shared(tmem_base));
        if constexpr (kClusterM == 1) {
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
                         "[%0], %1;" ::"r"(tmem_base_smem),
                "r"(kTmemAllocationColumns));
        } else {
            asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 "
                         "[%0], %1;" ::"r"(tmem_base_smem),
                "r"(kTmemAllocationColumns));
        }
    }
    if (warp_id == 0 && elect_sync()) {
        prefetch_tensor_map(&b_tensor_map);
        if constexpr (!kDirectGmem) {
            prefetch_tensor_map(&c_tensor_map);
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    const int tmem_addr = tmem_base[0];
    constexpr uint32_t instruction_desc = (1U << 4U) | (1U << 7U) | (1U << 10U) |
                                          (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
                                          (static_cast<uint32_t>(kClusterTileM) >> 4U << 24U);

    const int k_tiles = k / kTileK;

    auto tmem_to_smem = [&](int mainloop_stage, int n_block, int smem_stage, int valid_rows) {
        char *stage_smem = dynamic_smem + kMainloopSmemBytes + smem_stage * kEpilogueStageBytes;
        const int tmem_row = m_rank * kTileM + warp_id * kWarpSize;
        const int stage_tmem_addr = tmem_addr + mainloop_stage * kTileN;

        if (warp_id * kWarpSize < valid_rows) {
            for (int logical_col = n_block * kEpilogueTileN;
                logical_col < (n_block + 1) * kEpilogueTileN;
                logical_col += 8) {
                const int logical_block = logical_col / kInterleaveBlock;
                const int logical_in_block = logical_col % kInterleaveBlock;
                const int gate_tmem_col = logical_block * (2 * kInterleaveBlock) + logical_in_block;
                const int up_tmem_col = gate_tmem_col + kInterleaveBlock;
                float gate_vals[8];
                float up_vals[8];
                const int gate_load_addr = stage_tmem_addr + (tmem_row << 16) + gate_tmem_col;
                const int up_load_addr = stage_tmem_addr + (tmem_row << 16) + up_tmem_col;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                             "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=f"(gate_vals[0]),
                    "=f"(gate_vals[1]),
                    "=f"(gate_vals[2]),
                    "=f"(gate_vals[3]),
                    "=f"(gate_vals[4]),
                    "=f"(gate_vals[5]),
                    "=f"(gate_vals[6]),
                    "=f"(gate_vals[7])
                    : "r"(gate_load_addr));
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                             "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=f"(up_vals[0]),
                    "=f"(up_vals[1]),
                    "=f"(up_vals[2]),
                    "=f"(up_vals[3]),
                    "=f"(up_vals[4]),
                    "=f"(up_vals[5]),
                    "=f"(up_vals[6]),
                    "=f"(up_vals[7])
                    : "r"(up_load_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");

                nv_bfloat162 packed_out[4];
                for (int i = 0; i < 4; ++i) {
                    packed_out[i] = __float22bfloat162_rn({silu(gate_vals[2 * i]) * up_vals[2 * i],
                        silu(gate_vals[2 * i + 1]) * up_vals[2 * i + 1]});
                }
                const int group = (logical_col - n_block * kEpilogueTileN) / 8;
                const int chunk = tid * (kEpilogueTileN / 8) + group;
                const int swizzled_chunk = (chunk & ~7) + ((chunk & 7) ^ ((chunk >> 3) & 3));
                if (tid < valid_rows) {
                    reinterpret_cast<uint4 *>(stage_smem)[swizzled_chunk] = {
                        reinterpret_cast<uint32_t *>(&packed_out[0])[0],
                        reinterpret_cast<uint32_t *>(&packed_out[1])[0],
                        reinterpret_cast<uint32_t *>(&packed_out[2])[0],
                        reinterpret_cast<uint32_t *>(&packed_out[3])[0]};
                }
            }
        }
    };

    auto smem_to_gmem =
        [&](int n_block, int smem_stage, int cta_row, int output_tile_col, int valid_rows) {
            if (tid < valid_rows) {
                const char *stage_smem =
                    dynamic_smem + kMainloopSmemBytes + smem_stage * kEpilogueStageBytes;
                const int block_col = output_tile_col + n_block * kEpilogueTileN;
                for (int group = 0; group < kEpilogueTileN / 8; ++group) {
                    const int chunk = tid * (kEpilogueTileN / 8) + group;
                    const int swizzled_chunk = (chunk & ~7) + ((chunk & 7) ^ ((chunk >> 3) & 3));
                    reinterpret_cast<uint4 *>(out + (cta_row + tid) * n + block_col)[group] =
                        reinterpret_cast<const uint4 *>(stage_smem)[swizzled_chunk];
                }
            }
        };

    int producer_stage = 0;
    int producer_a_phase = 0;
    int producer_mma_phase = 1;
    int consumer_stage = 0;
    int consumer_a_phase = 0;
    int consumer_tma_phase = 0;
    int consumer_mainloop_stage = 0;
    int consumer_epilogue_phase = 1;
    int epilogue_mainloop_stage = 0;
    int epilogue_mainloop_phase = 0;
    int cached_source_row_start = -1;
    int clc_phase = 0;
    int dynamic_first_cluster_bid = static_first_cluster_bid;
    while (true) {
        const int first_cluster_bid = dynamic_first_cluster_bid;

        // Submit one cancellation request for the entire cluster. The response
        // is multicast to every CTA, so all ranks transition to the same stolen
        // cluster task after completing their current work.
        const int clc_response_addr = static_cast<int>(__cvta_generic_to_shared(clc_response));
        const int clc_mbar_addr = static_cast<int>(__cvta_generic_to_shared(clc_mbar));
        if (tid == 0) {
            asm volatile("mbarrier.arrive.expect_tx.relaxed.cluster."
                         "shared::cta.b64 _, [%0], 16;" :: "r"(clc_mbar_addr)
                : "memory");
        }
        asm volatile("barrier.cluster.arrive.relaxed.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
        if (cta_rank == 0 && tid == 0) {
            asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta."
                         "mbarrier::complete_tx::bytes.multicast::cluster::all."
                         "b128 [%0], [%1];" :: "r"(clc_response_addr),
                "r"(clc_mbar_addr)
                : "memory");
        }
        const bool has_work = first_cluster_bid < cluster_tiles;
        const int last_cluster_bid = has_work
                                         ? (first_cluster_bid + kWorkerTileCount < cluster_tiles
                                                   ? first_cluster_bid + kWorkerTileCount
                                                   : cluster_tiles)
                                         : first_cluster_bid;

        if (warp_id >= kProducerWarpBase && warp_id < kMmaWarp) {
            const int producer_rank = warp_id - kProducerWarpBase;
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                 ++cluster_bid) {
                const ExpertTileTask task = tasks[cluster_bid];
                const int tile_n = task.tile_n * kClusterN + n_rank;
                const int cta_row = task.row_start + task.tile_row + m_rank * kTileM;
                const int cta_tile_row = task.tile_row + m_rank * kTileM;
                const int remaining_rows = task.row_count - cta_tile_row;
                const int valid_rows =
                    remaining_rows < 0 ? 0 : (remaining_rows < kTileM ? remaining_rows : kTileM);
                const int tile_col = tile_n * kTileN;

                if constexpr (kClusterN == 1) {
                    if (cta_row != cached_source_row_start) {
                        constexpr int kRowsPerProducerWarp = kTileM / kProducerWarpCount;
                        const int lane = tid % kWarpSize;
                        if (lane < kRowsPerProducerWarp) {
                            const int row = producer_rank * kRowsPerProducerWarp + lane;
                            task_source_rows[row] =
                                row < valid_rows ? __ldg(source_rows + cta_row + row) : 0;
                        }
                        __syncwarp();
                        cached_source_row_start = cta_row;
                    }
                } else if (n_rank == 0 && cta_row != cached_source_row_start) {
                    constexpr int kRowsPerProducerWarp = kTileM / kProducerWarpCount;
                    const int lane = tid % kWarpSize;
                    if (lane < kRowsPerProducerWarp) {
                        const int row = producer_rank * kRowsPerProducerWarp + lane;
                        task_source_rows[row] =
                            row < valid_rows ? __ldg(source_rows + cta_row + row) : 0;
                    }
                    __syncwarp();
                    cached_source_row_start = cta_row;
                }

                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    const int stage = producer_stage;
                    const int mma_mbar_addr = base_mma_mbar_addr + stage * sizeof(uint64_t);
                    const int a_mbar_addr = base_a_mbar_addr + stage * sizeof(uint64_t);
                    const int tma_mbar_addr = base_tma_mbar_addr + stage * sizeof(uint64_t);
                    const int a_smem = mainloop_smem + stage * kStageSmemBytes;
                    const int b_smem = a_smem + kASmemBytes;

                    // Acquire the stage only after tcgen05 has consumed A and B.
                    mbarrier_wait_mma_reuse(mma_mbar_addr, producer_mma_phase);

                    // Start B TMA first so it overlaps the A LDGSTS traffic.
                    if (producer_rank == 0 && elect_sync()) {
                        const int pair_root_rank = cta_rank & ~1;
                        const int tma_transaction_mbar_addr =
                            map_shared_to_cluster_cta(tma_mbar_addr, pair_root_rank);
                        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
                                     "_, [%0], %1;" ::"r"(tma_transaction_mbar_addr),
                            "r"(kBSmemBytes)
                            : "memory");

                        const int k_slice = tile_k * (kTileK / 64);
                        tma_load_4d<kClusterM>(b_smem,
                            &b_tensor_map,
                            0,
                            tile_col + m_rank * (kTileN / kClusterM),
                            k_slice,
                            task.expert,
                            tma_transaction_mbar_addr);
                    }

                    if constexpr (kClusterN == 1) {
                        load_gather_a_tile<kTileM, kTileK>(a,
                            task_source_rows,
                            valid_rows,
                            tile_k,
                            k,
                            dynamic_smem + stage * kStageSmemBytes);
                        cp_async_a_mbarrier_arrive_noinc(a_mbar_addr);
                    } else if (n_rank == 0) {
                        load_gather_a_tile<kTileM, kTileK>(a,
                            task_source_rows,
                            valid_rows,
                            tile_k,
                            k,
                            dynamic_smem + stage * kStageSmemBytes);
                        cp_async_a_mbarrier_arrive_noinc(a_mbar_addr);
                    } else {
                        mbarrier_wait_producer_a_ready(a_mbar_addr, producer_a_phase);
                        if (producer_rank == 0 && elect_sync()) {
                            const int source_cta_rank = cta_rank - kClusterM;
                            mbarrier_arrive_on_cluster_cta(
                                base_mma_mbar_addr + stage * sizeof(uint64_t), source_cta_rank);
                            const int root_rank = n_rank * kClusterM;
                            mbarrier_arrive_on_cluster_cta(
                                base_a_cluster_mbar_addr + stage * sizeof(uint64_t), root_rank);
                        }
                    }

                    producer_stage = (producer_stage + 1) % kStageCount;
                    if (producer_stage == 0) {
                        producer_a_phase ^= 1;
                        producer_mma_phase ^= 1;
                    }
                }
            }
        }

        // CTA 1's MMA warp is otherwise idle.  Use it to forward that CTA's
        // local A completion to CTA 0 without blocking producer warp 0, which
        // must remain free to issue the next stage's B TMA load.
        if constexpr (kClusterM > 1) {
            if (n_rank == 0 && m_rank != 0 && warp_id == kMmaWarp && elect_sync()) {
                for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                     ++cluster_bid) {
                    for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                        mbarrier_wait_forward_a_ready(
                            base_a_mbar_addr + consumer_stage * sizeof(uint64_t),
                            consumer_a_phase);
                        mbarrier_arrive_on_cluster_cta(
                            base_a_cluster_mbar_addr + consumer_stage * sizeof(uint64_t),
                            n_rank * kClusterM);
                        if constexpr (kClusterN == 2) {
                            const int a_smem =
                                mainloop_smem + consumer_stage * kStageSmemBytes;
                            bulk_copy_a_stage_to_cluster_cta<kASmemBytes>(a_smem,
                                a_smem,
                                base_a_mbar_addr + consumer_stage * sizeof(uint64_t),
                                cta_rank + kClusterM);
                        }

                        consumer_stage = (consumer_stage + 1) % kStageCount;
                        if (consumer_stage == 0) {
                            consumer_a_phase ^= 1;
                            consumer_tma_phase ^= 1;
                        }
                    }
                }
            }
        }

        if (m_rank == 0 && warp_id == kMmaWarp && elect_sync()) {
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                 ++cluster_bid) {
                mbarrier_wait_epilogue_slot(base_epilogue_mbar_addr + consumer_mainloop_stage * sizeof(uint64_t),
                    consumer_epilogue_phase);

                const int stage_tmem_addr = tmem_addr + consumer_mainloop_stage * kTileN;
                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    const int tma_mbar_addr =
                        base_tma_mbar_addr + consumer_stage * sizeof(uint64_t);
                    const int mma_mbar_addr =
                        base_mma_mbar_addr + consumer_stage * sizeof(uint64_t);

                    if constexpr (kClusterN == 2) {
                        if (n_rank == 0) {
                            mbarrier_wait_consumer_a_ready(
                                base_a_mbar_addr + consumer_stage * sizeof(uint64_t),
                                consumer_a_phase);
                            mbarrier_arrive_on_cluster_cta(
                                base_a_cluster_mbar_addr + consumer_stage * sizeof(uint64_t),
                                n_rank * kClusterM);
                            const int a_smem =
                                mainloop_smem + consumer_stage * kStageSmemBytes;
                            bulk_copy_a_stage_to_cluster_cta<kASmemBytes>(a_smem,
                                a_smem,
                                base_a_mbar_addr + consumer_stage * sizeof(uint64_t),
                                cta_rank + kClusterM);
                            mbarrier_wait_consumer_cluster_a_ready(
                                base_a_cluster_mbar_addr + consumer_stage * sizeof(uint64_t),
                                consumer_a_phase);
                        } else {
                            mbarrier_wait_consumer_cluster_a_ready(
                                base_a_cluster_mbar_addr + consumer_stage * sizeof(uint64_t),
                                consumer_a_phase);
                        }
                    } else if constexpr (kClusterM == 1) {
                        mbarrier_wait_consumer_a_ready(
                            base_a_mbar_addr + consumer_stage * sizeof(uint64_t),
                            consumer_a_phase);
                    } else {
                        // Contribute CTA 0's local A completion.  CTA 1's idle
                        // MMA warp contributes its completion independently.
                        mbarrier_wait_consumer_a_ready(
                            base_a_mbar_addr + consumer_stage * sizeof(uint64_t),
                            consumer_a_phase);
                        mbarrier_arrive_on_cluster_cta(
                            base_a_cluster_mbar_addr + consumer_stage * sizeof(uint64_t), 0);
                        mbarrier_wait_consumer_cluster_a_ready(
                            base_a_cluster_mbar_addr + consumer_stage * sizeof(uint64_t),
                            consumer_a_phase);
                    }
                    mbarrier_wait_tma_b_ready(tma_mbar_addr, consumer_tma_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    {
                        const int a_smem = mainloop_smem + consumer_stage * kStageSmemBytes;
                        const int b_smem = a_smem + kASmemBytes;
                        for (int mma_k = 0; mma_k < kMmaStepsPerStage; ++mma_k) {
                            const int k_slab = mma_k / (64 / kMmaK);
                            const int k_in_slab = mma_k % (64 / kMmaK);
                            const int a_byte_offset = k_slab * kTileM * 64 * sizeof(nv_bfloat16) +
                                                      k_in_slab * kMmaK * sizeof(nv_bfloat16);
                            const int b_byte_offset =
                                k_slab * (kTileN / kClusterM) * 64 * sizeof(nv_bfloat16) +
                                k_in_slab * kMmaK * sizeof(nv_bfloat16);
                            const int accumulate = tile_k != 0 || mma_k != 0;
                            tcgen05_mma_bf16<kClusterM>(stage_tmem_addr,
                                make_k_major_128b_desc(a_smem + a_byte_offset),
                                make_k_major_128b_desc(b_smem + b_byte_offset),
                                instruction_desc,
                                accumulate);
                        }
                    }
                    tcgen05_commit<kClusterM>(mma_mbar_addr, cta_rank);

                    consumer_stage = (consumer_stage + 1) % kStageCount;
                    if (consumer_stage == 0) {
                        consumer_a_phase ^= 1;
                        consumer_tma_phase ^= 1;
                    }
                }

                const int mainloop_mbar_addr =
                    base_mainloop_mbar_addr + consumer_mainloop_stage * sizeof(uint64_t);
                tcgen05_commit<kClusterM>(mainloop_mbar_addr, cta_rank);

                consumer_mainloop_stage = (consumer_mainloop_stage + 1) % kTmemStageCount;
                if (consumer_mainloop_stage == 0) {
                    consumer_epilogue_phase ^= 1;
                }
            }
        }

        if (warp_id < 4) {
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                mbarrier_wait_mainloop_output(base_mainloop_mbar_addr + epilogue_mainloop_stage * sizeof(uint64_t),
                    epilogue_mainloop_phase);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

                const ExpertTileTask task = tasks[cluster_bid];
                const int tile_n = task.tile_n * kClusterN + n_rank;
                const int cta_tile_row = task.tile_row + m_rank * kTileM;
                const int cta_row = task.row_start + cta_tile_row;
                const int remaining_rows = task.row_count - cta_tile_row;
                const int valid_rows =
                    remaining_rows < 0 ? 0 : (remaining_rows < kTileM ? remaining_rows : kTileM);
                const int output_tile_col = tile_n * kLogicalTileN;
                const int tmem_row = m_rank * kTileM + warp_id * kWarpSize;

                if constexpr (kDirectGmem) {
                    // Baseline path for A/B comparison with the staged epilogue.
                    if (warp_id * kWarpSize < valid_rows) {
                        const int row = cta_row + tid;
                        const int stage_tmem_addr = tmem_addr + epilogue_mainloop_stage * kTileN;

                        for (int logical_col = 0; logical_col < kLogicalTileN; logical_col += 8) {
                            const int logical_block = logical_col / kInterleaveBlock;
                            const int logical_in_block = logical_col % kInterleaveBlock;
                            const int gate_tmem_col =
                                logical_block * (2 * kInterleaveBlock) + logical_in_block;
                            const int up_tmem_col = gate_tmem_col + kInterleaveBlock;

                            float gate_vals[8];
                            float up_vals[8];
                            const int gate_load_addr =
                                stage_tmem_addr + (tmem_row << 16) + gate_tmem_col;
                            const int up_load_addr =
                                stage_tmem_addr + (tmem_row << 16) + up_tmem_col;
                            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                                         "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=f"(gate_vals[0]),
                                "=f"(gate_vals[1]),
                                "=f"(gate_vals[2]),
                                "=f"(gate_vals[3]),
                                "=f"(gate_vals[4]),
                                "=f"(gate_vals[5]),
                                "=f"(gate_vals[6]),
                                "=f"(gate_vals[7])
                                : "r"(gate_load_addr));
                            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                                         "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=f"(up_vals[0]),
                                "=f"(up_vals[1]),
                                "=f"(up_vals[2]),
                                "=f"(up_vals[3]),
                                "=f"(up_vals[4]),
                                "=f"(up_vals[5]),
                                "=f"(up_vals[6]),
                                "=f"(up_vals[7])
                                : "r"(up_load_addr));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");

                            nv_bfloat162 packed_out[4];
                            for (int i = 0; i < 4; ++i) {
                                const float out0 = silu(gate_vals[2 * i]) * up_vals[2 * i];
                                const float out1 = silu(gate_vals[2 * i + 1]) * up_vals[2 * i + 1];
                                packed_out[i] = __float22bfloat162_rn({out0, out1});
                            }

                            const int out_col = output_tile_col + logical_col;
                            uint4 out_vec;
                            out_vec.x = reinterpret_cast<uint32_t *>(&packed_out[0])[0];
                            out_vec.y = reinterpret_cast<uint32_t *>(&packed_out[1])[0];
                            out_vec.z = reinterpret_cast<uint32_t *>(&packed_out[2])[0];
                            out_vec.w = reinterpret_cast<uint32_t *>(&packed_out[3])[0];
                            if (tid < valid_rows) {
                                reinterpret_cast<uint4 *>(out + row * n + out_col)[0] = out_vec;
                            }
                        }
                    }
                } else {
                    constexpr int kEpilogueBlockCount = kLogicalTileN / kEpilogueTileN;
                    tmem_to_smem(epilogue_mainloop_stage, 0, 0, valid_rows);
                    asm volatile("bar.sync 1, 128;" ::: "memory");

                    for (int n_block = 0; n_block < kEpilogueBlockCount; ++n_block) {
                        const int smem_stage = n_block % kEpilogueStageCount;
                        const int block_col = output_tile_col + n_block * kEpilogueTileN;
                        if (valid_rows == kTileM) {
                            if (tid == 0) {
                                const int stage_smem_addr = mainloop_smem + kMainloopSmemBytes +
                                                            smem_stage * kEpilogueStageBytes;
                                tma_store_3d(stage_smem_addr, &c_tensor_map, block_col, cta_row, 0);
                                tma_store_commit_group();
                            }
                        } else {
                            smem_to_gmem(n_block, smem_stage, cta_row, output_tile_col, valid_rows);
                        }

                        if (n_block + 1 < kEpilogueBlockCount) {
                            // A stage may be overwritten only after every warp has
                            // completed its SMEM -> GMEM/TMA reads from that stage.
                            if (kEpilogueStageCount == 1 || n_block > 0) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read %0;"
                                        :
                                        : "n"(kEpilogueStageCount - 1)
                                        : "memory");
                                }
                                asm volatile("bar.sync 1, 128;" ::: "memory");
                            }
                            const int next_stage = (n_block + 1) % kEpilogueStageCount;
                            tmem_to_smem(epilogue_mainloop_stage,
                                n_block + 1,
                                next_stage,
                                valid_rows);
                            asm volatile("bar.sync 1, 128;" ::: "memory");
                        }
                    }
                    if (tid == 0) {
                        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                }

                if (elect_sync()) {
                    const int pair_root_rank = cta_rank & ~1;
                    const int mbar_addr = map_shared_to_cluster_cta(
                        base_epilogue_mbar_addr + epilogue_mainloop_stage * sizeof(uint64_t),
                        pair_root_rank);
                    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 "
                                 "_, [%0];" ::"r"(mbar_addr)
                        : "memory");
                }

                epilogue_mainloop_stage = (epilogue_mainloop_stage + 1) % kTmemStageCount;
                if (epilogue_mainloop_stage == 0) {
                    epilogue_mainloop_phase ^= 1;
                }
            }
        }

        if (tid == 0) {
            clc_mbarrier_wait(clc_mbar_addr, clc_phase);

            const uint4 response = clc_response[0];
            int canceled;
            int canceled_cta_x = 0;
            asm volatile("{\n\t"
                         ".reg .pred was_canceled;\n\t"
                         ".reg .b128 response;\n\t"
                         "mov.b128 response, {%2, %3, %4, %5};\n\t"
                         "clusterlaunchcontrol.query_cancel.is_canceled.pred."
                         "b128 was_canceled, response;\n\t"
                         "selp.u32 %0, 1, 0, was_canceled;\n\t"
                         "@was_canceled "
                         "clusterlaunchcontrol.query_cancel.get_first_ctaid::x."
                         "b32.b128 %1, response;\n\t"
                         "}"
                : "=r"(canceled), "=r"(canceled_cta_x)
                : "r"(response.x), "r"(response.y), "r"(response.z), "r"(response.w)
                : "memory");
            next_work_smem[0] = canceled;
            next_work_smem[1] = canceled_cta_x / kClusterM * kWorkerTileCount;
        }
        __syncthreads();
        if (next_work_smem[0] == 0) {
            break;
        }
        dynamic_first_cluster_bid = next_work_smem[1];
        clc_phase ^= 1;

        // The next request reuses the response storage. Pair the generic read
        // with the CLC async proxy and keep every CTA in the cluster in lockstep.
        asm volatile("fence.proxy.async::generic.release."
                     "sync_restrict::shared::cta.cluster;" ::: "memory");
        asm volatile("barrier.cluster.arrive.relaxed.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
        asm volatile("fence.proxy.async::generic.acquire."
                     "sync_restrict::shared::cluster.cluster;" ::: "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    if (warp_id == 0) {
        if constexpr (kClusterM == 1) {
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 "
                         "%0, %1;" ::"r"(tmem_addr),
                "r"(kTmemAllocationColumns));
        } else {
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 "
                         "%0, %1;" ::"r"(tmem_addr),
                "r"(kTmemAllocationColumns));
        }
    }
}

cudaError_t launch_topk_router_impl(
    const __nv_bfloat16 *router_scores,
    int tokens,
    int experts,
    int topk,
    int *selected_experts,
    int *expert_counts,
    int *expert_offsets,
    cudaStream_t stream) {
    const size_t smem_bytes = 2 * kRoutingThreads * sizeof(int);
    auto kernel = topk_router_kernel;
    int blocks_per_sm = 0;
    cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,
        kernel,
        kRoutingThreads,
        smem_bytes);
    if (status != cudaSuccess || blocks_per_sm <= 0) {
        return status != cudaSuccess ? status : cudaErrorLaunchOutOfResources;
    }
    int device = 0;
    int sm_count = 0;
    status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) {
        return status;
    }
    const int resident_blocks = blocks_per_sm * sm_count;
    const int blocks = tokens < resident_blocks ? tokens : resident_blocks;
    void *args[] = {&router_scores,
        &tokens,
        &experts,
        &topk,
        &selected_experts,
        &expert_counts,
        &expert_offsets};
    return cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(kernel),
        dim3(blocks),
        dim3(kRoutingThreads),
        args,
        smem_bytes,
        stream);
}

cudaError_t launch_topk_router(
    const __nv_bfloat16 *d_router_scores,
    int tokens,
    int experts,
    int topk,
    int *selected_experts,
    int *expert_counts,
    int *expert_offsets,
    cudaStream_t stream) {
    return launch_topk_router_impl(d_router_scores,
        tokens,
        experts,
        topk,
        selected_experts,
        expert_counts,
        expert_offsets,
        stream);
}

// Breaking API: preparation owns every routing/task kernel; this function
// launches only the FC1 compute kernel from device-resident metadata.
template <int kTileM, int kTileN, int kTileK, int kClusterM, int kClusterN, bool kDirectGmem>
void launch_fc1_main_impl(const __nv_bfloat16* d_a, const __nv_bfloat16* d_b,
                          const int32_t* source_rows, const int32_t* task_words,
                          int32_t* task_state, __nv_bfloat16* d_out, int m, int experts,
                          int n, int k, int max_tasks, cudaStream_t stream) {
    using Config = KernelConfig<kTileM, kTileN, kTileK, kClusterM, kDirectGmem>;
    if (!d_a || !d_b || !source_rows || !task_words || !task_state || !d_out ||
        m <= 0 || experts <= 0 || n <= 0 || k <= 0 || max_tasks <= 0 ||
        n % ((kTileN / 2) * kClusterN) || k % kTileK) return;
    CUtensorMap b_tensor_map, c_tensor_map;
    if (!init_b_tensor_map<kTileN, kTileK, kClusterM>(&b_tensor_map, d_b, experts, 2 * n, k) ||
        !init_c_tensor_map<kTileM, Config::kEpilogueTileN>(&c_tensor_map, d_out, m, n)) return;
    constexpr int kDynamicSmemBytes = Config::kDynamicSmemBytes;
    const int persistent_clusters = (max_tasks + kPersistentTileCount - 1) / kPersistentTileCount;
    auto kernel = fc1_block_interleaved_swiglu_kernel<kTileM, kTileN, kTileK,
        kClusterM, kClusterN, kDirectGmem>;
    if constexpr (kDynamicSmemBytes > 48 * 1024)
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes);
    kernel<<<dim3(kClusterM * persistent_clusters, kClusterN, 1), kBlockThreads,
        kDynamicSmemBytes, stream>>>(d_a, source_rows, b_tensor_map, c_tensor_map,
        reinterpret_cast<const ExpertTileTask*>(task_words), task_state, d_out, n, k);
}

template <int kTileM, int kTileN, int kTileK, int kClusterM, int kClusterN, bool kDirectGmem>
void launch_fc1_metadata_impl(const __nv_bfloat16* router, int tokens, int experts, int topk,
                              int32_t* routing, int32_t* source_rows, int32_t* task_words,
                              int32_t* task_state, int max_tasks, int n, cudaStream_t stream) {
    if (!router || !routing || !source_rows || !task_words || !task_state || tokens <= 0 ||
        experts <= 0 || topk <= 0 || topk > experts || max_tasks <= 0 ||
        n % ((kTileN / 2) * kClusterN)) return;
    const int m = tokens * topk;
    int32_t* selected = nullptr;
    if (cudaMallocAsync(&selected, m * sizeof(int32_t), stream) != cudaSuccess) return;
    auto* counts = routing;
    auto* offsets = counts + experts;
    auto* grouped_layout = offsets + experts;
    auto* grouped_assignments = grouped_layout + m;
    auto* assignment_to_grouped = grouped_assignments + m;
    cudaMemsetAsync(counts, 0, experts * sizeof(int32_t), stream);
    cudaMemsetAsync(task_state, 0, 2 * sizeof(int32_t), stream);
    if (launch_topk_router(router, tokens, experts, topk, selected, counts, offsets, stream) != cudaSuccess) {
        cudaFreeAsync(selected, stream); return;
    }
    build_grouped_row_map_kernel<<<experts, kRoutingThreads, 0, stream>>>(selected, offsets,
        source_rows, grouped_assignments, m, topk);
    materialize_grouped_layout_kernel<<<experts, kRoutingThreads, 0, stream>>>(
        offsets, counts, experts, grouped_layout);
    invert_grouped_assignments_kernel<<<(m + kRoutingThreads - 1) / kRoutingThreads,
        kRoutingThreads, 0, stream>>>(grouped_assignments, m, assignment_to_grouped);
    constexpr int kTaskRows = kClusterM * kTileM;
    const int grid_n = n / ((kTileN / 2) * kClusterN);
    build_expert_tile_tasks<kTaskRows><<<(experts + kRoutingThreads - 1) / kRoutingThreads,
        kRoutingThreads, 0, stream>>>(offsets, counts, experts, grid_n,
        reinterpret_cast<ExpertTileTask*>(task_words), task_state);
    cudaFreeAsync(selected, stream);
}

} // namespace sm110a::bf16_fc1_block_interleaved_swiglu

using namespace sm110a::bf16_fc1_block_interleaved_swiglu;

extern "C" void sm110a_fc1_metadata_prepare(const __nv_bfloat16* router, int tokens,
    int experts, int topk, int32_t* routing, int32_t* source_rows, int32_t* task_words,
    int32_t* task_state, int max_tasks, int n, cudaStream_t stream) {
    launch_fc1_metadata_impl<SM110A_FC1_TILE_M, SM110A_FC1_TILE_N, SM110A_FC1_TILE_K,
        SM110A_FC1_CLUSTER_M, SM110A_FC1_CLUSTER_N, (SM110A_FC1_DIRECT_GMEM != 0)>(
            router, tokens, experts, topk, routing, source_rows, task_words, task_state,
            max_tasks, n, stream);
}

extern "C" void sm110a_fc1_main_launch(const __nv_bfloat16* a, const __nv_bfloat16* b,
    const int32_t* source_rows, const int32_t* task_words, int32_t* task_state,
    __nv_bfloat16* out, int m, int experts, int n, int k, int max_tasks, cudaStream_t stream) {
    launch_fc1_main_impl<SM110A_FC1_TILE_M, SM110A_FC1_TILE_N, SM110A_FC1_TILE_K,
        SM110A_FC1_CLUSTER_M, SM110A_FC1_CLUSTER_N, (SM110A_FC1_DIRECT_GMEM != 0)>(
            a, b, source_rows, task_words, task_state, out, m, experts, n, k, max_tasks, stream);
}

void launch_fc1_main(const __nv_bfloat16* a, const __nv_bfloat16* b,
    const int32_t* source_rows, const int32_t* tasks, int32_t* state,
    __nv_bfloat16* out, int m, int experts, int n, int k, int max_tasks,
    cudaStream_t stream) {
    sm110a_fc1_main_launch(a, b, source_rows, tasks, state, out, m, experts, n, k,
                           max_tasks, stream);
}
