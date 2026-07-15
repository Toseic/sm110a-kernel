/*
 * fc2_finalize.cu — routed MoE FC2/finalize entry points for SM110A.
 *
 * This translation unit owns FC2 routing metadata and a private physical copy
 * of the Blackwell BF16 FC2 grouped-GEMM implementation. Keeping it here lets
 * the fused FC2 epilogue be specialized without changing standalone grouped
 * GEMM, which remains the non-fused baseline.
 *
 * The copied kernel keeps its 6-warp TMA/tcgen05/TMEM pipeline and 1-CTA or
 * 2-CTA cluster row ownership, with the FC2-finalize reduction as its only
 * epilogue.
 */

#include <cmath>
#include <cstdint>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bf16_moe_fc2_finalize.h"

#ifndef SM110A_GROUPED_TILE_M
#define SM110A_GROUPED_TILE_M 128
#endif
#ifndef SM110A_GROUPED_TILE_N
#define SM110A_GROUPED_TILE_N 256
#endif
#ifndef SM110A_GROUPED_TILE_K
#define SM110A_GROUPED_TILE_K 128
#endif
#ifndef SM110A_GROUPED_EPILOGUE_TILE_N
#define SM110A_GROUPED_EPILOGUE_TILE_N 32
#endif
#ifndef SM110A_GROUPED_CLUSTER_M
#define SM110A_GROUPED_CLUSTER_M 2
#endif
#ifndef SM110A_FC2_REDUCE_TILE_N
#define SM110A_FC2_REDUCE_TILE_N 64
#endif

void launch_fc2_grouped_gemm_fused(
    const __nv_bfloat16* d_a, const __nv_bfloat16* d_b,
    const int32_t* d_tasks, int32_t* d_state,
    const int32_t* grouped_assignments, const __nv_bfloat16* routing_weights,
    __nv_bfloat16* final_output, int m, int experts, int n, int k, int topk,
    int max_tasks, cudaStream_t stream);

namespace {

constexpr int kRoutingThreads = 256;

static_assert(SM110A_GROUPED_TILE_M == 128,
              "the fused FC2 epilogue maps one 128-row CTA slice");
static_assert(SM110A_FC2_REDUCE_TILE_N == 64,
              "the initial fused FC2 epilogue uses 128-byte BF16 reductions");
static_assert(SM110A_GROUPED_TILE_N % SM110A_FC2_REDUCE_TILE_N == 0);

__device__ __forceinline__ void fence_smem_for_async_proxy() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void bulk_reduce_add_bf16(
    __nv_bfloat16* dst_gmem, const __nv_bfloat16* src_smem, uint32_t size_bytes) {
    const uint64_t dst_addr = reinterpret_cast<uint64_t>(dst_gmem);
    const uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(src_smem));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
                 "[%0], [%1], %2;"
                 :: "l"(dst_addr), "r"(src_addr), "r"(size_bytes) : "memory");
}

__device__ __forceinline__ void bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void bulk_wait_group_read_0() {
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

__device__ __forceinline__ void bulk_wait_group_0() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__global__ void select_topk_kernel(
    const __nv_bfloat16 *router_scores,
    int tokens,
    int experts,
    int topk,
    int32_t *selected_experts) {
    __shared__ float scores[kRoutingThreads];
    __shared__ int selected[kRoutingThreads];
    const int token = blockIdx.x;
    if (token >= tokens)
        return;

    for (int rank = 0; rank < topk; ++rank) {
        float best_score = -INFINITY;
        int best_expert = -1;
        for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
            bool already_selected = false;
            for (int previous = 0; previous < rank; ++previous) {
                already_selected |= selected_experts[token * topk + previous] == expert;
            }
            if (already_selected)
                continue;
            float score = __bfloat162float(router_scores[token * experts + expert]);
            if (isnan(score))
                score = -INFINITY;
            if (best_expert < 0 || score > best_score ||
                (score == best_score && expert < best_expert)) {
                best_score = score;
                best_expert = expert;
            }
        }
        scores[threadIdx.x] = best_score;
        selected[threadIdx.x] = best_expert;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const float other_score = scores[threadIdx.x + stride];
                const int other_expert = selected[threadIdx.x + stride];
                if (other_expert >= 0 &&
                    (selected[threadIdx.x] < 0 || other_score > scores[threadIdx.x] ||
                        (other_score == scores[threadIdx.x] &&
                            other_expert < selected[threadIdx.x]))) {
                    scores[threadIdx.x] = other_score;
                    selected[threadIdx.x] = other_expert;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
            selected_experts[token * topk + rank] = selected[0];
        __syncthreads();
    }
}

__global__ void count_experts_kernel(
    const int32_t *selected_experts,
    int m,
    int32_t *expert_counts) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m)
        atomicAdd(expert_counts + selected_experts[row], 1);
}

__global__ void build_offsets_kernel(
    const int32_t *expert_counts,
    int experts,
    int32_t *expert_offsets) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    int offset = 0;
    for (int expert = 0; expert < experts; ++expert) {
        expert_offsets[expert] = offset;
        offset += expert_counts[expert];
    }
}

__global__ void build_grouped_maps_kernel(
    const int32_t *selected_experts,
    const int32_t *expert_offsets,
    int m,
    int32_t *grouped_layout,
    int32_t *grouped_assignments) {
    const int expert = blockIdx.x;
    __shared__ int cursor;
    if (threadIdx.x == 0)
        cursor = expert_offsets[expert];
    __syncthreads();
    for (int base = 0; base < m; base += blockDim.x) {
        const int assignment = base + threadIdx.x;
        const bool matches = assignment < m && selected_experts[assignment] == expert;
        const unsigned mask = __ballot_sync(0xffffffff, matches);
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        __shared__ int warp_offsets[kRoutingThreads / 32];
        if (lane == 0)
            warp_offsets[warp] = __popc(mask);
        __syncthreads();
        if (threadIdx.x == 0) {
            int total = 0;
            for (int i = 0; i < kRoutingThreads / 32; ++i) {
                const int count = warp_offsets[i];
                warp_offsets[i] = total;
                total += count;
            }
            warp_offsets[0] = cursor;
            cursor += total;
        }
        __syncthreads();
        if (matches) {
            const int base_offset =
                warp == 0 ? warp_offsets[0] : warp_offsets[0] + warp_offsets[warp];
            const unsigned lower = lane == 0 ? 0 : ((1u << lane) - 1u);
            const int position = base_offset + __popc(mask & lower);
            grouped_layout[position] = expert;
            grouped_assignments[position] = assignment;
        }
        __syncthreads();
    }
}

__global__ void invert_grouped_assignments_kernel(
    const int32_t *grouped_assignments,
    int m,
    int32_t *assignment_to_grouped) {
    const int grouped_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (grouped_row < m)
        assignment_to_grouped[grouped_assignments[grouped_row]] = grouped_row;
}

} // namespace

void launch_fc2_finalize(
    const __nv_bfloat16 *d_fc1,
    const __nv_bfloat16 *d_b,
    const __nv_bfloat16 *d_routing_weights,
    __nv_bfloat16 *d_out,
    int tokens,
    int experts,
    int topk,
    int h,
    int i,
    const int32_t *d_routing_metadata,
    const int32_t *d_tasks,
    int32_t *d_task_state,
    int max_tasks,
    cudaStream_t stream) {
    if (!d_fc1 || !d_b || !d_routing_weights || !d_routing_metadata || !d_tasks || !d_task_state || !d_out || tokens <= 0 || experts <= 0 || topk <= 0 ||
        topk > experts || h <= 0 || i <= 0)
        return;
    const int m = tokens * topk;
    // `grouped_assignments` is the expanded assignment id written by the
    // routing packer: assignment = token * topk + topk_slot.  It is exactly
    // the metadata required by the fused epilogue to recover both output row
    // and router scale without materialising grouped FC2.
    auto *grouped_assignments = d_routing_metadata + 2 * experts + m;
    cudaMemsetAsync(d_out, 0, static_cast<size_t>(tokens) * h * sizeof(__nv_bfloat16), stream);
    launch_fc2_grouped_gemm_fused(d_fc1, d_b, d_tasks, d_task_state,
        grouped_assignments, d_routing_weights, d_out, m, experts, h, i, topk,
        max_tasks, stream);
}

extern "C" void sm110a_fc2_finalize_launch(
    const __nv_bfloat16 *d_fc1,
    const __nv_bfloat16 *d_b,
    const __nv_bfloat16 *d_routing_weights,
    __nv_bfloat16 *d_out,
    int tokens,
    int experts,
    int topk,
    int h,
    int i,
    const int32_t *d_routing_metadata,
    const int32_t *d_tasks,
    int32_t *d_task_state,
    int max_tasks,
    cudaStream_t stream) {
    launch_fc2_finalize(d_fc1, d_b, d_routing_weights, d_out, tokens, experts, topk, h, i,
        d_routing_metadata, d_tasks, d_task_state, max_tasks, stream);
}

namespace {

constexpr int kMmaK = 16;
constexpr int kWarpSize = 32;
constexpr int kWarpCount = 6;
constexpr int kBlockThreads = kWarpSize * kWarpCount;
constexpr int kTmemStageCount = 2;
// Two tiles amortize setup without letting persistent workers drift across
// unrelated experts and destroy the cross-cluster A reuse.
constexpr int kPersistentTileCount = 2;
constexpr int kMacroSwizzleM = 4;
// Keep the persistent working set below Thor's 32 MiB L2.  A 16-wide N band
// needs 25 MiB of B plus up to 12 MiB of A for four 256-row tasks, which
// evicts the B tiles before the next M task can reuse them.  An 8-wide band
// keeps the combined A/B footprint at or below 25 MiB for the large-M cases.
constexpr int kMacroSwizzleN = 8;

template <int kTileM, int kTileN, int kTileK, int kEpilogueTileN, int kClusterM>
struct Fc2GroupedGemmConfig {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert(kTileM == 4 * kWarpSize);
    // tcgen05.mma.kind::f16 supports N in steps of 8 for cta_group::1
    // and steps of 16 for cta_group::2, up to N=256. The instruction
    // descriptor encodes this dimension as N >> 3 in bits 17-22.
    static_assert(kTileN >= 8 * kClusterM);
    static_assert(kTileN <= 256);
    static_assert(kTileN % (8 * kClusterM) == 0);
    static_assert(kEpilogueTileN == 32 || kEpilogueTileN == 64 || kEpilogueTileN == 128);
    static_assert(kTileN % kEpilogueTileN == 0);
    static_assert(kTileN % kClusterM == 0);
    static_assert(kTileK % 64 == 0);
    static_assert(kTileK % kMmaK == 0);
    static constexpr int kClusterTileM = kClusterM * kTileM;
    static constexpr int kASmemBytes = kTileM * kTileK * sizeof(nv_bfloat16);
    static constexpr int kBSmemBytes = (kTileN / kClusterM) * kTileK * sizeof(nv_bfloat16);
    static constexpr int kPerStageSmemBytes = kASmemBytes + kBSmemBytes;
    static constexpr int kEpilogueStageBytes = kTileM * kEpilogueTileN * sizeof(nv_bfloat16);
    // A 128-column epilogue stage occupies 32 KiB. Keeping one such stage
    // leaves enough shared memory for two mainloop stages in the largest
    // supported configuration.
    static constexpr int kEpilogueStageCount = kEpilogueTileN == 128 ? 1 : 2;
    static constexpr int kEpilogueSmemBytes = kEpilogueStageCount * kEpilogueStageBytes;
    // Pack as many mainloop pipeline stages as shared memory allows for this
    // configuration. Blackwell has ~228 KB of SMEM per CTA; leave a small
    // margin for statically allocated barriers and bookkeeping.
    static constexpr int kMaxSmemBytes = 227 * 1024;
    static constexpr int kStageCount = (kMaxSmemBytes - kEpilogueSmemBytes) / kPerStageSmemBytes;
    static_assert(kStageCount >= 2);
    static constexpr int kMainloopSmemBytes = kStageCount * kPerStageSmemBytes;
    static constexpr int kDynamicSmemBytes = kMainloopSmemBytes + kEpilogueSmemBytes;
    static constexpr int kTmemColumnsNeeded = kTileN * kTmemStageCount;
    // tcgen05.alloc requires a power-of-two column count. In particular,
    // two N=192 accumulator stages need 384 columns but must allocate 512.
    static constexpr int kTmemAllocationColumns = kTmemColumnsNeeded <= 32    ? 32
                                                  : kTmemColumnsNeeded <= 64  ? 64
                                                  : kTmemColumnsNeeded <= 128 ? 128
                                                  : kTmemColumnsNeeded <= 256 ? 256
                                                                              : 512;
    static_assert(kTmemColumnsNeeded <= 512);
    // Backward-compat alias used inside the kernel function body.
    static constexpr int kStageSmemBytes = kPerStageSmemBytes;
};

struct Fc2ExpertTileTask {
    int row_start;
    int row_count;
    int tile_row;
    int expert;
    int tile_n;
};

__global__ void collect_expert_metadata(
    const int32_t *grouped_layout,
    int m,
    int experts,
    int *expert_offsets,
    int *expert_counts) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) {
        return;
    }
    const int expert = grouped_layout[row];
    if (expert < 0 || expert >= experts) {
        return;
    }
    atomicMin(expert_offsets + expert, row);
    atomicAdd(expert_counts + expert, 1);
}

template <int kTaskTileRows>
__global__ void build_expert_tile_tasks(
    const int *expert_offsets,
    const int *expert_counts,
    int experts,
    int grid_n,
    Fc2ExpertTileTask *tasks,
    int *task_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int next_task = 0;
    for (int expert = 0; expert < experts; ++expert) {
        const int row_count = expert_counts[expert];
        if (row_count == 0) {
            continue;
        }
        const int row_start = expert_offsets[expert];
        for (int group_row = 0; group_row < row_count;
            group_row += kMacroSwizzleM * kTaskTileRows) {
            const int group_row_end = group_row + kMacroSwizzleM * kTaskTileRows < row_count
                                          ? group_row + kMacroSwizzleM * kTaskTileRows
                                          : row_count;
            for (int macro_n = 0; macro_n < grid_n; macro_n += kMacroSwizzleN) {
                const int band_end =
                    macro_n + kMacroSwizzleN < grid_n ? macro_n + kMacroSwizzleN : grid_n;
                for (int tile_row = group_row; tile_row < group_row_end;
                    tile_row += kTaskTileRows) {
                    for (int tile_n = macro_n; tile_n < band_end; ++tile_n) {
                        tasks[next_task++] = {row_start, row_count, tile_row, expert, tile_n};
                    }
                }
            }
        }
    }
    *task_count = next_task;
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

__device__ inline void mbarrier_init(
    int mbar_addr,
    int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr), "r"(count));
}

__device__ inline void mbarrier_wait(
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
    int mbar_addr) {
    if constexpr (kClusterM == 1) {
        asm volatile("tcgen05.commit.cta_group::1."
                     "mbarrier::arrive::one.shared::cluster.b64 "
                     "[%0];" ::"r"(mbar_addr)
            : "memory");
    } else {
        asm volatile("tcgen05.commit.cta_group::2."
                     "mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
                     "[%0], %1;" ::"r"(mbar_addr),
            "h"(static_cast<int16_t>((1 << kClusterM) - 1))
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

template <
    int kTileM,
    int kTileN,
    int kTileK,
    int kEpilogueTileN,
    int kClusterM>
__global__ __cluster_dims__(kClusterM, 1, 1)
__launch_bounds__(kBlockThreads) void fc2_grouped_gemm_kernel(
    const __grid_constant__ CUtensorMap a_tensor_map,
    const __grid_constant__ CUtensorMap b_tensor_map,
    const Fc2ExpertTileTask *tasks,
    const int *task_count,
    int *next_cluster_tile,
    int n,
    int k,
    const int32_t *grouped_assignments,
    const nv_bfloat16 *routing_weights,
    nv_bfloat16 *final_output,
    int topk) {
    using Config = Fc2GroupedGemmConfig<kTileM, kTileN, kTileK, kEpilogueTileN, kClusterM>;
    constexpr int kClusterTileM = Config::kClusterTileM;
    constexpr int kStageCount = Config::kStageCount;
    constexpr int kASmemBytes = Config::kASmemBytes;
    constexpr int kBSmemBytes = Config::kBSmemBytes;
    constexpr int kStageSmemBytes = Config::kStageSmemBytes;
    constexpr int kMainloopSmemBytes = Config::kMainloopSmemBytes;
    constexpr int kMmaStepsPerStage = kTileK / kMmaK;
    constexpr int kTmemAllocationColumns = Config::kTmemAllocationColumns;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int cluster_tiles = *task_count;
    constexpr int kWorkerTileCount = kPersistentTileCount;

    __shared__ int tmem_base[1];
    __shared__ int next_work_smem[2];
    __shared__ uint64_t tma_mbars[kStageCount];
    __shared__ uint64_t mma_mbars[kStageCount];
    __shared__ uint64_t mainloop_mbars[kTmemStageCount];
    __shared__ uint64_t epilogue_mbars[kTmemStageCount];

    extern __shared__ __align__(1024) char dynamic_smem[];
    const int mainloop_smem = static_cast<int>(__cvta_generic_to_shared(dynamic_smem));
    const int base_tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(tma_mbars));
    const int base_mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mma_mbars));
    const int base_mainloop_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mainloop_mbars));
    const int base_epilogue_mbar_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_mbars));

    if (tid == 0) {
        for (int stage = 0; stage < kStageCount; ++stage) {
            mbarrier_init(base_tma_mbar_addr + stage * sizeof(uint64_t), kClusterM);
            mbarrier_init(base_mma_mbar_addr + stage * sizeof(uint64_t), 1);
        }
        for (int stage = 0; stage < kTmemStageCount; ++stage) {
            mbarrier_init(base_mainloop_mbar_addr + stage * sizeof(uint64_t), 1);
            mbarrier_init(base_epilogue_mbar_addr + stage * sizeof(uint64_t), 4 * kClusterM);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp_id == 5) {
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
        prefetch_tensor_map(&a_tensor_map);
        prefetch_tensor_map(&b_tensor_map);
    }
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    const int tmem_addr = tmem_base[0];
    constexpr uint32_t instruction_desc = (1U << 4U) | (1U << 7U) | (1U << 10U) |
                                          (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
                                          (static_cast<uint32_t>(kClusterTileM) >> 4U << 24U);

    const int k_tiles = k / kTileK;

    // FC2 fusion staging is deliberately physical row-major shared memory:
    // one epilogue thread owns one row and cp.reduce.async.bulk consumes a
    // contiguous 128-byte BF16 source range for that row.
    auto tmem_to_finalize_smem = [&](int mainloop_stage, int reduce_n, float scale) {
        auto *stage_smem = reinterpret_cast<nv_bfloat16 *>(
            dynamic_smem + kMainloopSmemBytes);
        for (int n_group = 0; n_group < SM110A_FC2_REDUCE_TILE_N / 8; ++n_group) {
            float values[8];
            const int tmem_row = cta_rank * kTileM + warp_id * kWarpSize;
            const int tmem_col = mainloop_stage * kTileN + reduce_n + n_group * 8;
            const int load_addr = tmem_addr + (tmem_row << 16) + tmem_col;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                         "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]),
                  "=f"(values[4]), "=f"(values[5]), "=f"(values[6]), "=f"(values[7])
                : "r"(load_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if (tid < kTileM) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    stage_smem[tid * SM110A_FC2_REDUCE_TILE_N + n_group * 8 + j] =
                        __float2bfloat16_rn(values[j] * scale);
                }
            }
        }
    };

    int producer_stage = 0;
    int producer_mma_phase = 1;
    int consumer_stage = 0;
    int consumer_tma_phase = 0;
    int consumer_mainloop_stage = 0;
    int consumer_epilogue_phase = 1;
    int epilogue_mainloop_stage = 0;
    int epilogue_mainloop_phase = 0;

    while (true) {
        int first_cluster_bid;
        if (cta_rank == 0 && tid == 0) {
            next_work_smem[0] = atomicAdd(next_cluster_tile, kPersistentTileCount);
        }
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");

        const int local_work_addr = static_cast<int>(__cvta_generic_to_shared(next_work_smem));
        int rank_zero_work_addr;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(rank_zero_work_addr)
            : "r"(local_work_addr), "r"(0));
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
            : "=r"(first_cluster_bid)
            : "r"(rank_zero_work_addr)
            : "memory");
        if (first_cluster_bid >= cluster_tiles) {
            break;
        }
        const bool has_work = first_cluster_bid < cluster_tiles;
        const int last_cluster_bid = has_work
                                         ? (first_cluster_bid + kWorkerTileCount < cluster_tiles
                                                   ? first_cluster_bid + kWorkerTileCount
                                                   : cluster_tiles)
                                         : first_cluster_bid;

        // The producer waits until MMA has released a stage before TMA overwrites it.
        if (warp_id == 4 && elect_sync()) {
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                const Fc2ExpertTileTask task = tasks[cluster_bid];
                const int tile_n = task.tile_n;
                const int cta_row = task.row_start + task.tile_row + cta_rank * kTileM;
                const int tile_col = tile_n * kTileN;

                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    const int mma_mbar_addr =
                        base_mma_mbar_addr + producer_stage * sizeof(uint64_t);
                    const int tma_mbar_addr =
                        base_tma_mbar_addr + producer_stage * sizeof(uint64_t);
                    const int a_smem = mainloop_smem + producer_stage * kStageSmemBytes;
                    const int b_smem = a_smem + kASmemBytes;

                    mbarrier_wait(mma_mbar_addr, producer_mma_phase);
                    const int tma_transaction_mbar_addr = tma_mbar_addr & 0xFEFFFFFF;
                    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
                                 "_, [%0], %1;" ::"r"(tma_transaction_mbar_addr),
                        "r"(kASmemBytes + kBSmemBytes)
                        : "memory");
                    const int k_slice = tile_k * (kTileK / 64);
                    tma_load_4d<kClusterM>(a_smem, &a_tensor_map, 0, cta_row,
                        k_slice, 0, tma_transaction_mbar_addr);
                    tma_load_4d<kClusterM>(b_smem, &b_tensor_map, 0,
                        tile_col + cta_rank * (kTileN / kClusterM), k_slice,
                        task.expert, tma_transaction_mbar_addr);

                    producer_stage = (producer_stage + 1) % kStageCount;
                    if (producer_stage == 0) {
                        producer_mma_phase ^= 1;
                    }
                }
            }
        }

        // The consumer advances independently and releases each stage after MMA has
        // finished reading it.
        if (cta_rank == 0 && warp_id == 5 && elect_sync()) {
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                mbarrier_wait(
                    base_epilogue_mbar_addr + consumer_mainloop_stage * sizeof(uint64_t),
                    consumer_epilogue_phase);

                const int stage_tmem_addr = tmem_addr + consumer_mainloop_stage * kTileN;
                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    const int tma_mbar_addr =
                        base_tma_mbar_addr + consumer_stage * sizeof(uint64_t);
                    const int mma_mbar_addr =
                        base_mma_mbar_addr + consumer_stage * sizeof(uint64_t);

                    mbarrier_wait(tma_mbar_addr, consumer_tma_phase);
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
                    tcgen05_commit<kClusterM>(mma_mbar_addr);

                    consumer_stage = (consumer_stage + 1) % kStageCount;
                    if (consumer_stage == 0) {
                        consumer_tma_phase ^= 1;
                    }
                }

                const int mainloop_mbar_addr =
                    base_mainloop_mbar_addr + consumer_mainloop_stage * sizeof(uint64_t);
                tcgen05_commit<kClusterM>(mainloop_mbar_addr);

                consumer_mainloop_stage = (consumer_mainloop_stage + 1) % kTmemStageCount;
                if (consumer_mainloop_stage == 0) {
                    consumer_epilogue_phase ^= 1;
                }
            }
        }

        if (warp_id < 4) {
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                mbarrier_wait(base_mainloop_mbar_addr + epilogue_mainloop_stage * sizeof(uint64_t),
                    epilogue_mainloop_phase);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

                const Fc2ExpertTileTask task = tasks[cluster_bid];
                const int tile_n = task.tile_n;
                const int cta_tile_row = task.tile_row + cta_rank * kTileM;
                const int cta_row = task.row_start + cta_tile_row;
                const int remaining_rows = task.row_count - cta_tile_row;
                const int valid_rows =
                    remaining_rows < 0 ? 0 : (remaining_rows < kTileM ? remaining_rows : kTileM);
                const int tile_col = tile_n * kTileN;

                auto release_tmem = [&]() {
                    if (elect_sync()) {
                        const int mbar_addr = (base_epilogue_mbar_addr +
                                                  epilogue_mainloop_stage * sizeof(uint64_t)) &
                                              0xFEFFFFFF;
                        asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 "
                                     "_, [%0];" ::"r"(mbar_addr)
                            : "memory");
                    }
                };

                // FC2 fusion epilogue.  Keep all four epilogue warps in the
                // tcgen05 load and bulk-group protocol; only invalid padded
                // rows skip metadata loads and the global reduction itself.
                const bool valid_row = tid < valid_rows;
                int token_idx = 0;
                float router_scale = 0.0f;
                if (valid_row) {
                    const int expanded_id = grouped_assignments[cta_row + tid];
                    token_idx = expanded_id / topk;
                    router_scale = __bfloat162float(routing_weights[expanded_id]);
                }
                for (int reduce_n = 0; reduce_n < kTileN;
                    reduce_n += SM110A_FC2_REDUCE_TILE_N) {
                    tmem_to_finalize_smem(epilogue_mainloop_stage, reduce_n, router_scale);
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    fence_smem_for_async_proxy();
                    if (valid_row) {
                        auto *source = reinterpret_cast<nv_bfloat16 *>(
                            dynamic_smem + kMainloopSmemBytes) +
                            tid * SM110A_FC2_REDUCE_TILE_N;
                        bulk_reduce_add_bf16(final_output +
                                                 static_cast<size_t>(token_idx) * n +
                                                 tile_col + reduce_n,
                                             source,
                                             SM110A_FC2_REDUCE_TILE_N * sizeof(nv_bfloat16));
                    }
                    bulk_commit_group();
                    bulk_wait_group_read_0();
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                }
                release_tmem();
                bulk_wait_group_0();
                asm volatile("bar.sync 1, 128;" ::: "memory");

                epilogue_mainloop_stage =
                    (epilogue_mainloop_stage + 1) % kTmemStageCount;
                if (epilogue_mainloop_stage == 0) {
                    epilogue_mainloop_phase ^= 1;
                }
            }
        }

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

} // namespace

void launch_fc2_grouped_gemm_fused(
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b,
    const int32_t *d_tasks,
    int32_t *d_state,
    const int32_t *grouped_assignments,
    const __nv_bfloat16 *routing_weights,
    __nv_bfloat16 *final_output,
    int m,
    int experts,
    int n,
    int k,
    int topk,
    int max_tasks,
    cudaStream_t stream) {
    using Config = Fc2GroupedGemmConfig<SM110A_GROUPED_TILE_M,
        SM110A_GROUPED_TILE_N,
        SM110A_GROUPED_TILE_K,
        SM110A_GROUPED_EPILOGUE_TILE_N,
        SM110A_GROUPED_CLUSTER_M>;
    constexpr int kDynamicSmemBytes = Config::kDynamicSmemBytes;
    if (!d_a || !d_b || !d_tasks || !d_state || !grouped_assignments ||
        !routing_weights || !final_output || m <= 0 || experts <= 0 ||
        n <= 0 || k <= 0 || topk <= 0 || max_tasks <= 0 ||
        n % SM110A_GROUPED_TILE_N != 0 || k % SM110A_GROUPED_TILE_K != 0) {
        return;
    }

    CUtensorMap a_tensor_map;
    CUtensorMap b_tensor_map;
    if (!init_a_tensor_map<SM110A_GROUPED_TILE_M, SM110A_GROUPED_TILE_K>(
            &a_tensor_map, d_a, m, k) ||
        !init_b_tensor_map<SM110A_GROUPED_TILE_N,
            SM110A_GROUPED_TILE_K,
            SM110A_GROUPED_CLUSTER_M>(&b_tensor_map, d_b, experts, n, k)) {
        return;
    }

    auto *tasks = reinterpret_cast<const Fc2ExpertTileTask *>(d_tasks);
    auto *task_count = d_state;
    auto *next_cluster_tile = d_state + 1;
    cudaMemsetAsync(next_cluster_tile, 0, sizeof(int32_t), stream);

    constexpr int kPersistentTileCount = 2;
    const int persistent_clusters =
        (max_tasks + kPersistentTileCount - 1) / kPersistentTileCount;
    auto kernel = fc2_grouped_gemm_kernel<SM110A_GROUPED_TILE_M,
        SM110A_GROUPED_TILE_N,
        SM110A_GROUPED_TILE_K,
        SM110A_GROUPED_EPILOGUE_TILE_N,
        SM110A_GROUPED_CLUSTER_M>;
    if constexpr (kDynamicSmemBytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            kDynamicSmemBytes);
    }
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = dim3(SM110A_GROUPED_CLUSTER_M * persistent_clusters, 1, 1);
    launch_config.blockDim = dim3(kBlockThreads, 1, 1);
    launch_config.dynamicSmemBytes = kDynamicSmemBytes;
    launch_config.stream = stream;
    int resident_clusters = 0;
    cudaOccupancyMaxActiveClusters(&resident_clusters, kernel, &launch_config);
    if (resident_clusters <= 0) {
        int device = 0;
        int sm_count = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        resident_clusters = sm_count / SM110A_GROUPED_CLUSTER_M;
        if (resident_clusters <= 0) resident_clusters = 1;
    }
    const int launched_clusters = resident_clusters < persistent_clusters
        ? resident_clusters : persistent_clusters;
    kernel<<<SM110A_GROUPED_CLUSTER_M * launched_clusters,
        kBlockThreads, kDynamicSmemBytes, stream>>>(
        a_tensor_map, b_tensor_map, tasks, task_count, next_cluster_tile, n, k,
        grouped_assignments, routing_weights, final_output, topk);
}
