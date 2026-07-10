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

#include <cstdint>
#include <cstdlib>
#include <cmath>
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
#ifndef SM110A_FC1_DIRECT_GMEM
#define SM110A_FC1_DIRECT_GMEM 0
#endif

namespace {

constexpr int kMmaK = 16;
constexpr int kWarpSize = 32;
constexpr int kWarpCount = 6;
constexpr int kBlockThreads = kWarpSize * kWarpCount;
constexpr int kTmemStageCount = 2;
constexpr int kPersistentTileCount = 2;
constexpr int kMacroSwizzleM = 4;
// Keep each expert-local A/B working set below Thor's 32 MiB L2 while the
// persistent workers traverse neighboring M tasks.
constexpr int kMacroSwizzleN = 8;
constexpr int kInterleaveBlock = 32;

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

__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

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
    for (int group_row = 0; group_row < row_count;
        group_row += kMacroSwizzleM * kTaskTileRows) {
        const int group_row_end =
            group_row + kMacroSwizzleM * kTaskTileRows < row_count
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

template <int kTileM, int kEpilogueTileN>
inline bool init_c_tensor_map(
    CUtensorMap *tensor_map,
    nv_bfloat16 *ptr,
    uint64_t m,
    uint64_t n) {
    constexpr uint32_t rank = 3;
    uint64_t global_dim[rank] = {n, m, 1};
    uint64_t global_stride[rank - 1] = {
        n * sizeof(nv_bfloat16), m * n * sizeof(nv_bfloat16)};
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

template <int kTileM, int kTileN, int kTileK, int kClusterM, bool kDirectGmem>
__global__ __cluster_dims__(kClusterM, 1, 1)
__launch_bounds__(kBlockThreads) void fc1_block_interleaved_swiglu_kernel(
    const __grid_constant__ CUtensorMap a_tensor_map,
    const __grid_constant__ CUtensorMap b_tensor_map,
    const __grid_constant__ CUtensorMap c_tensor_map,
    const ExpertTileTask *tasks,
    const int *task_count,
    int *next_cluster_tile,
    nv_bfloat16 *out,
    int n,
    int k) {
    using Config = KernelConfig<kTileM, kTileN, kTileK, kClusterM, kDirectGmem>;
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

    auto tmem_to_smem = [&](int mainloop_stage, int n_block, int smem_stage,
                            int valid_rows) {
        char *stage_smem = dynamic_smem + kMainloopSmemBytes +
                           smem_stage * kEpilogueStageBytes;
        const int tmem_row = cta_rank * kTileM + warp_id * kWarpSize;
        const int stage_tmem_addr = tmem_addr + mainloop_stage * kTileN;

        if (tid < valid_rows) {
            for (int logical_col = n_block * kEpilogueTileN;
                 logical_col < (n_block + 1) * kEpilogueTileN;
                 logical_col += 8) {
                const int logical_block = logical_col / kInterleaveBlock;
                const int logical_in_block = logical_col % kInterleaveBlock;
                const int gate_tmem_col =
                    logical_block * (2 * kInterleaveBlock) + logical_in_block;
                const int up_tmem_col = gate_tmem_col + kInterleaveBlock;
                float gate_vals[8];
                float up_vals[8];
                const int gate_load_addr = stage_tmem_addr + (tmem_row << 16) + gate_tmem_col;
                const int up_load_addr = stage_tmem_addr + (tmem_row << 16) + up_tmem_col;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                             "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=f"(gate_vals[0]), "=f"(gate_vals[1]), "=f"(gate_vals[2]),
                      "=f"(gate_vals[3]), "=f"(gate_vals[4]), "=f"(gate_vals[5]),
                      "=f"(gate_vals[6]), "=f"(gate_vals[7])
                    : "r"(gate_load_addr));
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                             "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=f"(up_vals[0]), "=f"(up_vals[1]), "=f"(up_vals[2]),
                      "=f"(up_vals[3]), "=f"(up_vals[4]), "=f"(up_vals[5]),
                      "=f"(up_vals[6]), "=f"(up_vals[7])
                    : "r"(up_load_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");

                nv_bfloat162 packed_out[4];
                for (int i = 0; i < 4; ++i) {
                    packed_out[i] = __float22bfloat162_rn(
                        {silu(gate_vals[2 * i]) * up_vals[2 * i],
                         silu(gate_vals[2 * i + 1]) * up_vals[2 * i + 1]});
                }
                const int group = (logical_col - n_block * kEpilogueTileN) / 8;
                const int chunk = tid * (kEpilogueTileN / 8) + group;
                const int swizzled_chunk =
                    (chunk & ~7) + ((chunk & 7) ^ ((chunk >> 3) & 3));
                reinterpret_cast<uint4 *>(stage_smem)[swizzled_chunk] =
                    {reinterpret_cast<uint32_t *>(&packed_out[0])[0],
                     reinterpret_cast<uint32_t *>(&packed_out[1])[0],
                     reinterpret_cast<uint32_t *>(&packed_out[2])[0],
                     reinterpret_cast<uint32_t *>(&packed_out[3])[0]};
            }
        }
    };

    auto smem_to_gmem = [&](int n_block, int smem_stage, int cta_row,
                            int output_tile_col, int valid_rows) {
        if (tid < valid_rows) {
            const char *stage_smem = dynamic_smem + kMainloopSmemBytes +
                                     smem_stage * kEpilogueStageBytes;
            const int block_col = output_tile_col + n_block * kEpilogueTileN;
            for (int group = 0; group < kEpilogueTileN / 8; ++group) {
                const int chunk = tid * (kEpilogueTileN / 8) + group;
                const int swizzled_chunk =
                    (chunk & ~7) + ((chunk & 7) ^ ((chunk >> 3) & 3));
                reinterpret_cast<uint4 *>(out + (cta_row + tid) * n + block_col)[group] =
                    reinterpret_cast<const uint4 *>(stage_smem)
                        [swizzled_chunk];
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
        if (cta_rank == 0 && tid == 0) {
            next_work_smem[0] = atomicAdd(next_cluster_tile, kWorkerTileCount);
        }
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");

        const int local_work_addr = static_cast<int>(__cvta_generic_to_shared(next_work_smem));
        int rank_zero_work_addr;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(rank_zero_work_addr)
            : "r"(local_work_addr), "r"(0));
        int first_cluster_bid;
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
            : "=r"(first_cluster_bid)
            : "r"(rank_zero_work_addr)
            : "memory");
        if (first_cluster_bid >= cluster_tiles) {
            break;
        }
        const int last_cluster_bid =
            first_cluster_bid + kWorkerTileCount < cluster_tiles
                ? first_cluster_bid + kWorkerTileCount
                : cluster_tiles;

        if (warp_id == 4 && elect_sync()) {
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                const ExpertTileTask task = tasks[cluster_bid];
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
                    tma_load_4d<kClusterM>(a_smem,
                        &a_tensor_map,
                        0,
                        cta_row,
                        k_slice,
                        0,
                        tma_transaction_mbar_addr);
                    tma_load_4d<kClusterM>(b_smem,
                        &b_tensor_map,
                        0,
                        tile_col + cta_rank * (kTileN / kClusterM),
                        k_slice,
                        task.expert,
                        tma_transaction_mbar_addr);

                    producer_stage = (producer_stage + 1) % kStageCount;
                    if (producer_stage == 0) {
                        producer_mma_phase ^= 1;
                    }
                }
            }
        }

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

                const ExpertTileTask task = tasks[cluster_bid];
                const int tile_n = task.tile_n;
                const int cta_tile_row = task.tile_row + cta_rank * kTileM;
                const int cta_row = task.row_start + cta_tile_row;
                const int remaining_rows = task.row_count - cta_tile_row;
                const int valid_rows =
                    remaining_rows < 0 ? 0 : (remaining_rows < kTileM ? remaining_rows : kTileM);
                const int output_tile_col = tile_n * kLogicalTileN;
                const int tmem_row = cta_rank * kTileM + warp_id * kWarpSize;

                if constexpr (kDirectGmem) {
                    // Baseline path for A/B comparison with the staged epilogue.
                    if (tid < valid_rows) {
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
                        reinterpret_cast<uint4 *>(out + row * n + out_col)[0] = out_vec;
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
                                const int stage_smem_addr =
                                    mainloop_smem + kMainloopSmemBytes +
                                    smem_stage * kEpilogueStageBytes;
                                tma_store_3d(
                                    stage_smem_addr, &c_tensor_map, block_col, cta_row, 0);
                                tma_store_commit_group();
                            }
                        } else {
                            smem_to_gmem(n_block,
                                smem_stage,
                                cta_row,
                                output_tile_col,
                                valid_rows);
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
                    const int mbar_addr =
                        (base_epilogue_mbar_addr + epilogue_mainloop_stage * sizeof(uint64_t)) &
                        0xFEFFFFFF;
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

template <int kTileM, int kTileN, int kTileK, int kClusterM, bool kDirectGmem>
void launch_fc1_block_interleaved_swiglu_impl(
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b_interleaved,
    const int32_t *d_grouped_layout,
    __nv_bfloat16 *d_out,
    int m,
    int experts,
    int n,
    int k,
    cudaStream_t stream) {
    using Config = KernelConfig<kTileM, kTileN, kTileK, kClusterM, kDirectGmem>;
    constexpr int kClusterTileM = Config::kClusterTileM;
    constexpr int kDynamicSmemBytes = Config::kDynamicSmemBytes;
    constexpr int kTaskTileRows = kClusterTileM;

    if (d_a == nullptr || d_b_interleaved == nullptr || d_grouped_layout == nullptr || d_out == nullptr) {
        return;
    }
    if (m <= 0 || experts <= 0 || n <= 0 || k <= 0) {
        return;
    }
    if (n % (kTileN / 2) != 0 || k % kTileK != 0) {
        return;
    }

    CUtensorMap a_tensor_map;
    CUtensorMap b_tensor_map;
    CUtensorMap c_tensor_map;
    const int physical_n = 2 * n;
    const bool tensor_maps_ok =
        init_a_tensor_map<kTileM, kTileK>(&a_tensor_map, d_a, m, k) &&
        init_b_tensor_map<kTileN, kTileK, kClusterM>(
            &b_tensor_map, d_b_interleaved, experts, physical_n, k) &&
        init_c_tensor_map<kTileM, Config::kEpilogueTileN>(
            &c_tensor_map, d_out, m, n);
    if (!tensor_maps_ok) {
        return;
    }

    const int grid_n = n / (kTileN / 2);
    const int max_m_tasks = (m + kTaskTileRows - 1) / kTaskTileRows + experts;
    const int max_tasks = max_m_tasks * grid_n;
    const size_t metadata_bytes = (2 * experts + 2) * sizeof(int);
    const size_t task_offset =
        (metadata_bytes + alignof(ExpertTileTask) - 1) & ~(alignof(ExpertTileTask) - 1);
    const size_t workspace_bytes = task_offset + max_tasks * sizeof(ExpertTileTask);
    void *workspace = nullptr;
    if (cudaMallocAsync(&workspace, workspace_bytes, stream) != cudaSuccess) {
        return;
    }

    auto *expert_offsets = static_cast<int *>(workspace);
    int *expert_counts = expert_offsets + experts;
    int *task_count = expert_counts + experts;
    int *next_cluster_tile = task_count + 1;
    auto *tasks = reinterpret_cast<ExpertTileTask *>(static_cast<char *>(workspace) + task_offset);
    cudaMemsetAsync(expert_offsets, 0x7f, experts * sizeof(int), stream);
    cudaMemsetAsync(expert_counts, 0, experts * sizeof(int), stream);
    cudaMemsetAsync(task_count, 0, 2 * sizeof(int), stream);

    constexpr int kMetadataThreads = 256;
    collect_expert_metadata<<<(m + kMetadataThreads - 1) / kMetadataThreads,
        kMetadataThreads,
        0,
        stream>>>(d_grouped_layout, m, experts, expert_offsets, expert_counts);
    build_expert_tile_tasks<kTaskTileRows>
        <<<(experts + kMetadataThreads - 1) / kMetadataThreads,
            kMetadataThreads,
            0,
            stream>>>(expert_offsets, expert_counts, experts, grid_n, tasks, task_count);

    const int max_cluster_tiles = max_tasks;
    const int persistent_clusters =
        (max_cluster_tiles + kPersistentTileCount - 1) / kPersistentTileCount;
    auto kernel =
        fc1_block_interleaved_swiglu_kernel<kTileM, kTileN, kTileK, kClusterM, kDirectGmem>;
    if constexpr (kDynamicSmemBytes > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes);
    }
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = dim3(kClusterM * persistent_clusters, 1, 1);
    launch_config.blockDim = dim3(kBlockThreads, 1, 1);
    launch_config.dynamicSmemBytes = kDynamicSmemBytes;
    launch_config.stream = stream;
    int resident_clusters = 0;
    const cudaError_t occupancy_status =
        cudaOccupancyMaxActiveClusters(&resident_clusters, kernel, &launch_config);
    if (occupancy_status != cudaSuccess || resident_clusters <= 0) {
        int device = 0;
        int sm_count = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        resident_clusters = sm_count / kClusterM > 0 ? sm_count / kClusterM : 1;
    }
    const int launched_clusters =
        resident_clusters < persistent_clusters ? resident_clusters : persistent_clusters;
    kernel<<<kClusterM * launched_clusters, kBlockThreads, kDynamicSmemBytes, stream>>>(
        a_tensor_map,
        b_tensor_map,
        c_tensor_map,
        tasks,
        task_count,
        next_cluster_tile,
        d_out,
        n,
        k);
    cudaFreeAsync(workspace, stream);
}

} // namespace

void launch_fc1_block_interleaved_swiglu(
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b_interleaved,
    const int32_t *d_grouped_layout,
    __nv_bfloat16 *d_out,
    int m,
    int experts,
    int n,
    int k,
    cudaStream_t stream) {
    launch_fc1_block_interleaved_swiglu_impl<SM110A_FC1_TILE_M,
        SM110A_FC1_TILE_N,
        SM110A_FC1_TILE_K,
        SM110A_FC1_CLUSTER_M,
        (SM110A_FC1_DIRECT_GMEM != 0)>(
        d_a, d_b_interleaved, d_grouped_layout, d_out, m, experts, n, k, stream);
}

extern "C" void sm110a_fc1_block_interleaved_swiglu_launch(
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b_interleaved,
    const int32_t *d_grouped_layout,
    __nv_bfloat16 *d_out,
    int m,
    int experts,
    int n,
    int k,
    cudaStream_t stream) {
    launch_fc1_block_interleaved_swiglu(
        d_a, d_b_interleaved, d_grouped_layout, d_out, m, experts, n, k, stream);
}
