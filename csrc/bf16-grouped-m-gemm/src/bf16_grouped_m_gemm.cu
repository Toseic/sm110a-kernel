#include <cstdlib>
#include <cstdint>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bf16_grouped_m_gemm.h"

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
#ifndef SM110A_GROUPED_ENABLE_CLC
#define SM110A_GROUPED_ENABLE_CLC 0
#endif

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
constexpr int kMacroSwizzleN = 16;

template <
    int kTileM,
    int kTileN,
    int kTileK,
    int kEpilogueTileN,
    int kClusterM>
struct KernelConfig {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert(kTileM == 4 * kWarpSize);
    static_assert(
        kEpilogueTileN == 32 ||
        kEpilogueTileN == 64 ||
        kEpilogueTileN == 128);
    static_assert(kTileN % kEpilogueTileN == 0);
    static_assert(kTileN % kClusterM == 0);
    static_assert(kTileK % 64 == 0);
    static_assert(kTileK % kMmaK == 0);
    static constexpr int kClusterTileM = kClusterM * kTileM;
    static constexpr int kASmemBytes =
        kTileM * kTileK * sizeof(nv_bfloat16);
    static constexpr int kBSmemBytes =
        (kTileN / kClusterM) * kTileK * sizeof(nv_bfloat16);
    static constexpr int kPerStageSmemBytes = kASmemBytes + kBSmemBytes;
    static constexpr int kEpilogueStageBytes =
        kTileM * kEpilogueTileN * sizeof(nv_bfloat16);
    // A 128-column epilogue stage occupies 32 KiB. Keeping one such stage
    // leaves enough shared memory for two mainloop stages in the largest
    // supported configuration.
    static constexpr int kEpilogueStageCount =
        kEpilogueTileN == 128 ? 1 : 2;
    static constexpr int kEpilogueSmemBytes =
        kEpilogueStageCount * kEpilogueStageBytes;
    // Pack as many mainloop pipeline stages as shared memory allows for this
    // configuration. Blackwell has ~228 KB of SMEM per CTA; leave a small
    // margin for statically allocated barriers and bookkeeping.
    static constexpr int kMaxSmemBytes = 227 * 1024;
    static constexpr int kStageCount =
        (kMaxSmemBytes - kEpilogueSmemBytes) / kPerStageSmemBytes;
    static_assert(kStageCount >= 2);
    static constexpr int kMainloopSmemBytes =
        kStageCount * kPerStageSmemBytes;
    static constexpr int kDynamicSmemBytes =
        kMainloopSmemBytes + kEpilogueSmemBytes;
    // Backward-compat alias used inside the kernel function body.
    static constexpr int kStageSmemBytes = kPerStageSmemBytes;
};

struct ExpertTileTask {
    int row_start;
    int row_count;
    int tile_row;
    int expert;
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

template <int kTileM, int kClusterM>
__global__ void build_expert_tile_tasks(
    const int *expert_offsets,
    const int *expert_counts,
    int experts,
    ExpertTileTask *tasks,
    int *task_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    constexpr int kClusterTileM =
        kTileM * kClusterM;
    int next_task = 0;
    for (int expert = 0; expert < experts; ++expert) {
        const int row_count = expert_counts[expert];
        if (row_count == 0) {
            continue;
        }
        const int row_start = expert_offsets[expert];
        for (int tile_row = 0; tile_row < row_count;
             tile_row += kClusterTileM) {
            tasks[next_task++] = {
                row_start, row_count, tile_row, expert};
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

__device__ inline constexpr uint64_t matrix_desc_encode(uint64_t value) {
    return (value & 0x3FFFFULL) >> 4ULL;
}

__device__ inline uint64_t make_k_major_128b_desc(int smem_addr) {
    constexpr int kStrideByteOffset = 8 * 64 * sizeof(nv_bfloat16);
    return matrix_desc_encode(smem_addr) |
           (matrix_desc_encode(kStrideByteOffset) << 32ULL) |
           (1ULL << 46ULL) |
           (2ULL << 61ULL);
}

__device__ inline void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr), "r"(count));
}

__device__ inline void mbarrier_wait(int mbar_addr, int phase) {
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

__device__ inline void clc_mbarrier_wait(int mbar_addr, int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred done;\n\t"
                 "WAIT_CLC:\n\t"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE_CLC;\n\t"
                 "bra.uni WAIT_CLC;\n\t"
                 "COMPLETE_CLC:\n\t"
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
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cluster.global."
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
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cluster.global."
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
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
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

__device__ inline void prefetch_tensor_map(const void *tensor_map) {
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
__device__ inline void tcgen05_commit(int mbar_addr) {
    if constexpr (kClusterM == 1) {
        asm volatile(
            "tcgen05.commit.cta_group::1."
            "mbarrier::arrive::one.shared::cluster.b64 "
            "[%0];" ::"r"(mbar_addr)
            : "memory");
    } else {
        asm volatile(
            "tcgen05.commit.cta_group::2."
            "mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
            "[%0], %1;" ::"r"(mbar_addr),
            "h"(static_cast<int16_t>((1 << kClusterM) - 1))
            : "memory");
    }
}

template <bool kUseMacroSwizzle>
__device__ inline void compute_macro_swizzled_tile(
    int cluster_bid,
    int grid_m,
    int grid_n,
    int &tile_m,
    int &tile_n) {
    if constexpr (!kUseMacroSwizzle) {
        tile_m = cluster_bid / grid_n;
        tile_n = cluster_bid % grid_n;
        return;
    }

    // Keep an 8x8 tile working set together in launch order. The N dimension
    // changes fastest so concurrently scheduled clusters share the same A tile.
    const int full_n_band_size = grid_m * kMacroSwizzleN;
    const int macro_n = cluster_bid / full_n_band_size;
    const int bid_in_n_band =
        cluster_bid - macro_n * full_n_band_size;
    const int remaining_n = grid_n - macro_n * kMacroSwizzleN;
    const int real_macro_n =
        remaining_n < kMacroSwizzleN ? remaining_n : kMacroSwizzleN;
    const int macro_size = kMacroSwizzleM * real_macro_n;
    const int macro_m = bid_in_n_band / macro_size;
    const int macro_lane = bid_in_n_band - macro_m * macro_size;

    tile_m = macro_m * kMacroSwizzleM + macro_lane / real_macro_n;
    tile_n = macro_n * kMacroSwizzleN + macro_lane % real_macro_n;
}

template <int kTileM, int kTileK>
inline bool init_a_tensor_map(
    CUtensorMap *tensor_map,
    const nv_bfloat16 *ptr,
    uint64_t m,
    uint64_t k) {
    constexpr uint32_t rank = 4;
    uint64_t global_dim[rank] = {64, m, k / 64, 1};
    uint64_t global_stride[rank - 1] = {
        k * sizeof(nv_bfloat16),
        128,
        m * k * sizeof(nv_bfloat16)};
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
    uint64_t global_stride[rank - 1] = {
        k * sizeof(nv_bfloat16),
        128,
        n * k * sizeof(nv_bfloat16)};
    uint32_t box_dim[rank] = {
        64, kTileN / kClusterM, kTileK / 64, 1};
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
        n * sizeof(nv_bfloat16),
        m * n * sizeof(nv_bfloat16)};
    constexpr int kStoreTileN =
        kEpilogueTileN < 64 ? kEpilogueTileN : 64;
    uint32_t box_dim[rank] = {kStoreTileN, kTileM, 1};
    uint32_t element_stride[rank] = {1, 1, 1};
    constexpr CUtensorMapSwizzle kSwizzle =
        kStoreTileN == 32
        ? CU_TENSOR_MAP_SWIZZLE_64B
        : CU_TENSOR_MAP_SWIZZLE_128B;

    return cuTensorMapEncodeTiled(tensor_map,
               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
               rank,
               ptr,
               global_dim,
               global_stride,
               box_dim,
               element_stride,
               CU_TENSOR_MAP_INTERLEAVE_NONE,
               kSwizzle,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <
    int kTileM,
    int kTileN,
    int kTileK,
    int kEpilogueTileN,
    int kClusterM,
    bool kPersistent,
    bool kDynamicClc = false>
__global__ __cluster_dims__(kClusterM, 1, 1)
__launch_bounds__(kBlockThreads) void bf16_grouped_m_gemm_kernel(
    const __grid_constant__ CUtensorMap a_tensor_map,
    const __grid_constant__ CUtensorMap b_tensor_map,
    const __grid_constant__ CUtensorMap c_tensor_map,
    const ExpertTileTask *tasks,
    const int *task_count,
    int *next_cluster_tile,
    nv_bfloat16 *c,
    int n,
    int k) {
    using Config = KernelConfig<
        kTileM, kTileN, kTileK, kEpilogueTileN, kClusterM>;
    static_assert(!kDynamicClc || kPersistent);
    constexpr int kClusterTileM = Config::kClusterTileM;
    constexpr int kStageCount = Config::kStageCount;
    constexpr int kASmemBytes = Config::kASmemBytes;
    constexpr int kBSmemBytes = Config::kBSmemBytes;
    constexpr int kStageSmemBytes = Config::kStageSmemBytes;
    constexpr int kMainloopSmemBytes = Config::kMainloopSmemBytes;
    constexpr int kEpilogueStageBytes = Config::kEpilogueStageBytes;
    constexpr int kEpilogueStageCount = Config::kEpilogueStageCount;
    constexpr int kEpilogueStoreTileN =
        kEpilogueTileN < 64 ? kEpilogueTileN : 64;
    constexpr int kEpilogueStoreTiles =
        kEpilogueTileN / kEpilogueStoreTileN;
    constexpr int kChunksPerStoreTile = kEpilogueStoreTileN / 8;
    constexpr int kSwizzlePhaseMask =
        kEpilogueStoreTileN == 32 ? 3 : 7;
    constexpr int kStoreTileBytes =
        kTileM * kEpilogueStoreTileN * sizeof(nv_bfloat16);
    constexpr int kMmaStepsPerStage = kTileK / kMmaK;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int worker_id = blockIdx.x / kClusterM;
    const int grid_m = *task_count;
    const int grid_n = n / kTileN;
    const int cluster_tiles = grid_m * grid_n;
    constexpr int kWorkerTileCount =
        kPersistent ? kPersistentTileCount : 1;
    const int static_first_cluster_bid =
        worker_id * kWorkerTileCount;
    if constexpr (!kPersistent) {
        if (static_first_cluster_bid >= cluster_tiles) {
            return;
        }
    }

    __shared__ int tmem_base[1];
    __shared__ int next_work_smem[2];
    __shared__ __align__(16) uint4 clc_response[1];
    __shared__ uint64_t clc_mbar[1];
    __shared__ uint64_t tma_mbars[kStageCount];
    __shared__ uint64_t mma_mbars[kStageCount];
    __shared__ uint64_t mainloop_mbars[kTmemStageCount];
    __shared__ uint64_t epilogue_mbars[kTmemStageCount];

    extern __shared__ __align__(1024) char dynamic_smem[];
    const int mainloop_smem = static_cast<int>(__cvta_generic_to_shared(dynamic_smem));
    const int epilogue_smem = mainloop_smem + kMainloopSmemBytes;
    const int base_tma_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(tma_mbars));
    const int base_mma_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(mma_mbars));
    const int base_mainloop_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(mainloop_mbars));
    const int base_epilogue_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(epilogue_mbars));

    if (tid == 0) {
        for (int stage = 0; stage < kStageCount; ++stage) {
            mbarrier_init(
                base_tma_mbar_addr + stage * sizeof(uint64_t), kClusterM);
            mbarrier_init(base_mma_mbar_addr + stage * sizeof(uint64_t), 1);
        }
        for (int stage = 0; stage < kTmemStageCount; ++stage) {
            mbarrier_init(
                base_mainloop_mbar_addr + stage * sizeof(uint64_t), 1);
            if constexpr (kPersistent) {
                mbarrier_init(
                    base_epilogue_mbar_addr + stage * sizeof(uint64_t),
                    4 * kClusterM);
            }
        }
        if constexpr (kDynamicClc) {
            mbarrier_init(
                static_cast<int>(__cvta_generic_to_shared(clc_mbar)), 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp_id == 5) {
        static_assert(kTileN * kTmemStageCount <= 512);
        const int tmem_base_smem = static_cast<int>(__cvta_generic_to_shared(tmem_base));
        if constexpr (kClusterM == 1) {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
                "[%0], %1;" ::"r"(tmem_base_smem),
                "r"(kTileN * kTmemStageCount));
        } else {
            asm volatile(
                "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 "
                "[%0], %1;" ::"r"(tmem_base_smem),
                "r"(kTileN * kTmemStageCount));
        }
    }
    if (warp_id == 0 && elect_sync()) {
        prefetch_tensor_map(&a_tensor_map);
        prefetch_tensor_map(&b_tensor_map);
        prefetch_tensor_map(&c_tensor_map);
    }
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    const int tmem_addr = tmem_base[0];
    constexpr uint32_t instruction_desc =
        (1U << 4U) |
        (1U << 7U) |
        (1U << 10U) |
        (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
        (static_cast<uint32_t>(kClusterTileM) >> 4U << 24U);

    const int k_tiles = k / kTileK;

    auto tmem_to_smem =
        [&](int mainloop_stage, int n_block, int smem_stage) {
        char *stage_smem =
            dynamic_smem + kMainloopSmemBytes + smem_stage * kEpilogueStageBytes;

        for (int n_group = 0; n_group < kEpilogueTileN / 8; ++n_group) {
            float values[8];
            const int tmem_row =
                cta_rank * kTileM + warp_id * kWarpSize;
            const int tmem_col =
                mainloop_stage * kTileN +
                n_block * kEpilogueTileN + n_group * 8;
            const int load_addr = tmem_addr + (tmem_row << 16) + tmem_col;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=f"(values[0]),
                "=f"(values[1]),
                "=f"(values[2]),
                "=f"(values[3]),
                "=f"(values[4]),
                "=f"(values[5]),
                "=f"(values[6]),
                "=f"(values[7])
                : "r"(load_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            nv_bfloat162 packed[4];
            for (int i = 0; i < 4; ++i) {
                packed[i] =
                    __float22bfloat162_rn({values[2 * i], values[2 * i + 1]});
            }

            const int store_tile = n_group / kChunksPerStoreTile;
            const int chunk_in_store =
                tid * kChunksPerStoreTile +
                n_group % kChunksPerStoreTile;
            const int swizzled_chunk =
                (chunk_in_store & ~7) +
                ((chunk_in_store & 7) ^
                    ((chunk_in_store >> 3) & kSwizzlePhaseMask));
            reinterpret_cast<int4 *>(
                stage_smem + store_tile * kStoreTileBytes)[swizzled_chunk] =
                reinterpret_cast<int4 *>(packed)[0];
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
    int clc_phase = 0;
    int dynamic_first_cluster_bid = static_first_cluster_bid;

    while (true) {
        int first_cluster_bid;
        if constexpr (kDynamicClc) {
            first_cluster_bid = dynamic_first_cluster_bid;

            const int clc_response_addr = static_cast<int>(
                __cvta_generic_to_shared(clc_response));
            const int clc_mbar_addr =
                static_cast<int>(__cvta_generic_to_shared(clc_mbar));
            if (tid == 0) {
                asm volatile(
                    "mbarrier.arrive.expect_tx.relaxed.cluster."
                    "shared::cta.b64 _, [%0], 16;"
                    ::"r"(clc_mbar_addr)
                    : "memory");
            }
            asm volatile("barrier.cluster.arrive.relaxed.aligned;");
            asm volatile("barrier.cluster.wait.aligned;");
            if (cta_rank == 0 && tid == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta."
                    "mbarrier::complete_tx::bytes.multicast::cluster::all."
                    "b128 [%0], [%1];"
                    ::"r"(clc_response_addr), "r"(clc_mbar_addr)
                    : "memory");
            }
        } else if constexpr (kPersistent) {
            if (cta_rank == 0 && tid == 0) {
                next_work_smem[0] =
                    atomicAdd(next_cluster_tile, kPersistentTileCount);
            }
            asm volatile("barrier.cluster.arrive.release.aligned;");
            asm volatile("barrier.cluster.wait.acquire.aligned;");

            const int local_work_addr =
                static_cast<int>(__cvta_generic_to_shared(next_work_smem));
            int rank_zero_work_addr;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(rank_zero_work_addr)
                : "r"(local_work_addr), "r"(0));
            asm volatile(
                "ld.shared::cluster.u32 %0, [%1];"
                : "=r"(first_cluster_bid)
                : "r"(rank_zero_work_addr)
                : "memory");
        } else {
            first_cluster_bid = static_first_cluster_bid;
        }
        if constexpr (!kDynamicClc) {
            if (first_cluster_bid >= cluster_tiles) {
                break;
            }
        }
        const bool has_work = first_cluster_bid < cluster_tiles;
        const int last_cluster_bid = has_work
            ? (first_cluster_bid + kWorkerTileCount < cluster_tiles
                      ? first_cluster_bid + kWorkerTileCount
                      : cluster_tiles)
            : first_cluster_bid;

    // The producer waits until MMA has released a stage before TMA overwrites it.
    if (warp_id == 4 && elect_sync()) {
        for (int cluster_bid = first_cluster_bid;
             cluster_bid < last_cluster_bid;
             ++cluster_bid) {
            int tile_m;
            int tile_n;
            compute_macro_swizzled_tile<kPersistent>(
                cluster_bid, grid_m, grid_n, tile_m, tile_n);
            const ExpertTileTask task = tasks[tile_m];
            const int cta_row =
                task.row_start + task.tile_row + cta_rank * kTileM;
            const int tile_col = tile_n * kTileN;

            for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                const int mma_mbar_addr =
                    base_mma_mbar_addr + producer_stage * sizeof(uint64_t);
                const int tma_mbar_addr =
                    base_tma_mbar_addr + producer_stage * sizeof(uint64_t);
                const int a_smem =
                    mainloop_smem + producer_stage * kStageSmemBytes;
                const int b_smem = a_smem + kASmemBytes;

                mbarrier_wait(mma_mbar_addr, producer_mma_phase);
                const int tma_transaction_mbar_addr =
                    tma_mbar_addr & 0xFEFFFFFF;
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
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

    // The consumer advances independently and releases each stage after MMA has
    // finished reading it.
    if (cta_rank == 0 && warp_id == 5 && elect_sync()) {
        for (int cluster_bid = first_cluster_bid;
             cluster_bid < last_cluster_bid;
             ++cluster_bid) {
            if constexpr (kPersistent) {
                mbarrier_wait(
                    base_epilogue_mbar_addr +
                        consumer_mainloop_stage * sizeof(uint64_t),
                    consumer_epilogue_phase);
            }

            const int stage_tmem_addr =
                tmem_addr + consumer_mainloop_stage * kTileN;
            for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                const int tma_mbar_addr =
                    base_tma_mbar_addr + consumer_stage * sizeof(uint64_t);
                const int mma_mbar_addr =
                    base_mma_mbar_addr + consumer_stage * sizeof(uint64_t);

                mbarrier_wait(tma_mbar_addr, consumer_tma_phase);
                asm volatile(
                    "tcgen05.fence::after_thread_sync;" ::: "memory");
                {
                    const int a_smem =
                        mainloop_smem +
                        consumer_stage * kStageSmemBytes;
                    const int b_smem = a_smem + kASmemBytes;
                    for (int mma_k = 0; mma_k < kMmaStepsPerStage;
                         ++mma_k) {
                        const int k_slab =
                            mma_k / (64 / kMmaK);
                        const int k_in_slab =
                            mma_k % (64 / kMmaK);
                        const int a_byte_offset =
                            k_slab * kTileM * 64 *
                                sizeof(nv_bfloat16) +
                            k_in_slab * kMmaK *
                                sizeof(nv_bfloat16);
                        const int b_byte_offset =
                            k_slab * (kTileN / kClusterM) *
                                64 * sizeof(nv_bfloat16) +
                            k_in_slab * kMmaK *
                                sizeof(nv_bfloat16);
                        const int accumulate =
                            tile_k != 0 || mma_k != 0;
                        tcgen05_mma_bf16<kClusterM>(
                            stage_tmem_addr,
                            make_k_major_128b_desc(
                                a_smem + a_byte_offset),
                            make_k_major_128b_desc(
                                b_smem + b_byte_offset),
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
                base_mainloop_mbar_addr +
                consumer_mainloop_stage * sizeof(uint64_t);
            tcgen05_commit<kClusterM>(mainloop_mbar_addr);

            consumer_mainloop_stage =
                (consumer_mainloop_stage + 1) % kTmemStageCount;
            if (consumer_mainloop_stage == 0) {
                if constexpr (kPersistent) {
                    consumer_epilogue_phase ^= 1;
                }
            }
        }
    }

    if (warp_id < 4) {
        constexpr int kEpilogueBlockCount = kTileN / kEpilogueTileN;

        for (int cluster_bid = first_cluster_bid;
             cluster_bid < last_cluster_bid;
             ++cluster_bid) {
            mbarrier_wait(base_mainloop_mbar_addr +
                    epilogue_mainloop_stage * sizeof(uint64_t),
                epilogue_mainloop_phase);
            asm volatile(
                "tcgen05.fence::after_thread_sync;" ::: "memory");

            int tile_m;
            int tile_n;
            compute_macro_swizzled_tile<kPersistent>(
                cluster_bid, grid_m, grid_n, tile_m, tile_n);
            const ExpertTileTask task = tasks[tile_m];
            const int cta_tile_row =
                task.tile_row + cta_rank * kTileM;
            const int cta_row =
                task.row_start + cta_tile_row;
            const int remaining_rows = task.row_count - cta_tile_row;
            const int valid_rows =
                remaining_rows < 0
                ? 0
                : (remaining_rows < kTileM ? remaining_rows : kTileM);
            const int tile_col = tile_n * kTileN;

            auto release_tmem = [&]() {
                if constexpr (kPersistent) {
                    if (elect_sync()) {
                        const int mbar_addr =
                            (base_epilogue_mbar_addr +
                                epilogue_mainloop_stage *
                                    sizeof(uint64_t)) &
                            0xFEFFFFFF;
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 "
                            "_, [%0];" ::"r"(mbar_addr)
                            : "memory");
                    }
                }
            };

            tmem_to_smem(epilogue_mainloop_stage, 0, 0);
            asm volatile("bar.sync 1, 128;" ::: "memory");
            if constexpr (kEpilogueBlockCount == 1) {
                release_tmem();
            }

            for (int n_block = 0; n_block < kEpilogueBlockCount;
                 ++n_block) {
                const int smem_stage = n_block % kEpilogueStageCount;
                const int stage_smem =
                    epilogue_smem + smem_stage * kEpilogueStageBytes;

                const int block_col =
                    tile_col + n_block * kEpilogueTileN;
                if (valid_rows == kTileM) {
                    if (tid == 0) {
                        for (int store_tile = 0;
                             store_tile < kEpilogueStoreTiles;
                             ++store_tile) {
                            tma_store_3d(
                                stage_smem +
                                    store_tile * kStoreTileBytes,
                                &c_tensor_map,
                                block_col +
                                    store_tile * kEpilogueStoreTileN,
                                cta_row,
                                0);
                        }
                        tma_store_commit_group();
                    }
                } else if (tid < valid_rows) {
                    char *stage_smem_ptr =
                        dynamic_smem + kMainloopSmemBytes +
                        smem_stage * kEpilogueStageBytes;
                    for (int n_group = 0;
                         n_group < kEpilogueTileN / 8;
                         ++n_group) {
                        const int store_tile =
                            n_group / kChunksPerStoreTile;
                        const int chunk_in_store =
                            tid * kChunksPerStoreTile +
                            n_group % kChunksPerStoreTile;
                        const int swizzled_chunk =
                            (chunk_in_store & ~7) +
                            ((chunk_in_store & 7) ^
                                ((chunk_in_store >> 3) &
                                    kSwizzlePhaseMask));
                        const int4 value =
                            reinterpret_cast<int4 *>(
                                stage_smem_ptr +
                                store_tile * kStoreTileBytes)[
                                    swizzled_chunk];
                        const int output_offset =
                            (cta_row + tid) * n +
                            block_col + n_group * 8;
                        reinterpret_cast<int4 *>(
                            c + output_offset)[0] = value;
                    }
                }

                if (n_block + 1 < kEpilogueBlockCount) {
                    if (kEpilogueStageCount == 1 || n_block > 0) {
                        if (tid == 0) {
                            asm volatile(
                                "cp.async.bulk.wait_group.read %0;"
                                :
                                : "n"(kEpilogueStageCount - 1)
                                : "memory");
                        }
                        asm volatile("bar.sync 1, 128;" ::: "memory");
                    }

                    const int next_stage =
                        (n_block + 1) % kEpilogueStageCount;
                    tmem_to_smem(epilogue_mainloop_stage,
                        n_block + 1,
                        next_stage);
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    if (n_block + 2 == kEpilogueBlockCount) {
                        release_tmem();
                    }
                }
            }

            if (tid == 0) {
                asm volatile(
                    "cp.async.bulk.wait_group 0;" ::: "memory");
            }
            asm volatile("bar.sync 1, 128;" ::: "memory");

            epilogue_mainloop_stage =
                (epilogue_mainloop_stage + 1) % kTmemStageCount;
            if (epilogue_mainloop_stage == 0) {
                epilogue_mainloop_phase ^= 1;
            }
        }
    }

        if constexpr (kDynamicClc) {
            const int clc_mbar_addr =
                static_cast<int>(__cvta_generic_to_shared(clc_mbar));
            if (tid == 0) {
                clc_mbarrier_wait(clc_mbar_addr, clc_phase);

                const uint4 response = clc_response[0];
                int canceled;
                int canceled_cta_x = 0;
                asm volatile(
                    "{\n\t"
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
                    : "r"(response.x),
                    "r"(response.y),
                    "r"(response.z),
                    "r"(response.w)
                    : "memory");
                next_work_smem[0] = canceled;
                next_work_smem[1] =
                    canceled_cta_x / kClusterM * kPersistentTileCount;
            }
            __syncthreads();
            if (next_work_smem[0] == 0) {
                break;
            }
            dynamic_first_cluster_bid = next_work_smem[1];
            clc_phase ^= 1;

            asm volatile(
                "fence.proxy.async::generic.release."
                "sync_restrict::shared::cta.cluster;"
                ::: "memory");
            asm volatile("barrier.cluster.arrive.relaxed.aligned;");
            asm volatile("barrier.cluster.wait.aligned;");
            asm volatile(
                "fence.proxy.async::generic.acquire."
                "sync_restrict::shared::cluster.cluster;"
                ::: "memory");
        } else if constexpr (!kPersistent) {
            break;
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    if (warp_id == 0) {
        if constexpr (kClusterM == 1) {
            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 "
                "%0, %1;" ::"r"(tmem_addr),
                "r"(kTileN * kTmemStageCount));
        } else {
            asm volatile(
                "tcgen05.dealloc.cta_group::2.sync.aligned.b32 "
                "%0, %1;" ::"r"(tmem_addr),
                "r"(kTileN * kTmemStageCount));
        }
    }
}

template <
    int kTileM,
    int kTileN,
    int kTileK,
    int kEpilogueTileN,
    int kClusterM>
void launch_bf16_grouped_m_gemm_impl(
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b,
    const int32_t *d_grouped_layout,
    __nv_bfloat16 *d_c,
    int m,
    int experts,
    int n,
    int k,
    cudaStream_t stream) {
    using Config = KernelConfig<
        kTileM, kTileN, kTileK, kEpilogueTileN, kClusterM>;
    constexpr int kClusterTileM = Config::kClusterTileM;
    constexpr int kDynamicSmemBytes = Config::kDynamicSmemBytes;
    if (n % kTileN != 0 || k % kTileK != 0) {
        return;
    }

    CUtensorMap a_tensor_map;
    CUtensorMap b_tensor_map;
    CUtensorMap c_tensor_map;
    const bool tensor_maps_ok =
        init_a_tensor_map<kTileM, kTileK>(
            &a_tensor_map, d_a, m, k) &&
        init_b_tensor_map<kTileN, kTileK, kClusterM>(
            &b_tensor_map, d_b, experts, n, k) &&
        init_c_tensor_map<kTileM, kEpilogueTileN>(
            &c_tensor_map, d_c, m, n);
    if (!tensor_maps_ok) {
        return;
    }

    const int max_tasks =
        (m + kClusterTileM - 1) / kClusterTileM + experts;
    const size_t metadata_bytes =
        (2 * experts + 2) * sizeof(int);
    const size_t task_offset =
        (metadata_bytes + alignof(ExpertTileTask) - 1) &
        ~(alignof(ExpertTileTask) - 1);
    const size_t workspace_bytes =
        task_offset + max_tasks * sizeof(ExpertTileTask);
    void *workspace = nullptr;
    if (cudaMallocAsync(&workspace, workspace_bytes, stream) !=
        cudaSuccess) {
        return;
    }

    auto *expert_offsets = static_cast<int *>(workspace);
    int *expert_counts = expert_offsets + experts;
    int *task_count = expert_counts + experts;
    int *next_cluster_tile = task_count + 1;
    auto *tasks = reinterpret_cast<ExpertTileTask *>(
        static_cast<char *>(workspace) + task_offset);
    cudaMemsetAsync(
        expert_offsets, 0x7f, experts * sizeof(int), stream);
    cudaMemsetAsync(
        expert_counts, 0, experts * sizeof(int), stream);
    cudaMemsetAsync(task_count, 0, 2 * sizeof(int), stream);
    constexpr int kMetadataThreads = 256;
    collect_expert_metadata<<<
        (m + kMetadataThreads - 1) / kMetadataThreads,
        kMetadataThreads,
        0,
        stream>>>(d_grouped_layout,
        m,
        experts,
        expert_offsets,
        expert_counts);
    build_expert_tile_tasks<kTileM, kClusterM><<<1, 1, 0, stream>>>(
        expert_offsets, expert_counts, experts, tasks, task_count);

    const int max_cluster_tiles = max_tasks * (n / kTileN);
    // Resident persistent clusters repeatedly claim two output tiles from a
    // global queue. Small average groups retain the one-tile row-major path.
    const bool use_persistent =
        m / experts >=
        kPersistentTileCount * kClusterTileM;
    if (use_persistent) {
#if SM110A_GROUPED_ENABLE_CLC
        const char *clc_env = std::getenv("SM110A_GROUPED_M_CLC");
        const bool use_dynamic_clc =
            clc_env != nullptr && clc_env[0] == '1';
#endif
        const int persistent_clusters =
            (max_cluster_tiles + kPersistentTileCount - 1) /
            kPersistentTileCount;
#if SM110A_GROUPED_ENABLE_CLC
        if (use_dynamic_clc) {
            auto kernel =
                bf16_grouped_m_gemm_kernel<
                    kTileM,
                    kTileN,
                    kTileK,
                    kEpilogueTileN,
                    kClusterM,
                    true,
                    true>;
            if constexpr (kDynamicSmemBytes > 48 * 1024) {
                cudaFuncSetAttribute(kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kDynamicSmemBytes);
            }
            // CLC needs a full logical grid: resident clusters steal work by
            // canceling clusters that have not started yet.
            kernel<<<kClusterM * persistent_clusters,
                kBlockThreads,
                kDynamicSmemBytes,
                stream>>>(a_tensor_map,
                b_tensor_map,
                c_tensor_map,
                tasks,
                task_count,
                next_cluster_tile,
                d_c,
                n,
                k);
            cudaFreeAsync(workspace, stream);
            return;
        }
#endif

        auto kernel = bf16_grouped_m_gemm_kernel<
            kTileM, kTileN, kTileK, kEpilogueTileN, kClusterM, true>;
        if constexpr (kDynamicSmemBytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kDynamicSmemBytes);
        }
        cudaLaunchConfig_t launch_config{};
        launch_config.gridDim =
            dim3(kClusterM * persistent_clusters, 1, 1);
        launch_config.blockDim = dim3(kBlockThreads, 1, 1);
        launch_config.dynamicSmemBytes = kDynamicSmemBytes;
        launch_config.stream = stream;
        int resident_clusters = 0;
        const cudaError_t occupancy_status =
            cudaOccupancyMaxActiveClusters(
                &resident_clusters, kernel, &launch_config);
        if (occupancy_status != cudaSuccess || resident_clusters <= 0) {
            int device = 0;
            int sm_count = 0;
            cudaGetDevice(&device);
            cudaDeviceGetAttribute(
                &sm_count, cudaDevAttrMultiProcessorCount, device);
            resident_clusters =
                sm_count / kClusterM > 0 ? sm_count / kClusterM : 1;
        }
        const int launched_clusters =
            resident_clusters < persistent_clusters
            ? resident_clusters
            : persistent_clusters;
        kernel<<<kClusterM * launched_clusters,
            kBlockThreads,
            kDynamicSmemBytes,
            stream>>>(a_tensor_map,
            b_tensor_map,
            c_tensor_map,
            tasks,
            task_count,
            next_cluster_tile,
            d_c,
            n,
            k);
    } else {
        auto kernel = bf16_grouped_m_gemm_kernel<
            kTileM, kTileN, kTileK, kEpilogueTileN, kClusterM, false>;
        if constexpr (kDynamicSmemBytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kDynamicSmemBytes);
        }
        kernel<<<kClusterM * max_cluster_tiles,
            kBlockThreads,
            kDynamicSmemBytes,
            stream>>>(a_tensor_map,
            b_tensor_map,
            c_tensor_map,
            tasks,
            task_count,
            next_cluster_tile,
            d_c,
            n,
            k);
    }
    cudaFreeAsync(workspace, stream);
}

} // namespace

void launch_bf16_grouped_m_gemm(
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b,
    const int32_t *d_grouped_layout,
    __nv_bfloat16 *d_c,
    int m,
    int experts,
    int n,
    int k,
    cudaStream_t stream) {
    if (d_a == nullptr || d_b == nullptr || d_grouped_layout == nullptr ||
        d_c == nullptr) {
        return;
    }
    if (m <= 0 || experts <= 0 || n <= 0 || k <= 0) {
        return;
    }

    // The JIT compiler emits exactly one template instance per configuration.
    launch_bf16_grouped_m_gemm_impl<
        SM110A_GROUPED_TILE_M,
        SM110A_GROUPED_TILE_N,
        SM110A_GROUPED_TILE_K,
        SM110A_GROUPED_EPILOGUE_TILE_N,
        SM110A_GROUPED_CLUSTER_M>(
        d_a, d_b, d_grouped_layout, d_c, m, experts, n, k, stream);
}

extern "C" void sm110a_bf16_grouped_m_gemm_launch(
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b,
    const int32_t *d_grouped_layout,
    __nv_bfloat16 *d_c,
    int m,
    int experts,
    int n,
    int k,
    cudaStream_t stream) {
    launch_bf16_grouped_m_gemm(
        d_a, d_b, d_grouped_layout, d_c, m, experts, n, k, stream);
}
