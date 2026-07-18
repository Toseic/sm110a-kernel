/*
 * Independent SM110A NVFP4 routed FC1 with fused token permutation.
 *
 * Computes C[grouped_row] = A[source_rows[grouped_row]] * B[expert]^T from
 * packed E2M1 operands and blocked UE4M3 scales. A and SFA are gathered into
 * the mainloop stage without materializing a permuted input tensor. B/SFB use
 * TMA, followed by tcgen05.cp scale staging and
 * tcgen05.mma.kind::mxf4nvf4.block_scale.block16.
 *
 * SFA remains blocked over the original token rows and SFB is blocked
 * independently for each expert.
 */

#include <cstdint>
#include <cstdlib>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include "nvfp4_moe_fc1.h"

#ifndef SM110A_NVFP4_FC1_TILE_M
#define SM110A_NVFP4_FC1_TILE_M 128
#endif
#ifndef SM110A_NVFP4_FC1_TILE_N
#define SM110A_NVFP4_FC1_TILE_N 128
#endif
#ifndef SM110A_NVFP4_FC1_TILE_K
#define SM110A_NVFP4_FC1_TILE_K 64
#endif
#ifndef SM110A_NVFP4_FC1_EPILOGUE_TILE_N
#define SM110A_NVFP4_FC1_EPILOGUE_TILE_N 32
#endif
#ifndef SM110A_NVFP4_FC1_CLUSTER_M
#define SM110A_NVFP4_FC1_CLUSTER_M 1
#endif
#ifndef SM110A_NVFP4_FC1_MODE
#define SM110A_NVFP4_FC1_MODE 0
#endif
#ifndef SM110A_NVFP4_FC1_DIRECT_GMEM
#define SM110A_NVFP4_FC1_DIRECT_GMEM 0
#endif

namespace {

constexpr int kMmaK = 64;
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
// Each mode-2 CLC worker cluster starts with two tasks.  Once it completes
// them it steals the work of an unstarted cluster through Cluster Launch Control.
constexpr int kClcWorkerTileCount = 2;

template <int kTileM, int kTileN, int kTileK, int kEpilogueTileN, int kClusterM>
struct KernelConfig {
    static_assert(kClusterM == 1, "NVFP4 FC1 gather currently requires cluster_m=1");
    static_assert(kTileM == 4 * kWarpSize);
    // tcgen05.mma.kind::f16 supports N in steps of 8 for cta_group::1
    // and steps of 16 for cta_group::2, up to N=256. The instruction
    // descriptor encodes this dimension as N >> 3 in bits 17-22.
    static_assert(kTileN == 64 || kTileN == 128);
    static_assert(kTileK == 64 || kTileK == 128 || kTileK == 256);
    static_assert(kEpilogueTileN == 32 || kEpilogueTileN == 64 || kEpilogueTileN == 128);
    static_assert(kTileN % kEpilogueTileN == 0);
    static_assert(kTileN % kClusterM == 0);
    static_assert(kTileK % 64 == 0);
    static_assert(kTileK % kMmaK == 0);
    static constexpr int kClusterTileM = kClusterM * kTileM;
    static constexpr int kATmaBytes = kTileM * kTileK / 2;
    static constexpr int kBTmaBytes = (kTileN / kClusterM) * kTileK / 2;
    static constexpr int kASmemBytes = kTileM * 128;
    static constexpr int kBSmemBytes = (kTileN / kClusterM) * 128;
    static constexpr int kMmaStepsPerStage = kTileK / 64;
    static constexpr int kSFASmemBytes = kMmaStepsPerStage * 512;
    static constexpr int kSFBScaleTileCount = (kTileN + 127) / 128;
    static constexpr int kSFBSmemBytes = kMmaStepsPerStage * kSFBScaleTileCount * 512;
    static constexpr int kPerStagePayloadBytes =
        kASmemBytes + kBSmemBytes + kSFASmemBytes + kSFBSmemBytes;
    static constexpr int kPerStageSmemBytes = (kPerStagePayloadBytes + 1023) & ~1023;
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
    static constexpr int kScaleColsPerMma = 4 + 4 * kSFBScaleTileCount;
    static constexpr int kScaleTmemCols = kStageCount * kMmaStepsPerStage * kScaleColsPerMma;
    static constexpr int kTmemColumnsNeeded = 512;
    static_assert(kTileN * kTmemStageCount + kScaleTmemCols <= kTmemColumnsNeeded);
    // tcgen05.alloc requires a power-of-two column count. In particular,
    // two N=192 accumulator stages need 384 columns but must allocate 512.
    static constexpr int kTmemAllocationColumns = 512;
    static_assert(kTmemColumnsNeeded <= 512);
    // Backward-compat alias used inside the kernel function body.
    static constexpr int kStageSmemBytes = kPerStageSmemBytes;
};

struct ExpertTileTask {
    int row_start;
    int row_count;
    int tile_row;
    int expert;
    int tile_n;
};

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

__device__ inline uint64_t make_k_major_128b_desc(int smem_addr) {
    constexpr int kStrideByteOffset = 8 * 128;
    const uint64_t base_offset = (smem_addr >> 7) & 7;
    return matrix_desc_encode(smem_addr) | (matrix_desc_encode(kStrideByteOffset) << 32ULL) |
           (1ULL << 46ULL) | (base_offset << 49ULL) | (2ULL << 61ULL);
}

__device__ inline uint64_t make_scale_desc(int smem_addr) {
    return matrix_desc_encode(smem_addr) | (matrix_desc_encode(128) << 32ULL) |
           (1ULL << 46ULL);
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

__device__ __forceinline__ void cp_async_a_mbarrier_arrive_noinc(int mbar_addr) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"
        :
        : "r"(mbar_addr)
        : "memory");
}

template <int kTileM, int kTileK>
__device__ __forceinline__ void load_gather_a_and_sfa(
    const uint8_t *__restrict__ a,
    const uint8_t *__restrict__ sfa,
    const int *__restrict__ source_rows,
    int valid_rows,
    int tile_k,
    int k,
    char *__restrict__ a_smem,
    char *__restrict__ sfa_smem,
    int a_mbar_addr) {
    constexpr int kVectorBytes = 16;
    constexpr int kVectorsPerRow = kTileK / 32;
    constexpr int kMmaSteps = kTileK / 64;
    constexpr int kRowsPerProducerWarp = kTileM / kProducerWarpCount;
    const int warp_id = threadIdx.x / kWarpSize;
    const int producer_rank = warp_id - kProducerWarpBase;
    const int lane = threadIdx.x & 31;
    const int row = producer_rank * kRowsPerProducerWarp + lane;

    static_assert(kTileM % kProducerWarpCount == 0);
    static_assert(kRowsPerProducerWarp == kWarpSize);

    // Match the 128B-swizzled TMA layout consumed by make_k_major_128b_desc.
    for (int vec = 0; vec < kVectorsPerRow; ++vec) {
        // The UMMA K-major SW128 atom reserves a full 128-byte row even when
        // this stage carries only K=64 or K=128. This is why kASmemBytes is
        // fixed at TileM*128 rather than the TMA transaction byte count.
        constexpr int kSmemVectorsPerRow = 128 / kVectorBytes;
        const int physical_linear = row * kSmemVectorsPerRow + vec;
        const int swizzled = (physical_linear & ~7) +
                             ((physical_linear & 7) ^ ((physical_linear >> 3) & 7));
        const bool valid = row < valid_rows;
        const int source_row = valid ? source_rows[row] : 0;
        const size_t offset = static_cast<size_t>(source_row) * (k / 2) +
                              static_cast<size_t>(tile_k) * (kTileK / 2) +
                              vec * kVectorBytes;
        const int dst_smem = static_cast<int>(__cvta_generic_to_shared(
            reinterpret_cast<uint4 *>(a_smem) + swizzled));
        cp_async_a_16b(dst_smem, a + offset, valid);
    }

    // Rebuild each gathered logical [128,4] scale tile directly in
    // SWIZZLE_32_4_4 order. Each thread moves all four scales for one row,
    // replacing the old byte-at-a-time gather.
    for (int mma_k = 0; mma_k < kMmaSteps; ++mma_k) {
        uint32_t value = 0;
        if (row < valid_rows) {
            const int source_row = source_rows[row];
            const int source_in_tile = (source_row & 31) * 16 +
                                       ((source_row >> 5) & 3) * 4;
            const size_t source_tile = static_cast<size_t>(source_row >> 7);
            const size_t global_mma_k = static_cast<size_t>(tile_k) * kMmaSteps + mma_k;
            value = *reinterpret_cast<const uint32_t *>(
                sfa + (source_tile * (k / 64) + global_mma_k) * 512 + source_in_tile);
        }
        const int dst_in_tile = (row & 31) * 16 + ((row >> 5) & 3) * 4;
        *reinterpret_cast<uint32_t *>(sfa_smem + mma_k * 512 + dst_in_tile) = value;
    }

    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    cp_async_a_mbarrier_arrive_noinc(a_mbar_addr);
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
__device__ inline void tcgen05_cp_scale(int tmem_addr, uint64_t smem_desc) {
    if constexpr (kClusterM == 1)
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_addr), "l"(smem_desc) : "memory");
    else
        asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_addr), "l"(smem_desc) : "memory");
}

template <int kClusterM>
__device__ inline void tcgen05_mma_nvfp4(
    int tmem_addr, uint64_t a_desc, uint64_t b_desc, uint32_t instruction_desc,
    int sfa_addr, int sfb_addr, int accumulate) {
    if constexpr (kClusterM == 1) {
        asm volatile("{\n\t.reg .pred input_d;\n\tsetp.ne.b32 input_d, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
                     "[%0], %1, %2, %3, [%4], [%5], input_d;\n\t}"
            :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(instruction_desc),
               "r"(sfa_addr), "r"(sfb_addr), "r"(accumulate) : "memory");
    } else {
        asm volatile("{\n\t.reg .pred input_d;\n\tsetp.ne.b32 input_d, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 "
                     "[%0], %1, %2, %3, [%4], [%5], input_d;\n\t}"
            :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(instruction_desc),
               "r"(sfa_addr), "r"(sfb_addr), "r"(accumulate) : "memory");
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

template <int kTileN, int kTileK, int kClusterM>
inline bool init_b_tensor_map(CUtensorMap *map, const uint8_t *ptr, uint64_t experts,
                              uint64_t n, uint64_t k) {
    constexpr uint32_t rank = 4;
    uint64_t dims[rank] = {k, n, 1, experts};
    uint64_t strides[rank - 1] = {k / 2, n * k / 2, n * k / 2};
    uint32_t box[rank] = {kTileK, kTileN / kClusterM, 1, 1};
    uint32_t elem[rank] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, rank,
        const_cast<uint8_t *>(ptr), dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

inline bool init_scale_tensor_map(CUtensorMap *map, const uint8_t *ptr,
                                  uint64_t row_tiles, uint64_t k_tiles,
                                  uint32_t k_box_tiles) {
    constexpr uint32_t rank = 4;
    uint64_t dims[rank] = {256, 2, k_tiles, row_tiles};
    uint64_t strides[rank - 1] = {256, 512, 512 * k_tiles};
    uint32_t box[rank] = {256, 2, k_box_tiles, 1};
    uint32_t elem[rank] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank,
        const_cast<uint8_t *>(ptr), dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
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
    constexpr int kStoreTileN = kEpilogueTileN < 64 ? kEpilogueTileN : 64;
    uint32_t box_dim[rank] = {kStoreTileN, kTileM, 1};
    uint32_t element_stride[rank] = {1, 1, 1};
    constexpr CUtensorMapSwizzle kSwizzle =
        kStoreTileN == 32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;

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

template <int kTileM, int kEpilogueTileN>
inline bool init_swapped_c_tensor_map(
    CUtensorMap *tensor_map,
    nv_bfloat16 *ptr,
    uint64_t m,
    uint64_t n) {
    constexpr uint32_t rank = 3;
    constexpr uint32_t kSwizzleElements = 64;
    uint64_t global_dim[rank] = {kSwizzleElements, n / kSwizzleElements, m};
    uint64_t global_stride[rank - 1] = {kSwizzleElements * sizeof(nv_bfloat16),
        n * sizeof(nv_bfloat16)};
    // One TMA store consumes one 64-column, 128B-swizzled atom.
    // Two atoms cover the full 128-column logical output tile.
    uint32_t box_dim[rank] = {kSwizzleElements, 1, kEpilogueTileN};
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
               CU_TENSOR_MAP_SWIZZLE_128B,
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
    bool kDynamicClc = false,
    bool kSwapAB = false,
    bool kDirectGmem = false>
__global__ __cluster_dims__(kClusterM, 1, 1)
__launch_bounds__(kBlockThreads) void nvfp4_fc1_kernel(
    const uint8_t *a,
    const __grid_constant__ CUtensorMap b_tensor_map,
    const uint8_t *sfa,
    const __grid_constant__ CUtensorMap sfb_tensor_map,
    const __grid_constant__ CUtensorMap c_tensor_map,
    const int *source_rows,
    const ExpertTileTask *tasks,
    const int *task_count,
    int *next_cluster_tile,
    nv_bfloat16 *c,
    int n,
    int k) {
    using Config = KernelConfig<kTileM, kTileN, kTileK, kEpilogueTileN, kClusterM>;
    constexpr int kClusterTileM = Config::kClusterTileM;
    constexpr int kStageCount = Config::kStageCount;
    constexpr int kASmemBytes = Config::kASmemBytes;
    constexpr int kBSmemBytes = Config::kBSmemBytes;
    constexpr int kSFASmemBytes = Config::kSFASmemBytes;
    constexpr int kSFBSmemBytes = Config::kSFBSmemBytes;
    constexpr int kSFBScaleTileCount = Config::kSFBScaleTileCount;
    constexpr int kStageSmemBytes = Config::kStageSmemBytes;
    constexpr int kMainloopSmemBytes = Config::kMainloopSmemBytes;
    constexpr int kEpilogueStageBytes = Config::kEpilogueStageBytes;
    constexpr int kEpilogueStageCount = Config::kEpilogueStageCount;
    constexpr int kEpilogueStoreTileN = kEpilogueTileN < 64 ? kEpilogueTileN : 64;
    constexpr int kEpilogueStoreTiles = kEpilogueTileN / kEpilogueStoreTileN;
    constexpr int kChunksPerStoreTile = kEpilogueStoreTileN / 8;
    constexpr int kSwizzlePhaseMask = kEpilogueStoreTileN == 32 ? 3 : 7;
    constexpr int kStoreTileBytes = kTileM * kEpilogueStoreTileN * sizeof(nv_bfloat16);
    constexpr int kMmaStepsPerStage = kTileK / kMmaK;
    constexpr int kTmemAllocationColumns = Config::kTmemAllocationColumns;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int worker_id = blockIdx.x / kClusterM;
    const int cluster_tiles = *task_count;
    constexpr int kWorkerTileCount = kDynamicClc ? kClcWorkerTileCount :
        (kPersistent ? kPersistentTileCount : 1);
    const int static_first_cluster_bid = worker_id * kWorkerTileCount;
    if constexpr (!kPersistent && !kDynamicClc) {
        if (static_first_cluster_bid >= cluster_tiles) {
            return;
        }
    }

    __shared__ int tmem_base[1];
    __shared__ int next_work_smem[2];
    __shared__ __align__(16) uint4 clc_response[1];
    __shared__ __align__(8) uint64_t clc_mbar[1];
    __shared__ int task_source_rows[kTileM];
    __shared__ uint64_t tma_mbars[kStageCount];
    __shared__ uint64_t a_mbars[kStageCount];
    __shared__ uint64_t mma_mbars[kStageCount];
    __shared__ uint64_t mainloop_mbars[kTmemStageCount];
    __shared__ uint64_t epilogue_mbars[kTmemStageCount];

    extern __shared__ __align__(1024) char dynamic_smem[];
    const int mainloop_smem = static_cast<int>(__cvta_generic_to_shared(dynamic_smem));
    const int epilogue_smem = mainloop_smem + kMainloopSmemBytes;
    const int base_tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(tma_mbars));
    const int base_a_mbar_addr = static_cast<int>(__cvta_generic_to_shared(a_mbars));
    const int base_mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mma_mbars));
    const int base_mainloop_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mainloop_mbars));
    const int base_epilogue_mbar_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_mbars));

    if (tid == 0) {
        for (int stage = 0; stage < kStageCount; ++stage) {
            mbarrier_init(base_tma_mbar_addr + stage * sizeof(uint64_t), 1);
            mbarrier_init(
                base_a_mbar_addr + stage * sizeof(uint64_t), kAProducerThreadCount);
            mbarrier_init(base_mma_mbar_addr + stage * sizeof(uint64_t), 1);
        }
        for (int stage = 0; stage < kTmemStageCount; ++stage) {
            mbarrier_init(base_mainloop_mbar_addr + stage * sizeof(uint64_t), 1);
            if constexpr (kPersistent || kDynamicClc) {
                mbarrier_init(base_epilogue_mbar_addr + stage * sizeof(uint64_t), 4 * kClusterM);
            }
        }
        if constexpr (kDynamicClc) {
            mbarrier_init(static_cast<int>(__cvta_generic_to_shared(clc_mbar)), 1);
        }
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
        prefetch_tensor_map(&sfb_tensor_map);
        prefetch_tensor_map(&c_tensor_map);
    }
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    const int tmem_addr = tmem_base[0];
    constexpr uint32_t instruction_desc = (1U << 7U) | (1U << 10U) |
                                          (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
                                          (static_cast<uint32_t>(kClusterTileM) >> 7U << 27U);

    const int k_tiles = k / kTileK;

    auto tmem_to_smem = [&](int mainloop_stage,
                            int n_block,
                            int smem_stage,
                            int cta_row = 0,
                            int tile_col = 0,
                            int valid_rows = kTileM) {
        char *stage_smem = dynamic_smem + kMainloopSmemBytes + smem_stage * kEpilogueStageBytes;

        for (int n_group = 0; n_group < kEpilogueTileN / 8; ++n_group) {
            const int tmem_col = mainloop_stage * kTileN + n_block * kEpilogueTileN + n_group * 8;

            if constexpr (!kSwapAB) {
                float values[8];
                const int tmem_row = cta_rank * kTileM + warp_id * kWarpSize;
                const int load_addr = tmem_addr + (tmem_row << 16) + tmem_col;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
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
                    packed[i] = __float22bfloat162_rn({values[2 * i], values[2 * i + 1]});
                }

                if constexpr (kDirectGmem) {
                    // Direct TMEM → RMEM → GMEM: write packed values straight
                    // to global memory, bypassing shared memory.
                    if (tid < valid_rows) {
                        const int gmem_row = cta_row + tid;
                        const int gmem_col = tile_col + n_block * kEpilogueTileN + n_group * 8;
                        reinterpret_cast<int4 *>(c + gmem_row * n + gmem_col)[0] =
                            reinterpret_cast<int4 *>(packed)[0];
                    }
                } else {
                    const int store_tile = n_group / kChunksPerStoreTile;
                    const int chunk_in_store =
                        tid * kChunksPerStoreTile + n_group % kChunksPerStoreTile;
                    const int swizzled_chunk =
                        (chunk_in_store & ~7) +
                        ((chunk_in_store & 7) ^ ((chunk_in_store >> 3) & kSwizzlePhaseMask));
                    reinterpret_cast<int4 *>(
                        stage_smem + store_tile * kStoreTileBytes)[swizzled_chunk] =
                        reinterpret_cast<int4 *>(packed)[0];
                }
            } else {
                // Match STSM's register fragment directly: each warp reads
                // lanes 0-15 and 16-31 as two .16x256b TMEM fragments.
                uint32_t values[8];
                const int row_base = cta_rank * kTileM + warp_id * kWarpSize;
                const int load_addr0 = tmem_addr + (row_base << 16) + tmem_col;
                const int load_addr1 = tmem_addr + ((row_base + 16) << 16) + tmem_col;
                asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 "
                             "{%0, %1, %2, %3}, [%4];"
                    : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
                    : "r"(load_addr0));
                asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 "
                             "{%0, %1, %2, %3}, [%4];"
                    : "=r"(values[4]), "=r"(values[5]), "=r"(values[6]), "=r"(values[7])
                    : "r"(load_addr1));
                asm volatile("tcgen05.wait::ld.sync.aligned;");

                uint32_t packed[4];
                for (int i = 0; i < 4; ++i) {
                    const float x = reinterpret_cast<float *>(values)[2 * i];
                    const float y = reinterpret_cast<float *>(values)[2 * i + 1];
                    const nv_bfloat162 pair = __float22bfloat162_rn({x, y});
                    packed[i] = reinterpret_cast<const uint32_t *>(&pair)[0];
                }

                // One pair of warps fills an 8x64 128B-swizzled atom.
                // STSM.T transposes the TMEM fragment while storing it.
                constexpr int kSwizzleAtomBytes = 128;
                constexpr int kBankGroupBytes = 16;
                const int lane = tid % kWarpSize;
                const int row = lane % 8;
                const int col = (warp_id % 2) * 4 + lane / 8;
                const int atom_offset = (warp_id / 2) * kEpilogueTileN * kSwizzleAtomBytes;
                const int row_group_offset = n_group * 8 * kSwizzleAtomBytes;
                const int smem_addr = static_cast<int>(__cvta_generic_to_shared(stage_smem)) +
                                      atom_offset + row_group_offset + row * kSwizzleAtomBytes +
                                      (col ^ row) * kBankGroupBytes;
                asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16.trans "
                             "[%0], {%1, %2, %3, %4};"
                    :
                    : "r"(smem_addr), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3])
                    : "memory");
            }
        }
    };

    int producer_stage = 0;
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
        int first_cluster_bid;
        int clc_mbar_addr = 0;
        if constexpr (kDynamicClc) {
            first_cluster_bid = dynamic_first_cluster_bid;

            // One request per cluster.  CLC multicasts the cancellation response
            // to every CTA so all ranks advance together after finishing work.
            const int clc_response_addr = static_cast<int>(__cvta_generic_to_shared(clc_response));
            clc_mbar_addr = static_cast<int>(__cvta_generic_to_shared(clc_mbar));
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
        } else if constexpr (kPersistent) {
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
        if (warp_id >= kProducerWarpBase && warp_id < kMmaWarp) {
            const int producer_rank = warp_id - kProducerWarpBase;
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                const ExpertTileTask task = tasks[cluster_bid];
                const int tile_n = task.tile_n;
                const int cta_row = task.row_start + task.tile_row + cta_rank * kTileM;
                const int tile_col = tile_n * kTileN;
                const int cta_tile_row = task.tile_row + cta_rank * kTileM;
                const int remaining_rows = task.row_count - cta_tile_row;
                const int valid_rows = remaining_rows < 0 ? 0 :
                    (remaining_rows < kTileM ? remaining_rows : kTileM);

                if (cta_row != cached_source_row_start) {
                    const int lane = tid & (kWarpSize - 1);
                    const int row = producer_rank * kWarpSize + lane;
                    task_source_rows[row] =
                        row < valid_rows ? __ldg(source_rows + cta_row + row) : 0;
                    __syncwarp();
                    cached_source_row_start = cta_row;
                }

                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    const int mma_mbar_addr =
                        base_mma_mbar_addr + producer_stage * sizeof(uint64_t);
                    const int a_mbar_addr =
                        base_a_mbar_addr + producer_stage * sizeof(uint64_t);
                    const int tma_mbar_addr =
                        base_tma_mbar_addr + producer_stage * sizeof(uint64_t);
                    const int a_smem = mainloop_smem + producer_stage * kStageSmemBytes;
                    const int b_smem = a_smem + kASmemBytes;
                    const int sfa_smem = b_smem + kBSmemBytes;
                    const int sfb_smem = sfa_smem + kSFASmemBytes;

                    mbarrier_wait(mma_mbar_addr, producer_mma_phase);
                    const int tma_transaction_mbar_addr = tma_mbar_addr & 0xFEFFFFFF;
                    const int off_k = tile_k * kTileK;
                    const int off_k64 = tile_k * Config::kMmaStepsPerStage;
                    if (producer_rank == 0 && elect_sync()) {
                        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
                                     "_, [%0], %1;" ::"r"(tma_transaction_mbar_addr),
                            "r"(Config::kBTmaBytes + kSFBSmemBytes)
                            : "memory");
                        tma_load_4d<kClusterM>(b_smem, &b_tensor_map, off_k,
                            tile_col, 0, task.expert, tma_transaction_mbar_addr);
                        for (int scale_tile = 0; scale_tile < kSFBScaleTileCount; ++scale_tile)
                            tma_load_4d<kClusterM>(
                                sfb_smem + scale_tile * Config::kMmaStepsPerStage * 512,
                                &sfb_tensor_map, 0, 0, off_k64,
                                task.expert * (n / 128) + tile_col / 128 + scale_tile,
                                tma_transaction_mbar_addr);
                    }

                    load_gather_a_and_sfa<kTileM, kTileK>(a, sfa, task_source_rows,
                        valid_rows, tile_k, k,
                        dynamic_smem + producer_stage * kStageSmemBytes,
                        dynamic_smem + producer_stage * kStageSmemBytes +
                            kASmemBytes + kBSmemBytes,
                        a_mbar_addr);

                    producer_stage = (producer_stage + 1) % kStageCount;
                    if (producer_stage == 0) {
                        producer_mma_phase ^= 1;
                    }
                }
            }
        }

        // The consumer advances independently and releases each stage after MMA has
        // finished reading it.
        if (cta_rank == 0 && warp_id == kMmaWarp && elect_sync()) {
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                if constexpr (kPersistent || kDynamicClc) {
                    mbarrier_wait(
                        base_epilogue_mbar_addr + consumer_mainloop_stage * sizeof(uint64_t),
                        consumer_epilogue_phase);
                }

                const int stage_tmem_addr = tmem_addr + consumer_mainloop_stage * kTileN;
                const int tile_col = tasks[cluster_bid].tile_n * kTileN;
                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    const int tma_mbar_addr =
                        base_tma_mbar_addr + consumer_stage * sizeof(uint64_t);
                    const int a_mbar_addr =
                        base_a_mbar_addr + consumer_stage * sizeof(uint64_t);
                    const int mma_mbar_addr =
                        base_mma_mbar_addr + consumer_stage * sizeof(uint64_t);

                    mbarrier_wait(a_mbar_addr, consumer_a_phase);
                    mbarrier_wait(tma_mbar_addr, consumer_tma_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    {
                        const int a_smem = mainloop_smem + consumer_stage * kStageSmemBytes;
                        const int b_smem = a_smem + kASmemBytes;
                        const int sfa_smem = b_smem + kBSmemBytes;
                        const int sfb_smem = sfa_smem + kSFASmemBytes;
                        for (int mma_k = 0; mma_k < kMmaStepsPerStage; ++mma_k) {
                            const int scale_addr = tmem_addr + 2 * kTileN +
                                (consumer_stage * kMmaStepsPerStage + mma_k) *
                                    Config::kScaleColsPerMma;
                            tcgen05_cp_scale<kClusterM>(scale_addr,
                                make_scale_desc(sfa_smem + mma_k * 512));
                            for (int scale_tile = 0; scale_tile < kSFBScaleTileCount; ++scale_tile)
                                tcgen05_cp_scale<kClusterM>(scale_addr + 4 + scale_tile * 4,
                                    make_scale_desc(sfb_smem +
                                        (scale_tile * kMmaStepsPerStage + mma_k) * 512));
                            const int accumulate = tile_k != 0 || mma_k != 0;
                            tcgen05_mma_nvfp4<kClusterM>(stage_tmem_addr,
                                make_k_major_128b_desc(a_smem + mma_k * 32),
                                make_k_major_128b_desc(b_smem + mma_k * 32),
                                instruction_desc, scale_addr,
                                scale_addr + 4 + (tile_col % 128) / 32, accumulate);
                        }
                    }
                    tcgen05_commit<kClusterM>(mma_mbar_addr);

                    consumer_stage = (consumer_stage + 1) % kStageCount;
                    if (consumer_stage == 0) {
                        consumer_a_phase ^= 1;
                        consumer_tma_phase ^= 1;
                    }
                }

                const int mainloop_mbar_addr =
                    base_mainloop_mbar_addr + consumer_mainloop_stage * sizeof(uint64_t);
                tcgen05_commit<kClusterM>(mainloop_mbar_addr);

                consumer_mainloop_stage = (consumer_mainloop_stage + 1) % kTmemStageCount;
                if (consumer_mainloop_stage == 0) {
                    if constexpr (kPersistent || kDynamicClc) {
                        consumer_epilogue_phase ^= 1;
                    }
                }
            }
        }

        if (warp_id < 4) {
            constexpr int kEpilogueBlockCount = kTileN / kEpilogueTileN;

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
                const int tile_col = tile_n * (kSwapAB ? kClusterTileM : kTileN);

                auto release_tmem = [&]() {
                    if constexpr (kPersistent || kDynamicClc) {
                        if (elect_sync()) {
                            const int mbar_addr = (base_epilogue_mbar_addr +
                                                      epilogue_mainloop_stage * sizeof(uint64_t)) &
                                                  0xFEFFFFFF;
                            asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 "
                                         "_, [%0];" ::"r"(mbar_addr)
                                : "memory");
                        }
                    }
                };

                tmem_to_smem(epilogue_mainloop_stage, 0, 0, cta_row, tile_col, valid_rows);
                asm volatile("bar.sync 1, 128;" ::: "memory");
                if constexpr (kEpilogueBlockCount == 1) {
                    release_tmem();
                }

                for (int n_block = 0; n_block < kEpilogueBlockCount; ++n_block) {
                    const int smem_stage = n_block % kEpilogueStageCount;
                    const int stage_smem = epilogue_smem + smem_stage * kEpilogueStageBytes;
                    if constexpr (!kSwapAB) {
                        // Direct gmem: data already written by tmem_to_smem.
                        // Only the SMEM→TMA path needs TMA store / fallback.
                        if constexpr (!kDirectGmem) {
                            const int block_col = tile_col + n_block * kEpilogueTileN;
                            if (valid_rows == kTileM) {
                                if (tid == 0) {
                                    for (int store_tile = 0; store_tile < kEpilogueStoreTiles;
                                        ++store_tile) {
                                        tma_store_3d(stage_smem + store_tile * kStoreTileBytes,
                                            &c_tensor_map,
                                            block_col + store_tile * kEpilogueStoreTileN,
                                            cta_row,
                                            0);
                                    }
                                    tma_store_commit_group();
                                }
                            } else if (tid < valid_rows) {
                                char *stage_smem_ptr = dynamic_smem + kMainloopSmemBytes +
                                                       smem_stage * kEpilogueStageBytes;
                                for (int n_group = 0; n_group < kEpilogueTileN / 8; ++n_group) {
                                    const int store_tile = n_group / kChunksPerStoreTile;
                                    const int chunk_in_store =
                                        tid * kChunksPerStoreTile + n_group % kChunksPerStoreTile;
                                    const int swizzled_chunk =
                                        (chunk_in_store & ~7) +
                                        ((chunk_in_store & 7) ^
                                            ((chunk_in_store >> 3) & kSwizzlePhaseMask));
                                    const int4 value = reinterpret_cast<int4 *>(
                                        stage_smem_ptr +
                                        store_tile * kStoreTileBytes)[swizzled_chunk];
                                    const int output_offset =
                                        (cta_row + tid) * n + block_col + n_group * 8;
                                    reinterpret_cast<int4 *>(c + output_offset)[0] = value;
                                }
                            }
                        }
                    } else {
                        const int block_row =
                            task.row_start + task.tile_row + n_block * kEpilogueTileN;
                        const int remaining_block_rows =
                            task.row_count - task.tile_row - n_block * kEpilogueTileN;
                        const int valid_block_rows =
                            remaining_block_rows < 0
                                ? 0
                                : (remaining_block_rows < kEpilogueTileN ? remaining_block_rows
                                                                         : kEpilogueTileN);
                        const int block_col = tile_col + cta_rank * kTileM;
                        if (valid_block_rows == kEpilogueTileN) {
                            if (tid == 0) {
                                constexpr int kSwappedStoreColumns = 64;
                                constexpr int kSwappedStoreBytes =
                                    kEpilogueTileN * kSwappedStoreColumns * sizeof(nv_bfloat16);
                                for (int atom = 0; atom < 2; ++atom) {
                                    tma_store_3d(stage_smem + atom * kSwappedStoreBytes,
                                        &c_tensor_map,
                                        0,
                                        block_col / kSwappedStoreColumns + atom,
                                        block_row);
                                }
                                tma_store_commit_group();
                            }
                        } else if (tid < kTileM) {
                            constexpr int kSwappedStoreColumns = 64;
                            constexpr int kSwizzleAtomBytes = 128;
                            constexpr int kBankGroupElements = 16 / sizeof(nv_bfloat16);
                            auto *stage_values =
                                reinterpret_cast<nv_bfloat16 *>(dynamic_smem + kMainloopSmemBytes +
                                                                smem_stage * kEpilogueStageBytes);
                            const int atom = tid / kSwappedStoreColumns;
                            const int col_in_atom = tid % kSwappedStoreColumns;
                            for (int row = 0; row < valid_block_rows; ++row) {
                                const int bank_group = col_in_atom / kBankGroupElements;
                                const int value_in_group = col_in_atom % kBankGroupElements;
                                const int atom_elements = kEpilogueTileN * kSwappedStoreColumns;
                                const int swizzled_index =
                                    atom * atom_elements +
                                    row * (kSwizzleAtomBytes / sizeof(nv_bfloat16)) +
                                    (bank_group ^ (row & 7)) * kBankGroupElements + value_in_group;
                                c[(block_row + row) * n + block_col + tid] =
                                    stage_values[swizzled_index];
                            }
                        }
                    }

                    if (n_block + 1 < kEpilogueBlockCount) {
                        if (kEpilogueStageCount == 1 || n_block > 0) {
                            if constexpr (!kDirectGmem || kSwapAB) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read %0;"
                                        :
                                        : "n"(kEpilogueStageCount - 1)
                                        : "memory");
                                }
                            }
                            asm volatile("bar.sync 1, 128;" ::: "memory");
                        }

                        const int next_stage = (n_block + 1) % kEpilogueStageCount;
                        tmem_to_smem(epilogue_mainloop_stage,
                            n_block + 1,
                            next_stage,
                            cta_row,
                            tile_col,
                            valid_rows);
                        asm volatile("bar.sync 1, 128;" ::: "memory");
                        if (n_block + 2 == kEpilogueBlockCount) {
                            release_tmem();
                        }
                    }
                }

                if (tid == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                }
                asm volatile("bar.sync 1, 128;" ::: "memory");

                epilogue_mainloop_stage = (epilogue_mainloop_stage + 1) % kTmemStageCount;
                if (epilogue_mainloop_stage == 0) {
                    epilogue_mainloop_phase ^= 1;
                }
            }
        }

        if constexpr (kDynamicClc) {
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

            asm volatile("fence.proxy.async::generic.release."
                         "sync_restrict::shared::cta.cluster;" ::: "memory");
            asm volatile("barrier.cluster.arrive.relaxed.aligned;");
            asm volatile("barrier.cluster.wait.aligned;");
            asm volatile("fence.proxy.async::generic.acquire."
                         "sync_restrict::shared::cluster.cluster;" ::: "memory");
        } else if constexpr (!kPersistent) {
            break;
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

template <int kTileM, int kTileN, int kTileK, int kEpilogueTileN, int kClusterM, bool kSwapAB>
void launch_nvfp4_fc1_impl(
    const uint8_t *d_a,
    const uint8_t *d_b,
    const uint8_t *d_sfa,
    const uint8_t *d_sfb,
    const int32_t *d_source_rows,
    const int32_t *d_tasks,
    int32_t *d_state,
    __nv_bfloat16 *d_c,
    int tokens,
    int m,
    int experts,
    int n,
    int k,
    int max_tasks,
    cudaStream_t stream) {
    using Config = KernelConfig<kTileM, kTileN, kTileK, kEpilogueTileN, kClusterM>;
    constexpr int kClusterTileM = Config::kClusterTileM;
    constexpr int kDynamicSmemBytes = Config::kDynamicSmemBytes;
    static_assert(!kSwapAB, "NVFP4 FC1 does not support swap_ab");
    if (tokens % 128 || n % kTileN != 0 || n % 128 || k % kTileK != 0) {
        return;
    }

    CUtensorMap b_tensor_map;
    CUtensorMap sfb_tensor_map;
    CUtensorMap c_tensor_map;
    bool tensor_maps_ok;
    tensor_maps_ok =
        init_b_tensor_map<kTileN, kTileK, kClusterM>(&b_tensor_map, d_b, experts, n, k) &&
        init_scale_tensor_map(&sfb_tensor_map, d_sfb, experts * (n / 128),
            k / 64, kTileK / 64) &&
        init_c_tensor_map<kTileM, kEpilogueTileN>(&c_tensor_map, d_c, m, n);
    if (!tensor_maps_ok) {
        return;
    }

    /* Metadata is prepared outside the timed main kernel. */
    // We pack metadata + tile-task queue into a single GPU allocation
    // so the kernel can consume everything from one base pointer.
    //
    //  workspace ─┐
    //             ▼
    //  ┌──────────────────────────────────────────┐
    //  │  expert_offsets[experts]     int32 []    │  experts × 4 B
    //  │    per-expert min row index              │
    //  ├──────────────────────────────────────────┤
    //  │  expert_counts[experts]     int32 []     │  experts × 4 B
    //  │    per-expert row count                  │
    //  ├──────────────────────────────────────────┤
    //  │  task_count                 int32        │  4 B
    //  │    actual #tasks written by builder      │
    //  ├──────────────────────────────────────────┤
    //  ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -  ─┤  ← task_offset
    //  │  tasks[max_tasks]        ExpertTileTask  │  max_tasks × 16 B
    //  │    {row_start, row_count,                │
    //  │     tile_row, expert}                    │
    //  └──────────────────────────────────────────┘
    //
    // Each row bundle produces one task per logical N tile.  The extra expert
    // slots account for the per-expert ceil when row counts are uneven.
    auto *tasks = reinterpret_cast<const ExpertTileTask *>(d_tasks);
    auto *task_count = d_state;
    auto *next_cluster_tile = d_state + 1;
    cudaMemsetAsync(next_cluster_tile, 0, sizeof(int32_t), stream);

    if constexpr (SM110A_NVFP4_FC1_MODE == 2) {
        const int worker_clusters =
            (max_tasks + kClcWorkerTileCount - 1) / kClcWorkerTileCount;
        auto kernel = nvfp4_fc1_kernel<kTileM,
            kTileN,
            kTileK,
            kEpilogueTileN,
            kClusterM,
            false,
            true,
            kSwapAB,
            (SM110A_NVFP4_FC1_DIRECT_GMEM != 0)>;
        if constexpr (kDynamicSmemBytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kDynamicSmemBytes);
        }
        kernel<<<kClusterM * worker_clusters, kBlockThreads, kDynamicSmemBytes, stream>>>(
            d_a, b_tensor_map, d_sfa, sfb_tensor_map, c_tensor_map, d_source_rows, tasks, task_count,
            next_cluster_tile, d_c, n, k);
    } else if constexpr (SM110A_NVFP4_FC1_MODE == 1) {
        const int persistent_clusters =
            (max_tasks + kPersistentTileCount - 1) / kPersistentTileCount;
        auto kernel = nvfp4_fc1_kernel<kTileM,
            kTileN,
            kTileK,
            kEpilogueTileN,
            kClusterM,
            true,
            false,
            kSwapAB,
            (SM110A_NVFP4_FC1_DIRECT_GMEM != 0)>;
        if constexpr (kDynamicSmemBytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kDynamicSmemBytes);
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
            d_a, b_tensor_map, d_sfa, sfb_tensor_map, c_tensor_map, d_source_rows, tasks, task_count,
            next_cluster_tile, d_c, n, k);
    } else {
        auto kernel = nvfp4_fc1_kernel<kTileM,
            kTileN,
            kTileK,
            kEpilogueTileN,
            kClusterM,
            false,
            false,
            kSwapAB,
            (SM110A_NVFP4_FC1_DIRECT_GMEM != 0)>;
        if constexpr (kDynamicSmemBytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kDynamicSmemBytes);
        }
        kernel<<<kClusterM * max_tasks, kBlockThreads, kDynamicSmemBytes, stream>>>(
            d_a, b_tensor_map, d_sfa, sfb_tensor_map, c_tensor_map, d_source_rows, tasks, task_count,
            next_cluster_tile, d_c, n, k);
    }
}

void launch_nvfp4_fc1_entry(
    const uint8_t *a, const uint8_t *b, const uint8_t *sfa, const uint8_t *sfb,
    const int32_t *source_rows, const int32_t *tasks, int32_t *state, nv_bfloat16 *c,
    int tokens, int m, int experts, int n, int k, int max_tasks, cudaStream_t stream) {
    if (!a || !b || !sfa || !sfb || !source_rows || !tasks || !state || !c ||
        tokens <= 0 || m <= 0 || experts <= 0 || n <= 0 || k <= 0 || max_tasks <= 0)
        return;
    launch_nvfp4_fc1_impl<SM110A_NVFP4_FC1_TILE_M,
        SM110A_NVFP4_FC1_TILE_N, SM110A_NVFP4_FC1_TILE_K,
        SM110A_NVFP4_FC1_EPILOGUE_TILE_N, SM110A_NVFP4_FC1_CLUSTER_M,
        false>(a, b, sfa, sfb, source_rows, tasks, state, c, tokens, m, experts,
            n, k, max_tasks, stream);
}

} // namespace

void launch_nvfp4_fc1_main(
    const uint8_t *a, const uint8_t *b, const uint8_t *sfa, const uint8_t *sfb,
    const int32_t *source_rows, const int32_t *tasks, int32_t *state, nv_bfloat16 *c,
    int tokens, int m, int experts, int n, int k, int max_tasks, cudaStream_t stream) {
    launch_nvfp4_fc1_entry(
        a, b, sfa, sfb, source_rows, tasks, state, c, tokens, m, experts, n, k,
        max_tasks, stream);
}

extern "C" void sm110a_nvfp4_fc1_launch(
    const uint8_t *a, const uint8_t *b, const uint8_t *sfa, const uint8_t *sfb,
    const int32_t *source_rows, const int32_t *tasks, int32_t *state, nv_bfloat16 *c,
    int tokens, int m, int experts, int n, int k, int max_tasks, cudaStream_t stream) {
    launch_nvfp4_fc1_main(
        a, b, sfa, sfb, source_rows, tasks, state, c, tokens, m, experts, n, k,
        max_tasks, stream);
}
