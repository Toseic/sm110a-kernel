/*
 * One/two-CTA MXFP4 W4A8 FC1 + FC2 kernel.
 *
 * FC1 publishes swizzled FP8 / E8M0 blocks to a reusable global image.
 * FC2 streams that image through a TMA ring, preserving full FP32 sums.
 * Per-row resident partitions exchange FC1-I and split FC2-H.
 * Tiled scheduling distributes each phase across all resident workers.
 */

#include <cmath>
#include <cstdint>

#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "mxfp4_fc1fc2.h"
#include "mxfp4_fc1fc2_scale.cuh"

namespace sm110a::mxfp4_fc1fc2 {

#ifndef SM110A_MXFP4_FC1FC2_HIDDEN_DIM
#define SM110A_MXFP4_FC1FC2_HIDDEN_DIM 6144
#endif
constexpr int kH = SM110A_MXFP4_FC1FC2_HIDDEN_DIM;
static_assert(kH > 0 && kH % 256 == 0);
constexpr int kI = 2048;
constexpr int kPhysicalFc1N = 2 * kI;
#ifndef SM110A_MXFP4_FC1FC2_H_SPLIT
#define SM110A_MXFP4_FC1FC2_H_SPLIT 1
#endif
#ifndef SM110A_MXFP4_FC1FC2_STAGES
#define SM110A_MXFP4_FC1FC2_STAGES 5
#endif
#ifndef SM110A_MXFP4_FC1FC2_FUSE_ROUTING
#define SM110A_MXFP4_FC1FC2_FUSE_ROUTING 0
#endif
#ifndef SM110A_MXFP4_FC1FC2_WEIGHT_IN_FC1
#define SM110A_MXFP4_FC1FC2_WEIGHT_IN_FC1 0
#endif
#ifndef SM110A_MXFP4_FC1FC2_ACTIVATION_BF16_ROUNDING
#define SM110A_MXFP4_FC1FC2_ACTIVATION_BF16_ROUNDING 0
#endif
#ifndef SM110A_MXFP4_FC1FC2_COMPACT_WEIGHT_SCALES
#define SM110A_MXFP4_FC1FC2_COMPACT_WEIGHT_SCALES 0
#endif
constexpr bool kCompactWeightScales = SM110A_MXFP4_FC1FC2_COMPACT_WEIGHT_SCALES;
constexpr int kHSplit = SM110A_MXFP4_FC1FC2_H_SPLIT;
static_assert(kH % (256 * kHSplit) == 0);
#ifndef SM110A_MXFP4_FC1FC2_TILED
#define SM110A_MXFP4_FC1FC2_TILED 0
#endif
constexpr bool kTiledSchedule = SM110A_MXFP4_FC1FC2_TILED;
constexpr bool kFuseRouting = SM110A_MXFP4_FC1FC2_FUSE_ROUTING;
constexpr bool kWeightInFc1 = SM110A_MXFP4_FC1FC2_WEIGHT_IN_FC1;
constexpr bool kActivationBf16Rounding = SM110A_MXFP4_FC1FC2_ACTIVATION_BF16_ROUNDING;
constexpr int kTileM = 128;
#ifndef SM110A_MXFP4_FC1FC2_CLUSTER_M
#define SM110A_MXFP4_FC1FC2_CLUSTER_M 2
#endif
constexpr int kClusterM = SM110A_MXFP4_FC1FC2_CLUSTER_M;
static_assert(kClusterM == 1 || kClusterM == 2);
#if SM110A_MXFP4_FC1FC2_CLUSTER_M == 1
#define MXFP4_FC1FC2_CTA_GROUP "1"
#else
#define MXFP4_FC1FC2_CTA_GROUP "2"
#endif
constexpr int kClusterCtas = kClusterM;
constexpr int kClusterRows = kTileM * kClusterM;
constexpr int kPairTileM = kClusterRows;
constexpr int kTileN = 256;
constexpr int kStageK = 128;
constexpr int kMmaK = 32;
constexpr int kMmaSteps = 4;
constexpr int kWarpSize = 32;
constexpr int kEpilogueWarps = 4;
constexpr int kMmaWarp = 4;
constexpr int kTmaWarp = 5;
#ifndef SM110A_MXFP4_FC1FC2_GATHER_WARPS
#define SM110A_MXFP4_FC1FC2_GATHER_WARPS 4
#endif
constexpr int kGatherWarpCount = SM110A_MXFP4_FC1FC2_GATHER_WARPS;
// The peer CTA's MMA warp is free after TMEM allocation. For two gather
// warps, reuse it to forward peer readiness and fit the CTA in eight warps.
constexpr bool kReusePeerMmaWarp = kClusterM == 2 && kGatherWarpCount == 2;
constexpr int kTmaReadyWarp = kReusePeerMmaWarp ? kMmaWarp : 6;
constexpr int kGatherWarpBase = kClusterM == 2 && !kReusePeerMmaWarp ? 7 : 6;
constexpr int kReadyArrivals = kReusePeerMmaWarp ? 1 : kClusterM;
static_assert(kGatherWarpCount == 2 || kGatherWarpCount == 4);
constexpr int kGatherThreadCount = kGatherWarpCount * kWarpSize;
constexpr int kWarpCount = kFuseRouting ? kGatherWarpBase + kGatherWarpCount
                                        : (kClusterM == 1 || kReusePeerMmaWarp ? 6 : 7);
constexpr int kFc1TasksPerRowTask = kPhysicalFc1N / kTileN;
constexpr int kMatrixSmemBytes = kTileM * 128;
constexpr int kWeightSmemBytes = (kTileN / kClusterM) * 128;
constexpr int kWeightTmaBytes = kWeightSmemBytes / 2;
constexpr int kScaleStageBytes = kMmaSteps * 512;
constexpr int kWeightScaleBytes = 2 * kScaleStageBytes;
constexpr int kWeightScaleTmaBytes = kWeightScaleBytes / (kCompactWeightScales ? 4 : 1);
// Global image stores swizzled FP8 atoms and padded replicated E8M0 scales.
constexpr int kIntermediateAtoms = kI / kStageK;
constexpr int kIntermediateValueBytes = kIntermediateAtoms * kMatrixSmemBytes;
constexpr int kIntermediateBytes = kIntermediateValueBytes + kIntermediateAtoms * kScaleStageBytes;
#ifndef SM110A_MXFP4_FC1FC2_REDUCE_TILE_N
#define SM110A_MXFP4_FC1FC2_REDUCE_TILE_N 64
#endif
constexpr int kReduceN = SM110A_MXFP4_FC1FC2_REDUCE_TILE_N;
static_assert(kReduceN == 64 || kReduceN == 128);
// One compact slot shared by the FC1 quantizer and FC2 BF16 reduction.
constexpr int kFc1CompactIntermediateBytes = kMatrixSmemBytes + kScaleStageBytes;
constexpr int kFc2ReduceSmemBytes = kTileM * kReduceN * sizeof(__nv_bfloat16);
constexpr int kEpilogueSmemBytes = kFc1CompactIntermediateBytes > kFc2ReduceSmemBytes
                                       ? kFc1CompactIntermediateBytes
                                       : kFc2ReduceSmemBytes;
constexpr int kFc1ScaleStagingOffset = kMatrixSmemBytes;
constexpr int kWorkspaceBytes =
    kMatrixSmemBytes + kWeightSmemBytes + kScaleStageBytes + kWeightScaleBytes;
constexpr int kFc2WorkspaceBytes = kWorkspaceBytes;
constexpr int kFc1StageCount = SM110A_MXFP4_FC1FC2_STAGES;
constexpr int kFc2StageCount = kFc1StageCount;
constexpr int kBarrierStageCount = kFc1StageCount;
constexpr int kDynamicSmemBytes = kEpilogueSmemBytes + kFc1StageCount * kWorkspaceBytes;
constexpr int kTmemColumns = 512;
#ifndef SM110A_MXFP4_FC1FC2_ACCUMULATOR_STAGES
#define SM110A_MXFP4_FC1FC2_ACCUMULATOR_STAGES 2
#endif
constexpr int kAccumulatorStages = SM110A_MXFP4_FC1FC2_ACCUMULATOR_STAGES;
constexpr int kAccumulatorStride = kAccumulatorStages == 2 ? 128 : 256;
static_assert(kAccumulatorStages == 1 || kAccumulatorStages == 2);
constexpr int kScaleTmemCol = kAccumulatorStages == 2 ? 384 : 256;
constexpr int kFc2AccumulatorStride = kAccumulatorStages == 2 ? 240 : 256;
constexpr int kFc2ScaleTmemCol = kAccumulatorStages == 2 ? 496 : 256;
static_assert(kDynamicSmemBytes + 2048 <= 232448);
static_assert(kFc1StageCount >= 2 && kFc1StageCount <= 5);
static_assert(kHSplit == 1 || kHSplit == 2 || kHSplit == 4);

struct ExpertTileTask {
    int row_start;
    int row_count;
    int tile_row;
    int expert;
    int tile_n;
};

__device__ __forceinline__ float silu(
    float x) {
    constexpr float kLog2E = 1.4426950408889634f;
    float exponential;
    float reciprocal;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(-kLog2E * x));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(1.0f + exponential));
    return x * reciprocal;
}

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t elected = 0;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "elect.sync _|p, %1;\n\t"
                 "@p mov.s32 %0, 1;\n\t}"
        : "+r"(elected)
        : "r"(0xffffffffu));
    return elected;
}

__device__ __forceinline__ constexpr uint64_t matrix_desc_encode(
    uint64_t value) {
    return (value & 0x3ffffULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t make_k_major_128b_desc(
    int smem_addr) {
    constexpr int kStrideByteOffset = 8 * 128;
    const uint64_t base_offset = (smem_addr >> 7) & 7;
    return matrix_desc_encode(smem_addr) | (matrix_desc_encode(kStrideByteOffset) << 32ULL) |
           (1ULL << 46ULL) | (base_offset << 49ULL) | (2ULL << 61ULL);
}

__device__ __forceinline__ uint64_t make_scale_desc(
    int smem_addr) {
    return matrix_desc_encode(smem_addr) | (matrix_desc_encode(128) << 32ULL) | (1ULL << 46ULL);
}

__device__ __forceinline__ void mbarrier_init(
    int addr) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(addr));
}

__device__ __forceinline__ void mbarrier_wait(
    int addr,
    int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t.reg .pred done;\n\t"
                 "WAIT_FUSED_MMA:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE_FUSED_MMA;\n\t"
                 "bra.uni WAIT_FUSED_MMA;\n\t"
                 "COMPLETE_FUSED_MMA:\n\t}" ::"r"(addr),
        "r"(phase),
        "r"(kSuspendTicks)
        : "memory");
}

__device__ __forceinline__ int swizzled_vector(
    int row,
    int vector) {
    const int linear = row * 8 + vector;
    return (linear & ~7) + ((linear & 7) ^ ((linear >> 3) & 7));
}

__device__ __forceinline__ int scale_offset(
    int row,
    int group) {
    return (row & 31) * 16 + ((row >> 5) & 3) * 4 + group;
}

__device__ __forceinline__ void cluster_barrier() {
    if constexpr (kClusterM == 1) {
        __syncthreads();
    } else {
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
    }
}

__device__ __forceinline__ int map_shared_to_cluster_cta(
    int local_smem_addr,
    int target_cta_rank) {
    if constexpr (kClusterM == 1)
        return local_smem_addr;
    int remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote_addr)
        : "r"(local_smem_addr), "r"(target_cta_rank));
    return remote_addr;
}

__device__ __forceinline__ void mbarrier_arrive_on_cluster_cta(
    int local_mbar_addr,
    int target_cta_rank) {
    const int remote_addr = map_shared_to_cluster_cta(local_mbar_addr, target_cta_rank);
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(remote_addr)
        : "memory");
}

__device__ __forceinline__ void pair_barrier(
    int gather_addr,
    int release_addr,
    int pair_root,
    int phase,
    int tid) {
    if constexpr (kClusterM == 1) {
        __syncthreads();
        return;
    }
    __syncthreads();
    if (tid == 0) {
        mbarrier_arrive_on_cluster_cta(gather_addr, pair_root);
        int cta_rank;
        asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
        if (cta_rank == pair_root) {
            mbarrier_wait(gather_addr, phase);
            mbarrier_arrive_on_cluster_cta(release_addr, pair_root);
            mbarrier_arrive_on_cluster_cta(release_addr, pair_root + 1);
        }
        mbarrier_wait(release_addr, phase);
    }
    __syncthreads();
}

__device__ __forceinline__ void mbarrier_expect_tx(
    int addr,
    int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
                 "_, [%0], %1;" ::"r"(addr),
        "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void cp_async_16b(
    int destination,
    const void *source,
    bool valid) {
    const uint32_t ignore = valid ? 0U : 1U;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "setp.ne.u32 p, %2, 0;\n\t"
                 "cp.async.cg.shared::cta.global [%0], [%1], 16, p;\n\t}" ::"r"(destination),
        "l"(source),
        "r"(ignore)
        : "memory");
}

__device__ __forceinline__ void cp_async_4b(
    int destination,
    const void *source,
    bool valid) {
    const uint32_t ignore = valid ? 0U : 1U;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "setp.ne.u32 p, %2, 0;\n\t"
                 "cp.async.ca.shared::cta.global [%0], [%1], 4, p;\n\t}" ::"r"(destination),
        "l"(source),
        "r"(ignore)
        : "memory");
}

__device__ __forceinline__ void mbarrier_invalidate(
    int addr) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(addr) : "memory");
}

__device__ __forceinline__ void cp_async_mbarrier_arrive_noinc(
    int addr) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(addr) : "memory");
}

__device__ __forceinline__ void tma_load_4d(
    int destination,
    const void *tensor_map,
    int x,
    int y,
    int z,
    int w,
    int mbar_addr) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global."
                 "mbarrier::complete_tx::bytes.cta_group::1 "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(destination),
        "l"(tensor_map),
        "r"(x),
        "r"(y),
        "r"(z),
        "r"(w),
        "r"(mbar_addr)
        : "memory");
}

// SFB is identical for both CTA ranks. One multicast uses independent
// cta_group::1 completion signals for each rank's local TMA barrier.
__device__ __forceinline__ void tma_load_sfb(
    int destination,
    const void *map,
    int k_tile,
    int n_tile,
    int mbar_addr,
    int cta_rank) {
    if constexpr (kClusterM == 1) {
        tma_load_4d(destination, map, 0, 0, k_tile, n_tile, mbar_addr);
    } else if (cta_rank == 0) {
        constexpr uint16_t mask = 3;
        asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global."
                     "mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1 "
                     "[%0], [%1, {0, 0, %2, %3}], [%4], %5;" ::"r"(destination),
            "l"(map),
            "r"(k_tile),
            "r"(n_tile),
            "r"(mbar_addr),
            "h"(mask)
            : "memory");
    }
}

__device__ __forceinline__ void bulk_store_bf16(
    __nv_bfloat16 *destination,
    const __nv_bfloat16 *source,
    int bytes) {
    const uint64_t destination_addr = reinterpret_cast<uint64_t>(destination);
    const uint32_t source_addr = static_cast<uint32_t>(__cvta_generic_to_shared(source));
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group "
                 "[%0], [%1], %2;" ::"l"(destination_addr),
        "r"(source_addr),
        "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void bulk_load_bytes(
    int destination_smem_addr,
    const void *source,
    int bytes,
    int mbar_addr) {
    const uint64_t source_addr = reinterpret_cast<uint64_t>(source);
    asm volatile("cp.async.bulk.shared::cluster.global."
                 "mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];" ::"r"(destination_smem_addr),
        "l"(source_addr),
        "r"(bytes),
        "r"(mbar_addr)
        : "memory");
}

__device__ __forceinline__ void grid_barrier(
    uint32_t *count,
    uint32_t *sense,
    int expected_clusters,
    uint32_t phase,
    int cta_rank,
    int tid) {
    cluster_barrier();
    if (cta_rank == 0 && tid == 0) {
        __threadfence();
        const uint32_t ticket = atomicAdd(count, 1U);
        if (ticket + 1U == static_cast<uint32_t>(expected_clusters)) {
            atomicExch(count, 0U);
            __threadfence();
            atomicExch(sense, phase);
        } else {
            while (atomicAdd(sense, 0U) != phase) {
            }
        }
    }
    cluster_barrier();
}

__device__ __forceinline__ void bulk_reduce_add_bf16(
    __nv_bfloat16 *destination,
    const __nv_bfloat16 *source,
    int bytes) {
    const uint64_t destination_addr = reinterpret_cast<uint64_t>(destination);
    const uint32_t source_addr = static_cast<uint32_t>(__cvta_generic_to_shared(source));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
                 "[%0], [%1], %2;" ::"l"(destination_addr),
        "r"(source_addr),
        "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void bulk_commit_and_wait_read() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

__device__ __forceinline__ void issue_mma_step(
    int tmem_addr,
    int scale_addr,
    int a_smem_addr,
    int b_smem_addr,
    int sfa_smem_addr,
    int sfb_smem_addr,
    int mma_k,
    int accumulate) {
    asm volatile("tcgen05.cp.cta_group::" MXFP4_FC1FC2_CTA_GROUP
                 ".32x128b.warpx4 [%0], %1;" ::"r"(scale_addr),
        "l"(make_scale_desc(sfa_smem_addr + mma_k * 512))
        : "memory");
    asm volatile("tcgen05.cp.cta_group::" MXFP4_FC1FC2_CTA_GROUP
                 ".32x128b.warpx4 [%0], %1;" ::"r"(scale_addr + 4),
        "l"(make_scale_desc(sfb_smem_addr + mma_k * 512))
        : "memory");
    asm volatile("tcgen05.cp.cta_group::" MXFP4_FC1FC2_CTA_GROUP
                 ".32x128b.warpx4 [%0], %1;" ::"r"(scale_addr + 8),
        "l"(make_scale_desc(sfb_smem_addr + kScaleStageBytes + mma_k * 512))
        : "memory");
    constexpr uint32_t instruction_desc = (5U << 10U) | (1U << 23U) |
                                          (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
                                          (static_cast<uint32_t>(kPairTileM) >> 7U << 27U);
    const uint64_t a_desc = make_k_major_128b_desc(a_smem_addr + mma_k * 32);
    const uint64_t b_desc = make_k_major_128b_desc(b_smem_addr + mma_k * 32);
    asm volatile("{\n\t.reg .pred input_d;\n\t"
                 "setp.ne.b32 input_d, %6, 0;\n\t"
                 "tcgen05.mma.cta_group::" MXFP4_FC1FC2_CTA_GROUP
                 ".kind::mxf8f6f4.block_scale.scale_vec::1X "
                 "[%0], %1, %2, %3, [%4], [%5], input_d;\n\t}" ::"r"(tmem_addr),
        "l"(a_desc),
        "l"(b_desc),
        "r"(instruction_desc),
        "r"(scale_addr),
        "r"(scale_addr + 4),
        "r"(accumulate)
        : "memory");
}

__device__ __forceinline__ void unpack_weight_scales(
    uint8_t *shared_scales) {
    // Prefetch the entire compact tile before any overlapping expanded store.
    // One warp owns each stage; the producer cannot reuse it until MMA commits.
    static_assert(kWeightScaleBytes == 4096);
    const int lane = threadIdx.x & 31;
    uint32_t packed[8];
#pragma unroll
    for (int block = 0; block < 8; ++block)
        packed[block] = reinterpret_cast<const uint32_t *>(shared_scales)[block * 32 + lane];
    __syncwarp();
#pragma unroll
    for (int block = 0; block < 8; ++block) {
        const uint32_t raw = packed[block];
        reinterpret_cast<uint4 *>(shared_scales)[block * 32 + lane] =
            uint4{raw & 255U, (raw >> 8) & 255U, (raw >> 16) & 255U, raw >> 24};
    }
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void commit_pair_mma(
    int done_mbar_addr,
    int pair_id) {
    if constexpr (kClusterM == 1) {
        asm volatile("tcgen05.commit.cta_group::1."
                     "mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(done_mbar_addr)
            : "memory");
    } else {
        const int16_t mask = static_cast<int16_t>(3 << (pair_id * 2));
        asm volatile("tcgen05.commit.cta_group::2."
                     "mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
                     "[%0], %1;" ::"r"(done_mbar_addr),
            "h"(mask)
            : "memory");
    }
}

#if SM110A_MXFP4_FC1FC2_TILED
#include "mxfp4_fc1fc2_tiled.cuh"
#else
__global__ __cluster_dims__(kClusterCtas, 1, 1)
__launch_bounds__(kWarpCount * kWarpSize, 1)
void mxfp4_fc1fc2_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap sfa_map,
    const __grid_constant__ CUtensorMap sfw1_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap sfw2_map,
    const uint8_t* raw_a, const uint8_t* raw_sfa,
    const int32_t* source_rows, const ExpertTileTask* tasks,
    const int32_t* task_state, const int32_t* grouped_assignments,
    const void* routing_weights,
    __nv_bfloat16* output, uint8_t* global_workspace, float swiglu_limit,
    int tokens, int m, int topk, int row_task_count_arg) {
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    constexpr int pair_id = 0;
    const int pair_rank = cta_rank;
    const int worker_id = blockIdx.x / kClusterCtas;
    const int worker_count = gridDim.x / kClusterCtas;
    const int row_task_count = kFuseRouting ? task_state[0] / kFc1TasksPerRowTask
                                           : row_task_count_arg;
    auto *grid_barrier_words =
        reinterpret_cast<uint32_t *>(global_workspace + static_cast<size_t>(row_task_count_arg) *
                                                            kClusterM * kIntermediateBytes);

    __shared__ int tmem_base[1];
    __shared__ uint64_t pair_ready[kBarrierStageCount];
    __shared__ uint64_t tma_done[kBarrierStageCount];
    __shared__ uint64_t stage_free[kBarrierStageCount];
    __shared__ uint64_t gather_done[kFc1StageCount];
    __shared__ uint64_t output_done[kAccumulatorStages];
    __shared__ uint64_t epilogue_free[kAccumulatorStages];
    __shared__ uint64_t overlap_free[kAccumulatorStages];
    __shared__ uint64_t pair_gather[1];
    __shared__ uint64_t pair_release[1];
    __shared__ int task_source_rows[kTileM];
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t *intermediate_values = smem;
    // The compact epilogue slot is disjoint from the live mainloop ring.
    uint8_t *fc1_intermediate_scales = smem + kFc1ScaleStagingOffset;
    uint8_t *fc1_workspace = smem + kEpilogueSmemBytes;
    uint8_t *fc2_workspace = smem + kEpilogueSmemBytes;
    auto *finalize_smem = reinterpret_cast<__nv_bfloat16 *>(smem);
    const int fc1_workspace_addr = static_cast<int>(__cvta_generic_to_shared(fc1_workspace));
    const int fc2_workspace_addr = static_cast<int>(__cvta_generic_to_shared(fc2_workspace));
    const int pair_ready_addr = static_cast<int>(__cvta_generic_to_shared(pair_ready));
    const int tma_done_addr = static_cast<int>(__cvta_generic_to_shared(tma_done));
    const int stage_free_addr = static_cast<int>(__cvta_generic_to_shared(stage_free));
    const int gather_done_addr = static_cast<int>(__cvta_generic_to_shared(gather_done));
    const int output_done_addr = static_cast<int>(__cvta_generic_to_shared(output_done));
    const int epilogue_free_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_free));
    const int overlap_free_addr = static_cast<int>(__cvta_generic_to_shared(overlap_free));
    const int pair_gather_addr = static_cast<int>(__cvta_generic_to_shared(pair_gather));
    const int pair_release_addr = static_cast<int>(__cvta_generic_to_shared(pair_release));
    if (warp == kMmaWarp) {
        const int tmem_base_addr = static_cast<int>(__cvta_generic_to_shared(tmem_base));
        asm volatile("tcgen05.alloc.cta_group::" MXFP4_FC1FC2_CTA_GROUP
                     ".sync.aligned.shared::cta.b32 "
                     "[%0], %1;" ::"r"(tmem_base_addr),
            "r"(kTmemColumns));
    }
    cluster_barrier();
    const int tmem_addr = tmem_base[0];

    const int task_count = row_task_count * kHSplit;
    for (int task = worker_id; task < task_count; task += worker_count) {
        const int row_task = task / kHSplit;
        const int h_partition = task % kHSplit;
        const int h_tile_begin = h_partition * (kH / kTileN / kHSplit);
        const int h_tile_end = h_tile_begin + kH / kTileN / kHSplit;
        const int pair_row = row_task * kClusterRows;
        ExpertTileTask row_info{pair_row, kClusterRows, 0, 0, 0};
        if constexpr (kFuseRouting) {
            row_info = tasks[row_task * kFc1TasksPerRowTask];
        }
        const int cta_tile_row = row_info.tile_row + pair_rank * kTileM;
        const int cta_row = row_info.row_start + cta_tile_row;
        const int remaining_rows = row_info.row_count - cta_tile_row;
        const int valid_rows =
            remaining_rows <= 0 ? 0 : (remaining_rows < kTileM ? remaining_rows : kTileM);
        const int expert = row_info.expert;
        if constexpr (kFuseRouting) {
            if (warp >= kGatherWarpBase && warp < kGatherWarpBase + kGatherWarpCount) {
                for (int row = (warp - kGatherWarpBase) * kWarpSize + (tid & 31); row < kTileM;
                    row += kGatherThreadCount)
                    task_source_rows[row] =
                        row < valid_rows ? __ldg(source_rows + cta_row + row) : 0;
                asm volatile("bar.sync 4, %0;" ::"r"(kGatherThreadCount) : "memory");
            }
        }
        if (tid == 0) {
            for (int stage = 0; stage < kBarrierStageCount; ++stage) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                                 pair_ready_addr + stage * static_cast<int>(sizeof(uint64_t))),
                    "r"(kReadyArrivals));
                mbarrier_init(tma_done_addr + stage * sizeof(uint64_t));
                mbarrier_init(stage_free_addr + stage * sizeof(uint64_t));
                if (stage < kFc1StageCount) {
                    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                                     gather_done_addr + stage * static_cast<int>(sizeof(uint64_t))),
                        "r"(kGatherThreadCount));
                }
            }
            for (int stage = 0; stage < kAccumulatorStages; ++stage) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                                 overlap_free_addr + stage * static_cast<int>(sizeof(uint64_t))),
                    "r"(kClusterM));
                mbarrier_init(output_done_addr + stage * sizeof(uint64_t));
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                                 epilogue_free_addr + stage * static_cast<int>(sizeof(uint64_t))),
                    "r"(kClusterM));
            }
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(pair_gather_addr),
                "r"(kClusterM));
            mbarrier_init(pair_release_addr);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        __syncthreads();
        cluster_barrier();
        int tma_stage = 0;
        int tma_free_phase = 1;
        int ready_tma_stage = 0;
        int ready_tma_phase = 0;
        int gather_stage = 0;
        int gather_free_phase = 1;
        int gather_done_phase = 0;
        int mma_stage = 0;
        int mma_ready_phase = 0;
        int output_phase = 0;
        int pair_sync_phase = 0;

        const int pair_i_block_begin = h_partition * (kIntermediateAtoms / kHSplit);
        const int pair_i_block_end = pair_i_block_begin + kIntermediateAtoms / kHSplit;
        auto *scratch = global_workspace +
                        static_cast<size_t>(row_task * kClusterM + pair_rank) * kIntermediateBytes;
        // FC1: a physical 256-column tile yields one logical 128-column block.
        if (warp == kTmaWarp && elect_sync()) {
            for (int i_block = pair_i_block_begin; i_block < pair_i_block_end; ++i_block) {
                const int physical_n = i_block * kTileN;
                for (int h_stage = 0; h_stage < kH / kStageK; ++h_stage) {
                    const int h = h_stage * kStageK;
                    const int stage_base = fc1_workspace_addr + tma_stage * kWorkspaceBytes;
                    const int stage_a = stage_base;
                    const int stage_b = stage_a + kMatrixSmemBytes;
                    const int stage_sfa = stage_b + kWeightSmemBytes;
                    const int stage_sfb = stage_sfa + kScaleStageBytes;
                    const int local_tma_done = tma_done_addr + tma_stage * sizeof(uint64_t);
                    mbarrier_wait(stage_free_addr + tma_stage * sizeof(uint64_t), tma_free_phase);
                    mbarrier_expect_tx(local_tma_done,
                        kFuseRouting ? kWeightTmaBytes + kWeightScaleTmaBytes
                                     : kMatrixSmemBytes + kWeightTmaBytes + kScaleStageBytes +
                                           kWeightScaleTmaBytes);
                    if constexpr (!kFuseRouting) {
                        tma_load_4d(stage_a, &a_map, h, cta_row, 0, 0, local_tma_done);
                    }
                    tma_load_4d(stage_b,
                        &w1_map,
                        h,
                        expert * kPhysicalFc1N + physical_n + pair_rank * (kTileN / kClusterM),
                        0,
                        0,
                        local_tma_done);
                    if constexpr (!kFuseRouting) {
                        tma_load_4d(stage_sfa,
                            &sfa_map,
                            0,
                            0,
                            h / kMmaK,
                            cta_row / kTileM,
                            local_tma_done);
                    }
                    tma_load_sfb(stage_sfb,
                        &sfw1_map,
                        h / kMmaK,
                        expert * (kPhysicalFc1N / 128) + physical_n / 128,
                        local_tma_done,
                        cta_rank);
                    tma_stage = (tma_stage + 1) % kFc1StageCount;
                    if (tma_stage == 0) {
                        tma_free_phase ^= 1;
                    }
                }
            }
        }
        if constexpr (kFuseRouting) {
            if (warp >= kGatherWarpBase && warp < kGatherWarpBase + kGatherWarpCount) {
                const int gather_rank = warp - kGatherWarpBase;
                const int lane = tid & (kWarpSize - 1);
                constexpr int kVectorsPerRow = kStageK / 16;
                constexpr int kRowsPerIssue = kWarpSize / kVectorsPerRow;
                constexpr int kRowsPerGatherWarp = kTileM / kGatherWarpCount;
                constexpr int kRowGroupCount = kRowsPerGatherWarp / kRowsPerIssue;
                int a_source_row_cache[kRowGroupCount];
                constexpr int kScaleRowsPerThread = kTileM / kGatherThreadCount;
                int sfa_source_rows[kScaleRowsPerThread];
                {
#pragma unroll
                    for (int row_group = 0; row_group < kRowGroupCount; ++row_group) {
                        const int local_row = gather_rank * kRowsPerGatherWarp +
                                              row_group * kRowsPerIssue + lane / kVectorsPerRow;
                        a_source_row_cache[row_group] = task_source_rows[local_row];
                    }
                }
                const int sfa_local_row = gather_rank * kWarpSize + lane;
                {
#pragma unroll
                    for (int row_index = 0; row_index < kScaleRowsPerThread; ++row_index)
                        sfa_source_rows[row_index] =
                            task_source_rows[sfa_local_row + row_index * kGatherThreadCount];
                }
                for (int i_block = pair_i_block_begin; i_block < pair_i_block_end; ++i_block) {
                    for (int h_stage = 0; h_stage < kH / kStageK; ++h_stage) {
                        const int h = h_stage * kStageK;
                        const int stage_base = fc1_workspace_addr + gather_stage * kWorkspaceBytes;
                        const int stage_a = stage_base;
                        const int stage_sfa = stage_base + kMatrixSmemBytes + kWeightSmemBytes;
                        mbarrier_wait(stage_free_addr + gather_stage * sizeof(uint64_t),
                            gather_free_phase);
                        const int vec = lane % kVectorsPerRow;
#pragma unroll
                        for (int row_group = 0; row_group < kRowGroupCount; ++row_group) {
                            const int local_row = gather_rank * kRowsPerGatherWarp +
                                                  row_group * kRowsPerIssue + lane / kVectorsPerRow;
                            const bool valid = local_row < valid_rows;
                            const int source_row = a_source_row_cache[row_group];
                            const size_t source_offset =
                                static_cast<size_t>(source_row) * kH + h + vec * 16;
                            const int destination = stage_a + swizzled_vector(local_row, vec) * 16;
                            cp_async_16b(destination, raw_a + source_offset, valid);
                        }
#pragma unroll
                        for (int row_index = 0; row_index < kScaleRowsPerThread; ++row_index) {
                            const int local_row = sfa_local_row + row_index * kGatherThreadCount;
                            const bool valid = local_row < valid_rows;
                            const int source_row = sfa_source_rows[row_index];
#pragma unroll
                            for (int mma_k = 0; mma_k < kMmaSteps; ++mma_k) {
                                const int source_in_tile =
                                    (source_row & 31) * 16 + ((source_row >> 5) & 3) * 4;
                                const size_t source_tile = static_cast<size_t>(source_row >> 7);
                                const size_t global_mma_k = h / kMmaK + mma_k;
                                const int destination =
                                    stage_sfa + mma_k * 512 + scale_offset(local_row, 0);
                                cp_async_4b(destination,
                                    raw_sfa + (source_tile * (kH / kMmaK) + global_mma_k) * 512 +
                                        source_in_tile,
                                    valid);
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        cp_async_mbarrier_arrive_noinc(
                            gather_done_addr + gather_stage * sizeof(uint64_t));
                        gather_stage = (gather_stage + 1) % kFc1StageCount;
                        if (gather_stage == 0) {
                            gather_free_phase ^= 1;
                            gather_done_phase ^= 1;
                        }
                    }
                }
            }
        }
        if (kClusterM > 1 && (!kReusePeerMmaWarp || pair_rank == 1) && warp == kTmaReadyWarp &&
            (kCompactWeightScales || elect_sync())) {
            for (int i_block = pair_i_block_begin; i_block < pair_i_block_end; ++i_block) {
                for (int h_stage = 0; h_stage < kH / kStageK; ++h_stage) {
                    mbarrier_wait(tma_done_addr + ready_tma_stage * sizeof(uint64_t),
                        ready_tma_phase);
                    if constexpr (kFuseRouting) {
                        mbarrier_wait(gather_done_addr + ready_tma_stage * sizeof(uint64_t),
                            gather_done_phase);
                    }
                    if constexpr (kCompactWeightScales)
                        unpack_weight_scales(fc1_workspace + ready_tma_stage * kWorkspaceBytes +
                                             kMatrixSmemBytes + kWeightSmemBytes +
                                             kScaleStageBytes);
                    if (!kCompactWeightScales || elect_sync())
                        mbarrier_arrive_on_cluster_cta(
                            pair_ready_addr + ready_tma_stage * sizeof(uint64_t),
                            pair_id * kClusterM);
                    ready_tma_stage = (ready_tma_stage + 1) % kFc1StageCount;
                    if (ready_tma_stage == 0)
                        ready_tma_phase ^= 1;
                    if constexpr (kFuseRouting) {
                        if (ready_tma_stage == 0)
                            gather_done_phase ^= 1;
                    }
                }
            }
        }
        if (pair_rank == 0 && warp == kMmaWarp && (kCompactWeightScales || elect_sync())) {
            int accumulator_stage = 0;
            int accumulator_free_phase = 1;
            int overlap_phase = 0;
            bool has_previous = false;
            for (int i_block = pair_i_block_begin; i_block < pair_i_block_end; ++i_block) {
                mbarrier_wait(epilogue_free_addr + accumulator_stage * sizeof(uint64_t),
                    accumulator_free_phase);
                if constexpr (kAccumulatorStages == 2) {
                    if (has_previous) {
                        const int previous_stage = accumulator_stage ^ 1;
                        mbarrier_wait(overlap_free_addr + previous_stage * sizeof(uint64_t),
                            overlap_phase);
                        if (previous_stage == 1)
                            overlap_phase ^= 1;
                    }
                    has_previous = true;
                }
                const int accumulator_addr = tmem_addr + accumulator_stage * kAccumulatorStride;
                for (int h_stage = 0; h_stage < kH / kStageK; ++h_stage) {
                    const int stage_base = fc1_workspace_addr + mma_stage * kWorkspaceBytes;
                    if constexpr (kClusterM == 1 || kReusePeerMmaWarp) {
                        mbarrier_wait(tma_done_addr + mma_stage * sizeof(uint64_t),
                            mma_ready_phase);
                        if constexpr (kFuseRouting)
                            mbarrier_wait(gather_done_addr + mma_stage * sizeof(uint64_t),
                                mma_ready_phase);
                        if constexpr (kReusePeerMmaWarp)
                            mbarrier_wait(pair_ready_addr + mma_stage * sizeof(uint64_t),
                                mma_ready_phase);
                    } else {
                        mbarrier_wait(pair_ready_addr + mma_stage * sizeof(uint64_t),
                            mma_ready_phase);
                    }
                    if constexpr (kCompactWeightScales && (kClusterM == 1 || kReusePeerMmaWarp))
                        unpack_weight_scales(fc1_workspace + mma_stage * kWorkspaceBytes +
                                             kMatrixSmemBytes + kWeightSmemBytes +
                                             kScaleStageBytes);
                    if (!kCompactWeightScales || elect_sync()) {
                        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
                        for (int mma_k = 0; mma_k < kMmaSteps; ++mma_k) {
                            issue_mma_step(accumulator_addr,
                                tmem_addr + kScaleTmemCol,
                                stage_base,
                                stage_base + kMatrixSmemBytes,
                                stage_base + kMatrixSmemBytes + kWeightSmemBytes,
                                stage_base + kMatrixSmemBytes + kWeightSmemBytes + kScaleStageBytes,
                                mma_k,
                                h_stage != 0 || mma_k != 0);
                        }
                        commit_pair_mma(stage_free_addr + mma_stage * sizeof(uint64_t), pair_id);
                    }
                    mma_stage = (mma_stage + 1) % kFc1StageCount;
                    if (mma_stage == 0)
                        mma_ready_phase ^= 1;
                }
                if (!kCompactWeightScales || elect_sync())
                    commit_pair_mma(output_done_addr + accumulator_stage * sizeof(uint64_t),
                        pair_id);
                accumulator_stage = (accumulator_stage + 1) % kAccumulatorStages;
                if (accumulator_stage == 0)
                    accumulator_free_phase ^= 1;
            }
        }

        if (warp < kEpilogueWarps) {
            int accumulator_stage = 0;
            for (int i_block = pair_i_block_begin; i_block < pair_i_block_end; ++i_block) {
                const int accumulator_addr = tmem_addr + accumulator_stage * kAccumulatorStride;
                mbarrier_wait(output_done_addr + accumulator_stage * sizeof(uint64_t),
                    output_phase);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const int local_row = warp * kWarpSize + (tid & 31);
                const int tmem_row = pair_rank * kTileM + warp * kWarpSize;
                float activated[128];
                float abs_max = 0.0f;
                float router_scale = 1.0f;
                if constexpr (kFuseRouting && kWeightInFc1) {
                    if (local_row < valid_rows)
                        router_scale = static_cast<const float *>(
                            routing_weights)[grouped_assignments[cta_row + local_row]];
                }
                // tcgen05.ld is warp collective; whole padded warps can
                // skip it while all four warps retain the release barriers.
                if (!kFuseRouting || warp * kWarpSize < valid_rows) {
#pragma unroll
                    for (int group = 0; group < 8; ++group) {
                        float gate_values[16], up_values[16];
                        const int gate_addr = accumulator_addr + (tmem_row << 16) +
                                              (group / 4) * 128 + (group % 4) * 16;
                        const int up_addr = gate_addr + 64;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                            : "=f"(gate_values[0]),
                            "=f"(gate_values[1]),
                            "=f"(gate_values[2]),
                            "=f"(gate_values[3]),
                            "=f"(gate_values[4]),
                            "=f"(gate_values[5]),
                            "=f"(gate_values[6]),
                            "=f"(gate_values[7]),
                            "=f"(gate_values[8]),
                            "=f"(gate_values[9]),
                            "=f"(gate_values[10]),
                            "=f"(gate_values[11]),
                            "=f"(gate_values[12]),
                            "=f"(gate_values[13]),
                            "=f"(gate_values[14]),
                            "=f"(gate_values[15])
                            : "r"(gate_addr));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                            : "=f"(up_values[0]),
                            "=f"(up_values[1]),
                            "=f"(up_values[2]),
                            "=f"(up_values[3]),
                            "=f"(up_values[4]),
                            "=f"(up_values[5]),
                            "=f"(up_values[6]),
                            "=f"(up_values[7]),
                            "=f"(up_values[8]),
                            "=f"(up_values[9]),
                            "=f"(up_values[10]),
                            "=f"(up_values[11]),
                            "=f"(up_values[12]),
                            "=f"(up_values[13]),
                            "=f"(up_values[14]),
                            "=f"(up_values[15])
                            : "r"(up_addr));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
                        for (int col = 0; col < 16; ++col) {
                            float gate = gate_values[col], up = up_values[col];
                            if constexpr (kActivationBf16Rounding) {
                                gate = __bfloat162float(__float2bfloat16_rn(gate));
                                up = __bfloat162float(__float2bfloat16_rn(up));
                            }
                            if (swiglu_limit > 0.0f) {
                                gate = fminf(gate, swiglu_limit);
                                up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
                            }
                            float value = silu(gate) * up;
                            if constexpr (kWeightInFc1)
                                value *= router_scale;
                            if constexpr (kActivationBf16Rounding)
                                value = __bfloat162float(__float2bfloat16_rn(value));
                            activated[group * 16 + col] = value;
                            abs_max = fmaxf(abs_max, fabsf(value));
                        }
                    }
                }
                if constexpr (kAccumulatorStages == 2) {
                    asm volatile("bar.sync 2, 128;" ::: "memory");
                    if (warp == 0 && (tid & 31) == 0)
                        mbarrier_arrive_on_cluster_cta(
                            overlap_free_addr + accumulator_stage * sizeof(uint64_t),
                            0);
                }
                uint8_t scale_raw = 0;
                if (local_row < valid_rows) {
                    if constexpr (kActivationBf16Rounding)
                        abs_max = fmaxf(abs_max, 1.0e-4f);
                    scale_raw = encode_activation_scale(abs_max);
                    const float inverse_scale = reciprocal_e8m0(scale_raw);
                    alignas(16) uint16_t row_quantized[64];
#pragma unroll
                    for (int col = 0; col < 128; col += 2)
                        row_quantized[col / 2] =
                            __nv_cvt_float2_to_fp8x2(make_float2(activated[col] * inverse_scale,
                                                         activated[col + 1] * inverse_scale),
                                __NV_SATFINITE,
                                __NV_E4M3);
#pragma unroll
                    for (int vector = 0; vector < 8; ++vector)
                        reinterpret_cast<uint4 *>(
                            intermediate_values)[swizzled_vector(local_row, vector)] =
                            reinterpret_cast<const uint4 *>(row_quantized)[vector];
                }
#pragma unroll
                for (int replica = 0; replica < 4; ++replica)
                    fc1_intermediate_scales[replica * 512 + scale_offset(local_row, 0)] = scale_raw;
                asm volatile("bar.sync 2, 128;" ::: "memory");
                if (warp == 0 && (tid & 31) == 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (valid_rows > 0)
                        bulk_store_bf16(
                            reinterpret_cast<__nv_bfloat16 *>(scratch + i_block * kMatrixSmemBytes),
                            reinterpret_cast<const __nv_bfloat16 *>(intermediate_values),
                            valid_rows * kStageK);
                    bulk_store_bf16(
                        reinterpret_cast<__nv_bfloat16 *>(
                            scratch + kIntermediateValueBytes + i_block * kScaleStageBytes),
                        reinterpret_cast<const __nv_bfloat16 *>(fc1_intermediate_scales),
                        kScaleStageBytes);
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                }
                asm volatile("bar.sync 3, 128;" ::: "memory");
                if (warp == 0 && (tid & 31) == 0) {
                    mbarrier_arrive_on_cluster_cta(
                        epilogue_free_addr + accumulator_stage * sizeof(uint64_t),
                        pair_id * kClusterM);
                }
                accumulator_stage = (accumulator_stage + 1) % kAccumulatorStages;
                if (accumulator_stage == 0)
                    output_phase ^= 1;
            }
        }
        pair_barrier(pair_gather_addr, pair_release_addr, 0, pair_sync_phase, tid);
        pair_sync_phase ^= 1;
        if constexpr (kHSplit > 1)
            grid_barrier(grid_barrier_words + row_task * 2,
                grid_barrier_words + row_task * 2 + 1,
                kHSplit,
                1,
                cta_rank,
                tid);

        // Drain every multicast stage-release before reinitializing its
        // barrier. Output completion alone does not wait for remote arrivals.
        if (tid == 0) {
            const int tiles = pair_i_block_end - pair_i_block_begin;
            if (cta_rank == 0)
                for (int stage = 0; stage < kAccumulatorStages; ++stage) {
                    const int last_phase = ((tiles / kAccumulatorStages) - 1) & 1;
                    mbarrier_wait(epilogue_free_addr + stage * sizeof(uint64_t), last_phase);
                    if constexpr (kAccumulatorStages == 2)
                        mbarrier_wait(overlap_free_addr + stage * sizeof(uint64_t), last_phase);
                }
            const int fc1_steps = (pair_i_block_end - pair_i_block_begin) * (kH / kStageK);
            for (int stage = 0; stage < kFc1StageCount; ++stage) {
                const int uses = (fc1_steps + kFc1StageCount - 1 - stage) / kFc1StageCount;
                mbarrier_wait(stage_free_addr + stage * sizeof(uint64_t), (uses - 1) & 1);
            }
        }
        __syncthreads();
        cluster_barrier();

        // Drain, invalidate, and reinitialize before switching roles. PTX
        // forbids mbarrier.init on a valid object, even after completion.
        if (tid == 0) {
            for (int stage = 0; stage < kBarrierStageCount; ++stage) {
                mbarrier_invalidate(pair_ready_addr + stage * sizeof(uint64_t));
                mbarrier_invalidate(tma_done_addr + stage * sizeof(uint64_t));
                mbarrier_invalidate(stage_free_addr + stage * sizeof(uint64_t));
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                                 pair_ready_addr + stage * static_cast<int>(sizeof(uint64_t))),
                    "r"(kReadyArrivals));
                mbarrier_init(tma_done_addr + stage * sizeof(uint64_t));
                mbarrier_init(stage_free_addr + stage * sizeof(uint64_t));
            }
            for (int stage = 0; stage < kAccumulatorStages; ++stage) {
                mbarrier_invalidate(output_done_addr + stage * sizeof(uint64_t));
                mbarrier_invalidate(epilogue_free_addr + stage * sizeof(uint64_t));
                mbarrier_invalidate(overlap_free_addr + stage * sizeof(uint64_t));
                mbarrier_init(output_done_addr + stage * sizeof(uint64_t));
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                                 epilogue_free_addr + stage * static_cast<int>(sizeof(uint64_t))),
                    "r"(kClusterM));
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                                 overlap_free_addr + stage * static_cast<int>(sizeof(uint64_t))),
                    "r"(kClusterM));
            }
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        __syncthreads();
        cluster_barrier();

        // Reset the role-local cursors before reinterpreting the same
        // workspace as compact FC2 B/SFB stages.
        tma_stage = 0;
        tma_free_phase = 1;
        ready_tma_stage = 0;
        ready_tma_phase = 0;
        mma_stage = 0;
        mma_ready_phase = 0;
        output_phase = 0;

        // FC2 keeps complete FP32 accumulations in TMEM while the ring
        // streams FP8 activations and MXFP4 weights from the global image.
        if (warp == kTmaWarp && elect_sync()) {
            for (int h_tile = h_tile_begin; h_tile < h_tile_end; ++h_tile) {
                const int h = h_tile * kTileN;
                for (int i_stage = 0; i_stage < kI / kStageK; ++i_stage) {
                    const int global_i = i_stage * kStageK;
                    const int stage_base = fc2_workspace_addr + tma_stage * kFc2WorkspaceBytes;
                    const int stage_a = stage_base;
                    const int stage_b = stage_a + kMatrixSmemBytes;
                    const int stage_sfa = stage_b + kWeightSmemBytes;
                    const int stage_sfb = stage_sfa + kScaleStageBytes;
                    const int local_tma_done = tma_done_addr + tma_stage * sizeof(uint64_t);
                    mbarrier_wait(stage_free_addr + tma_stage * sizeof(uint64_t), tma_free_phase);
                    // FC1's gather zero-filled padded rows in every reused
                    // A slot. Preserve those zeros and transfer only live rows;
                    // otherwise sparse experts repeatedly stream large holes.
                    const int image_value_bytes = valid_rows * kStageK;
                    mbarrier_expect_tx(local_tma_done,
                        image_value_bytes + kWeightTmaBytes + kScaleStageBytes +
                            kWeightScaleTmaBytes);
                    if (image_value_bytes > 0)
                        bulk_load_bytes(stage_a,
                            scratch + i_stage * kMatrixSmemBytes,
                            image_value_bytes,
                            local_tma_done);
                    bulk_load_bytes(stage_sfa,
                        scratch + kIntermediateValueBytes + i_stage * kScaleStageBytes,
                        kScaleStageBytes,
                        local_tma_done);
                    tma_load_4d(stage_b,
                        &w2_map,
                        global_i,
                        expert * kH + h + pair_rank * (kTileN / kClusterM),
                        0,
                        0,
                        local_tma_done);
                    tma_load_sfb(stage_sfb,
                        &sfw2_map,
                        global_i / kMmaK,
                        expert * (kH / 128) + h / 128,
                        local_tma_done,
                        cta_rank);
                    tma_stage = (tma_stage + 1) % kFc2StageCount;
                    if (tma_stage == 0) {
                        tma_free_phase ^= 1;
                    }
                }
            }
        }
        if (kClusterM > 1 && (!kReusePeerMmaWarp || pair_rank == 1) && warp == kTmaReadyWarp &&
            (kCompactWeightScales || elect_sync())) {
            for (int h_tile = h_tile_begin; h_tile < h_tile_end; ++h_tile) {
                for (int i_stage = 0; i_stage < kI / kStageK; ++i_stage) {
                    mbarrier_wait(tma_done_addr + ready_tma_stage * sizeof(uint64_t),
                        ready_tma_phase);
                    if constexpr (kCompactWeightScales)
                        unpack_weight_scales(fc2_workspace + ready_tma_stage * kWorkspaceBytes +
                                             kMatrixSmemBytes + kWeightSmemBytes +
                                             kScaleStageBytes);
                    if (!kCompactWeightScales || elect_sync())
                        mbarrier_arrive_on_cluster_cta(
                            pair_ready_addr + ready_tma_stage * sizeof(uint64_t),
                            pair_id * kClusterM);
                    ready_tma_stage = (ready_tma_stage + 1) % kFc2StageCount;
                    if (ready_tma_stage == 0)
                        ready_tma_phase ^= 1;
                }
            }
        }
        if (pair_rank == 0 && warp == kMmaWarp && (kCompactWeightScales || elect_sync())) {
            int accumulator_stage = 0;
            int accumulator_free_phase = 1;
            int overlap_phase = 0;
            bool has_previous = false;
            for (int h_tile = h_tile_begin; h_tile < h_tile_end; ++h_tile) {
                mbarrier_wait(epilogue_free_addr + accumulator_stage * sizeof(uint64_t),
                    accumulator_free_phase);
                if constexpr (kAccumulatorStages == 2) {
                    if (has_previous) {
                        const int previous_stage = accumulator_stage ^ 1;
                        mbarrier_wait(overlap_free_addr + previous_stage * sizeof(uint64_t),
                            overlap_phase);
                        if (previous_stage == 1)
                            overlap_phase ^= 1;
                    }
                    has_previous = true;
                }
                const int accumulator_addr = tmem_addr + accumulator_stage * kFc2AccumulatorStride;
                for (int i_stage = 0; i_stage < kI / kStageK; ++i_stage) {
                    const int stage_base = fc2_workspace_addr + mma_stage * kFc2WorkspaceBytes;
                    if constexpr (kClusterM == 1 || kReusePeerMmaWarp)
                        mbarrier_wait(tma_done_addr + mma_stage * sizeof(uint64_t),
                            mma_ready_phase);
                    if constexpr (kClusterM > 1)
                        mbarrier_wait(pair_ready_addr + mma_stage * sizeof(uint64_t),
                            mma_ready_phase);
                    if constexpr (kCompactWeightScales && (kClusterM == 1 || kReusePeerMmaWarp))
                        unpack_weight_scales(fc2_workspace + mma_stage * kWorkspaceBytes +
                                             kMatrixSmemBytes + kWeightSmemBytes +
                                             kScaleStageBytes);
                    if (!kCompactWeightScales || elect_sync()) {
                        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
                        for (int mma_k = 0; mma_k < kMmaSteps; ++mma_k) {
                            issue_mma_step(accumulator_addr,
                                tmem_addr + kFc2ScaleTmemCol,
                                stage_base,
                                stage_base + kMatrixSmemBytes,
                                stage_base + kMatrixSmemBytes + kWeightSmemBytes,
                                stage_base + kMatrixSmemBytes + kWeightSmemBytes + kScaleStageBytes,
                                mma_k,
                                i_stage != 0 || mma_k != 0);
                        }
                        commit_pair_mma(stage_free_addr + mma_stage * sizeof(uint64_t), pair_id);
                    }
                    mma_stage = (mma_stage + 1) % kFc2StageCount;
                    if (mma_stage == 0)
                        mma_ready_phase ^= 1;
                }
                if (!kCompactWeightScales || elect_sync())
                    commit_pair_mma(output_done_addr + accumulator_stage * sizeof(uint64_t),
                        pair_id);
                accumulator_stage = (accumulator_stage + 1) % kAccumulatorStages;
                if (accumulator_stage == 0)
                    accumulator_free_phase ^= 1;
            }
        }

        if (warp < kEpilogueWarps) {
            int accumulator_stage = 0;
            for (int h_tile = h_tile_begin; h_tile < h_tile_end; ++h_tile) {
                const int h = h_tile * kTileN;
                const int accumulator_addr = tmem_addr + accumulator_stage * kFc2AccumulatorStride;
                mbarrier_wait(output_done_addr + accumulator_stage * sizeof(uint64_t),
                    output_phase);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                // A padded CTA with no routed rows has no observable TMEM
                // values. Release both slots without staging BF16 slabs.
                if constexpr (kFuseRouting) {
                    if (valid_rows == 0) {
                        asm volatile("bar.sync 2, 128;" ::: "memory");
                        if (warp == 0 && (tid & 31) == 0) {
                            if constexpr (kAccumulatorStages == 2)
                                mbarrier_arrive_on_cluster_cta(
                                    overlap_free_addr + accumulator_stage * sizeof(uint64_t),
                                    0);
                            mbarrier_arrive_on_cluster_cta(
                                epilogue_free_addr + accumulator_stage * sizeof(uint64_t),
                                0);
                        }
                        accumulator_stage = (accumulator_stage + 1) % kAccumulatorStages;
                        if (accumulator_stage == 0)
                            output_phase ^= 1;
                        continue;
                    }
                }
                const int local_row = warp * kWarpSize + (tid & 31);
                const int global_row = cta_row + local_row;
                const int tmem_row = pair_rank * kTileM + warp * kWarpSize;
                const bool valid_row = local_row < valid_rows;
                int token_row = global_row;
                float router_scale = 1.0f;
                if constexpr (kFuseRouting) {
                    if (valid_row) {
                        const int assignment = grouped_assignments[global_row];
                        token_row = assignment / topk;
                        if constexpr (!kWeightInFc1)
                            router_scale = __bfloat162float(
                                static_cast<const __nv_bfloat16 *>(routing_weights)[assignment]);
                    }
                }
                auto stage_and_store = [&](int slab, int group_begin, int group_end) {
                    for (int group = group_begin; group < group_end; ++group) {
                        float values[8];
                        const int load_addr =
                            accumulator_addr + (tmem_row << 16) + slab + group * 8;
                        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                                     "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                            : "=f"(values[0]),
                            "=f"(values[1]),
                            "=f"(values[2]),
                            "=f"(values[3]),
                            "=f"(values[4]),
                            "=f"(values[5]),
                            "=f"(values[6]),
                            "=f"(values[7])
                            : "r"(load_addr));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
                        for (int col = 0; col < 8; ++col) {
                            finalize_smem[local_row * kReduceN + group * 8 + col] =
                                __float2bfloat16_rn(values[col] * router_scale);
                        }
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (valid_row) {
                        auto *destination = output + static_cast<size_t>(token_row) * kH + h +
                                            slab + group_begin * 8;
                        auto *source = finalize_smem + local_row * kReduceN + group_begin * 8;
                        const int bytes = (group_end - group_begin) * 8 * sizeof(__nv_bfloat16);
                        if constexpr (kFuseRouting)
                            bulk_reduce_add_bf16(destination, source, bytes);
                        else
                            bulk_store_bf16(destination, source, bytes);
                    }
                    bulk_commit_and_wait_read();
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                };
                if constexpr (kAccumulatorStages == 2) {
                    const int overlap_slab = accumulator_stage == 0 ? kTileN - kReduceN : 0;
                    const int overlap_begin = accumulator_stage == 0 ? kReduceN / 8 - 2 : 0;
                    stage_and_store(overlap_slab, overlap_begin, overlap_begin + 2);
                    if (warp == 0 && (tid & 31) == 0)
                        mbarrier_arrive_on_cluster_cta(
                            overlap_free_addr + accumulator_stage * sizeof(uint64_t),
                            0);
                    if (accumulator_stage == 0)
                        stage_and_store(overlap_slab, 0, kReduceN / 8 - 2);
                    else
                        stage_and_store(overlap_slab, 2, kReduceN / 8);
#pragma unroll
                    for (int slab = 0; slab < kTileN; slab += kReduceN)
                        if (slab != overlap_slab)
                            stage_and_store(slab, 0, kReduceN / 8);
                } else {
#pragma unroll
                    for (int slab = 0; slab < kTileN; slab += kReduceN)
                        stage_and_store(slab, 0, kReduceN / 8);
                }
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                asm volatile("bar.sync 2, 128;" ::: "memory");
                if (warp == 0 && elect_sync()) {
                    mbarrier_arrive_on_cluster_cta(
                        epilogue_free_addr + accumulator_stage * sizeof(uint64_t),
                        pair_id * kClusterM);
                }
                accumulator_stage = (accumulator_stage + 1) % kAccumulatorStages;
                if (accumulator_stage == 0)
                    output_phase ^= 1;
            }
        }
        pair_barrier(pair_gather_addr,
            pair_release_addr,
            pair_id * kClusterM,
            pair_sync_phase,
            tid);
        pair_sync_phase ^= 1;

        if (tid == 0) {
            const int fc2_steps = (h_tile_end - h_tile_begin) * kIntermediateAtoms;
            for (int stage = 0; stage < kFc2StageCount; ++stage) {
                const int uses = (fc2_steps + kFc2StageCount - 1 - stage) / kFc2StageCount;
                mbarrier_wait(stage_free_addr + stage * sizeof(uint64_t), (uses - 1) & 1);
            }
            if (cta_rank == 0)
                for (int stage = 0; stage < kAccumulatorStages; ++stage) {
                    const int last_phase =
                        ((h_tile_end - h_tile_begin + kAccumulatorStages - 1 - stage) /
                                kAccumulatorStages - 1) & 1;
                    mbarrier_wait(epilogue_free_addr + stage * sizeof(uint64_t), last_phase);
                    if constexpr (kAccumulatorStages == 2)
                        mbarrier_wait(overlap_free_addr + stage * sizeof(uint64_t), last_phase);
                }
        }
        __syncthreads();
        cluster_barrier();
        if (tid == 0) {
            for (int stage = 0; stage < kBarrierStageCount; ++stage) {
                mbarrier_invalidate(pair_ready_addr + stage * sizeof(uint64_t));
                mbarrier_invalidate(tma_done_addr + stage * sizeof(uint64_t));
                mbarrier_invalidate(stage_free_addr + stage * sizeof(uint64_t));
                mbarrier_invalidate(gather_done_addr + stage * sizeof(uint64_t));
            }
            for (int stage = 0; stage < kAccumulatorStages; ++stage) {
                mbarrier_invalidate(output_done_addr + stage * sizeof(uint64_t));
                mbarrier_invalidate(epilogue_free_addr + stage * sizeof(uint64_t));
                mbarrier_invalidate(overlap_free_addr + stage * sizeof(uint64_t));
            }
            mbarrier_invalidate(pair_gather_addr);
            mbarrier_invalidate(pair_release_addr);
        }
        __syncthreads();
        cluster_barrier();
    }

    cluster_barrier();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::" MXFP4_FC1FC2_CTA_GROUP
                     ".sync.aligned.b32 %0, %1;" ::"r"(tmem_addr),
            "r"(kTmemColumns));
    }
}

#endif

template <int kBoxRows, bool kFp8 = false>
bool init_matrix_map(
    CUtensorMap *map,
    const uint8_t *pointer,
    uint64_t rows,
    uint64_t cols) {
    constexpr uint32_t rank = 4;
    uint64_t dimensions[rank] = {cols, rows, 1, 1};
    uint64_t strides[rank - 1] = {cols / (kFp8 ? 1 : 2),
        rows * cols / (kFp8 ? 1 : 2),
        rows * cols / (kFp8 ? 1 : 2)};
    uint32_t box[rank] = {kStageK, kBoxRows, 1, 1};
    uint32_t elements[rank] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(map,
               kFp8 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
               rank,
               const_cast<uint8_t *>(pointer),
               dimensions,
               strides,
               box,
               elements,
               CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_128B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <int kRowTiles = 1>
bool init_scale_map(
    CUtensorMap *map,
    const uint8_t *pointer,
    uint64_t row_tiles,
    uint64_t k_tiles) {
    constexpr uint32_t rank = 4;
    uint64_t dimensions[rank] = {256, 2, k_tiles, row_tiles};
    uint64_t strides[rank - 1] = {256, 512, 512 * k_tiles};
    uint32_t box[rank] = {256, 2, kMmaSteps, kRowTiles};
    uint32_t elements[rank] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(map,
               CU_TENSOR_MAP_DATA_TYPE_UINT8,
               rank,
               const_cast<uint8_t *>(pointer),
               dimensions,
               strides,
               box,
               elements,
               CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_NONE,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <int kRowTiles = 1>
bool init_compact_scale_map(
    CUtensorMap *map,
    const uint8_t *pointer,
    uint64_t row_tiles,
    uint64_t k_tiles) {
    constexpr uint32_t rank = 4;
    uint64_t dimensions[rank] = {128, 1, k_tiles, row_tiles};
    uint64_t strides[rank - 1] = {128, 128, 128 * k_tiles};
    uint32_t box[rank] = {128, 1, kMmaSteps, kRowTiles};
    uint32_t elements[rank] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(map,
               CU_TENSOR_MAP_DATA_TYPE_UINT8,
               rank,
               const_cast<uint8_t *>(pointer),
               dimensions,
               strides,
               box,
               elements,
               CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_NONE,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

} // namespace sm110a::mxfp4_fc1fc2

using namespace sm110a::mxfp4_fc1fc2;

cudaError_t launch_mxfp4_fc1fc2(
    const uint8_t *a,
    const uint8_t *w1,
    const uint8_t *sfa,
    const uint8_t *sfw1,
    const uint8_t *w2,
    const uint8_t *sfw2,
    const int32_t *source_rows,
    const int32_t *grouped_assignments,
    const int32_t *tasks,
    const int32_t *state,
    const void *routing_weights,
    __nv_bfloat16 *output,
    uint8_t *global_workspace,
    float swiglu_limit,
    int tokens,
    int m,
    int experts,
    int topk,
    int row_task_count,
    int max_tasks,
    cudaStream_t stream) {
    if (!a || !w1 || !sfa || !sfw1 || !w2 || !sfw2 || !output || !global_workspace || tokens <= 0 ||
        m <= 0 || experts <= 0 || topk <= 0) {
        return cudaErrorInvalidValue;
    }
    if constexpr (kFuseRouting) {
        if (!source_rows || !grouped_assignments || !tasks || !state || !routing_weights ||
            row_task_count <= 0 || max_tasks <= 0 || max_tasks % kFc1TasksPerRowTask)
            return cudaErrorInvalidValue;
    } else {
        if (m % kClusterRows)
            return cudaErrorInvalidValue;
    }
    CUtensorMap a_map;
    CUtensorMap w1_map;
    CUtensorMap sfa_map;
    CUtensorMap sfw1_map;
    CUtensorMap w2_map;
    CUtensorMap sfw2_map;
    const bool maps_ok =
        init_matrix_map<kTileM, true>(&a_map, a, tokens, kH) &&
        init_matrix_map<kTileN / kClusterM>(&w1_map, w1, experts * kPhysicalFc1N, kH) &&
        init_scale_map(&sfa_map, sfa, tokens / kTileM, kH / kMmaK) &&
        (kCompactWeightScales ? init_compact_scale_map<2> : init_scale_map<2>)(&sfw1_map,
            sfw1,
            experts * (kPhysicalFc1N / 128),
            kH / kMmaK) &&
        init_matrix_map<kTileN / kClusterM>(&w2_map, w2, experts * kH, kI) &&
        (kCompactWeightScales
                ? init_compact_scale_map<2>
                : init_scale_map<2>)(&sfw2_map, sfw2, experts * (kH / 128), kI / kMmaK);
    if (!maps_ok) {
        return cudaErrorInvalidValue;
    }
    auto kernel = mxfp4_fc1fc2_kernel;
    const cudaError_t attribute_status = cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSmemBytes);
    if (attribute_status != cudaSuccess)
        return attribute_status;
    const int row_tasks = kFuseRouting ? row_task_count : m / kClusterRows;
    const int launch_task_count = row_tasks * (kTiledSchedule ? kH / kTileN : kHSplit);
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = dim3(kClusterCtas * launch_task_count, 1, 1);
    launch_config.blockDim = dim3(kWarpCount * kWarpSize, 1, 1);
    launch_config.dynamicSmemBytes = kDynamicSmemBytes;
    launch_config.stream = stream;
    int resident_clusters = 0;
    const cudaError_t occupancy_status =
        cudaOccupancyMaxActiveClusters(&resident_clusters, kernel, &launch_config);
    if (occupancy_status != cudaSuccess)
        return occupancy_status;
    if (resident_clusters <= 0)
        return cudaErrorInvalidConfiguration;
    if (resident_clusters < kHSplit)
        return cudaErrorInvalidConfiguration;
    // Keep every row task's partitions resident together, including later
    // waves. Each row owns its image and a separate one-shot grid barrier.
    const int resident_rows = resident_clusters / kHSplit;
    const int workers =
        kTiledSchedule
            ? (launch_task_count < resident_clusters ? launch_task_count : resident_clusters)
            : (row_tasks < resident_rows ? row_tasks : resident_rows) * kHSplit;
    auto *workspace_control =
        global_workspace + static_cast<size_t>(row_tasks) * kClusterM * kIntermediateBytes;
    if constexpr (kFuseRouting) {
        const auto output_status = cudaMemsetAsync(output,
            0,
            static_cast<size_t>(tokens) * kH * sizeof(__nv_bfloat16),
            stream);
        if (output_status != cudaSuccess)
            return output_status;
    }
    if constexpr (kTiledSchedule || kHSplit > 1) {
        const auto control_status =
            cudaMemsetAsync(workspace_control, 0, static_cast<size_t>(row_tasks) * 8, stream);
        if (control_status != cudaSuccess)
            return control_status;
    }
    kernel<<<dim3(kClusterCtas * workers, 1, 1),
        kWarpCount * kWarpSize,
        kDynamicSmemBytes,
        stream>>>(a_map,
        w1_map,
        sfa_map,
        sfw1_map,
        w2_map,
        sfw2_map,
        a,
        sfa,
        source_rows,
        reinterpret_cast<const ExpertTileTask *>(tasks),
        state,
        grouped_assignments,
        routing_weights,
        output,
        global_workspace,
        swiglu_limit,
        tokens,
        m,
        topk,
        row_tasks);
    return cudaGetLastError();
}

extern "C" cudaError_t sm110a_mxfp4_fc1fc2_launch(
    const uint8_t *a,
    const uint8_t *w1,
    const uint8_t *sfa,
    const uint8_t *sfw1,
    const uint8_t *w2,
    const uint8_t *sfw2,
    const int32_t *source_rows,
    const int32_t *grouped_assignments,
    const int32_t *tasks,
    const int32_t *state,
    const void *routing_weights,
    __nv_bfloat16 *output,
    uint8_t *workspace,
    float swiglu_limit,
    int tokens,
    int m,
    int experts,
    int topk,
    int row_task_count,
    int max_tasks,
    cudaStream_t stream) {
    return launch_mxfp4_fc1fc2(a,
        w1,
        sfa,
        sfw1,
        w2,
        sfw2,
        source_rows,
        grouped_assignments,
        tasks,
        state,
        routing_weights,
        output,
        workspace,
        swiglu_limit,
        tokens,
        m,
        experts,
        topk,
        row_task_count,
        max_tasks,
        stream);
}
