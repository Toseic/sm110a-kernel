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
#define SM110A_NVFP4_FC1_EPILOGUE_TILE_N 64
#endif
#ifndef SM110A_NVFP4_FC1_CLUSTER_M
#define SM110A_NVFP4_FC1_CLUSTER_M 1
#endif
#ifndef SM110A_NVFP4_FC1_CLUSTER_N
#define SM110A_NVFP4_FC1_CLUSTER_N 1
#endif
#ifndef SM110A_NVFP4_FC1_DIRECT_GMEM
#define SM110A_NVFP4_FC1_DIRECT_GMEM 0
#endif
#ifndef SM110A_NVFP4_FC1_NUM_C_STAGE
#define SM110A_NVFP4_FC1_NUM_C_STAGE 0
#endif
namespace {

constexpr int kMmaK = 64;
constexpr int kWarpSize = 32;
constexpr int kEpilogueWarpCount = 4;
// Keep the A/SFA gather group independent from the B/SFB TMA issuer.  This
// matches the producer decomposition used by the CuTE baseline: four gather
// warps, one TMA warp, one MMA warp, and four epilogue warps.  In particular,
// cta_group::2 no longer doubles the gather group merely to overlap A/SFA.
constexpr int kAProducerWarpCount = 4;
constexpr int kSfaProducerWarpCount = kAProducerWarpCount;
constexpr int kProducerWarpCount = kAProducerWarpCount;
constexpr bool kSplitGatherWarps = false;
constexpr int kProducerWarpBase = kEpilogueWarpCount;
constexpr int kMmaWarp = kProducerWarpBase + kProducerWarpCount;
constexpr int kTmaWarp = kMmaWarp + 1;
// A dedicated scheduler warp owns persistent work acquisition and publishes
// task records into a small CTA-local ring.  Keeping it out of the producer
// group lets gather/TMA/MMA/epilogue advance independently across tasks.
constexpr int kSchedulerWarp = kTmaWarp + 1;
// The 2-CTA tcgen05 path needs one more warp to turn each CTA-local A/SFA
// completion into the paired-CTA ready event consumed by the leader MMA warp.
// Keep this physically absent from the 1-CTA launch so its occupancy is
// unchanged.
constexpr int kSyncTransformWarp = kSchedulerWarp + 1;
constexpr int kBaseWarpCount = kSchedulerWarp + 1;
constexpr bool kEnableTaskPipeline = true;
constexpr int kTaskPipelineStageCount = 2;
constexpr int kProducerThreadCount = kProducerWarpCount * kWarpSize;
constexpr int kPersistentTileCount = 2;
constexpr int kInterleaveBlock = 64;

__device__ inline float silu(float x) {
    constexpr float kLog2E = 1.4426950408889634f;
    float exp2_value;
    float reciprocal;
    const float exp2_arg = -kLog2E * x;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exp2_value) : "f"(exp2_arg));
    const float denominator = 1.0f + exp2_value;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(denominator));
    return x * reciprocal;
}

template <int kTileM, int kTileN, int kTileK, int kEpilogueTileN, int kClusterM,
          int kClusterN = 1>
struct KernelConfig {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert(kClusterN == 1, "NVFP4 FC1 only supports cluster_n=1");
    static_assert(kTileM == 4 * kWarpSize);
    // tcgen05.mma.kind::f16 supports N in steps of 8 for cta_group::1
    // and steps of 16 for cta_group::2, up to N=256. The instruction
    // descriptor encodes this dimension as N >> 3 in bits 17-22.
    static_assert(kTileN == 64 || kTileN == 128 || kTileN == 256);
    static_assert(kTileK == 64 || kTileK == 128 || kTileK == 256);
    // The physical tile contains interleaved gate/up columns.  A 64-column
    // physical epilogue tile therefore produces a 32-column SwiGLU tile,
    // matching the BF16 FC1 epilogue and its TMA store granularity.
    static_assert(kEpilogueTileN == 64 || kEpilogueTileN == 128);
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
    static constexpr int kOutputEpilogueTileN = kEpilogueTileN / 2;
    static constexpr int kEpilogueStageBytes =
        kTileM * kOutputEpilogueTileN * sizeof(nv_bfloat16);
    // The CuTE SMEM heuristic varies C depth by tactic (for example, it picks
    // four stages for N=256/cluster_m=1). FC1 measurements show that a fixed
    // three-stage C pipeline is better across the selected tactics: it keeps
    // two TMA stores in flight without needlessly deepening the store FIFO.
    // The macro remains an experiment override; zero selects this policy.
    static constexpr int kDefaultCStageCount = 3;
    static constexpr int kEpilogueStageCount = SM110A_NVFP4_FC1_NUM_C_STAGE == 0
        ? kDefaultCStageCount
        : SM110A_NVFP4_FC1_NUM_C_STAGE;
    static_assert(kEpilogueStageCount >= kDefaultCStageCount && kEpilogueStageCount <= 5,
                  "NVFP4 FC1 supports three through five C epilogue stages");
    static constexpr int kEpilogueSmemBytes = kEpilogueStageCount * kEpilogueStageBytes;
    // Dynamic and static shared memory share the same per-CTA budget.  Use
    // every mainloop stage that fits after reserving the epilogue and the
    // kernel's static/driver shared-memory allocations.
    static constexpr int kMaxOptInSmemBytes = 232448;
    static constexpr int kStaticSmemReserve = 3 * 1024;
    static constexpr int kSmemLimitedStageCount =
        (kMaxOptInSmemBytes - kStaticSmemReserve - kEpilogueSmemBytes) /
        kPerStageSmemBytes;
    // Each cta_group::2 M-pair in an N cluster owns one TMEM allocation. Cap
    // the pipeline depth so every N-pair fits in the 512-column TMEM pool
    // (cluster_n=2: 256 columns per pair, cluster_n=4: 128 per pair).
    static constexpr int kStageCount = kClusterN == 1 ? kSmemLimitedStageCount :
        (kClusterN == 2 ? (kSmemLimitedStageCount < 3 ? kSmemLimitedStageCount : 3) :
            (kSmemLimitedStageCount < 2 ? kSmemLimitedStageCount : 2));
    static_assert(kStageCount >= 1);
    static constexpr int kMainloopSmemBytes = kStageCount * kPerStageSmemBytes;
    static constexpr int kDynamicSmemBytes = kMainloopSmemBytes + kEpilogueSmemBytes;
    static constexpr int kScaleColsPerMma = 4 + 4 * kSFBScaleTileCount;
    // CuTE's 2-CTA N=256 path keeps two logical accumulator stages by
    // aliasing 128 physical TMEM columns. The stage stride is therefore 128,
    // matching CuTE's C-fragment layout rather than a 256-column double buffer.
    static constexpr int kAccumulatorOverlapCols =
        kTileN == 256 && kEpilogueTileN == 128 ? 128 : 0;
    static constexpr int kTmemStageCount = kClusterN == 1 ? 2 : 1;
    static constexpr int kAccumulatorStageStride =
        kTileN - kAccumulatorOverlapCols;
    static constexpr int kScaleTmemSlotCount =
        kTileN < 256 ? kStageCount * kMmaStepsPerStage : 1;
    static constexpr int kScaleTmemCols = kScaleTmemSlotCount * kScaleColsPerMma;
    static constexpr int kAccumulatorTmemCols =
        kTileN * kTmemStageCount - kAccumulatorOverlapCols;
    static constexpr int kTmemColumnsNeeded = 512;
    static_assert(kAccumulatorTmemCols + kScaleTmemCols <= kTmemColumnsNeeded);
    // tcgen05.alloc requires a power-of-two column count. In particular,
    // two N=192 accumulator stages need 384 columns but must allocate 512.
    static constexpr int kTmemAllocationColumns = kTmemColumnsNeeded <= 128 ? 128 :
        kTmemColumnsNeeded <= 256 ? 256 : 512;
    static_assert(kTmemColumnsNeeded <= kTmemAllocationColumns);
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

// `cluster_n` keeps each MMA CTA-group along M. Its N-pairs compute separate
// N tiles, while N-rank 0 distributes both M-ranks' gathered A/SFA stages
// through cluster shared memory.
template <int kClusterM, int kClusterN>
struct ClusterShape {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert(kClusterN == 1, "NVFP4 FC1 only supports cluster_n=1");
};

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

// Analysis-only split: these helpers deliberately keep identical PTX while
// giving SourceCounters a distinct line-info location for each pipeline edge.
__device__ inline void mbarrier_wait_mma_reuse(int mbar_addr, int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile(
        "{ .reg .pred done; WAIT_MMA_REUSE: "
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1, %2; "
        "@done bra.uni COMPLETE_MMA_REUSE; bra.uni WAIT_MMA_REUSE; "
        "COMPLETE_MMA_REUSE: }" ::"r"(mbar_addr),
        "r"(phase), "r"(kSuspendTicks) : "memory");
}

__device__ inline void mbarrier_wait_forward_a_ready(int mbar_addr, int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile(
        "{ .reg .pred done; WAIT_FORWARD_A: "
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1, %2; "
        "@done bra.uni COMPLETE_FORWARD_A; bra.uni WAIT_FORWARD_A; "
        "COMPLETE_FORWARD_A: }" ::"r"(mbar_addr),
        "r"(phase), "r"(kSuspendTicks) : "memory");
}

__device__ inline void mbarrier_wait_epilogue_slot(int mbar_addr, int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile(
        "{ .reg .pred done; WAIT_EPILOGUE_SLOT: "
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1, %2; "
        "@done bra.uni COMPLETE_EPILOGUE_SLOT; bra.uni WAIT_EPILOGUE_SLOT; "
        "COMPLETE_EPILOGUE_SLOT: }" ::"r"(mbar_addr),
        "r"(phase), "r"(kSuspendTicks) : "memory");
}

__device__ inline void mbarrier_wait_consumer_a_ready(int mbar_addr, int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile(
        "{ .reg .pred done; WAIT_CONSUMER_A: "
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1, %2; "
        "@done bra.uni COMPLETE_CONSUMER_A; bra.uni WAIT_CONSUMER_A; "
        "COMPLETE_CONSUMER_A: }" ::"r"(mbar_addr),
        "r"(phase), "r"(kSuspendTicks) : "memory");
}

__device__ inline void mbarrier_wait_consumer_cluster_a_ready(int mbar_addr, int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile(
        "{ .reg .pred done; WAIT_CLUSTER_A: "
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1, %2; "
        "@done bra.uni COMPLETE_CLUSTER_A; bra.uni WAIT_CLUSTER_A; "
        "COMPLETE_CLUSTER_A: }" ::"r"(mbar_addr),
        "r"(phase), "r"(kSuspendTicks) : "memory");
}

__device__ inline void mbarrier_wait_tma_ready(int mbar_addr, int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile(
        "{ .reg .pred done; WAIT_TMA: "
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1, %2; "
        "@done bra.uni COMPLETE_TMA; bra.uni WAIT_TMA; "
        "COMPLETE_TMA: }" ::"r"(mbar_addr),
        "r"(phase), "r"(kSuspendTicks) : "memory");
}

__device__ inline void mbarrier_wait_tma_issue(int mbar_addr, int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile(
        "{ .reg .pred done; WAIT_TMA_ISSUE: "
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1, %2; "
        "@done bra.uni COMPLETE_TMA_ISSUE; bra.uni WAIT_TMA_ISSUE; "
        "COMPLETE_TMA_ISSUE: }" ::"r"(mbar_addr),
        "r"(phase), "r"(kSuspendTicks) : "memory");
}

__device__ inline void mbarrier_arrive_tma_issue(int mbar_addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::
        "r"(mbar_addr) : "memory");
}

__device__ inline void mbarrier_wait_mainloop_output(int mbar_addr, int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile(
        "{ .reg .pred done; WAIT_MAINLOOP_OUTPUT: "
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1, %2; "
        "@done bra.uni COMPLETE_MAINLOOP_OUTPUT; bra.uni WAIT_MAINLOOP_OUTPUT; "
        "COMPLETE_MAINLOOP_OUTPUT: }" ::"r"(mbar_addr),
        "r"(phase), "r"(kSuspendTicks) : "memory");
}

__device__ __forceinline__ void cp_async_a_16b(
    int dst_smem,
    const void *src_gmem,
    bool valid) {
    const uint32_t ignore_src = valid ? 0u : 1u;
    asm volatile("{\n\t"
                 ".reg .pred p_ignore_src;\n\t"
                 "setp.ne.u32 p_ignore_src, %2, 0;\n\t"
                 "cp.async.cg.shared::cta.global [%0], [%1], 16, p_ignore_src;\n\t"
                 "}\n"
        :
        : "r"(dst_smem), "l"(src_gmem), "r"(ignore_src)
        : "memory");
}

__device__ __forceinline__ void cp_async_sfa_4b(
    int dst_smem,
    const void *src_gmem,
    bool valid) {
    const uint32_t ignore_src = valid ? 0u : 1u;
    asm volatile("{\n\t"
                 ".reg .pred p_ignore_src;\n\t"
                 "setp.ne.u32 p_ignore_src, %2, 0;\n\t"
                 "cp.async.ca.shared::cta.global [%0], [%1], 4, p_ignore_src;\n\t"
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
    const int *__restrict__ a_source_rows,
    int sfa_source_row,
    int valid_rows,
    int tile_k,
    int k,
    char *__restrict__ a_smem,
    char *__restrict__ sfa_smem,
    int a_mbar_addr,
    bool a_is_tma,
    bool sfa_is_tma) {
    constexpr int kVectorBytes = 16;
    constexpr int kVectorsPerRow = kTileK / 32;
    constexpr int kMmaSteps = kTileK / 64;
    constexpr int kRowsPerAWarp = kTileM / kAProducerWarpCount;
    constexpr int kRowsPerSfaWarp = kTileM / kSfaProducerWarpCount;
    constexpr int kRowsPerIssue = kWarpSize / kVectorsPerRow;
    constexpr int kRowGroupCount = kRowsPerAWarp / kRowsPerIssue;
    const int warp_id = threadIdx.x / kWarpSize;
    const int producer_rank = warp_id - kProducerWarpBase;
    const int lane = threadIdx.x & 31;

    if constexpr (!kSplitGatherWarps) {
        constexpr int kRowsPerIssue = kProducerThreadCount / kVectorsPerRow;
        const int producer_thread = producer_rank * kWarpSize + lane;
        const int vec = producer_thread % kVectorsPerRow;
        const int sfa_row = producer_thread;

        static_assert(kTileM == kProducerThreadCount);
        static_assert(kProducerThreadCount % kVectorsPerRow == 0);
        static_assert(kTileM % kRowsPerIssue == 0);

        // CuTE-style cooperative gather: the lanes in each V-lane group
        // jointly load one logical row, where V = TileK / 32.  For TileK=256,
        // eight lanes issue adjacent 16B copies and cover the row's full
        // 128B. A warp therefore touches four random source rows per issue,
        // rather than 32 random rows in the old one-lane-per-row mapping.
        if (!a_is_tma)
            #pragma unroll
            for (int issue = 0; issue < kVectorsPerRow; ++issue) {
                const int row = producer_thread / kVectorsPerRow + issue * kRowsPerIssue;
                const bool valid = row < valid_rows;
                constexpr int kSmemVectorsPerRow = 128 / kVectorBytes;
                const int physical_linear =
                    row * kSmemVectorsPerRow + vec;
                const int swizzled =
                    (physical_linear & ~7) +
                    ((physical_linear & 7) ^
                        ((physical_linear >> 3) & 7));
                const size_t offset =
                    static_cast<size_t>(a_source_rows[issue]) * (k / 2) +
                    static_cast<size_t>(tile_k) * (kTileK / 2) +
                    vec * kVectorBytes;
                const int dst_smem =
                    static_cast<int>(__cvta_generic_to_shared(
                        reinterpret_cast<uint4 *>(a_smem) + swizzled));
                cp_async_a_16b(dst_smem, a + offset, valid);
            }

        if (!sfa_is_tma)
            for (int mma_k = 0; mma_k < kMmaSteps; ++mma_k) {
                const int source_in_tile =
                    (sfa_source_row & 31) * 16 +
                    ((sfa_source_row >> 5) & 3) * 4;
                const size_t source_tile =
                    static_cast<size_t>(sfa_source_row >> 7);
                const size_t global_mma_k =
                    static_cast<size_t>(tile_k) * kMmaSteps + mma_k;
                const int dst_in_tile =
                    (sfa_row & 31) * 16 + ((sfa_row >> 5) & 3) * 4;
                const int dst_smem =
                    static_cast<int>(__cvta_generic_to_shared(
                        sfa_smem + mma_k * 512 + dst_in_tile));
                cp_async_sfa_4b(dst_smem,
                    sfa + (source_tile * (k / 64) + global_mma_k) * 512 +
                        source_in_tile,
                    sfa_row < valid_rows);
            }

        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        cp_async_a_mbarrier_arrive_noinc(a_mbar_addr);
        return;
    }

    const bool a_producer =
        !kSplitGatherWarps || producer_rank < kAProducerWarpCount;
    const bool sfa_producer =
        !kSplitGatherWarps || producer_rank >= kAProducerWarpCount;
    const int a_rank = producer_rank;
    const int sfa_rank = kSplitGatherWarps
        ? producer_rank - kAProducerWarpCount
        : producer_rank;
    const int a_row_begin = a_rank * kRowsPerAWarp;
    const int sfa_row_begin = sfa_rank * kRowsPerSfaWarp;

    static_assert(kTileM % kAProducerWarpCount == 0);
    static_assert(kTileM % kSfaProducerWarpCount == 0);
    static_assert(kRowsPerAWarp == kWarpSize);
    static_assert(kWarpSize % kVectorsPerRow == 0);
    static_assert(kRowsPerAWarp % kRowsPerIssue == 0);

    // Rebuild each gathered logical [128,4] scale tile directly in
    // SWIZZLE_32_4_4 order. Each thread moves all four scales for one row.
    if (sfa_producer && !sfa_is_tma && lane < kRowsPerSfaWarp)
        for (int mma_k = 0; mma_k < kMmaSteps; ++mma_k) {
            const int row = sfa_row_begin + lane;
            const bool valid = row < valid_rows;
            // Rank 1 is often empty for small expert buckets. Its output rows
            // are discarded, so do not spend issue slots on zero-fill copies.
            if (!kSplitGatherWarps || valid) {
                const int source_row = valid ? source_rows[row] : 0;
                const int source_in_tile = (source_row & 31) * 16 +
                                           ((source_row >> 5) & 3) * 4;
                const size_t source_tile =
                    static_cast<size_t>(source_row >> 7);
                const size_t global_mma_k =
                    static_cast<size_t>(tile_k) * kMmaSteps + mma_k;
                const int dst_in_tile =
                    (row & 31) * 16 + ((row >> 5) & 3) * 4;
                const int dst_smem =
                    static_cast<int>(__cvta_generic_to_shared(
                        sfa_smem + mma_k * 512 + dst_in_tile));
                cp_async_sfa_4b(dst_smem,
                    sfa + (source_tile * (k / 64) + global_mma_k) * 512 +
                        source_in_tile,
                    valid);
            }
        }

    // Match the 128B-swizzled TMA layout consumed by make_k_major_128b_desc.
    if (a_producer && !a_is_tma)
        for (int row_group = 0; row_group < kRowGroupCount; ++row_group) {
        const int row = a_row_begin + row_group * kRowsPerIssue +
                        lane / kVectorsPerRow;
        const int vec = lane % kVectorsPerRow;
        // The UMMA K-major SW128 atom reserves a full 128-byte row even when
        // this stage carries only K=64 or K=128. This is why kASmemBytes is
        // fixed at TileM*128 rather than the TMA transaction byte count.
        constexpr int kSmemVectorsPerRow = 128 / kVectorBytes;
        const int physical_linear = row * kSmemVectorsPerRow + vec;
        const int swizzled = (physical_linear & ~7) +
                             ((physical_linear & 7) ^ ((physical_linear >> 3) & 7));
        const bool valid = row < valid_rows;
        // Preserve the original c1 zero-fill behavior; c2 can skip invalid
        // rows because those accumulator rows are never stored.
        if (!kSplitGatherWarps || valid) {
            const int source_row = valid ? source_rows[row] : 0;
            const size_t offset = static_cast<size_t>(source_row) * (k / 2) +
                                  static_cast<size_t>(tile_k) * (kTileK / 2) +
                                  vec * kVectorBytes;
            const int dst_smem = static_cast<int>(__cvta_generic_to_shared(
                reinterpret_cast<uint4 *>(a_smem) + swizzled));
            cp_async_a_16b(dst_smem, a + offset, valid);
        }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    cp_async_a_mbarrier_arrive_noinc(a_mbar_addr);
}

// Map a shared-memory object to the address at which it is hosted by another
// CTA in the cluster.  The CTA-group::2 pipeline uses rank 0 as the owner for
// barriers that aggregate work completed independently by both CTAs.
__device__ inline int map_shared_to_cluster_cta(
    int local_smem_addr,
    int target_cta_rank) {
    int remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote_addr)
        : "r"(local_smem_addr), "r"(target_cta_rank));
    return remote_addr;
}

__device__ inline void mbarrier_arrive_on_cluster_cta(
    int local_mbar_addr,
    int target_cta_rank) {
    const int remote_mbar_addr =
        map_shared_to_cluster_cta(local_mbar_addr, target_cta_rank);
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 "
                 "_, [%0];"
        :
        : "r"(remote_mbar_addr)
        : "memory");
}

// Complete a bulk shared-memory transfer into one peer CTA and signal that
// peer's local ready barrier.
template <int kBytes>
__device__ inline void bulk_copy_stage_to_cluster_cta(
    int src_smem,
    int dst_smem,
    int local_mbar_addr,
    int target_cta_rank) {
    static_assert(kBytes % 16 == 0);
    const int remote_dst = map_shared_to_cluster_cta(dst_smem, target_cta_rank);
    const int remote_mbar = map_shared_to_cluster_cta(local_mbar_addr, target_cta_rank);
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
                 "_, [%0], %1;" :: "r"(remote_mbar), "r"(kBytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.shared::cta."
                 "mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
                 "r"(remote_dst), "r"(src_smem), "r"(kBytes), "r"(remote_mbar) : "memory");
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

template <int kClusterM>
__device__ inline void tma_load_4d_multicast(
    int dst_smem,
    const void *tensor_map,
    int x,
    int y,
    int z,
    int w,
    int mbar_addr,
    int16_t cta_mask) {
    static_assert(kClusterM == 2);
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global."
                 "mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2 "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::
            "r"(dst_smem), "l"(tensor_map), "r"(x), "r"(y), "r"(z), "r"(w),
            "r"(mbar_addr), "h"(cta_mask) : "memory");
}

__device__ inline void store_output_16b(int4 *ptr, int4 value) {
    ptr[0] = value;
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
        "r"(x), "r"(y), "r"(z), "r"(src_smem) : "memory");
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
    int mbar_addr,
    int cta_rank = 0) {
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
inline bool init_a_tensor_map(CUtensorMap *map, const uint8_t *ptr, uint64_t m, uint64_t k) {
    constexpr uint32_t rank = 4;
    uint64_t dims[rank] = {k, m, 1, 1};
    uint64_t strides[rank - 1] = {k / 2, m * k / 2, m * k / 2};
    uint32_t box[rank] = {kTileK, kTileM, 1, 1};
    uint32_t elem[rank] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, rank,
        const_cast<uint8_t *>(ptr), dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
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
    int kClusterN,
    bool kSwapAB = false,
    bool kDirectGmem = false>
__global__ __cluster_dims__(kClusterM, kClusterN, 1)
__launch_bounds__(kWarpSize * (kBaseWarpCount + (kClusterM > 1 ? 1 : 0))) void nvfp4_fc1_kernel(
    const uint8_t *a,
    const __grid_constant__ CUtensorMap a_tensor_map,
    const __grid_constant__ CUtensorMap b_tensor_map,
    const uint8_t *sfa,
    const __grid_constant__ CUtensorMap sfa_tensor_map,
    const __grid_constant__ CUtensorMap sfb_tensor_map,
    const __grid_constant__ CUtensorMap c_tensor_map,
    const int *source_rows,
    const ExpertTileTask *tasks,
    const int *task_count,
    int *next_cluster_tile,
    nv_bfloat16 *c,
    int n,
    int k) {
    using Config = KernelConfig<kTileM, kTileN, kTileK, kEpilogueTileN, kClusterM, kClusterN>;
    using Shape = ClusterShape<kClusterM, kClusterN>;
    (void)sizeof(Shape);
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
    constexpr int kOutputEpilogueTileN = Config::kOutputEpilogueTileN;
    constexpr int kEpilogueStoreTileN = kOutputEpilogueTileN < 64 ? kOutputEpilogueTileN : 64;
    constexpr int kEpilogueStoreTiles = kOutputEpilogueTileN / kEpilogueStoreTileN;
    constexpr int kChunksPerStoreTile = kEpilogueStoreTileN / 8;
    constexpr int kSwizzlePhaseMask = kEpilogueStoreTileN == 32 ? 3 : 7;
    constexpr int kStoreTileBytes = kTileM * kEpilogueStoreTileN * sizeof(nv_bfloat16);
    constexpr int kMmaStepsPerStage = kTileK / kMmaK;
    constexpr int kTmemStageCount = Config::kTmemStageCount;
    constexpr int kTmemAllocationColumns = Config::kTmemAllocationColumns;
    constexpr bool kUseTaskPipeline = kEnableTaskPipeline && kClusterM == 1 &&
        kClusterN == 1;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int m_rank = cta_rank % kClusterM;
    const int n_rank = cta_rank / kClusterM;
    const int cluster_tiles = *task_count;
    constexpr int kWorkerTileCount = kPersistentTileCount;

    __shared__ int tmem_base[1];
    __shared__ int next_work_smem[2];
    // Persistent c1 task pipeline.  `task_ready` is published by the
    // scheduler and observed independently by every role; `task_free` is
    // returned by the epilogue once no role needs the metadata slot.
    __shared__ int scheduled_cluster_bids[kTaskPipelineStageCount];
    __shared__ uint64_t task_ready_mbars[kTaskPipelineStageCount];
    __shared__ uint64_t task_free_mbars[kTaskPipelineStageCount];
    __shared__ int task_source_rows[kTileM];
    __shared__ uint64_t tma_mbars[kStageCount];
    __shared__ uint64_t tma_issue_mbars[kStageCount];
    __shared__ uint64_t a_mbars[kStageCount];
    // In the cluster-N path, SFA uses a second transaction barrier. A single
    // mbarrier cannot have two outstanding expect_tx transactions at once.
    __shared__ uint64_t sfa_mbars[kStageCount];
    // CTA 0 owns this barrier.  Each CTA arrives only after its local A/SFA
    // gather is complete, so the CTA-group::2 MMA cannot consume a partially
    // produced operand from CTA 1.
    __shared__ uint64_t a_cluster_mbars[kStageCount];
    __shared__ uint64_t mma_mbars[kStageCount];
    __shared__ uint64_t mainloop_mbars[kTmemStageCount];
    __shared__ uint64_t epilogue_mbars[kTmemStageCount];
    __shared__ uint64_t overlap_mbars[kTmemStageCount];

    extern __shared__ __align__(1024) char dynamic_smem[];
    const int mainloop_smem = static_cast<int>(__cvta_generic_to_shared(dynamic_smem));
    const int epilogue_smem = mainloop_smem + kMainloopSmemBytes;
    const int base_tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(tma_mbars));
    const int base_task_ready_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(task_ready_mbars));
    const int base_task_free_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(task_free_mbars));
    const int base_tma_issue_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(tma_issue_mbars));
    const int base_a_mbar_addr = static_cast<int>(__cvta_generic_to_shared(a_mbars));
    const int base_sfa_mbar_addr = static_cast<int>(__cvta_generic_to_shared(sfa_mbars));
    const int base_a_cluster_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(a_cluster_mbars));
    const int base_mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mma_mbars));
    const int base_mainloop_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mainloop_mbars));
    const int base_epilogue_mbar_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_mbars));
    const int base_overlap_mbar_addr = static_cast<int>(__cvta_generic_to_shared(overlap_mbars));

    if (tid == 0) {
        for (int stage = 0; stage < kStageCount; ++stage) {
            mbarrier_init(base_tma_mbar_addr + stage * sizeof(uint64_t), kClusterM);
            mbarrier_init(base_tma_issue_mbar_addr + stage * sizeof(uint64_t), 1);
            // In an N cluster, only rank 0 gathers A/SFA. Each peer receives
            // A via its local A barrier and SFA via the separate barrier below.
            const int a_arrival_count =
                kClusterN > 1 && n_rank != 0 ? 1 : kProducerThreadCount;
            mbarrier_init(base_a_mbar_addr + stage * sizeof(uint64_t), a_arrival_count);
            if constexpr (kClusterN > 1)
                mbarrier_init(base_sfa_mbar_addr + stage * sizeof(uint64_t), 1);
            if constexpr (kClusterM > 1) {
                if (m_rank == 0) {
                    mbarrier_init(
                        base_a_cluster_mbar_addr + stage * sizeof(uint64_t), kClusterM);
                }
            }
            mbarrier_init(base_mma_mbar_addr + stage * sizeof(uint64_t), 1);
        }
        for (int stage = 0; stage < kTmemStageCount; ++stage) {
            mbarrier_init(base_mainloop_mbar_addr + stage * sizeof(uint64_t), 1);
            mbarrier_init(base_epilogue_mbar_addr + stage * sizeof(uint64_t), 4 * kClusterM);
            if constexpr (Config::kAccumulatorOverlapCols != 0) {
                mbarrier_init(base_overlap_mbar_addr + stage * sizeof(uint64_t),
                    4 * kClusterM);
            }
        }
        if constexpr (kUseTaskPipeline) {
            for (int stage = 0; stage < kTaskPipelineStageCount; ++stage) {
                mbarrier_init(base_task_ready_mbar_addr + stage * sizeof(uint64_t), 1);
                mbarrier_init(base_task_free_mbar_addr + stage * sizeof(uint64_t),
                    kEpilogueWarpCount);
            }
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
        prefetch_tensor_map(&a_tensor_map);
        prefetch_tensor_map(&b_tensor_map);
        prefetch_tensor_map(&sfa_tensor_map);
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

        constexpr int kTmemLoadColumns = kSwapAB ? 8 : 16;
        for (int n_group = 0; n_group < kOutputEpilogueTileN / kTmemLoadColumns;
             ++n_group) {
            const int logical_col =
                n_block * kOutputEpilogueTileN + n_group * kTmemLoadColumns;
            const int logical_block = logical_col / kInterleaveBlock;
            const int logical_in_block = logical_col % kInterleaveBlock;
            const int gate_tmem_col =
                mainloop_stage * Config::kAccumulatorStageStride +
                logical_block * (2 * kInterleaveBlock) + logical_in_block;
            const int up_tmem_col = gate_tmem_col + kInterleaveBlock;
            // Kept for the unused swap-AB specialization below.
            const int tmem_col = gate_tmem_col;

            if constexpr (!kSwapAB) {
                float gate_values[kTmemLoadColumns];
                float up_values[kTmemLoadColumns];
                const int tmem_row = m_rank * kTileM + warp_id * kWarpSize;
                const int gate_load_addr = tmem_addr + (tmem_row << 16) + gate_tmem_col;
                const int up_load_addr = tmem_addr + (tmem_row << 16) + up_tmem_col;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                             "{%0, %1, %2, %3, %4, %5, %6, %7, "
                             "%8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=f"(gate_values[0]), "=f"(gate_values[1]),
                    "=f"(gate_values[2]), "=f"(gate_values[3]),
                    "=f"(gate_values[4]), "=f"(gate_values[5]),
                    "=f"(gate_values[6]), "=f"(gate_values[7]),
                    "=f"(gate_values[8]), "=f"(gate_values[9]),
                    "=f"(gate_values[10]), "=f"(gate_values[11]),
                    "=f"(gate_values[12]), "=f"(gate_values[13]),
                    "=f"(gate_values[14]), "=f"(gate_values[15])
                    : "r"(gate_load_addr));
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                             "{%0, %1, %2, %3, %4, %5, %6, %7, "
                             "%8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=f"(up_values[0]), "=f"(up_values[1]),
                    "=f"(up_values[2]), "=f"(up_values[3]),
                    "=f"(up_values[4]), "=f"(up_values[5]),
                    "=f"(up_values[6]), "=f"(up_values[7]),
                    "=f"(up_values[8]), "=f"(up_values[9]),
                    "=f"(up_values[10]), "=f"(up_values[11]),
                    "=f"(up_values[12]), "=f"(up_values[13]),
                    "=f"(up_values[14]), "=f"(up_values[15])
                    : "r"(up_load_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");

                nv_bfloat162 packed[kTmemLoadColumns / 2];
                for (int i = 0; i < kTmemLoadColumns / 2; ++i) {
                    packed[i] = __float22bfloat162_rn({
                        silu(gate_values[2 * i]) * up_values[2 * i],
                        silu(gate_values[2 * i + 1]) * up_values[2 * i + 1]});
                }

                if constexpr (kDirectGmem) {
                    // Direct TMEM → RMEM → GMEM: write packed values straight
                    // to global memory, bypassing shared memory.
                    if (tid < valid_rows) {
                        const int gmem_row = cta_row + tid;
                        const int gmem_col = tile_col / 2 + logical_col;
                        for (int half = 0; half < 2; ++half) {
                            store_output_16b(
                                reinterpret_cast<int4 *>(c + gmem_row * (n / 2) +
                                    gmem_col + half * 8),
                                reinterpret_cast<int4 *>(packed)[half]);
                        }
                    }
                } else {
                    // One x16 TMEM fragment produces two adjacent 16-byte
                    // chunks. Map each half independently through the TMA
                    // swizzle; treating registers 8-15 as a linear append to
                    // the first chunk would corrupt the shared-memory tile.
                    for (int half = 0; half < 2; ++half) {
                        const int chunk_group = n_group * 2 + half;
                        const int store_tile = chunk_group / kChunksPerStoreTile;
                        const int chunk_in_store = tid * kChunksPerStoreTile +
                            chunk_group % kChunksPerStoreTile;
                        const int swizzled_chunk =
                            (chunk_in_store & ~7) +
                            ((chunk_in_store & 7) ^
                                ((chunk_in_store >> 3) & kSwizzlePhaseMask));
                        reinterpret_cast<int4 *>(
                            stage_smem + store_tile * kStoreTileBytes)[swizzled_chunk] =
                            reinterpret_cast<int4 *>(packed)[half];
                    }
                }
            } else {
                // Match STSM's register fragment directly: each warp reads
                // lanes 0-15 and 16-31 as two .16x256b TMEM fragments.
                uint32_t values[8];
                const int row_base = m_rank * kTileM + warp_id * kWarpSize;
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
    int tma_stage = 0;
    int tma_issue_phase = 0;
    int consumer_stage = 0;
    int consumer_a_phase = 0;
    int consumer_tma_phase = 0;
    int consumer_mainloop_stage = 0;
    int consumer_epilogue_phase = 1;
    int consumer_overlap_phase = 0;
    bool consumer_has_previous_overlap = false;
    int epilogue_mainloop_stage = 0;
    int epilogue_mainloop_phase = 0;
    // C stages follow the TMA bulk-group FIFO, independently of the logical N
    // subtile order (which reverses on alternate overlapping TMEM stages).
    int epilogue_c_stage = 0;
    int cached_source_row_start = -1;
    // Each role tracks the task-ring parity independently.  It is intentionally
    // separate from the mainloop SMEM/TMEM stage state: a task spans all K
    // stages, while its metadata must remain stable until the epilogue ends.
    int task_pipe_stage = 0;
    int task_ready_phase = 0;
    int scheduler_free_phase = 1;

    while (true) {
        int first_cluster_bid;
        if constexpr (kUseTaskPipeline) {
            if (warp_id == kSchedulerWarp) {
                // The scheduler is the sole task-source producer.
                mbarrier_wait(base_task_free_mbar_addr +
                                  task_pipe_stage * sizeof(uint64_t),
                              scheduler_free_phase);
                if (elect_sync()) {
                    const int scheduled_bid = atomicAdd(next_cluster_tile, 1);

                    const int task_ready_mbar_addr = base_task_ready_mbar_addr +
                        task_pipe_stage * static_cast<int>(sizeof(uint64_t));
                    scheduled_cluster_bids[task_pipe_stage] = scheduled_bid;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::
                        "r"(task_ready_mbar_addr) : "memory");
                }
                __syncwarp();
                const bool scheduler_done =
                    scheduled_cluster_bids[task_pipe_stage] >= cluster_tiles;
                task_pipe_stage = (task_pipe_stage + 1) % kTaskPipelineStageCount;
                if (task_pipe_stage == 0) {
                    scheduler_free_phase ^= 1;
                }
                if (scheduler_done) {
                    break;
                }
                continue;
            }

            mbarrier_wait(base_task_ready_mbar_addr +
                              task_pipe_stage * sizeof(uint64_t),
                          task_ready_phase);
            first_cluster_bid = scheduled_cluster_bids[task_pipe_stage];
            if (first_cluster_bid >= cluster_tiles) {
                break;
            }
        } else {
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
        }
        if (first_cluster_bid >= cluster_tiles) {
            break;
        }
        const bool has_work = first_cluster_bid < cluster_tiles;
        constexpr int kTasksPerPipelineIteration = kUseTaskPipeline ? 1 : kWorkerTileCount;
        const int last_cluster_bid = has_work
                                         ? (first_cluster_bid + kTasksPerPipelineIteration < cluster_tiles
                                                   ? first_cluster_bid + kTasksPerPipelineIteration
                                                   : cluster_tiles)
                                         : first_cluster_bid;

        // The producer waits until MMA has released a stage before TMA overwrites it.
        if (warp_id >= kProducerWarpBase && warp_id < kMmaWarp) {
            const int producer_rank = warp_id - kProducerWarpBase;
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                const ExpertTileTask task = tasks[cluster_bid];
                const int tile_n = task.tile_n * kClusterN + n_rank;
                const int cta_row = task.row_start + task.tile_row + m_rank * kTileM;
                const int tile_col = tile_n * kTileN;
                const int cta_tile_row = task.tile_row + m_rank * kTileM;
                const int remaining_rows = task.row_count - cta_tile_row;
                const int valid_rows = remaining_rows < 0 ? 0 :
                    (remaining_rows < kTileM ? remaining_rows : kTileM);

                if ((kClusterN == 1 || n_rank == 0) && cta_row != cached_source_row_start) {
                    const int lane = tid & (kWarpSize - 1);
                    const int row = producer_rank * kWarpSize + lane;
                    if (producer_rank < kAProducerWarpCount)
                        task_source_rows[row] =
                            row < valid_rows ? __ldg(source_rows + cta_row + row) : 0;
                    __syncwarp();
                    cached_source_row_start = cta_row;
                    asm volatile("bar.sync 2, %0;" :: "n"(kProducerThreadCount) : "memory");
                }

                constexpr int kAVectorsPerRow = kTileK / 32;
                constexpr int kARowsPerIssue = kProducerThreadCount / kAVectorsPerRow;
                const int producer_thread = producer_rank * kWarpSize + (tid & (kWarpSize - 1));
                int a_source_row_cache[kAVectorsPerRow];
                int sfa_source_row_cache = 0;
                if (kClusterN == 1 || n_rank == 0) {
                    #pragma unroll
                    for (int issue = 0; issue < kAVectorsPerRow; ++issue) {
                        const int row = producer_thread / kAVectorsPerRow +
                            issue * kARowsPerIssue;
                        a_source_row_cache[issue] = row < valid_rows ? task_source_rows[row] : 0;
                    }
                    sfa_source_row_cache =
                        producer_thread < valid_rows ? task_source_rows[producer_thread] : 0;
                }
                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    const int mma_mbar_addr =
                        base_mma_mbar_addr + producer_stage * sizeof(uint64_t);
                    const int a_mbar_addr =
                        base_a_mbar_addr + producer_stage * sizeof(uint64_t);

                    mbarrier_wait_mma_reuse(mma_mbar_addr, producer_mma_phase);
                    if (producer_rank == 0 && elect_sync()) {
                        mbarrier_arrive_tma_issue(
                            base_tma_issue_mbar_addr + producer_stage * sizeof(uint64_t));
                    }
                    if constexpr (kClusterN == 1) {
                        load_gather_a_and_sfa<kTileM, kTileK>(a, sfa, task_source_rows,
                            a_source_row_cache, sfa_source_row_cache,
                            valid_rows, tile_k, k,
                            dynamic_smem + producer_stage * kStageSmemBytes,
                            dynamic_smem + producer_stage * kStageSmemBytes +
                                kASmemBytes + kBSmemBytes,
                            a_mbar_addr, false, false);
                    } else if (n_rank == 0) {
                        load_gather_a_and_sfa<kTileM, kTileK>(a, sfa, task_source_rows,
                            a_source_row_cache, sfa_source_row_cache,
                            valid_rows, tile_k, k,
                            dynamic_smem + producer_stage * kStageSmemBytes,
                            dynamic_smem + producer_stage * kStageSmemBytes +
                                kASmemBytes + kBSmemBytes,
                            a_mbar_addr, false, false);
                    }

                    producer_stage = (producer_stage + 1) % kStageCount;
                    if (producer_stage == 0) {
                        producer_mma_phase ^= 1;
                    }
                }
            }
        }

        // The gather producer owns stage acquisition.  Once it has observed
        // the MMA-reuse barrier it hands the same stage to this dedicated
        // B/SFB TMA warp through tma_issue_mbars.
        if (warp_id == kTmaWarp) {
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                const ExpertTileTask task = tasks[cluster_bid];
                const int tile_n = task.tile_n * kClusterN + n_rank;
                const int tile_col = tile_n * kTileN;
                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    const int issue_mbar_addr =
                        base_tma_issue_mbar_addr + tma_stage * sizeof(uint64_t);
                    const int tma_mbar_addr =
                        base_tma_mbar_addr + tma_stage * sizeof(uint64_t);
                    const int a_smem = mainloop_smem + tma_stage * kStageSmemBytes;
                    const int b_smem = a_smem + kASmemBytes;
                    const int sfa_smem = b_smem + kBSmemBytes;
                    const int sfb_smem = sfa_smem + kSFASmemBytes;

                    mbarrier_wait_tma_issue(issue_mbar_addr, tma_issue_phase);
                    if (elect_sync()) {
                        const int tma_transaction_mbar_addr = tma_mbar_addr & 0xFEFFFFFF;
                        const int off_k = tile_k * kTileK;
                        const int off_k64 = tile_k * Config::kMmaStepsPerStage;
                        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
                                     "_, [%0], %1;" ::"r"(tma_transaction_mbar_addr),
                            "r"(Config::kBTmaBytes + kSFBSmemBytes) : "memory");
                        tma_load_4d<kClusterM>(b_smem, &b_tensor_map, off_k,
                            tile_col + m_rank * (kTileN / kClusterM), 0, task.expert,
                            tma_transaction_mbar_addr);
                        for (int scale_tile = 0; scale_tile < kSFBScaleTileCount; ++scale_tile) {
                            const int sfb_tile_smem =
                                sfb_smem + scale_tile * Config::kMmaStepsPerStage * 512;
                            const int sfb_tile_index =
                                task.expert * (n / 128) + tile_col / 128 + scale_tile;
                            if constexpr (kClusterM == 1) {
                                tma_load_4d<1>(sfb_tile_smem, &sfb_tensor_map, 0, 0, off_k64,
                                    sfb_tile_index, tma_transaction_mbar_addr);
                            } else if (m_rank == 0) {
                                tma_load_4d_multicast<2>(sfb_tile_smem, &sfb_tensor_map,
                                    0, 0, off_k64, sfb_tile_index, tma_transaction_mbar_addr,
                                    static_cast<int16_t>(3 << (n_rank * kClusterM)));
                            }
                        }
                    }

                    tma_stage = (tma_stage + 1) % kStageCount;
                    if (tma_stage == 0) {
                        tma_issue_phase ^= 1;
                    }
                }
            }
        }

        // The dedicated sync-transform warp is the 2-CTA equivalent of the
        // CuTe baseline's warp 11.  It does not move data: each CTA waits for
        // its local LDGSTS A/SFA stage, then both warps arrive on CTA 0's
        // aggregate barrier.  The leader MMA warp can consequently wait once
        // for a whole cta_group::2 A operand instead of participating in the
        // local completion protocol itself.
        if constexpr (kClusterM > 1) {
            if (warp_id == kSyncTransformWarp && elect_sync()) {
                for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                    ++cluster_bid) {
                    for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                        mbarrier_wait_forward_a_ready(
                            base_a_mbar_addr + consumer_stage * sizeof(uint64_t),
                            consumer_a_phase);
                        if constexpr (kClusterN > 1) {
                            if (n_rank != 0) {
                                mbarrier_wait_forward_a_ready(
                                    base_sfa_mbar_addr + consumer_stage * sizeof(uint64_t),
                                    consumer_a_phase);
                            }
                        }
                        mbarrier_arrive_on_cluster_cta(
                            base_a_cluster_mbar_addr + consumer_stage * sizeof(uint64_t),
                            n_rank * kClusterM);
                        consumer_stage = (consumer_stage + 1) % kStageCount;
                        if (consumer_stage == 0) {
                            consumer_a_phase ^= 1;
                            consumer_tma_phase ^= 1;
                        }
                    }
                }
            }
        }

        // The consumer advances independently and releases each stage after MMA has
        // finished reading it.
        if (m_rank == 0 && warp_id == kMmaWarp && elect_sync()) {
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                mbarrier_wait_epilogue_slot(
                    base_epilogue_mbar_addr + consumer_mainloop_stage * sizeof(uint64_t),
                    consumer_epilogue_phase);
                if constexpr (Config::kAccumulatorOverlapCols != 0) {
                    if (consumer_has_previous_overlap) {
                        const int previous_stage =
                            (consumer_mainloop_stage + kTmemStageCount - 1) %
                            kTmemStageCount;
                        mbarrier_wait_epilogue_slot(
                            base_overlap_mbar_addr + previous_stage * sizeof(uint64_t),
                            consumer_overlap_phase);
                        if (previous_stage == kTmemStageCount - 1) {
                            consumer_overlap_phase ^= 1;
                        }
                    }
                    consumer_has_previous_overlap = true;
                }

                const int stage_tmem_addr =
                    tmem_addr + consumer_mainloop_stage * Config::kAccumulatorStageStride;
                const int tile_col =
                    (tasks[cluster_bid].tile_n * kClusterN + n_rank) * kTileN;
                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    const int tma_mbar_addr =
                        base_tma_mbar_addr + consumer_stage * sizeof(uint64_t);
                    const int a_mbar_addr =
                        base_a_mbar_addr + consumer_stage * sizeof(uint64_t);
                    const int sfa_mbar_addr =
                        base_sfa_mbar_addr + consumer_stage * sizeof(uint64_t);
                    const int mma_mbar_addr =
                        base_mma_mbar_addr + consumer_stage * sizeof(uint64_t);

                    if constexpr (kClusterM == 1) {
                        mbarrier_wait_consumer_a_ready(a_mbar_addr, consumer_a_phase);
                    }
                    if constexpr (kClusterN > 1) {
                        // N-rank 0 owns the gathered TileM. Once its local
                        // gather is visible, fan out A and SFA to every N
                        // rank. Peers wait on separate A and SFA transaction
                        // barriers before their pair enters MMA.
                        if (n_rank == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            const int a_smem =
                                mainloop_smem + consumer_stage * kStageSmemBytes;
                            const int sfa_smem = a_smem + kASmemBytes + kBSmemBytes;
                            for (int target_n_rank = 1; target_n_rank < kClusterN;
                                 ++target_n_rank) {
                                const int target_rank = target_n_rank * kClusterM + m_rank;
                                bulk_copy_stage_to_cluster_cta<kASmemBytes>(a_smem, a_smem,
                                    a_mbar_addr, target_rank);
                                bulk_copy_stage_to_cluster_cta<kSFASmemBytes>(sfa_smem, sfa_smem,
                                    sfa_mbar_addr, target_rank);
                            }
                        } else {
                            mbarrier_wait_consumer_a_ready(sfa_mbar_addr, consumer_a_phase);
                        }
                    }
                    if constexpr (kClusterM > 1) {
                        mbarrier_wait_consumer_cluster_a_ready(
                            base_a_cluster_mbar_addr + consumer_stage * sizeof(uint64_t),
                            consumer_a_phase);
                    }
                    mbarrier_wait_tma_ready(tma_mbar_addr, consumer_tma_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    {
                        const int a_smem = mainloop_smem + consumer_stage * kStageSmemBytes;
                        const int b_smem = a_smem + kASmemBytes;
                        const int sfa_smem = b_smem + kBSmemBytes;
                        const int sfb_smem = sfa_smem + kSFASmemBytes;
                        for (int mma_k = 0; mma_k < kMmaStepsPerStage; ++mma_k) {
                            const int scale_slot = kTileN < 256
                                ? consumer_stage * kMmaStepsPerStage + mma_k
                                : 0;
                            const int scale_addr = tmem_addr + Config::kAccumulatorTmemCols +
                                scale_slot * Config::kScaleColsPerMma;
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
            constexpr int kEpilogueBlockCount = kTileN / kEpilogueTileN;

            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                mbarrier_wait_mainloop_output(
                    base_mainloop_mbar_addr + epilogue_mainloop_stage * sizeof(uint64_t),
                    epilogue_mainloop_phase);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

                const ExpertTileTask task = tasks[cluster_bid];
                const int tile_n = task.tile_n * kClusterN + n_rank;
                const int cta_tile_row = task.tile_row + m_rank * kTileM;
                const int cta_row = task.row_start + cta_tile_row;
                const int remaining_rows = task.row_count - cta_tile_row;
                const int valid_rows =
                    remaining_rows < 0 ? 0 : (remaining_rows < kTileM ? remaining_rows : kTileM);
                const int tile_col = tile_n * (kSwapAB ? kClusterTileM : kTileN);

                auto release_tmem = [&]() {
                    if (elect_sync()) {
                            const int pair_root_rank = cta_rank & ~1;
                            const int mbar_addr = map_shared_to_cluster_cta(
                                base_epilogue_mbar_addr +
                                    epilogue_mainloop_stage * sizeof(uint64_t),
                                pair_root_rank);
                            asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 "
                                         "_, [%0];" ::"r"(mbar_addr)
                                : "memory");
                    }
                };

                auto release_overlap = [&]() {
                    if constexpr (Config::kAccumulatorOverlapCols != 0) {
                        if (elect_sync()) {
                            const int pair_root_rank = cta_rank & ~1;
                            const int mbar_addr = map_shared_to_cluster_cta(
                                base_overlap_mbar_addr +
                                    epilogue_mainloop_stage * sizeof(uint64_t),
                                pair_root_rank);
                            asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 "
                                         "_, [%0];" :: "r"(mbar_addr) : "memory");
                        }
                    }
                };

                // A 128-column logical-stage shift aliases stage 0's upper
                // half with stage 1's lower half. Drain that half first:
                // stage 0 consumes block 1, stage 1 consumes block 0.
                const int first_n_block = Config::kAccumulatorOverlapCols != 0 &&
                        epilogue_mainloop_stage == 0 ? 1 : 0;
                // The C slot follows a monotonically advancing bulk-group
                // ring, not n_block: overlap reverses the latter on alternate
                // accumulator stages.
                tmem_to_smem(epilogue_mainloop_stage, first_n_block,
                    epilogue_c_stage,
                    cta_row, tile_col, valid_rows);
                if constexpr (!kDirectGmem || kSwapAB) {
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                }
                release_overlap();
                if constexpr (kEpilogueBlockCount == 1) {
                    release_tmem();
                }

                for (int n_block_index = 0; n_block_index < kEpilogueBlockCount;
                     ++n_block_index) {
                    const int n_block = Config::kAccumulatorOverlapCols != 0
                        ? (first_n_block ^ n_block_index) : n_block_index;
                    const int smem_stage = epilogue_c_stage;
                    const int stage_smem = epilogue_smem + smem_stage * kEpilogueStageBytes;
                    if constexpr (!kSwapAB) {
                        // Direct gmem: data already written by tmem_to_smem.
                        // Only the SMEM→TMA path needs TMA store / fallback.
                        if constexpr (!kDirectGmem) {
                            const int block_col = tile_col / 2 + n_block * kOutputEpilogueTileN;
                            if (valid_rows == kTileM) {
                                if (tid == 0) {
                                    for (int store_tile = 0; store_tile < kEpilogueStoreTiles;
                                        ++store_tile) {
                                        tma_store_3d(
                                            stage_smem + store_tile * kStoreTileBytes,
                                            &c_tensor_map,
                                            block_col + store_tile * kEpilogueStoreTileN,
                                            cta_row,
                                            0);
                                    }
                                }
                            } else if (tid < valid_rows) {
                                char *stage_smem_ptr = dynamic_smem + kMainloopSmemBytes +
                                                       smem_stage * kEpilogueStageBytes;
                                for (int n_group = 0; n_group < kOutputEpilogueTileN / 8; ++n_group) {
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
                                        (cta_row + tid) * (n / 2) + block_col + n_group * 8;
                                    store_output_16b(
                                        reinterpret_cast<int4 *>(c + output_offset), value);
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
                        const int block_col = tile_col + m_rank * kTileM;
                        if (valid_block_rows == kEpilogueTileN) {
                            if (tid == 0) {
                                constexpr int kSwappedStoreColumns = 64;
                                constexpr int kSwappedStoreBytes =
                                    kEpilogueTileN * kSwappedStoreColumns * sizeof(nv_bfloat16);
                                for (int atom = 0; atom < 2; ++atom) {
                                    tma_store_3d(
                                        stage_smem + atom * kSwappedStoreBytes,
                                        &c_tensor_map,
                                        0,
                                        block_col / kSwappedStoreColumns + atom,
                                        block_row);
                                }
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

                    if constexpr (!kDirectGmem || kSwapAB) {
                        if (tid == 0) {
                            // Commit one group for every logical C stage.  A
                            // boundary tile that used scalar stores commits an
                            // empty group, preserving the FIFO-to-slot mapping.
                            tma_store_commit_group();
                            asm volatile("cp.async.bulk.wait_group.read %0;"
                                :
                                : "n"(kEpilogueStageCount - 1)
                                : "memory");
                        }
                        // Warp 0 has made the oldest C slot safe.  Do not let
                        // the other epilogue warps begin the next TMEM drain
                        // until they observe that completion.
                        asm volatile("bar.sync 1, 128;" ::: "memory");
                    }

                    epilogue_c_stage = (epilogue_c_stage + 1) % kEpilogueStageCount;

                    if (n_block_index + 1 < kEpilogueBlockCount) {
                        const int next_n_block = Config::kAccumulatorOverlapCols != 0
                            ? (first_n_block ^ (n_block_index + 1)) : n_block_index + 1;
                        tmem_to_smem(epilogue_mainloop_stage,
                            next_n_block,
                            epilogue_c_stage,
                            cta_row,
                            tile_col,
                            valid_rows);
                        if constexpr (!kDirectGmem || kSwapAB) {
                            asm volatile("bar.sync 1, 128;" ::: "memory");
                        }
                        if (n_block_index + 2 == kEpilogueBlockCount) {
                            release_tmem();
                        }
                    }
                }

                epilogue_mainloop_stage = (epilogue_mainloop_stage + 1) % kTmemStageCount;
                if (epilogue_mainloop_stage == 0) {
                    epilogue_mainloop_phase ^= 1;
                }
            }
        }

        if constexpr (kUseTaskPipeline) {
            // The epilogue is the final user of a task record.  Returning the
            // ring slot here lets the scheduler acquire the next task without
            // waiting for the other roles to reconverge at a CTA barrier.  All
            // four epilogue warps must arrive: direct-gmem has no warp-group
            // barrier, so warp 0 alone may otherwise lap a slower epilogue warp
            // and make it miss a task_ready phase transition.
            if (warp_id < kEpilogueWarpCount && elect_sync()) {
                const int task_free_mbar_addr = base_task_free_mbar_addr +
                    task_pipe_stage * static_cast<int>(sizeof(uint64_t));
                asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::
                    "r"(task_free_mbar_addr) : "memory");
            }
            task_pipe_stage = (task_pipe_stage + 1) % kTaskPipelineStageCount;
            if (task_pipe_stage == 0) {
                task_ready_phase ^= 1;
            }
        }

    }

    // No later C-slot acquisition exists to retire the final bulk group(s).
    // Drain them before the persistent kernel reaches its shared-memory/TMEM
    // teardown; this is the only full C-store wait in the steady-state path.
    if constexpr (!kDirectGmem || kSwapAB) {
        if (warp_id < kEpilogueWarpCount) {
            if (tid == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            }
            asm volatile("bar.sync 1, 128;" ::: "memory");
        }
    }

    if constexpr (!kUseTaskPipeline) {
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
    } else {
        // Scheduler reaches the tail before the data-path warps after it
        // publishes the sentinel.  Re-converge locally before TMEM dealloc.
        __syncthreads();
    }
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

template <int kTileM, int kTileN, int kTileK, int kEpilogueTileN, int kClusterM,
          int kClusterN, bool kSwapAB>
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
    int max_tasks, int max_active_clusters,
    cudaStream_t stream) {
    using Config = KernelConfig<kTileM, kTileN, kTileK, kEpilogueTileN, kClusterM, kClusterN>;
    using Shape = ClusterShape<kClusterM, kClusterN>;
    (void)sizeof(Shape);
    constexpr int kClusterTileM = Config::kClusterTileM;
    constexpr int kBlockThreads =
        kWarpSize * (kBaseWarpCount + (kClusterM > 1 ? 1 : 0));
    constexpr int kDynamicSmemBytes = Config::kDynamicSmemBytes;
    static_assert(!kSwapAB, "NVFP4 FC1 does not support swap_ab");
    if (tokens % 128 || n % (kTileN * kClusterN) != 0 || n % 128 || k % kTileK != 0) {
        return;
    }

    CUtensorMap a_tensor_map;
    CUtensorMap b_tensor_map;
    CUtensorMap sfa_tensor_map;
    CUtensorMap sfb_tensor_map;
    CUtensorMap c_tensor_map;
    bool tensor_maps_ok;
    tensor_maps_ok =
        init_a_tensor_map<kTileM, kTileK>(&a_tensor_map, d_a, tokens, k) &&
        init_b_tensor_map<kTileN, kTileK, kClusterM>(&b_tensor_map, d_b, experts, n, k) &&
        init_scale_tensor_map(&sfa_tensor_map, d_sfa, tokens / 128,
            k / 64, kTileK / 64) &&
        init_scale_tensor_map(&sfb_tensor_map, d_sfb, experts * (n / 128),
            k / 64, kTileK / 64) &&
        init_c_tensor_map<kTileM, Config::kOutputEpilogueTileN>(&c_tensor_map, d_c, m, n / 2);
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

    const int persistent_clusters =
        (max_tasks + kPersistentTileCount - 1) / kPersistentTileCount;
    auto kernel = nvfp4_fc1_kernel<kTileM,
            kTileN,
            kTileK,
            kEpilogueTileN,
            kClusterM,
            kClusterN, kSwapAB,
            (SM110A_NVFP4_FC1_DIRECT_GMEM != 0)>;
    if constexpr (kDynamicSmemBytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kDynamicSmemBytes);
    }
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = dim3(kClusterM * persistent_clusters, kClusterN, 1);
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
            constexpr int kClusterCtas = kClusterM * kClusterN;
            resident_clusters = sm_count / kClusterCtas > 0 ? sm_count / kClusterCtas : 1;
    }
    const int cluster_limit = max_active_clusters > 0 ? max_active_clusters : resident_clusters;
    const int launched_clusters = cluster_limit < persistent_clusters
        ? cluster_limit : persistent_clusters;
    kernel<<<dim3(kClusterM * launched_clusters, kClusterN, 1), kBlockThreads,
        kDynamicSmemBytes, stream>>>(
        d_a, a_tensor_map, b_tensor_map, d_sfa, sfa_tensor_map, sfb_tensor_map, c_tensor_map, d_source_rows, tasks, task_count,
        next_cluster_tile, d_c, n, k);
}

void launch_nvfp4_fc1_entry(
    const uint8_t *a, const uint8_t *b, const uint8_t *sfa, const uint8_t *sfb,
    const int32_t *source_rows, const int32_t *tasks, int32_t *state, nv_bfloat16 *c,
    int tokens, int m, int experts, int n, int k, int max_tasks, int max_active_clusters, cudaStream_t stream) {
    if (!a || !b || !sfa || !sfb || !source_rows || !tasks || !state || !c ||
        tokens <= 0 || m <= 0 || experts <= 0 || n <= 0 || k <= 0 || max_tasks <= 0)
        return;
    launch_nvfp4_fc1_impl<SM110A_NVFP4_FC1_TILE_M,
        SM110A_NVFP4_FC1_TILE_N, SM110A_NVFP4_FC1_TILE_K,
        SM110A_NVFP4_FC1_EPILOGUE_TILE_N, SM110A_NVFP4_FC1_CLUSTER_M,
        SM110A_NVFP4_FC1_CLUSTER_N,
        false>(a, b, sfa, sfb, source_rows, tasks, state, c, tokens, m, experts,
            n, k, max_tasks, max_active_clusters, stream);
}

} // namespace

void launch_nvfp4_fc1_main(
    const uint8_t *a, const uint8_t *b, const uint8_t *sfa, const uint8_t *sfb,
    const int32_t *source_rows, const int32_t *tasks, int32_t *state, nv_bfloat16 *c,
    int tokens, int m, int experts, int n, int k, int max_tasks, int max_active_clusters, cudaStream_t stream) {
    launch_nvfp4_fc1_entry(
        a, b, sfa, sfb, source_rows, tasks, state, c, tokens, m, experts, n, k,
        max_tasks, max_active_clusters, stream);
}

extern "C" void sm110a_nvfp4_fc1_launch(
    const uint8_t *a, const uint8_t *b, const uint8_t *sfa, const uint8_t *sfb,
    const int32_t *source_rows, const int32_t *tasks, int32_t *state, nv_bfloat16 *c,
    int tokens, int m, int experts, int n, int k, int max_tasks, int max_active_clusters, cudaStream_t stream) {
    launch_nvfp4_fc1_main(
        a, b, sfa, sfb, source_rows, tasks, state, c, tokens, m, experts, n, k,
        max_tasks, max_active_clusters, stream);
}
