#include "pdl.h"
#ifndef SM110A_MXFP4_FC1_PDL_INPUT
#define SM110A_MXFP4_FC1_PDL_INPUT 0
#endif
#ifndef SM110A_MXFP4_FC1_PDL
#define SM110A_MXFP4_FC1_PDL 0
#endif
#ifndef SM110A_MXFP4_FC1_PDL_TRIGGER
#define SM110A_MXFP4_FC1_PDL_TRIGGER 0
#endif

/*
 * Independent SM110A MXFP4 (W4A8) routed FC1 with fused token permutation.
 *
 * Port of nvfp4_fc1_kernel.cu; only the quantization formats change:
 *   - A is FP8 E4M3 [tokens, K] (dense bytes) with blocked UE8M0 scales.
 *   - B stays the interleaved packed-E2M1 [experts, 2N, K/2] weight with
 *     blocked UE8M0 scales; its TMA uses the b4x16_p64 (ALIGN16B) expansion
 *     required by kind::mxf8f6f4.
 *   - Scales are UE8M0, one per 32 contiguous K elements, stored as 512-byte
 *     SWIZZLE_32_4_4 tiles per (128 rows, K=32 step) with the live byte in
 *     column 0 (the fp8mxfp4_gemm layout). A per-128-element activation scale
 *     is legal: the producer replicates it across the four K=32 tiles.
 *
 * The MMA is tcgen05.mma.kind::mxf8f6f4.block_scale.scale_vec::1X, so the
 * mainloop advances K in 32-element steps and TileK is capped at 128 (the
 * FP8 operand must fit a 128B-swizzle UMMA atom). TileN is fixed at 256 so
 * one accumulator stage holds exactly one 128-column logical output block,
 * the E8M0 scale granularity of the quantized FC2 input this kernel emits.
 *
 * Epilogue: block-interleaved SwiGLU with the DeepSeek swiglu_limit clamps
 * (gate clamped from above, up from both sides), quantized to FP8 E4M3 with
 * one shared E8M0 scale per 128 logical output elements. The scale requires
 * amax over the whole block, so each block is visited twice per row: a
 * TMEM-only pass accumulates amax, then a second pass re-reads, applies the
 * scale, and stores FP8 bytes (TMA-swizzled slot or direct GMEM). SFC is
 * written in the blocked layout consumed by the MXFP4 FC2 kernel, each
 * 128-element scale replicated across its four K=32 scale tiles.
 */

#include <cstdint>
#include <cstdlib>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "mxfp4_moe_fc1.h"
#include "mxfp4_fc1_pipeline_delay.cuh"
#include "pipeline_delay.h"

#ifndef SM110A_MXFP4_FC1_TILE_M
#define SM110A_MXFP4_FC1_TILE_M 128
#endif
#ifndef SM110A_MXFP4_FC1_TILE_N
#define SM110A_MXFP4_FC1_TILE_N 256
#endif
#ifndef SM110A_MXFP4_FC1_TILE_K
#define SM110A_MXFP4_FC1_TILE_K 128
#endif
#ifndef SM110A_MXFP4_FC1_EPILOGUE_TILE_N
#define SM110A_MXFP4_FC1_EPILOGUE_TILE_N 128
#endif
#ifndef SM110A_MXFP4_FC1_CLUSTER_M
#define SM110A_MXFP4_FC1_CLUSTER_M 1
#endif
#ifndef SM110A_MXFP4_FC1_CLUSTER_N
#define SM110A_MXFP4_FC1_CLUSTER_N 1
#endif
#ifndef SM110A_MXFP4_FC1_DIRECT_GMEM
#define SM110A_MXFP4_FC1_DIRECT_GMEM 0
#endif
#ifndef SM110A_MXFP4_FC1_WEIGHT_IN_FC1
#define SM110A_MXFP4_FC1_WEIGHT_IN_FC1 0
#endif
#ifndef SM110A_MXFP4_FC1_ACTIVATION_BF16_ROUNDING
#define SM110A_MXFP4_FC1_ACTIVATION_BF16_ROUNDING 0
#endif
#include "mxfp4_weight_scales.cuh"
#ifndef SM110A_MXFP4_FC1_COMPACT_WEIGHT_SCALES
#define SM110A_MXFP4_FC1_COMPACT_WEIGHT_SCALES 0
#endif

namespace sm110a::mxfp4_fc1 {
constexpr bool kCompactWeightScales = SM110A_MXFP4_FC1_COMPACT_WEIGHT_SCALES;


constexpr int kMmaK = 32;
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
// E8M0 output-scale granularity of the quantized FC2 input (logical columns).
constexpr int kScaleBlockN = 128;
constexpr float kE4M3Max = 448.0f;

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

// DeepSeek-style limited SwiGLU: gate is clamped from above, up from both
// sides (models/deepseek_v41_flash Expert.forward). limit <= 0 disables.
__device__ __forceinline__ float swiglu_activate(float gate, float up, float limit) {
    if (limit > 0.0f) {
        gate = fminf(gate, limit);
        up = fminf(fmaxf(up, -limit), limit);
    }
    return silu(gate) * up;
}

template <int kTileM, int kTileN, int kTileK, int kEpilogueTileN, int kClusterM,
          int kClusterN = 1>
struct KernelConfig {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert(kClusterN == 1, "MXFP4 FC1 only supports cluster_n=1");
    static_assert(kTileM == 4 * kWarpSize);
    static_assert(kTileN == 256,
                  "MXFP4 FC1 fixes TileN=256 so one accumulator stage covers "
                  "exactly one 128-column E8M0 output-scale block");
    static_assert(kEpilogueTileN == 128,
                  "MXFP4 FC1 drains two 64-column logical blocks per stage");
    // K is fixed at 128: FP8 activations ride a 128B-swizzle UMMA atom (the
    // NVFP4 kernel allowed 256 for half-width FP4 rows), and the b4x16_p64
    // TMA expansion of the packed E2M1 weights requires a 128-element box.
    static_assert(kTileK == 128);
    static_assert(kTileN % kEpilogueTileN == 0);
    static_assert(kTileN % kClusterM == 0);
    static_assert(kTileK % 32 == 0);
    static_assert(kTileK % kMmaK == 0);
    static constexpr int kClusterTileM = kClusterM * kTileM;
    static constexpr int kATmaBytes = kTileM * kTileK;
    static constexpr int kBTmaBytes = (kTileN / kClusterM) * kTileK / 2;
    static constexpr int kASmemBytes = kTileM * 128;
    // B is expanded by TMA to b4x16_p64 (8 data + 8 padding bytes per 16B
    // group), so the SMEM span stays 128B per row even at K=128 FP4.
    static constexpr int kBSmemBytes = (kTileN / kClusterM) * 128;
    static constexpr int kMmaStepsPerStage = kTileK / kMmaK;
    static constexpr int kSFASmemBytes = kMmaStepsPerStage * 512;
    static constexpr int kSFBScaleTileCount = (kTileN + 127) / 128;
    static constexpr int kSFBSmemBytes = kMmaStepsPerStage * kSFBScaleTileCount * 512;
    static constexpr int kPerStagePayloadBytes =
        kASmemBytes + kBSmemBytes + kSFASmemBytes + kSFBSmemBytes;
    static constexpr int kPerStageSmemBytes = (kPerStagePayloadBytes + 1023) & ~1023;
    static constexpr int kOutputEpilogueTileN = kEpilogueTileN / 2;
    static_assert(kScaleBlockN % kOutputEpilogueTileN == 0);
    // One C slot holds one full 128-column output-scale block of FP8 rows in
    // the 128B-swizzled TMA store layout; two slots double-buffer tiles.
    static constexpr int kEpilogueStageBytes = kTileM * kScaleBlockN;
    static constexpr int kEpilogueStageCount = 2;
    static constexpr int kEpilogueSmemBytes = kEpilogueStageCount * kEpilogueStageBytes;
    // Dynamic and static shared memory share the same per-CTA budget.  Use
    // every mainloop stage that fits after reserving the epilogue and the
    // kernel's static/driver shared-memory allocations.
    static constexpr int kMaxOptInSmemBytes = 232448;
    static constexpr int kStaticSmemReserve = 3 * 1024;
    static constexpr int kSmemLimitedStageCount =
        (kMaxOptInSmemBytes - kStaticSmemReserve - kEpilogueSmemBytes) /
        kPerStageSmemBytes;
    static constexpr int kStageCount = kSmemLimitedStageCount;
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
    static_assert(kClusterN == 1, "MXFP4 FC1 only supports cluster_n=1");
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
    constexpr int kVectorsPerRow = kTileK / 16;
    constexpr int kMmaSteps = kTileK / 32;
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
        // jointly load one logical row, where V = TileK / 16 for the dense
        // FP8 rows. A warp therefore touches fewer random source rows per
        // issue than the FP4 kernel's V = TileK / 32 mapping.
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
                    static_cast<size_t>(a_source_rows[issue]) * k +
                    static_cast<size_t>(tile_k) * kTileK +
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
                    sfa + (source_tile * (k / 32) + global_mma_k) * 512 +
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
                    sfa + (source_tile * (k / 32) + global_mma_k) * 512 +
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
        // this stage carries only K=64. This is why kASmemBytes is fixed at
        // TileM*128 rather than the TMA transaction byte count.
        constexpr int kSmemVectorsPerRow = 128 / kVectorBytes;
        const int physical_linear = row * kSmemVectorsPerRow + vec;
        const int swizzled = (physical_linear & ~7) +
                             ((physical_linear & 7) ^ ((physical_linear >> 3) & 7));
        const bool valid = row < valid_rows;
        // Preserve the original c1 zero-fill behavior; c2 can skip invalid
        // rows because those accumulator rows are never stored.
        if (!kSplitGatherWarps || valid) {
            const int source_row = valid ? source_rows[row] : 0;
            const size_t offset = static_cast<size_t>(source_row) * k +
                                  static_cast<size_t>(tile_k) * kTileK +
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
__device__ inline void tcgen05_mma_mxfp4_w4a8(
    int tmem_addr, uint64_t a_desc, uint64_t b_desc, uint32_t instruction_desc,
    int sfa_addr, int sfb_addr, int accumulate) {
    if constexpr (kClusterM == 1) {
        asm volatile("{\n\t.reg .pred input_d;\n\tsetp.ne.b32 input_d, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
                     "[%0], %1, %2, %3, [%4], [%5], input_d;\n\t}"
            :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(instruction_desc),
               "r"(sfa_addr), "r"(sfb_addr), "r"(accumulate) : "memory");
    } else {
        asm volatile("{\n\t.reg .pred input_d;\n\tsetp.ne.b32 input_d, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X "
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

// FP8 activations are dense bytes.
template <int kTileM, int kTileK>
inline bool init_a_tensor_map(CUtensorMap *map, const uint8_t *ptr, uint64_t m, uint64_t k) {
    constexpr uint32_t rank = 4;
    uint64_t dims[rank] = {k, m, 1, 1};
    uint64_t strides[rank - 1] = {k, m * k, m * k};
    uint32_t box[rank] = {kTileK, kTileM, 1, 1};
    uint32_t elem[rank] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank,
        const_cast<uint8_t *>(ptr), dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// Packed E2M1 weights use the b4x16_p64 TMA expansion (ALIGN16B): every 8B
// source group becomes a 16B SMEM group with an appended padding region, the
// operand layout required by kind::mxf8f6f4.
template <int kTileN, int kTileK, int kClusterM>
inline bool init_b_tensor_map(CUtensorMap *map, const uint8_t *ptr, uint64_t experts,
                              uint64_t n, uint64_t k) {
    constexpr uint32_t rank = 4;
    uint64_t dims[rank] = {k, n, 1, experts};
    uint64_t strides[rank - 1] = {k / 2, n * k / 2, n * k / 2};
    uint32_t box[rank] = {kTileK, kTileN / kClusterM, 1, 1};
    uint32_t elem[rank] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, rank,
        const_cast<uint8_t *>(ptr), dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <bool Compact = false>
inline bool init_scale_tensor_map(CUtensorMap *map, const uint8_t *ptr,
                                  uint64_t row_tiles, uint64_t k_tiles,
                                  uint32_t k_box_tiles) {
    constexpr uint32_t rank = 4;
    uint64_t dims[rank] = {Compact ? 128u : 256u, Compact ? 1u : 2u, k_tiles, row_tiles};
    uint64_t strides[rank - 1] = {Compact ? 128u : 256u, Compact ? 128u : 512u, (Compact ? 128u : 512u) * k_tiles};
    uint32_t box[rank] = {Compact ? 128u : 256u, Compact ? 1u : 2u, k_box_tiles, 1};
    uint32_t elem[rank] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank,
        const_cast<uint8_t *>(ptr), dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// C is the FP8 E4M3 quantized FC2 input, stored row-major [M, N]. One TMA
// store covers one full 128-column output-scale block per 128 rows.
template <int kTileM>
inline bool init_c_tensor_map(
    CUtensorMap *tensor_map,
    uint8_t *ptr,
    uint64_t m,
    uint64_t n) {
    constexpr uint32_t rank = 3;
    uint64_t global_dim[rank] = {n, m, 1};
    uint64_t global_stride[rank - 1] = {n, m * n};
    uint32_t box_dim[rank] = {kScaleBlockN, kTileM, 1};
    uint32_t element_stride[rank] = {1, 1, 1};

    return cuTensorMapEncodeTiled(tensor_map,
               CU_TENSOR_MAP_DATA_TYPE_UINT8,
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
    bool kDirectGmem = false,
    bool kWeightInFc1 = false,
    bool kActivationBf16Rounding = false>
__global__ __cluster_dims__(kClusterM, kClusterN, 1)
__launch_bounds__(kWarpSize * (kBaseWarpCount + (kClusterM > 1 ? 1 : 0))) void mxfp4_fc1_kernel(
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
    uint8_t *c,
    uint8_t *sfc,
    int n,
    int k,
    float swiglu_limit,
    const int32_t *grouped_assignments,
    const float *routing_weights) {
    if constexpr (SM110A_MXFP4_FC1_PDL && SM110A_MXFP4_FC1_PDL_TRIGGER == 0) {
        if (threadIdx.x == 0) sm110a::pdl_trigger<true>(true);
    }
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
    constexpr int kEpilogueBlockCount = kTileN / kEpilogueTileN;
    constexpr int kGroupsPerNBlock = kOutputEpilogueTileN / 16;
    constexpr int kChunksPerRow = kScaleBlockN / 16;
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
    __shared__ uint64_t weight_ready_mbars[kCompactWeightScales && kClusterM > 1 ? kStageCount : 1];
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
    const int base_weight_ready_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(weight_ready_mbars));
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
            if constexpr (kCompactWeightScales && kClusterM > 1)
                mbarrier_init(base_weight_ready_mbar_addr + stage * sizeof(uint64_t), 1);
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

    if constexpr (SM110A_MXFP4_FC1_PDL && SM110A_MXFP4_FC1_PDL_TRIGGER == 1) {
        if (tid == 0) sm110a::pdl_trigger<true>(true);
    }

    sm110a::pdl_wait<SM110A_MXFP4_FC1_PDL_INPUT>();
    const int tmem_addr = tmem_base[0];
    // kind::mxf8f6f4 instruction descriptor: atype = E4M3 (field 0) and
    // btype = E2M1 (field 5), input-D enabled. Identical encoding to
    // fp8mxfp4_gemm_kernel.cu.
    constexpr uint32_t instruction_desc = (5U << 10U) | (1U << 23U) |
                                          (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
                                          (static_cast<uint32_t>(kClusterTileM) >> 7U << 27U);

    const int k_tiles = k / kTileK;

    // One pass over a 64-logical-column block: TMEM gate/up x16 loads only.
    // Each lane owns one row, so the returned amax is that row's block amax.
    auto activation = [&](float gate, float up, float router_scale) -> float {
        if constexpr (kActivationBf16Rounding) {
            gate = __bfloat162float(__float2bfloat16_rn(gate));
            up = __bfloat162float(__float2bfloat16_rn(up));
        }
        float value = swiglu_activate(gate, up, swiglu_limit);
        if constexpr (kWeightInFc1) {
            value *= router_scale;
        }
        if constexpr (kActivationBf16Rounding) {
            value = __bfloat162float(__float2bfloat16_rn(value));
        }
        return value;
    };
    auto block_amax = [&](int mainloop_stage, int n_block, float router_scale) -> float {
        float amax = 0.0f;
        const int tmem_row = m_rank * kTileM + warp_id * kWarpSize;
        for (int group = 0; group < kGroupsPerNBlock; ++group) {
            const int logical_col = n_block * kOutputEpilogueTileN + group * 16;
            const int logical_block = logical_col / kInterleaveBlock;
            const int logical_in_block = logical_col % kInterleaveBlock;
            const int gate_tmem_col =
                mainloop_stage * Config::kAccumulatorStageStride +
                logical_block * (2 * kInterleaveBlock) + logical_in_block;
            const int up_tmem_col = gate_tmem_col + kInterleaveBlock;
            float gate_values[16];
            float up_values[16];
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
            for (int i = 0; i < 16; ++i) {
                amax = fmaxf(amax, fabsf(
                    activation(gate_values[i], up_values[i], router_scale)));
            }
        }
        return amax;
    };

    // Second pass over one 64-logical-column block: re-read TMEM, apply the
    // shared block scale, and write FP8 bytes either directly to GMEM or into
    // the 128B-swizzled C slot (16 bytes per lane per group).
    auto quantize_block = [&](int mainloop_stage, int n_block, float inverse_scale,
                              int cta_row, int tile_col, int valid_rows,
                              char *slot_base, float router_scale) {
        const int tmem_row = m_rank * kTileM + warp_id * kWarpSize;
        for (int group = 0; group < kGroupsPerNBlock; ++group) {
            const int logical_col = n_block * kOutputEpilogueTileN + group * 16;
            const int logical_block = logical_col / kInterleaveBlock;
            const int logical_in_block = logical_col % kInterleaveBlock;
            const int gate_tmem_col =
                mainloop_stage * Config::kAccumulatorStageStride +
                logical_block * (2 * kInterleaveBlock) + logical_in_block;
            const int up_tmem_col = gate_tmem_col + kInterleaveBlock;
            float gate_values[16];
            float up_values[16];
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

            uint8_t packed[16];
            for (int i = 0; i < 16; i += 2) {
                const float2 scaled = make_float2(
                    activation(gate_values[i], up_values[i], router_scale) *
                        inverse_scale,
                    activation(gate_values[i + 1], up_values[i + 1],
                               router_scale) * inverse_scale);
                const __nv_fp8x2_storage_t fp8x2 =
                    __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
                packed[i] = static_cast<uint8_t>(fp8x2 & 0xFF);
                packed[i + 1] = static_cast<uint8_t>(fp8x2 >> 8);
            }

            if (tid < valid_rows) {
                if constexpr (kDirectGmem) {
                    // Direct TMEM -> RMEM -> FP8 GMEM. One 16-byte store per
                    // group; the row stride is the logical N.
                    const int gmem_row = cta_row + tid;
                    const int gmem_col = tile_col / 2 + logical_col;
                    *reinterpret_cast<uint4 *>(
                        c + static_cast<size_t>(gmem_row) * (n / 2) + gmem_col) =
                        *reinterpret_cast<const uint4 *>(packed);
                } else {
                    // One 16-byte chunk per group in the 128B-swizzled slot.
                    const int chunk = tid * kChunksPerRow + logical_col / 16;
                    const int swizzled_chunk = (chunk & ~7) +
                        ((chunk & 7) ^ ((chunk >> 3) & 7));
                    *reinterpret_cast<uint4 *>(
                        slot_base + swizzled_chunk * 16) =
                        *reinterpret_cast<const uint4 *>(packed);
                }
            }
        }
    };

    // Only the elected TMA lane uses this; signal once in the first local
    // task. Both ranks issue B TMA, unlike the rank-0-only MMA warp.
    bool pdl_triggered = false;
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
    // C slots follow the TMA bulk-group FIFO. One slot (one full 128-column
    // output-scale block) is consumed per accumulator stage.
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
                sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_TASK_WAIT>();
                if (elect_sync()) {
                    const int scheduled_bid = atomicAdd(next_cluster_tile, 1);

                    const int task_ready_mbar_addr = base_task_ready_mbar_addr +
                        task_pipe_stage * static_cast<int>(sizeof(uint64_t));
                    scheduled_cluster_bids[task_pipe_stage] = scheduled_bid;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::
                        "r"(task_ready_mbar_addr) : "memory");
                    sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_TASK_PUBLISH>();
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

                constexpr int kAVectorsPerRow = kTileK / 16;
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
                    sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_PRODUCER_WAIT_REUSE>();
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
                    // A/SFA is gathered by four producer warps.  Delay only
                    // after the gather has published its stage completion.
                    sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_LOAD_A>();

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
                    sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_B_WAIT_ISSUE>();
                    if (elect_sync()) {
                        const int tma_transaction_mbar_addr = tma_mbar_addr & 0xFEFFFFFF;
                        const int off_k = tile_k * kTileK;
                        const int off_k32 = tile_k * Config::kMmaStepsPerStage;
                        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
                                     "_, [%0], %1;" ::"r"(tma_transaction_mbar_addr),
                            "r"(Config::kBTmaBytes + kSFBSmemBytes / (kCompactWeightScales ? 4 : 1)) : "memory");
                        tma_load_4d<kClusterM>(b_smem, &b_tensor_map, off_k,
                            tile_col + m_rank * (kTileN / kClusterM), 0, task.expert,
                            tma_transaction_mbar_addr);
                        for (int scale_tile = 0; scale_tile < kSFBScaleTileCount; ++scale_tile) {
                            const int sfb_tile_smem =
                                sfb_smem + scale_tile * Config::kMmaStepsPerStage * 512;
                            const int sfb_tile_index =
                                task.expert * (n / 128) + tile_col / 128 + scale_tile;
                            if constexpr (kClusterM == 1) {
                                tma_load_4d<1>(sfb_tile_smem, &sfb_tensor_map, 0, 0, off_k32,
                                    sfb_tile_index, tma_transaction_mbar_addr);
                            } else if (m_rank == 0) {
                                tma_load_4d_multicast<2>(sfb_tile_smem, &sfb_tensor_map,
                                    0, 0, off_k32, sfb_tile_index, tma_transaction_mbar_addr,
                                    static_cast<int16_t>(3 << (n_rank * kClusterM)));
                            }
                        }
                        // B/SFB TMA requests have been issued for this stage.
                        sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_LOAD_B>();
                    }

                    if constexpr (SM110A_MXFP4_FC1_PDL &&
                                  SM110A_MXFP4_FC1_PDL_TRIGGER >= 2 &&
                                  SM110A_MXFP4_FC1_PDL_TRIGGER <= 5) {
                        if (tid == kTmaWarp * kWarpSize && !pdl_triggered &&
                            (tile_k + 1) * 4 >=
                                k_tiles * (SM110A_MXFP4_FC1_PDL_TRIGGER - 1)) {
                            sm110a::pdl_trigger<true>(true);
                            pdl_triggered = true;
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
        // CuTE baseline's warp 11.  It does not move data: each CTA waits for
        // its local LDGSTS A/SFA stage, then both warps arrive on CTA 0's
        // aggregate barrier.  The leader MMA warp can consequently wait once
        // for a whole cta_group::2 A operand instead of participating in the
        // local completion protocol itself.
        if constexpr (kClusterM > 1) {
            if (warp_id == kSyncTransformWarp && (kCompactWeightScales || elect_sync())) {
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
                        if constexpr (kCompactWeightScales) {
                            // Group-2 TMA completes on the leader only.
                            // Relay completion to the peer before it expands SFB.
                            if (m_rank == 0) {
                                mbarrier_wait_tma_ready(
                                    base_tma_mbar_addr + consumer_stage * sizeof(uint64_t),
                                    consumer_tma_phase);
                                if (elect_sync())
                                    mbarrier_arrive_on_cluster_cta(
                                        base_weight_ready_mbar_addr + consumer_stage * sizeof(uint64_t),
                                        n_rank * kClusterM + 1);
                            } else {
                                mbarrier_wait_tma_ready(
                                    base_weight_ready_mbar_addr + consumer_stage * sizeof(uint64_t),
                                    consumer_tma_phase);
                            }
                            auto *scales = reinterpret_cast<uint8_t *>(__cvta_shared_to_generic(
                                mainloop_smem + consumer_stage * kStageSmemBytes +
                                kASmemBytes + kBSmemBytes + kSFASmemBytes));
                            for (int tile = 0; tile < kSFBScaleTileCount; ++tile)
                                unpack_mxfp4_weight_scales<kMmaStepsPerStage>(
                                    scales + tile * kMmaStepsPerStage * 512);
                        }
                        if (!kCompactWeightScales || elect_sync())
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
        if (m_rank == 0 && warp_id == kMmaWarp &&
            ((kCompactWeightScales && kClusterM == 1) || elect_sync())) {
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                mbarrier_wait_epilogue_slot(
                    base_epilogue_mbar_addr + consumer_mainloop_stage * sizeof(uint64_t),
                    consumer_epilogue_phase);
                sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_CONSUMER_WAIT_EPILOGUE>();
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
                    sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_CONSUMER_WAIT_A>();
                    mbarrier_wait_tma_ready(tma_mbar_addr, consumer_tma_phase);
                    sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_CONSUMER_WAIT_B>();
                    if constexpr (kCompactWeightScales && kClusterM == 1) {
                        auto *scales = reinterpret_cast<uint8_t *>(__cvta_shared_to_generic(
                            mainloop_smem + consumer_stage * kStageSmemBytes +
                            kASmemBytes + kBSmemBytes + kSFASmemBytes));
                        for (int tile = 0; tile < kSFBScaleTileCount; ++tile)
                            unpack_mxfp4_weight_scales<kMmaStepsPerStage>(
                                scales + tile * kMmaStepsPerStage * 512);
                    }
                    if (!kCompactWeightScales || kClusterM > 1 || elect_sync()) {
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
                                tcgen05_mma_mxfp4_w4a8<kClusterM>(stage_tmem_addr,
                                    make_k_major_128b_desc(a_smem + mma_k * 32),
                                    make_k_major_128b_desc(b_smem + mma_k * 32),
                                    instruction_desc, scale_addr,
                                    scale_addr + 4 + (tile_col % 128) / 32, accumulate);
                            }
                        }
                        tcgen05_commit<kClusterM>(mma_mbar_addr, cta_rank);
                        sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_MMA>();
                    }

                    consumer_stage = (consumer_stage + 1) % kStageCount;
                    if (consumer_stage == 0) {
                        consumer_a_phase ^= 1;
                        consumer_tma_phase ^= 1;
                    }
                }

                const int mainloop_mbar_addr =
                    base_mainloop_mbar_addr + consumer_mainloop_stage * sizeof(uint64_t);
                if (!kCompactWeightScales || kClusterM > 1 || elect_sync())
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
                mbarrier_wait_mainloop_output(
                    base_mainloop_mbar_addr + epilogue_mainloop_stage * sizeof(uint64_t),
                    epilogue_mainloop_phase);
                sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_EPILOGUE_WAIT>();
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

                // Pass 1: row-local amax over both 64-column blocks of this
                // stage's single 128-column output-scale block.
                float router_scale = 1.0f;
                if constexpr (kWeightInFc1) {
                    if (tid < valid_rows) {
                        router_scale = routing_weights[grouped_assignments[cta_row + tid]];
                    }
                }
                float block_max = block_amax(epilogue_mainloop_stage, first_n_block,
                                             router_scale);
                block_max = fmaxf(block_max,
                    block_amax(epilogue_mainloop_stage, first_n_block ^ 1, router_scale));
                if constexpr (kActivationBf16Rounding) {
                    // Match the official activation quantizer even for zero
                    // or tiny weighted experts (the weight can be exactly zero).
                    block_max = fmaxf(block_max, 1.0e-4f);
                }

                // Shared E8M0 scale for the whole 128-column block, computed
                // exactly like bf16tofp8_quant (round-towards-positive-infinity power
                // of two via the hardware E8M0 conversion of amax / E4M3 max).
                uint8_t scale_raw = 0;
                float inverse_scale = 1.0f;
                if (tid < valid_rows) {
                    const __nv_fp8_e8m0 scale_e8m0(block_max / kE4M3Max);
                    scale_raw = scale_e8m0.__x;
                    inverse_scale = ldexpf(1.0f, 127 - static_cast<int>(scale_raw));
                }

                // SFC: one 128-column scale replicated across the four K=32
                // scale tiles the block covers, in the blocked layout the
                // MXFP4 FC2 kernel consumes via TMA.
                if (tid < valid_rows && !kSwapAB) {
                    const int output_row = cta_row + tid;
                    const int scale_cols = (n / 2) / 32;
                    const int scale_col = tile_col / 2 / 32;
                    const int tile_base =
                        (output_row / 128) * scale_cols + scale_col;
                    const int in_tile =
                        (output_row % 32) * 16 + ((output_row % 128) / 32) * 4;
                    #pragma unroll
                    for (int replica = 0; replica < 4; ++replica) {
                        sfc[(tile_base + replica) * 512 + in_tile] = scale_raw;
                    }
                }

                char *slot_base = kDirectGmem
                    ? nullptr
                    : dynamic_smem + kMainloopSmemBytes +
                          epilogue_c_stage * kEpilogueStageBytes;

                // Pass 2a: quantize the first-drained block, then release the
                // aliased TMEM columns (both of this block's passes have read
                // them; the disjoint block's reads have not started).
                quantize_block(epilogue_mainloop_stage, first_n_block, inverse_scale,
                               cta_row, tile_col, valid_rows, slot_base, router_scale);
                if constexpr (!kDirectGmem || kSwapAB) {
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                }
                release_overlap();
                sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_EPILOGUE_TMEM_TO_SMEM>();

                // Pass 2b: quantize the remaining block, then release the
                // whole accumulator stage once its last TMEM reads are done.
                quantize_block(epilogue_mainloop_stage, first_n_block ^ 1, inverse_scale,
                               cta_row, tile_col, valid_rows, slot_base, router_scale);
                if constexpr (!kDirectGmem || kSwapAB) {
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                }
                release_tmem();
                sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_MXFP4_FC1_EPILOGUE_STORE>();

                if constexpr (!kDirectGmem || kSwapAB) {
                    if constexpr (!kSwapAB) {
                        if (valid_rows == kTileM) {
                            if (tid == 0) {
                                tma_store_3d(
                                    static_cast<int>(__cvta_generic_to_shared(slot_base)),
                                    &c_tensor_map,
                                    tile_col / 2,
                                    cta_row,
                                    0);
                            }
                        } else if (tid < valid_rows) {
                            // Boundary tile: copy the swizzled slot rows out
                            // with scalar 16-byte stores.
                            for (int chunk_i = 0; chunk_i < kChunksPerRow; ++chunk_i) {
                                const int chunk = tid * kChunksPerRow + chunk_i;
                                const int swizzled_chunk = (chunk & ~7) +
                                    ((chunk & 7) ^ ((chunk >> 3) & 7));
                                const int4 value = reinterpret_cast<const int4 *>(
                                    slot_base)[swizzled_chunk];
                                const size_t output_offset =
                                    static_cast<size_t>(cta_row + tid) * (n / 2) +
                                    tile_col / 2 + chunk_i * 16;
                                store_output_16b(
                                    reinterpret_cast<int4 *>(c + output_offset), value);
                            }
                        }
                    }
                    if (tid == 0) {
                        // Commit one group per logical C stage.  A boundary
                        // tile that used scalar stores commits an empty group,
                        // preserving the FIFO-to-slot mapping.
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

    if constexpr (SM110A_MXFP4_FC1_PDL && SM110A_MXFP4_FC1_PDL_TRIGGER == 6) {
        if (tid == 0) sm110a::pdl_trigger<true>(true);
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
          int kClusterN, bool kSwapAB, bool kWeightInFc1, bool kActivationBf16Rounding>
void launch_mxfp4_fc1_impl(
    const uint8_t *d_a,
    const uint8_t *d_b,
    const uint8_t *d_sfa,
    const uint8_t *d_sfb,
    const int32_t *d_source_rows,
    const int32_t *d_tasks,
    int32_t *d_state,
    uint8_t *d_c,
    uint8_t *d_sfc,
    int tokens,
    int m,
    int experts,
    int n,
    int k,
    float swiglu_limit,
    int max_tasks, int max_active_clusters,
    cudaStream_t stream, const int32_t *grouped_assignments,
    const float *routing_weights) {
    using Config = KernelConfig<kTileM, kTileN, kTileK, kEpilogueTileN, kClusterM, kClusterN>;
    using Shape = ClusterShape<kClusterM, kClusterN>;
    (void)sizeof(Shape);
    constexpr int kClusterTileM = Config::kClusterTileM;
    (void)kClusterTileM;
    constexpr int kBlockThreads =
        kWarpSize * (kBaseWarpCount + (kClusterM > 1 ? 1 : 0));
    constexpr int kDynamicSmemBytes = Config::kDynamicSmemBytes;
    static_assert(!kSwapAB, "MXFP4 FC1 does not support swap_ab");
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
            k / 32, kTileK / 32) &&
        init_scale_tensor_map<kCompactWeightScales>(&sfb_tensor_map, d_sfb, experts * (n / 128),
            k / 32, kTileK / 32) &&
        init_c_tensor_map<kTileM>(
            &c_tensor_map, d_c, m, n / 2);
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
    if constexpr (!SM110A_MXFP4_FC1_PDL_INPUT)
        cudaMemsetAsync(next_cluster_tile, 0, sizeof(int32_t), stream);

    const int persistent_clusters =
        (max_tasks + kPersistentTileCount - 1) / kPersistentTileCount;
    auto kernel = mxfp4_fc1_kernel<kTileM,
            kTileN,
            kTileK,
            kEpilogueTileN,
            kClusterM,
            kClusterN, kSwapAB,
            (SM110A_MXFP4_FC1_DIRECT_GMEM != 0), kWeightInFc1, kActivationBf16Rounding>;
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
    launch_config.gridDim = dim3(kClusterM * launched_clusters, kClusterN, 1);
    sm110a::launch_pdl<SM110A_MXFP4_FC1_PDL_INPUT>(&launch_config, kernel,
        d_a, a_tensor_map, b_tensor_map, d_sfa, sfa_tensor_map, sfb_tensor_map, c_tensor_map, d_source_rows, tasks, task_count,
        next_cluster_tile, d_c, d_sfc, n, k, swiglu_limit,
        grouped_assignments, routing_weights);
}

void launch_mxfp4_fc1_entry(
    const uint8_t *a, const uint8_t *b, const uint8_t *sfa, const uint8_t *sfb,
    const int32_t *source_rows, const int32_t *tasks, int32_t *state, uint8_t *c,
    uint8_t *sfc,
    int tokens, int m, int experts, int n, int k, float swiglu_limit,
    int max_tasks, int max_active_clusters, cudaStream_t stream,
    const int32_t *grouped_assignments, const float *routing_weights) {
    if (!a || !b || !sfa || !sfb || !source_rows || !tasks || !state || !c || !sfc ||
        tokens <= 0 || m <= 0 || experts <= 0 || n <= 0 || k <= 0 || max_tasks <= 0)
        return;
    if constexpr (SM110A_MXFP4_FC1_WEIGHT_IN_FC1 != 0) {
        if (!grouped_assignments || !routing_weights) return;
    }
    launch_mxfp4_fc1_impl<SM110A_MXFP4_FC1_TILE_M,
        SM110A_MXFP4_FC1_TILE_N, SM110A_MXFP4_FC1_TILE_K,
        SM110A_MXFP4_FC1_EPILOGUE_TILE_N, SM110A_MXFP4_FC1_CLUSTER_M,
        SM110A_MXFP4_FC1_CLUSTER_N,
        false, (SM110A_MXFP4_FC1_WEIGHT_IN_FC1 != 0),
        (SM110A_MXFP4_FC1_ACTIVATION_BF16_ROUNDING != 0)>(
            a, b, sfa, sfb, source_rows, tasks, state, c, sfc, tokens, m, experts,
            n, k, swiglu_limit, max_tasks, max_active_clusters, stream,
            grouped_assignments, routing_weights);
}

} // namespace sm110a::mxfp4_fc1

using namespace sm110a::mxfp4_fc1;

void launch_mxfp4_fc1_main(
    const uint8_t *a, const uint8_t *b, const uint8_t *sfa, const uint8_t *sfb,
    const int32_t *source_rows, const int32_t *tasks, int32_t *state, uint8_t *c,
    uint8_t *sfc,
    int tokens, int m, int experts, int n, int k, float swiglu_limit,
    int max_tasks, int max_active_clusters, cudaStream_t stream,
    const int32_t *grouped_assignments, const float *routing_weights) {
    launch_mxfp4_fc1_entry(
        a, b, sfa, sfb, source_rows, tasks, state, c, sfc, tokens, m, experts, n, k,
        swiglu_limit, max_tasks, max_active_clusters, stream, grouped_assignments, routing_weights);
}

extern "C" void sm110a_mxfp4_fc1_launch(
    const uint8_t *a, const uint8_t *b, const uint8_t *sfa, const uint8_t *sfb,
    const int32_t *source_rows, const int32_t *tasks, int32_t *state, uint8_t *c,
    uint8_t *sfc,
    int tokens, int m, int experts, int n, int k, float swiglu_limit,
    int max_tasks, int max_active_clusters, cudaStream_t stream,
    const int32_t *grouped_assignments, const float *routing_weights) {
    launch_mxfp4_fc1_main(
        a, b, sfa, sfb, source_rows, tasks, state, c, sfc, tokens, m, experts, n, k,
        swiglu_limit, max_tasks, max_active_clusters, stream, grouped_assignments, routing_weights);
}
