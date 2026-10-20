#include "pdl.h"
#ifndef SM110A_MXFP4_FC2_PDL
#define SM110A_MXFP4_FC2_PDL 0
#endif
#ifndef SM110A_MXFP4_FC2_EXTERNAL_PREPARE
#define SM110A_MXFP4_FC2_EXTERNAL_PREPARE SM110A_MXFP4_FC2_PDL
#endif
#ifndef SM110A_MXFP4_FC2_PDL_WAIT_BEFORE_A
#define SM110A_MXFP4_FC2_PDL_WAIT_BEFORE_A 0
#endif

/*
 * Fused SM110A MXFP4 (W4A8) routed FC2: grouped GEMM with token-major routing
 * reduction fused into the epilogue.
 *
 * Port of nvfp4_fc2_kernel.cu; only the quantization formats change:
 *   - A is FP8 E4M3 [M, K] (dense bytes) with blocked UE8M0 scales.
 *   - B stays packed E2M1 [experts, N, K/2] with blocked UE8M0 scales.
 *   - Scales are UE8M0, one per 32 contiguous K elements, stored as 512-byte
 *     SWIZZLE_32_4_4 tiles per (128 rows, K=32 step) with the live byte in
 *     column 0 — the same layout consumed by fp8mxfp4_gemm_kernel.cu.
 * The MMA is tcgen05.mma.kind::mxf8f6f4.block_scale.scale_vec::1X, so the
 * mainloop advances K in 32-element steps and B's TMA uses the b4x16_p64
 * (ALIGN16B) expansion that pads each 16-byte group to 16 data + 16 pad
 * bytes in shared memory. Everything else (tile scheduling, task pipeline,
 * clustering, SFA staging for unaligned expert slices, and the routing-weight
 * epilogue) is unchanged from the NVFP4 version.
 *
 * Computes C[token] = sum_rank routing_weight[assignment] *
 *                     A[grouped_row] * B[expert]^T from packed E2M1 operands
 * and blocked UE8M0 scales. A is the expert-grouped FC1 output, so no token
 * permutation is required on the A side: A/SFA use TMA just like the
 * standalone grouped-M GEMM. The token-major reduction is folded into the
 * epilogue, which scales each TMEM accumulator row by its routing weight and
 * accumulates it into the destination with cp.reduce.async.bulk, removing the
 * [M, N] workspace round-trip of the non-fused baseline.
 *
 * SFA is blocked over A's physical rows and SFB is blocked independently for
 * each expert. Expert segments in grouped A must begin on 128-row boundaries.
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include "mxfp4_moe_fc2.h"

#ifndef SM110A_MXFP4_FC2_TILE_M
#define SM110A_MXFP4_FC2_TILE_M 128
#endif
#ifndef SM110A_MXFP4_FC2_TILE_N
#define SM110A_MXFP4_FC2_TILE_N 256
#endif
#ifndef SM110A_MXFP4_FC2_TILE_K
#define SM110A_MXFP4_FC2_TILE_K 128
#endif
#ifndef SM110A_MXFP4_FC2_CLUSTER_M
#define SM110A_MXFP4_FC2_CLUSTER_M 2
#endif
#ifndef SM110A_MXFP4_FC2_REDUCE_TILE_N
#define SM110A_MXFP4_FC2_REDUCE_TILE_N 64
#endif
#ifndef SM110A_MXFP4_FC2_WEIGHT_IN_FC1
#define SM110A_MXFP4_FC2_WEIGHT_IN_FC1 0
#endif
#ifndef SM110A_MXFP4_FC2_DETERMINISTIC
#define SM110A_MXFP4_FC2_DETERMINISTIC 0
#endif

#include "mxfp4_weight_scales.cuh"
#ifndef SM110A_MXFP4_FC2_COMPACT_WEIGHT_SCALES
#define SM110A_MXFP4_FC2_COMPACT_WEIGHT_SCALES 0
#endif

namespace sm110a::mxfp4_fc2 {
constexpr bool kCompactWeightScales = SM110A_MXFP4_FC2_COMPACT_WEIGHT_SCALES;


constexpr int kMmaK = 32;
constexpr int kWarpSize = 32;
constexpr int kEpilogueWarpCount = 4;
constexpr int kTmaWarp = kEpilogueWarpCount;
constexpr int kSfaWarp = kTmaWarp + 1;
constexpr int kMmaWarp = kSfaWarp + 1;
constexpr int kSchedulerWarp = kMmaWarp + 1;
constexpr int kBaseWarpCount = kSchedulerWarp + 1;
constexpr int kTaskPipelineStageCount = 2;
constexpr int kPersistentTileCount = 2;
constexpr int kTaskPacketTileCount = 4;
constexpr int kMinRowsPerExpertForTaskPacket = 256;
// Each mode-2 CLC worker cluster starts with two tasks.  Once it completes
// them it steals the work of an unstarted cluster through Cluster Launch Control.
constexpr int kClcWorkerTileCount = 2;

static_assert(SM110A_MXFP4_FC2_REDUCE_TILE_N % 8 == 0,
              "reduce_tile_n must be a multiple of 8 for tcgen05.ld.x8");

template <int kTileM, int kTileN, int kTileK, int kClusterM>
struct Mxfp4Fc2Config {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert(kTileM == 4 * kWarpSize);
    static_assert(kTileN == 256);
    // K is fixed at 128: FP8 activations ride a 128B-swizzle atom (the NVFP4
    // kernel allowed 256 for half-width FP4 rows), and the b4x16_p64 TMA
    // expansion of the packed E2M1 weights requires a 128-element box.
    static_assert(kTileK == 128);
    static_assert(kTileN % kClusterM == 0);
    static_assert(kTileK % 32 == 0);
    static_assert(kTileK % kMmaK == 0);
    static_assert(kTileN % SM110A_MXFP4_FC2_REDUCE_TILE_N == 0);
    static constexpr int kClusterTileM = kClusterM * kTileM;
    static constexpr int kATmaBytes = kTileM * kTileK;
    static constexpr int kBTmaBytes = (kTileN / kClusterM) * kTileK / 2;
    static constexpr int kASmemBytes = kTileM * 128;
    // B is expanded by TMA to b4x16_p64 (8 data + 8 padding bytes per 16B
    // group), so the SMEM span stays 128B per row even at K=128 FP4.
    static constexpr int kBSmemBytes = (kTileN / kClusterM) * 128;
    static constexpr int kMmaStepsPerStage = kTileK / kMmaK;
    // Padded scale layout (one 512-byte SWIZZLE_32_4_4 tile per 128 rows and
    // K=32 step, live byte in column 0) — the same layout consumed by
    // fp8mxfp4_gemm_kernel.cu. A compact per-K=128 variant (four live
    // columns, one quarter of the scale traffic) was evaluated: the
    // scale_vec::1X MMA faults on non-4-aligned TMEM scale columns, and the
    // smem rotation needed to re-align compact tiles is left as follow-up.
    static constexpr int kSFASmemBytes = kMmaStepsPerStage * 512;
    static constexpr int kSFBScaleTileCount = (kTileN + 127) / 128;
    static constexpr int kSFBSmemBytes = kMmaStepsPerStage * kSFBScaleTileCount * 512;
    // A grouped expert slice can start in the middle of a blocked 128-row
    // scale tile. cta_group::1 stages the two adjacent source tiles with TMA
    // and reconstructs the requested slice entirely in shared memory.
    static constexpr int kSFAStagingBytes = 2 * kSFASmemBytes;
    static constexpr int kPerStagePayloadBytes =
        kASmemBytes + kBSmemBytes + kSFASmemBytes + kSFBSmemBytes +
        kSFAStagingBytes;
    static constexpr int kPerStageSmemBytes = (kPerStagePayloadBytes + 1023) & ~1023;
    // The fused finalize stages one physical 128-row × reduce_tile_n BF16 slab
    // in row-major shared memory; cp.reduce.async.bulk consumes it per row.
    static constexpr int kFinalizeSmemBytes =
        kTileM * SM110A_MXFP4_FC2_REDUCE_TILE_N * sizeof(nv_bfloat16);
    // Pack as many mainloop pipeline stages as shared memory allows for this
    // configuration. Blackwell has ~228 KB of SMEM per CTA; leave a small
    // margin for statically allocated barriers and bookkeeping.
    static constexpr int kMaxSmemBytes = 227 * 1024;
    static constexpr int kStageCount = (kMaxSmemBytes - kFinalizeSmemBytes) / kPerStageSmemBytes;
    static_assert(kStageCount >= 2);
    static constexpr int kMainloopSmemBytes = kStageCount * kPerStageSmemBytes;
    static constexpr int kDynamicSmemBytes = kMainloopSmemBytes + kFinalizeSmemBytes;
    // Each compact 512-byte scale tile is copied into four 4-column-aligned
    // TMEM copies (one per K=32 MMA of the K=128 span; the MMA's scale
    // operand requires a 4-column-aligned start, so the live slice must sit
    // at column +4*k of its own copy).
    static constexpr int kScaleColsPerMma = 4 + 4 * kSFBScaleTileCount;
    // Keep two accumulator stages even at N=256 by overlapping the last 16
    // columns of stage 0 with the first 16 columns of stage 1.  The epilogue
    // drains these columns before releasing a stage, so the next MMA can
    // overwrite the overlap while finalize continues through the remaining
    // columns.  This matches the standalone GEMM TMEM pipeline.
    static constexpr int kAccumulatorOverlapCols = kTileN == 256 ? 16 : 0;
    static constexpr int kTmemStageCount = 2;
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

__device__ inline void clc_issue_request(int response_addr, int mbar_addr) {
    asm volatile("mbarrier.arrive.expect_tx.relaxed.cluster."
                 "shared::cta.b64 _, [%0], 16;" :: "r"(mbar_addr)
        : "memory");
    asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta."
                 "mbarrier::complete_tx::bytes.multicast::cluster::all."
                 "b128 [%0], [%1];" :: "r"(response_addr), "r"(mbar_addr)
        : "memory");
}

__device__ inline bool clc_wait_and_query(
    const uint4 *response, int mbar_addr, int phase, int *canceled_cta_x) {
    clc_mbarrier_wait(mbar_addr, phase);
    const uint4 value = response[0];
    int canceled;
    int first_cta_x = 0;
    asm volatile("{\n\t"
                 ".reg .pred was_canceled;\n\t"
                 ".reg .b128 response;\n\t"
                 "mov.b128 response, {%2, %3, %4, %5};\n\t"
                 "clusterlaunchcontrol.query_cancel.is_canceled.pred."
                 "b128 was_canceled, response;\n\t"
                 "selp.u32 %0, 1, 0, was_canceled;\n\t"
                 "@was_canceled clusterlaunchcontrol.query_cancel."
                 "get_first_ctaid::x.b32.b128 %1, response;\n\t"
                 "}"
        : "=r"(canceled), "=r"(first_cta_x)
        : "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
        : "memory");
    *canceled_cta_x = first_cta_x;
    return canceled != 0;
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
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global."
                 "mbarrier::complete_tx::bytes.multicast::cluster.cta_group::%8 "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::
            "r"(dst_smem),
            "l"(tensor_map),
            "r"(x),
            "r"(y),
            "r"(z),
            "r"(w),
            "r"(mbar_addr),
            "h"(cta_mask),
            "n"(kClusterM)
            : "memory");
}

__device__ inline void prefetch_tensor_map(
    const void *tensor_map) {
    const uint64_t ptr = reinterpret_cast<uint64_t>(tensor_map);
    asm volatile("prefetch.tensormap [%0];" ::"l"(ptr) : "memory");
}

__device__ inline int map_shared_to_cluster_cta(
    int local_smem_addr, int target_cta_rank) {
    int remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote_addr)
        : "r"(local_smem_addr), "r"(target_cta_rank));
    return remote_addr;
}

__device__ inline void mbarrier_arrive_on_cluster_cta(
    int local_mbar_addr, int target_cta_rank) {
    const int remote_mbar_addr =
        map_shared_to_cluster_cta(local_mbar_addr, target_cta_rank);
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
        :
        : "r"(remote_mbar_addr)
        : "memory");
}

__device__ inline void store_shared_on_cluster_cta(
    int local_smem_addr, int target_cta_rank, int value) {
    const int remote_smem_addr =
        map_shared_to_cluster_cta(local_smem_addr, target_cta_rank);
    asm volatile("st.shared::cluster.u32 [%0], %1;" ::
        "r"(remote_smem_addr), "r"(value) : "memory");
}

__device__ inline void mbarrier_arrive_release_cluster_on_cluster_cta(
    int local_mbar_addr, int target_cta_rank) {
    const int remote_mbar_addr =
        map_shared_to_cluster_cta(local_mbar_addr, target_cta_rank);
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::
        "r"(remote_mbar_addr) : "memory");
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

// Fused-finalize helpers: fence the shared-memory source into the async proxy
// and accumulate one row via cp.reduce.async.bulk add (no FTZ, BF16).
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

template <
    int kTileM,
    int kTileN,
    int kTileK,
    int kClusterM,
    bool kPersistent,
    bool kDynamicClc = false,
    int kTaskPacketSize = 1,
    bool kWeightInFc1 = false,
    bool kDeterministic = false>
__global__ __cluster_dims__(kClusterM, 1, 1)
__launch_bounds__(kWarpSize * kBaseWarpCount)
void mxfp4_fc2_kernel(
    const uint8_t *sfa,
    const __grid_constant__ CUtensorMap a_tensor_map,
    const __grid_constant__ CUtensorMap b_tensor_map,
    const __grid_constant__ CUtensorMap sfa_tensor_map,
    const __grid_constant__ CUtensorMap sfb_tensor_map,
    const ExpertTileTask *tasks,
    const int *task_count,
    int *next_cluster_tile,
    const int32_t *grouped_assignments,
    const __nv_bfloat16 *routing_weights,
    __nv_bfloat16 *final_output,
    int n,
    int k,
    int topk) {
    using Config = Mxfp4Fc2Config<kTileM, kTileN, kTileK, kClusterM>;
    constexpr int kClusterTileM = Config::kClusterTileM;
    constexpr int kStageCount = Config::kStageCount;
    constexpr int kASmemBytes = Config::kASmemBytes;
    constexpr int kBSmemBytes = Config::kBSmemBytes;
    constexpr int kSFASmemBytes = Config::kSFASmemBytes;
    constexpr int kSFBSmemBytes = Config::kSFBSmemBytes;
    constexpr int kSFAStagingBytes = Config::kSFAStagingBytes;
    constexpr int kSFBScaleTileCount = Config::kSFBScaleTileCount;
    constexpr int kStageSmemBytes = Config::kStageSmemBytes;
    constexpr int kMainloopSmemBytes = Config::kMainloopSmemBytes;
    constexpr int kMmaStepsPerStage = kTileK / kMmaK;
    constexpr int kTmemStageCount = Config::kTmemStageCount;
    constexpr int kTmemAllocationColumns = Config::kTmemAllocationColumns;
    constexpr int kAccumulatorOverlapCols = Config::kAccumulatorOverlapCols;
    constexpr int kAccumulatorStageStride = Config::kAccumulatorStageStride;
    constexpr bool kUseTaskPipeline = kPersistent;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int worker_id = blockIdx.x / kClusterM;
    const int cluster_tiles = *task_count;
    static_assert(kTaskPacketSize > 0);
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
    __shared__ int scheduled_cluster_bids[kTaskPipelineStageCount];
    __shared__ uint64_t task_ready_mbars[kTaskPipelineStageCount];
    __shared__ uint64_t task_free_mbars[kTaskPipelineStageCount];
    __shared__ __align__(16) uint4 clc_response[1];
    __shared__ __align__(8) uint64_t clc_mbar[1];
    __shared__ uint64_t tma_mbars[kStageCount];
    __shared__ uint64_t weight_ready_mbars[kCompactWeightScales && kClusterM > 1 ? kStageCount : 1];
    __shared__ uint64_t sfa_tma_mbars[kStageCount];
    __shared__ uint64_t sfa_mbars[kStageCount];
    __shared__ uint64_t mma_mbars[kStageCount];
    __shared__ uint64_t mainloop_mbars[kTmemStageCount];
    __shared__ uint64_t epilogue_mbars[kTmemStageCount];
    __shared__ uint64_t overlap_mbars[kTmemStageCount];

    extern __shared__ __align__(1024) char dynamic_smem[];
    const int mainloop_smem = static_cast<int>(__cvta_generic_to_shared(dynamic_smem));
    const int base_tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(tma_mbars));
    const int base_weight_ready_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(weight_ready_mbars));
    const int base_sfa_tma_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(sfa_tma_mbars));
    const int base_task_ready_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(task_ready_mbars));
    const int base_task_free_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(task_free_mbars));
    const int base_sfa_mbar_addr = static_cast<int>(__cvta_generic_to_shared(sfa_mbars));
    const int base_mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mma_mbars));
    const int base_mainloop_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mainloop_mbars));
    const int base_epilogue_mbar_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_mbars));
    const int base_overlap_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(overlap_mbars));

    if (tid == 0) {
        for (int stage = 0; stage < kStageCount; ++stage) {
            mbarrier_init(base_tma_mbar_addr + stage * sizeof(uint64_t), kClusterM);
            if constexpr (kCompactWeightScales && kClusterM > 1)
                mbarrier_init(base_weight_ready_mbar_addr + stage * sizeof(uint64_t), 1);
            mbarrier_init(base_sfa_tma_mbar_addr + stage * sizeof(uint64_t), 1);
            // cta_group::2 aggregates one completed SFA gather arrival from
            // each CTA on rank 0. cta_group::1 folds SFA into tma_mbars.
            mbarrier_init(base_sfa_mbar_addr + stage * sizeof(uint64_t), kClusterM);
            mbarrier_init(base_mma_mbar_addr + stage * sizeof(uint64_t), 1);
        }
        for (int stage = 0; stage < kTmemStageCount; ++stage) {
            mbarrier_init(base_mainloop_mbar_addr + stage * sizeof(uint64_t), 1);
            if constexpr (kPersistent || kDynamicClc) {
                mbarrier_init(base_epilogue_mbar_addr + stage * sizeof(uint64_t), 4 * kClusterM);
                if constexpr (kAccumulatorOverlapCols != 0) {
                    mbarrier_init(base_overlap_mbar_addr + stage * sizeof(uint64_t),
                        4 * kClusterM);
                }
            }
        }
        if constexpr (kUseTaskPipeline) {
            for (int stage = 0; stage < kTaskPipelineStageCount; ++stage) {
                mbarrier_init(base_task_ready_mbar_addr + stage * sizeof(uint64_t), 1);
                mbarrier_init(base_task_free_mbar_addr + stage * sizeof(uint64_t),
                    kClusterM);
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
        prefetch_tensor_map(&a_tensor_map);
        prefetch_tensor_map(&b_tensor_map);
        prefetch_tensor_map(&sfa_tensor_map);
        prefetch_tensor_map(&sfb_tensor_map);
    }
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if constexpr (SM110A_MXFP4_FC2_PDL && !SM110A_MXFP4_FC2_PDL_WAIT_BEFORE_A) {
        cudaGridDependencySynchronize();
    }

    const int tmem_addr = tmem_base[0];
    // kind::mxf8f6f4 instruction descriptor: atype = E4M3 (field 0) and
    // btype = E2M1 (field 5), input-D enabled. Identical encoding to
    // fp8mxfp4_gemm_kernel.cu.
    constexpr uint32_t instruction_desc = (5U << 10U) | (1U << 23U) |
                                          (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
                                          (static_cast<uint32_t>(kClusterTileM) >> 7U << 27U);

    const int k_tiles = k / kTileK;

    bool pdl_dependency_ready = false;
    int producer_stage = 0;
    int producer_mma_phase = 1;
    int sfa_stage = 0;
    int sfa_tma_phase = 0;
    int consumer_stage = 0;
    int consumer_tma_phase = 0;
    int consumer_mainloop_stage = 0;
    int consumer_epilogue_phase = 1;
    int consumer_overlap_phase = 0;
    bool consumer_has_previous_overlap = false;
    int epilogue_mainloop_stage = 0;
    int epilogue_mainloop_phase = 0;
    int clc_phase = 0;
    int dynamic_first_cluster_bid = static_first_cluster_bid;
    int task_pipe_stage = 0;
    int task_ready_phase = 0;
    int scheduler_free_phase = 1;
    int scheduler_packet_next = static_first_cluster_bid;
    int scheduler_packet_end =
        min(static_first_cluster_bid + kWorkerTileCount, cluster_tiles);
    bool scheduler_clc_pending = false;
    bool scheduler_clc_done = false;

    while (true) {
        int first_cluster_bid;
        int clc_mbar_addr = 0;
        if constexpr (kUseTaskPipeline) {
            if (warp_id == kSchedulerWarp) {
                if constexpr (kClusterM == 1) {
                    mbarrier_wait(base_task_free_mbar_addr +
                                      task_pipe_stage * sizeof(uint64_t),
                                  scheduler_free_phase);
                    if (elect_sync()) {
                        int scheduled_bid;
                        if constexpr (kDynamicClc) {
                            const int response_addr =
                                static_cast<int>(__cvta_generic_to_shared(clc_response));
                            const int scheduler_mbar_addr =
                                static_cast<int>(__cvta_generic_to_shared(clc_mbar));
                            while (scheduler_packet_next >= scheduler_packet_end &&
                                   !scheduler_clc_done) {
                                if (!scheduler_clc_pending) {
                                    clc_issue_request(response_addr, scheduler_mbar_addr);
                                    scheduler_clc_pending = true;
                                }
                                int canceled_cta_x = 0;
                                const bool canceled = clc_wait_and_query(clc_response,
                                    scheduler_mbar_addr, clc_phase, &canceled_cta_x);
                                scheduler_clc_pending = false;
                                clc_phase ^= 1;
                                if (!canceled) {
                                    scheduler_clc_done = true;
                                    break;
                                }
                                scheduler_packet_next =
                                    canceled_cta_x / kClusterM * kWorkerTileCount;
                                scheduler_packet_end = min(
                                    scheduler_packet_next + kWorkerTileCount, cluster_tiles);
                            }
                            scheduled_bid = scheduler_packet_next < scheduler_packet_end
                                ? scheduler_packet_next++
                                : cluster_tiles;
                        } else {
                            scheduled_bid =
                                atomicAdd(next_cluster_tile, kTaskPacketSize);
                        }
                        scheduled_cluster_bids[task_pipe_stage] = scheduled_bid;
                        const int ready_addr = base_task_ready_mbar_addr +
                            task_pipe_stage * sizeof(uint64_t);
                        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::
                            "r"(ready_addr) : "memory");
                    }
                } else if (cta_rank == 0) {
                    // Rank 0 is the cluster's sole task-source producer.  It
                    // publishes the same task into each CTA-local ring so all
                    // data-path roles can keep using cheap CTA-local waits.
                    clc_mbarrier_wait(base_task_free_mbar_addr +
                                          task_pipe_stage * sizeof(uint64_t),
                                      scheduler_free_phase);
                    if (elect_sync()) {
                        const int scheduled_bid =
                            atomicAdd(next_cluster_tile, kTaskPacketSize);
                        const int scheduled_bid_addr = static_cast<int>(
                            __cvta_generic_to_shared(
                                scheduled_cluster_bids + task_pipe_stage));
                        const int ready_addr = base_task_ready_mbar_addr +
                            task_pipe_stage * sizeof(uint64_t);
                        scheduled_cluster_bids[task_pipe_stage] = scheduled_bid;
                        store_shared_on_cluster_cta(
                            scheduled_bid_addr, 1, scheduled_bid);
                        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::
                            "r"(ready_addr) : "memory");
                        mbarrier_arrive_release_cluster_on_cluster_cta(
                            ready_addr, 1);
                    }
                } else {
                    clc_mbarrier_wait(base_task_ready_mbar_addr +
                                          task_pipe_stage * sizeof(uint64_t),
                                      task_ready_phase);
                }
                __syncwarp();
                const bool done =
                    scheduled_cluster_bids[task_pipe_stage] >= cluster_tiles;
                task_pipe_stage =
                    (task_pipe_stage + 1) % kTaskPipelineStageCount;
                if (task_pipe_stage == 0) {
                    scheduler_free_phase ^= 1;
                    if constexpr (kClusterM > 1) {
                        if (cta_rank != 0) task_ready_phase ^= 1;
                    }
                }
                if (done) break;
                continue;
            }
            if constexpr (kClusterM == 1) {
                mbarrier_wait(base_task_ready_mbar_addr +
                                  task_pipe_stage * sizeof(uint64_t),
                              task_ready_phase);
            } else {
                clc_mbarrier_wait(base_task_ready_mbar_addr +
                                      task_pipe_stage * sizeof(uint64_t),
                                  task_ready_phase);
            }
            first_cluster_bid = scheduled_cluster_bids[task_pipe_stage];
            if (first_cluster_bid >= cluster_tiles) break;
        } else if constexpr (kDynamicClc) {
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
        constexpr int tasks_per_iteration = kUseTaskPipeline
            ? (kDynamicClc ? 1 : kTaskPacketSize)
            : kWorkerTileCount;
        const int last_cluster_bid = has_work
                                         ? (first_cluster_bid + tasks_per_iteration < cluster_tiles
                                                   ? first_cluster_bid + tasks_per_iteration
                                                   : cluster_tiles)
                                         : first_cluster_bid;

        // The producer waits until MMA has released a stage before TMA overwrites it.
        if (warp_id == kTmaWarp && elect_sync()) {
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
                    const int sfa_smem = b_smem + kBSmemBytes;
                    const int sfb_smem = sfa_smem + kSFASmemBytes;
                    const int sfa_staging_smem = sfb_smem + kSFBSmemBytes;

                    const int off_k = tile_k * kTileK;
                    const int off_k32 = tile_k * Config::kMmaStepsPerStage;
                    const int valid_rows =
                        min(kTileM, max(0, task.row_count - task.tile_row -
                                                  cta_rank * kTileM));
                    const bool sfa_aligned = (cta_row & (kTileM - 1)) == 0;
                    const int first_sfa_tile = cta_row / kTileM;
                    const int last_sfa_row = cta_row + valid_rows - 1;
                    const int sfa_source_tiles = valid_rows > 0 &&
                            (last_sfa_row / kTileM) != first_sfa_tile
                        ? 2
                        : 1;
                    const int sfa_tma_bytes = kClusterM == 1
                        ? (sfa_aligned ? kSFASmemBytes
                                       : sfa_source_tiles * kSFASmemBytes)
                        : 0;
                    mbarrier_wait(mma_mbar_addr, producer_mma_phase);
                    const int tma_transaction_mbar_addr = tma_mbar_addr & 0xFEFFFFFF;
                    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
                                 "_, [%0], %1;" ::"r"(tma_transaction_mbar_addr),
                        "r"(Config::kATmaBytes + Config::kBTmaBytes +
                            kSFBSmemBytes / (kCompactWeightScales ? 4 : 1) + sfa_tma_bytes)
                        : "memory");
                    // Only A/SFA depend on FC1. Start independent weight
                    // traffic before waiting; downstream warps remain gated
                    // by the existing TMA/SFA completion barriers.
                    if constexpr (SM110A_MXFP4_FC2_PDL && SM110A_MXFP4_FC2_PDL_WAIT_BEFORE_A) {
                        tma_load_4d<kClusterM>(b_smem, &b_tensor_map, off_k,
                            tile_col + cta_rank * (kTileN / kClusterM), 0, task.expert,
                            tma_transaction_mbar_addr);
                        for (int scale_tile = 0; scale_tile < kSFBScaleTileCount; ++scale_tile) {
                            const int sfb_tile_smem =
                                sfb_smem + scale_tile * Config::kMmaStepsPerStage * 512;
                            const int sfb_tile_index =
                                task.expert * (n / 128) + tile_col / 128 + scale_tile;
                            if constexpr (kClusterM == 1) {
                                tma_load_4d<1>(sfb_tile_smem, &sfb_tensor_map, 0, 0, off_k32,
                                    sfb_tile_index, tma_transaction_mbar_addr);
                            } else if (cta_rank == 0) {
                                constexpr int16_t cta_mask = (1 << kClusterM) - 1;
                                tma_load_4d_multicast<kClusterM>(sfb_tile_smem, &sfb_tensor_map,
                                    0, 0, off_k32, sfb_tile_index, tma_transaction_mbar_addr,
                                    cta_mask);
                            }
                        }
                        if (!pdl_dependency_ready) {
                            cudaGridDependencySynchronize();
                            pdl_dependency_ready = true;
                        }
                    }
                    tma_load_4d<kClusterM>(a_smem, &a_tensor_map, off_k, cta_row, 0, 0,
                        tma_transaction_mbar_addr);
                    if constexpr (!(SM110A_MXFP4_FC2_PDL && SM110A_MXFP4_FC2_PDL_WAIT_BEFORE_A)) {
                        tma_load_4d<kClusterM>(b_smem, &b_tensor_map, off_k,
                            tile_col + cta_rank * (kTileN / kClusterM), 0, task.expert,
                            tma_transaction_mbar_addr);
                    }
                    if constexpr (kClusterM == 1) {
                        if (sfa_aligned) {
                            tma_load_4d<1>(sfa_smem, &sfa_tensor_map,
                                0, 0, off_k32, first_sfa_tile,
                                tma_transaction_mbar_addr);
                        } else {
                            tma_load_4d<1>(sfa_staging_smem, &sfa_tensor_map,
                                0, 0, off_k32, first_sfa_tile,
                                tma_transaction_mbar_addr);
                        }
                        if (!sfa_aligned && sfa_source_tiles == 2) {
                            tma_load_4d<1>(sfa_staging_smem + kSFASmemBytes,
                                &sfa_tensor_map, 0, 0, off_k32,
                                first_sfa_tile + 1, tma_transaction_mbar_addr);
                        }
                    } else {
                        // Each CTA owns a different 128-row A slice. Issue a
                        // local CTA-group::1 SFA TMA transaction and aggregate
                        // the transformed completion separately on rank 0.
                        const int local_sfa_mbar_addr =
                            base_sfa_tma_mbar_addr + producer_stage * sizeof(uint64_t);
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
                            "_, [%0], %1;" :: "r"(local_sfa_mbar_addr),
                            "r"((sfa_aligned ? 1 : sfa_source_tiles) * kSFASmemBytes)
                            : "memory");
                        if (sfa_aligned) {
                            tma_load_4d<1>(sfa_smem, &sfa_tensor_map,
                                0, 0, off_k32, first_sfa_tile,
                                local_sfa_mbar_addr);
                        } else {
                            tma_load_4d<1>(sfa_staging_smem, &sfa_tensor_map,
                                0, 0, off_k32, first_sfa_tile,
                                local_sfa_mbar_addr);
                            if (sfa_source_tiles == 2) {
                                tma_load_4d<1>(sfa_staging_smem + kSFASmemBytes,
                                    &sfa_tensor_map, 0, 0, off_k32,
                                    first_sfa_tile + 1, local_sfa_mbar_addr);
                            }
                        }
                    }
                    if constexpr (!(SM110A_MXFP4_FC2_PDL && SM110A_MXFP4_FC2_PDL_WAIT_BEFORE_A)) {
                        for (int scale_tile = 0; scale_tile < kSFBScaleTileCount; ++scale_tile) {
                            const int sfb_tile_smem =
                                sfb_smem + scale_tile * Config::kMmaStepsPerStage * 512;
                            const int sfb_tile_index =
                                task.expert * (n / 128) + tile_col / 128 + scale_tile;
                            if constexpr (kClusterM == 1) {
                                tma_load_4d<1>(sfb_tile_smem, &sfb_tensor_map, 0, 0, off_k32,
                                    sfb_tile_index, tma_transaction_mbar_addr);
                            } else if (cta_rank == 0) {
                                constexpr int16_t cta_mask = (1 << kClusterM) - 1;
                                tma_load_4d_multicast<kClusterM>(sfb_tile_smem, &sfb_tensor_map,
                                    0, 0, off_k32, sfb_tile_index, tma_transaction_mbar_addr,
                                    cta_mask);
                            }
                        }
                    }

                    producer_stage = (producer_stage + 1) % kStageCount;
                    if (producer_stage == 0) {
                        producer_mma_phase ^= 1;
                    }
                }
            }
        }

        // SFA transformation is kept off the MMA warp. cta_group::1 waits for
        // the staged TMA source and reconstructs only unaligned slices;
        // cta_group::2 gathers each rank's physical rows independently.
        if (warp_id == kSfaWarp) {
            const int lane = tid & 31;
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                const ExpertTileTask task = tasks[cluster_bid];
                const int cta_row = task.row_start + task.tile_row + cta_rank * kTileM;
                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    const int a_smem = mainloop_smem + sfa_stage * kStageSmemBytes;
                    const int sfa_smem = a_smem + kASmemBytes + kBSmemBytes;
                    const int sfa_staging_smem =
                        sfa_smem + kSFASmemBytes + kSFBSmemBytes;
                    if constexpr (kClusterM == 1) {
                        mbarrier_wait(base_tma_mbar_addr + sfa_stage * sizeof(uint64_t),
                            sfa_tma_phase);
                        if ((cta_row & (kTileM - 1)) != 0) {
                            auto *dst = reinterpret_cast<uint32_t *>(
                                dynamic_smem + (sfa_smem - mainloop_smem));
                            auto *src = reinterpret_cast<const uint32_t *>(
                                dynamic_smem + (sfa_staging_smem - mainloop_smem));
                            const int first_source_tile = cta_row >> 7;
                            const int valid_rows = min(kTileM,
                                max(0, task.row_count - task.tile_row));
                            for (int row = lane; row < valid_rows; row += kWarpSize) {
                                const int source_row = cta_row + row;
                                const int source_tile =
                                    (source_row >> 7) - first_source_tile;
                                const int source_in_tile =
                                    (source_row & 31) * 4 + ((source_row >> 5) & 3);
                                const int dst_in_tile =
                                    (row & 31) * 4 + ((row >> 5) & 3);
                                for (int mma_k = 0; mma_k < kMmaStepsPerStage; ++mma_k) {
                                    dst[mma_k * 128 + dst_in_tile] =
                                        src[source_tile * (kSFASmemBytes / 4) +
                                            mma_k * 128 + source_in_tile];
                                }
                            }
                        }
                    } else {
                        mbarrier_wait(
                            base_sfa_tma_mbar_addr + sfa_stage * sizeof(uint64_t),
                            sfa_tma_phase);
                        const int valid_rows = min(kTileM,
                            max(0, task.row_count - task.tile_row - cta_rank * kTileM));
                        if ((cta_row & (kTileM - 1)) != 0) {
                            auto *dst = reinterpret_cast<uint32_t *>(
                                dynamic_smem + (sfa_smem - mainloop_smem));
                            auto *src = reinterpret_cast<const uint32_t *>(
                                dynamic_smem + (sfa_staging_smem - mainloop_smem));
                            const int first_source_tile = cta_row >> 7;
                            for (int row = lane; row < valid_rows; row += kWarpSize) {
                                const int source_row = cta_row + row;
                                const int source_tile =
                                    (source_row >> 7) - first_source_tile;
                                const int source_in_tile =
                                    (source_row & 31) * 4 + ((source_row >> 5) & 3);
                                const int dst_in_tile =
                                    (row & 31) * 4 + ((row >> 5) & 3);
                                for (int mma_k = 0; mma_k < kMmaStepsPerStage; ++mma_k) {
                                    dst[mma_k * 128 + dst_in_tile] =
                                        src[source_tile * (kSFASmemBytes / 4) +
                                            mma_k * 128 + source_in_tile];
                                }
                            }
                        }
                    }
                    if constexpr (kCompactWeightScales) {
                        // Both CTAs expand their local multicast SFB before
                        // publishing the existing paired-operand ready event.
                        if constexpr (kClusterM == 1) {
                            mbarrier_wait(base_tma_mbar_addr + sfa_stage * sizeof(uint64_t),
                                sfa_tma_phase);
                        } else if (cta_rank == 0) {
                            mbarrier_wait(base_tma_mbar_addr + sfa_stage * sizeof(uint64_t),
                                sfa_tma_phase);
                            if (elect_sync())
                                mbarrier_arrive_on_cluster_cta(
                                    base_weight_ready_mbar_addr + sfa_stage * sizeof(uint64_t), 1);
                        } else {
                            mbarrier_wait(base_weight_ready_mbar_addr + sfa_stage * sizeof(uint64_t),
                                sfa_tma_phase);
                        }
                        auto *scales = reinterpret_cast<uint8_t *>(__cvta_shared_to_generic(
                            sfa_smem + kSFASmemBytes));
                        for (int tile = 0; tile < kSFBScaleTileCount; ++tile)
                            unpack_mxfp4_weight_scales<kMmaStepsPerStage>(
                                scales + tile * kMmaStepsPerStage * 512);
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_on_cluster_cta(
                            base_sfa_mbar_addr + sfa_stage * sizeof(uint64_t), 0);
                    }
                    sfa_stage = (sfa_stage + 1) % kStageCount;
                    if (sfa_stage == 0) {
                        sfa_tma_phase ^= 1;
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
                    if constexpr (kAccumulatorOverlapCols != 0) {
                        if (consumer_has_previous_overlap) {
                            // The adjacent accumulator stage aliases these 16
                            // columns.  Wait only until the previous epilogue
                            // has copied that overlap into shared memory; its
                            // disjoint columns may continue finalizing.
                            const int previous_stage =
                                (consumer_mainloop_stage + kTmemStageCount - 1) %
                                kTmemStageCount;
                            mbarrier_wait(base_overlap_mbar_addr +
                                              previous_stage * sizeof(uint64_t),
                                          consumer_overlap_phase);
                            if (previous_stage == kTmemStageCount - 1) {
                                consumer_overlap_phase ^= 1;
                            }
                        }
                        consumer_has_previous_overlap = true;
                    }
                }

                const ExpertTileTask task = tasks[cluster_bid];
                const int tile_col = task.tile_n * kTileN;
                const int stage_tmem_addr =
                    tmem_addr + consumer_mainloop_stage * kAccumulatorStageStride;
                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    const int tma_mbar_addr =
                        base_tma_mbar_addr + consumer_stage * sizeof(uint64_t);
                    const int mma_mbar_addr =
                        base_mma_mbar_addr + consumer_stage * sizeof(uint64_t);

                    mbarrier_wait(tma_mbar_addr, consumer_tma_phase);
                    mbarrier_wait(
                        base_sfa_mbar_addr + consumer_stage * sizeof(uint64_t),
                        consumer_tma_phase);
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
                            // SFB is staged in 128-column blocks.  A 64-column
                            // output tile in the upper half starts two TMEM
                            // scale columns into that block.
                            const int sfb_tmem_addr =
                                scale_addr + 4 + (tile_col % 128) / 32;
                            const int accumulate = tile_k != 0 || mma_k != 0;
                            tcgen05_mma_mxfp4_w4a8<kClusterM>(stage_tmem_addr,
                                make_k_major_128b_desc(a_smem + mma_k * 32),
                                make_k_major_128b_desc(b_smem + mma_k * 32),
                                instruction_desc, scale_addr, sfb_tmem_addr, accumulate);
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
                    if constexpr (kPersistent || kDynamicClc) {
                        consumer_epilogue_phase ^= 1;
                    }
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
                const int tile_col = tile_n * kTileN;

                auto release_tmem = [&]() {
                    if constexpr (kPersistent || kDynamicClc) {
                        if (elect_sync()) {
                            // The MMA warp lives on cluster rank 0.  Aggregate
                            // all epilogue-warp releases from both CTAs onto
                            // rank 0's barrier so cta_group::2 cannot reuse an
                            // overlapping accumulator stage early.
                            mbarrier_arrive_on_cluster_cta(
                                base_epilogue_mbar_addr +
                                    epilogue_mainloop_stage * sizeof(uint64_t),
                                0);
                        }
                    }
                };

                auto release_overlap = [&]() {
                    if constexpr ((kPersistent || kDynamicClc) &&
                                  kAccumulatorOverlapCols != 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_on_cluster_cta(
                                base_overlap_mbar_addr +
                                    epilogue_mainloop_stage * sizeof(uint64_t),
                                0);
                        }
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
                    token_idx = kDeterministic ? expanded_id : expanded_id / topk;
                    if constexpr (kWeightInFc1) {
                        router_scale = 1.0f;
                    } else {
                        router_scale = __bfloat162float(routing_weights[expanded_id]);
                    }
                }
                if (valid_rows == 0) {
                    // A 2-CTA M=256 tile frequently has no rows in rank 1 for
                    // sparse expert buckets.  Its accumulator values are
                    // unobservable, so release both handshakes without doing
                    // four slabs of TMEM loads and shared/global reduction.
                    release_overlap();
                    release_tmem();
                } else {
                    auto *stage_smem = reinterpret_cast<nv_bfloat16 *>(
                        dynamic_smem + kMainloopSmemBytes);
                    auto stage_tmem_groups = [&](int reduce_n, int group_begin,
                                                 int group_end) {
                        for (int n_group = group_begin; n_group < group_end; ++n_group) {
                            float values[8];
                            const int tmem_row = cta_rank * kTileM + warp_id * kWarpSize;
                            const int tmem_col =
                                epilogue_mainloop_stage * kAccumulatorStageStride +
                                reduce_n + n_group * 8;
                            const int load_addr = tmem_addr + (tmem_row << 16) + tmem_col;
                            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                                         "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]),
                                  "=f"(values[3]), "=f"(values[4]), "=f"(values[5]),
                                  "=f"(values[6]), "=f"(values[7])
                                : "r"(load_addr));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            if (tid < kTileM) {
#pragma unroll
                                for (int j = 0; j < 8; ++j) {
                                    stage_smem[tid * SM110A_MXFP4_FC2_REDUCE_TILE_N +
                                               n_group * 8 + j] =
                                        __float2bfloat16_rn(values[j] * router_scale);
                                }
                            }
                        }
                        asm volatile("bar.sync 1, 128;" ::: "memory");
                    };
                    auto reduce_smem_groups = [&](int reduce_n, int group_begin,
                                                  int group_end) {
                        fence_smem_for_async_proxy();
                        if (valid_row) {
                            auto *source = stage_smem +
                                tid * SM110A_MXFP4_FC2_REDUCE_TILE_N + group_begin * 8;
                            auto *destination = final_output +
                                static_cast<size_t>(token_idx) * n +
                                tile_col + reduce_n + group_begin * 8;
                            if constexpr (kDeterministic) {
                                // Each expanded assignment owns a unique row:
                                // no atomic accumulation or schedule-dependent rounding.
                                for (int group = group_begin; group < group_end; ++group) {
                                    const int offset = (group - group_begin) * 8;
                                    *reinterpret_cast<uint4*>(destination + offset) =
                                        *reinterpret_cast<const uint4*>(source + offset);
                                }
                            } else {
                                bulk_reduce_add_bf16(destination, source,
                                    (group_end - group_begin) * 8 * sizeof(nv_bfloat16));
                            }
                        }
                        bulk_commit_group();
                        bulk_wait_group_read_0();
                        asm volatile("bar.sync 1, 128;" ::: "memory");
                    };
                    auto stage_and_reduce = [&](int reduce_n) {
                        stage_tmem_groups(reduce_n, 0,
                            SM110A_MXFP4_FC2_REDUCE_TILE_N / 8);
                        reduce_smem_groups(reduce_n, 0,
                            SM110A_MXFP4_FC2_REDUCE_TILE_N / 8);
                    };

                    constexpr int kGroupsPerReduce =
                        SM110A_MXFP4_FC2_REDUCE_TILE_N / 8;
                    if constexpr (kAccumulatorOverlapCols != 0) {
                        static_assert(kAccumulatorOverlapCols == 16);
                        const int overlap_reduce_n = epilogue_mainloop_stage == 0
                            ? kTileN - SM110A_MXFP4_FC2_REDUCE_TILE_N
                            : 0;
                        const int overlap_group_begin = epilogue_mainloop_stage == 0
                            ? kGroupsPerReduce - kAccumulatorOverlapCols / 8
                            : 0;
                        const int overlap_group_end =
                            overlap_group_begin + kAccumulatorOverlapCols / 8;
                        stage_tmem_groups(overlap_reduce_n, overlap_group_begin,
                                          overlap_group_end);
                        release_overlap();

                        // Complete the same reduce slab after releasing the
                        // aliased TMEM columns, then issue one full-width global
                        // reduction instead of splitting it into 16+48 columns.
                        if (overlap_group_begin != 0) {
                            stage_tmem_groups(overlap_reduce_n, 0,
                                              overlap_group_begin);
                        }
                        if (overlap_group_end != kGroupsPerReduce) {
                            stage_tmem_groups(overlap_reduce_n, overlap_group_end,
                                              kGroupsPerReduce);
                        }
                        reduce_smem_groups(overlap_reduce_n, 0, kGroupsPerReduce);

                        for (int reduce_n = 0; reduce_n < kTileN;
                             reduce_n += SM110A_MXFP4_FC2_REDUCE_TILE_N) {
                            if (reduce_n != overlap_reduce_n) {
                                stage_and_reduce(reduce_n);
                            }
                        }
                        release_tmem();
                    } else {
                        for (int reduce_n = 0; reduce_n < kTileN;
                             reduce_n += SM110A_MXFP4_FC2_REDUCE_TILE_N) {
                            stage_and_reduce(reduce_n);
                        }
                        release_tmem();
                    }
                }
                bulk_wait_group_0();
                asm volatile("bar.sync 1, 128;" ::: "memory");

                epilogue_mainloop_stage =
                    (epilogue_mainloop_stage + 1) % kTmemStageCount;
                if (epilogue_mainloop_stage == 0) {
                    epilogue_mainloop_phase ^= 1;
                }
            }
        }

        if constexpr (kUseTaskPipeline) {
            // The four epilogue warps have already reconverged at bar.sync 1,
            // so one elected lane per CTA is sufficient to release the task
            // slot.  The 2-CTA barrier counts one arrival from each rank.
            if (warp_id == 0 && elect_sync()) {
                const int free_addr = base_task_free_mbar_addr +
                    task_pipe_stage * sizeof(uint64_t);
                if constexpr (kClusterM == 1) {
                    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::
                        "r"(free_addr) : "memory");
                } else {
                    mbarrier_arrive_release_cluster_on_cluster_cta(
                        free_addr, 0);
                }
            }
            task_pipe_stage = (task_pipe_stage + 1) % kTaskPipelineStageCount;
            if (task_pipe_stage == 0) task_ready_phase ^= 1;
        }

        if constexpr (kDynamicClc && !kUseTaskPipeline) {
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

template <int kTileM, int kTileN, int kTileK, int kClusterM,
          bool kWeightInFc1, bool kDeterministic>
void launch_mxfp4_fc2_impl(
    const uint8_t *d_a,
    const uint8_t *d_b,
    const uint8_t *d_sfa,
    const uint8_t *d_sfb,
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
    int max_tasks, int max_active_clusters,
    cudaStream_t stream) {
    using Config = Mxfp4Fc2Config<kTileM, kTileN, kTileK, kClusterM>;
    constexpr int kDynamicSmemBytes = Config::kDynamicSmemBytes;
    constexpr int kBlockThreads = kWarpSize * kBaseWarpCount;
    if (m % 128 || n % kTileN != 0 || n % 128 || k % kTileK != 0) {
        return;
    }

    CUtensorMap a_tensor_map;
    CUtensorMap b_tensor_map;
    CUtensorMap sfa_tensor_map;
    CUtensorMap sfb_tensor_map;
    bool tensor_maps_ok;
    tensor_maps_ok =
        init_a_tensor_map<kTileM, kTileK>(&a_tensor_map, d_a, m, k) &&
        init_b_tensor_map<kTileN, kTileK, kClusterM>(&b_tensor_map, d_b, experts, n, k) &&
        init_scale_tensor_map(&sfa_tensor_map, d_sfa, m / 128,
            k / 32, kTileK / 32) &&
        init_scale_tensor_map<kCompactWeightScales>(&sfb_tensor_map, d_sfb, experts * (n / 128),
            k / 32, kTileK / 32);
    if (!tensor_maps_ok) {
        return;
    }

    auto *tasks = reinterpret_cast<const ExpertTileTask *>(d_tasks);
    auto *task_count = d_state;
    auto *next_cluster_tile = d_state + 1;
    if constexpr (!SM110A_MXFP4_FC2_EXTERNAL_PREPARE) {
        cudaMemsetAsync(next_cluster_tile, 0, sizeof(int32_t), stream);
    }

    const int persistent_clusters =
        (max_tasks + kPersistentTileCount - 1) / kPersistentTileCount;
    auto single_tile_kernel = mxfp4_fc2_kernel<kTileM,
        kTileN,
        kTileK,
        kClusterM,
        true,
        false,
        1, kWeightInFc1, kDeterministic>;
    auto packet_kernel = mxfp4_fc2_kernel<kTileM,
        kTileN,
        kTileK,
        kClusterM,
        true,
        false,
        kTaskPacketTileCount, kWeightInFc1, kDeterministic>;
    if constexpr (kDynamicSmemBytes > 48 * 1024) {
        cudaFuncSetAttribute(single_tile_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kDynamicSmemBytes);
        cudaFuncSetAttribute(packet_kernel,
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
        cudaOccupancyMaxActiveClusters(
            &resident_clusters, single_tile_kernel, &launch_config);
    if (occupancy_status != cudaSuccess || resident_clusters <= 0) {
        int device = 0;
        int sm_count = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        resident_clusters = sm_count / kClusterM > 0 ? sm_count / kClusterM : 1;
    }
    const int cluster_limit = max_active_clusters > 0 ? max_active_clusters : resident_clusters;
    const int launched_clusters = cluster_limit < persistent_clusters
        ? cluster_limit : persistent_clusters;
    // Packetized scheduling wins once each expert has enough row tiles to keep
    // the role-specialized pipeline in steady state.  Keep the compile-time
    // single-tile specialization for short workloads so the large-M
    // optimization adds no device-side branch or tail-granularity regression.
    const bool use_task_packets =
        m / experts >= kMinRowsPerExpertForTaskPacket;
    auto kernel = use_task_packets ? packet_kernel : single_tile_kernel;
    launch_config.gridDim = dim3(kClusterM * launched_clusters, 1, 1);
    sm110a::launch_pdl<SM110A_MXFP4_FC2_PDL>(&launch_config, kernel,
        d_sfa, a_tensor_map, b_tensor_map, sfa_tensor_map, sfb_tensor_map,
        tasks, task_count,
        next_cluster_tile, grouped_assignments, routing_weights, final_output, n, k, topk);
}

void launch_mxfp4_fc2_entry(
    const uint8_t *a, const uint8_t *b, const uint8_t *sfa, const uint8_t *sfb,
    const int32_t *grouped_assignments, const int32_t *tasks, int32_t *state,
    __nv_bfloat16 *c, const __nv_bfloat16 *routing_weights,
    int tokens, int m, int experts, int n, int k, int topk, int max_tasks, int max_active_clusters,
    cudaStream_t stream) {
    if (!a || !b || !sfa || !sfb || !grouped_assignments || !tasks || !state || !c ||
        !routing_weights || tokens <= 0 || m <= 0 || experts <= 0 || n <= 0 || k <= 0 ||
        topk <= 0 || max_tasks <= 0)
        return;
    launch_mxfp4_fc2_impl<SM110A_MXFP4_FC2_TILE_M,
        SM110A_MXFP4_FC2_TILE_N, SM110A_MXFP4_FC2_TILE_K,
        SM110A_MXFP4_FC2_CLUSTER_M,
        (SM110A_MXFP4_FC2_WEIGHT_IN_FC1 != 0),
        (SM110A_MXFP4_FC2_DETERMINISTIC != 0)>(
        a, b, sfa, sfb, tasks, state, grouped_assignments, routing_weights, c,
        m, experts, n, k, topk, max_tasks, max_active_clusters, stream);
}

__global__ void mxfp4_fc2_finalize_deterministic(
    const __nv_bfloat16* assignments, __nv_bfloat16* output,
    int tokens, int n, int topk) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(tokens) * n) return;
    const int token = index / n;
    const int column = index % n;
    float sum = 0.0f;
    for (int rank = 0; rank < topk; ++rank) {
        sum = __fadd_rn(sum, __bfloat162float(
            assignments[(static_cast<size_t>(token) * topk + rank) * n + column]));
    }
    output[index] = __float2bfloat16_rn(sum);
}

} // namespace sm110a::mxfp4_fc2

using namespace sm110a::mxfp4_fc2;

void launch_mxfp4_fc2_main(
    const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
    const uint8_t* sfb, const int32_t* grouped_assignments,
    const int32_t* tasks, int32_t* state,
    __nv_bfloat16* c, const __nv_bfloat16* routing_weights, int tokens,
    int m, int experts, int n, int k, int topk, int max_tasks, int max_active_clusters,
    cudaStream_t stream, __nv_bfloat16* assignment_scratch) {
    if (!a || !b || !sfa || !sfb || !grouped_assignments || !tasks ||
        !state || !c || !routing_weights || tokens <= 0 ||
        m <= 0 || experts <= 0 || n <= 0 || k <= 0 || topk <= 0 ||
        max_tasks <= 0) {
        return;
    }
    if constexpr (SM110A_MXFP4_FC2_DETERMINISTIC != 0) {
        if (!assignment_scratch) return;
    } else if constexpr (!SM110A_MXFP4_FC2_EXTERNAL_PREPARE) {
        // BF16 cp.reduce requires zeroed token-major output.
        cudaMemsetAsync(c, 0, static_cast<size_t>(tokens) * n * sizeof(__nv_bfloat16), stream);
    }
    launch_mxfp4_fc2_entry(a, b, sfa, sfb, grouped_assignments, tasks,
                           state, SM110A_MXFP4_FC2_DETERMINISTIC ? assignment_scratch : c,
                           routing_weights, tokens, m, experts, n, k,
                           topk, max_tasks, max_active_clusters, stream);
    if constexpr (SM110A_MXFP4_FC2_DETERMINISTIC != 0) {
        constexpr int threads = 256;
        const int blocks = (static_cast<size_t>(tokens) * n + threads - 1) / threads;
        mxfp4_fc2_finalize_deterministic<<<blocks, threads, 0, stream>>>(
            assignment_scratch, c, tokens, n, topk);
    }
}

extern "C" void sm110a_mxfp4_fc2_launch(
    const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
    const uint8_t* sfb, const int32_t* grouped_assignments,
    const int32_t* tasks, int32_t* state,
    __nv_bfloat16* c, const __nv_bfloat16* routing_weights, int tokens,
    int m, int experts, int n, int k, int topk, int max_tasks, int max_active_clusters,
    cudaStream_t stream, __nv_bfloat16* assignment_scratch) {
    launch_mxfp4_fc2_main(a, b, sfa, sfb, grouped_assignments, tasks,
                          state, c, routing_weights, tokens, m, experts, n, k,
                          topk, max_tasks, max_active_clusters, stream, assignment_scratch);
}
