#ifndef SM110A_FP8_SHARED_FC1_SWIGLU_LIMIT
#define SM110A_FP8_SHARED_FC1_SWIGLU_LIMIT 0.0f
#endif
#include "pdl.h"
#ifndef SM110A_FP8_SHARED_FC1_PDL
#define SM110A_FP8_SHARED_FC1_PDL 0
#endif
// Dense shared-expert FC1. FP8 weights are interleaved gate[64], up[64].
// Reuses the FP8 GEMM pipeline design in this independent translation unit.
// The epilogue rounds projections to BF16, evaluates clamp/SwiGLU in FP32,
// then stores BF16 [M,N/2] without materializing either GEMM output.

#include <cassert>
#include <cstdint>
#include <cudaTypedefs.h> // required for CUtensorMap

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tuple>

#include "common.h"

// Keep the pipeline implementation structurally aligned with FP8 x MXFP4
// while giving this independently-built JIT target a distinct configuration
// ABI.
#ifndef SM110A_FP8_SHARED_FC1_TILE_M
#define SM110A_FP8_SHARED_FC1_TILE_M 128
#endif
#ifndef SM110A_FP8_SHARED_FC1_TILE_N
#define SM110A_FP8_SHARED_FC1_TILE_N 128
#endif
#ifndef SM110A_FP8_SHARED_FC1_TILE_K
#define SM110A_FP8_SHARED_FC1_TILE_K 128
#endif
#ifndef SM110A_FP8_SHARED_FC1_CLUSTER_M
#define SM110A_FP8_SHARED_FC1_CLUSTER_M 2
#endif
#ifndef SM110A_FP8_SHARED_FC1_EPILOGUE_BLOCK_N
#define SM110A_FP8_SHARED_FC1_EPILOGUE_BLOCK_N 32
#endif
#ifndef SM110A_FP8_SHARED_FC1_SPLIT_K
#define SM110A_FP8_SHARED_FC1_SPLIT_K 1
#endif
#ifndef SM110A_FP8_SHARED_FC1_USE_TMA_PREFETCH
#define SM110A_FP8_SHARED_FC1_USE_TMA_PREFETCH 0
#endif

namespace sm110a::fp8_shared_fc1 {

constexpr int kMmaK = 32;
constexpr int kWarpSize = 32;
constexpr int kWarpCount = 6;
constexpr int kBlockThreads = kWarpSize * kWarpCount;
constexpr int kEpilogueStageCount = 2;
constexpr int kTmemColumnCapacity = 512;
// cudaDevAttrMaxSharedMemoryPerBlockOptin on SM110A/Thor.  Dynamic shared
// memory, the launch reserve, and statically allocated barriers all consume
// this same per-block budget.
constexpr int kMaxOptInSmemBytes = 232448;
constexpr int kMbarrierSmemBytes = 256;
constexpr int kStaticBarrierSmemBytes = 1024;
constexpr bool kPrefetchTmaDesc = true;
// Keep the TMA load pipeline's worth of future K tiles resident in L2.  The
// producer issues an initial window and then replenishes it after every
// regular TMA load, so every K tile is hinted exactly once.
constexpr bool kPrefetchTmaData = SM110A_FP8_SHARED_FC1_USE_TMA_PREFETCH != 0;
constexpr int kSmCount = 20;

constexpr int kMacroSwizzleM = 8;
constexpr int kMacroSwizzleN = 8;
constexpr bool kMacroSwizzleMFast = true;

template <int kTileM, int kTileN, int kTileK, int kClusterM, int kEpilogueBlockN, int kSplitK>
struct KernelConfig {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert(
        (kSplitK == 1 || kSplitK == 2 || kSplitK == 4 || kSplitK == 8) && kClusterM * kSplitK <= 8);
    static_assert(kTileM == 128);
    static_assert(kTileN == 64 || kTileN == 128 || kTileN == 256);
    static_assert(kTileK == 64 || kTileK == 128);
    static_assert(kEpilogueBlockN == 32 || kEpilogueBlockN == 64 || kEpilogueBlockN == 128);
    static_assert(kTileN % kEpilogueBlockN == 0);
    static_assert(kTileN % kClusterM == 0);
    static_assert(kTileK % 32 == 0);
    static_assert(kTileK % kMmaK == 0);

    static constexpr int kClusterTileM = kClusterM * kTileM;
    static constexpr int kATmaBytes = kTileM * kTileK;
    static constexpr int kBTmaBytes = (kTileN / kClusterM) * kTileK;
    // A 128B swizzle always reserves a full 128B span per row, even when a
    // TileK=64 transfer only populates 64B of that span.
    static constexpr int kASmemBytes = kTileM * 128;
    static constexpr int kBSmemBytes = (kTileN / kClusterM) * 128;
    static constexpr int kMmaStepsPerStage = kTileK / kMmaK;
    // K=32 mxf8f6f4 supplies 128x1 live E8M0 scales.  Keep the 128x4 physical
    // tile (512 B) required by tcgen05.cp.32x128b.warpx4.
    static constexpr int kSFASmemBytes = kTileM * (kTileK / 8);
    // Scale storage is blocked in 128-row, 512-byte units.  TileN=64 loads
    // its containing 128-row tile and selects the appropriate half in TMEM.
    static constexpr int kSFBScaleTileCount = (kTileN + 127) / 128;
    static constexpr int kSFBSmemBytes = kMmaStepsPerStage * kSFBScaleTileCount * 512;
    static constexpr int kPerStagePayloadBytes =
        kASmemBytes + kBSmemBytes + kSFASmemBytes + kSFBSmemBytes;
    static constexpr int kPerStageSmemBytes = (kPerStagePayloadBytes + 1023) & ~1023;
    static constexpr int kEpilogueStageBytes =
        kTileM * kEpilogueBlockN * static_cast<int>(sizeof(nv_bfloat16));
    static constexpr int kEpilogueSmemBytes = kEpilogueStageCount * kEpilogueStageBytes;
    static constexpr int kReductionSmemBytes =
        kSplitK == 1 ? 0 : kTileM * kEpilogueBlockN * sizeof(float);
    // Cap the ring at six stages, while reducing it for wider K tiles so the
    // mainloop and epilogue remain within Blackwell's per-CTA SMEM budget.
    static constexpr int kSmemLimitedStageCount =
        (kMaxOptInSmemBytes - kStaticBarrierSmemBytes - kMbarrierSmemBytes - kEpilogueSmemBytes -
            kReductionSmemBytes) /
        kPerStageSmemBytes;
    static constexpr int kStageCount = kSmemLimitedStageCount < 6 ? kSmemLimitedStageCount : 6;
    static_assert(kStageCount >= 2);
    // A uses one live E8M0 subcolumn but its 32x128b transfer occupies a
    // four-column TMEM tile.  Keep the three padding columns before B so both
    // tcgen05.cp destinations remain 4-column aligned.
    static constexpr int kScaleColsPerMma = 4 + 4 * kSFBScaleTileCount;
    // Narrow-N kernels have ample TMEM after their two accumulator stages.
    // Keep every in-flight SMEM stage and K=32 slab's scales in a distinct
    // slot. The mma_mbar handshake prevents a stage's SMEM and TMEM scale
    // slots from being reused until its asynchronous MMA has completed.
    // TileN=256 is capacity constrained and reuses a single scale slot.
    static constexpr int kScaleTmemSlotCount = kTileN < 256 ? kStageCount * kMmaStepsPerStage : 1;
    static constexpr int kScaleTmemSlotStride = kScaleColsPerMma;
    static constexpr int kScaleTmemCols = kScaleTmemSlotStride * kScaleTmemSlotCount;
    // A 16-column overlap is aligned for tcgen05.ld (.x8 values per thread)
    // and fits in every supported epilogue block size.  At TileN=256 this
    // changes the footprint from 2*256 + 12 (impossible) to
    // 2*256 - 16 + 12 = 508 columns.
    static constexpr int kAccumulatorOverlapCols = kTileN == 256 ? 16 : 0;
    static constexpr int kAccumulatorStageCount = 2;
    static constexpr int kAccumulatorStageStride = kTileN - kAccumulatorOverlapCols;
    static constexpr int kAccumulatorTmemCols =
        kAccumulatorStageCount * kTileN - kAccumulatorOverlapCols;
    static_assert(kAccumulatorTmemCols + kScaleTmemCols <= kTmemColumnCapacity);
    static_assert(kAccumulatorOverlapCols == 0 ||
                  (kAccumulatorOverlapCols == 16 && kEpilogueBlockN >= kAccumulatorOverlapCols));
    static constexpr int kMainloopSmemBytes = kStageCount * kPerStageSmemBytes;
    static constexpr int kDynamicSmemBytes =
        kMainloopSmemBytes + kEpilogueSmemBytes + kReductionSmemBytes;
    static_assert(
        kStaticBarrierSmemBytes + kDynamicSmemBytes + kMbarrierSmemBytes <= kMaxOptInSmemBytes);
};

template <int CTA_GROUP = 1>
__device__ inline void tma_3d_gmem2smem(
    int dst,
    const void *tmap_ptr,
    int x,
    int y,
    int z,
    int _base_tma_mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6 "
        "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
        "l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(z),
        "r"(_base_tma_mbar_addr),
        "n"(CTA_GROUP)
        : "memory");
}

template <int CTA_GROUP = 1>
__device__ inline void tma_4d_gmem2smem(
    int dst,
    const void *tmap_ptr,
    int x,
    int y,
    int z,
    int w,
    int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%7 "
        "[%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
        "l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(z),
        "r"(w),
        "r"(mbar_addr),
        "n"(CTA_GROUP)
        : "memory");
}

template <int CTA_GROUP = 1>
__device__ inline void tma_4d_gmem2smem_multicast(
    int dst,
    const void *tmap_ptr,
    int x,
    int y,
    int z,
    int w,
    int mbar_addr,
    int16_t cta_mask) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes."
                 "multicast::cluster.cta_group::%8 "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(dst),
        "l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(z),
        "r"(w),
        "r"(mbar_addr),
        "h"(cta_mask),
        "n"(CTA_GROUP)
        : "memory");
}

__device__ inline void tma_3d_prefetch_l2(
    const void *tmap_ptr,
    int x,
    int y,
    int z) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(z)
        : "memory");
}

__device__ inline void tma_4d_prefetch_l2(
    const void *tmap_ptr,
    int x,
    int y,
    int z,
    int w) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" ::"l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(z),
        "r"(w)
        : "memory");
}

__device__ inline uint64_t make_epilogue_cache_policy() {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

__device__ inline void tma_3d_smem2gmem(
    uint64_t policy,
    int src,
    const void *tmap_ptr,
    int x,
    int y,
    int z) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint "
                 "[%0, {%1, %2, %3}], [%4], %5;" ::"l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(z),
        "r"(src),
        "l"(policy)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ inline void tma_2d_smem2gmem(
    uint64_t policy,
    int src,
    const void *tmap_ptr,
    int x,
    int y) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint "
                 "[%0, {%1, %2}], [%3], %4;" ::"l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(src),
        "l"(policy)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// Keep these waits as separate functions for Nsight Compute source
// correlation.  They intentionally contain identical PTX: only the
// producer/consumer role differs, and the separate bodies let lineinfo
// distinguish the four pipeline dependencies.
__device__ inline void wait_mma_slot(
    int mbar_addr,
    int phase) {
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni DONE;\n\t"
                 "bra.uni LAB_WAIT;\n\t"
                 "DONE:\n\t"
                 "}" ::"r"(mbar_addr),
        "r"(phase),
        "r"(ticks)
        : "memory");
}

__device__ inline void wait_tma(
    int mbar_addr,
    int phase) {
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni DONE;\n\t"
                 "bra.uni LAB_WAIT;\n\t"
                 "DONE:\n\t"
                 "}" ::"r"(mbar_addr),
        "r"(phase),
        "r"(ticks)
        : "memory");
}

__device__ inline void wait_epilogue(
    int mbar_addr,
    int phase) {
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni DONE;\n\t"
                 "bra.uni LAB_WAIT;\n\t"
                 "DONE:\n\t"
                 "}" ::"r"(mbar_addr),
        "r"(phase),
        "r"(ticks)
        : "memory");
}

__device__ inline void wait_mainloop(
    int mbar_addr,
    int phase) {
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni DONE;\n\t"
                 "bra.uni LAB_WAIT;\n\t"
                 "DONE:\n\t"
                 "}" ::"r"(mbar_addr),
        "r"(phase),
        "r"(ticks)
        : "memory");
}

template <int kClusterM>
__device__ inline void tcgen05_cp_scale(
    int taddr,
    uint64_t smem_desc) {
    if constexpr (kClusterM == 1) {
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(smem_desc)
            : "memory");
    } else {
        asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(smem_desc)
            : "memory");
    }
}

template <int kClusterM>
__device__ inline void tcgen05_mma_mxf8f6f4(
    int taddr,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int sfa_taddr,
    int sfb_taddr,
    int enable_input_d) {
    if constexpr (kClusterM == 1) {
        asm volatile("{\n\t"
                     ".reg .pred p;\n\t"
                     "setp.ne.b32 p, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
                     "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
                     "}" ::"r"(taddr),
            "l"(a_desc),
            "l"(b_desc),
            "r"(i_desc),
            "r"(sfa_taddr),
            "r"(sfb_taddr),
            "r"(enable_input_d)
            : "memory");
    } else {
        asm volatile("{\n\t"
                     ".reg .pred p;\n\t"
                     "setp.ne.b32 p, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X "
                     "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
                     "}" ::"r"(taddr),
            "l"(a_desc),
            "l"(b_desc),
            "r"(i_desc),
            "r"(sfa_taddr),
            "r"(sfb_taddr),
            "r"(enable_input_d)
            : "memory");
    }
}

template <int kClusterM, int kSplitK = 1>
__device__ inline void tcgen05_commit(
    int mbar_addr,
    int split_rank = 0) {
    if constexpr (kClusterM == 1) {
        asm volatile("tcgen05.commit.cta_group::1."
                     "mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar_addr)
            : "memory");
    } else {
        if constexpr (kSplitK == 1) {
            constexpr int16_t cta_mask = (1 << kClusterM) - 1;
            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
                         "multicast::cluster.b64 [%0], %1;" ::"r"(mbar_addr),
                "h"(cta_mask)
                : "memory");
        } else {
            const int16_t cta_mask = ((1 << kClusterM) - 1) << (split_rank * kClusterM);
            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
                         "multicast::cluster.b64 [%0], %1;" ::"r"(mbar_addr),
                "h"(cta_mask)
                : "memory");
        }
    }
}

__device__ inline uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile("{\n\t"
                 ".reg .pred %%px;\n\t"
                 "elect.sync _|%%px, %1;\n\t"
                 "@%%px mov.s32 %0, 1;\n\t"
                 "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

__device__ inline constexpr uint64_t desc_encode(
    uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
};

__device__ inline void mbarrier_init(
    int mbar_addr,
    int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr), "r"(count));
}

__device__ inline void prefetch_tensormap(
    const void *desc) {
    uint64_t ptr = reinterpret_cast<uint64_t>(desc);
    asm volatile("prefetch.tensormap [%0];" ::"l"(ptr) : "memory");
}

inline void init_fp8_tmap(
    CUtensorMap *tmap,
    const uint8_t *ptr,
    uint64_t global_height,
    uint64_t global_width,
    uint32_t shared_height,
    uint32_t shared_width,
    CUtensorMapSwizzle swizzle) {

    constexpr uint32_t rank = 3;

    uint64_t global_dim[rank] = {global_width, global_height, 1};

    uint64_t global_strides[rank - 1] = {global_width, global_width * global_height};

    uint32_t box_dim[rank] = {shared_width, shared_height, 1};
    uint32_t element_strides[rank] = {1, 1, 1};

    auto err = cuTensorMapEncodeTiled(tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
        rank,
        (void *)ptr,
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(err == CUDA_SUCCESS);
}

// `scale_ptr` uses the standard Blackwell SWIZZLE_32_4_4 layout.  Every
// 512-byte tile is a 32x16-byte matrix ready for tcgen05.cp ... warpx4.
inline void init_scale_tmap(
    CUtensorMap *tmap,
    const uint8_t *scale_ptr,
    uint64_t row_tiles,
    uint64_t k_tiles,
    uint32_t k_box_tiles) {
    constexpr uint32_t rank = 4;
    uint64_t global_dim[rank] = {256, 2, k_tiles, row_tiles};
    uint64_t global_strides[rank - 1] = {256, 512, 512 * k_tiles};
    // K=32 scale tiles are contiguous in both global and shared memory.  Make
    // the whole TileK span one TMA box instead of issuing one 512B TMA per MMA.
    uint32_t box_dim[rank] = {256, 2, k_box_tiles, 1};
    uint32_t element_strides[rank] = {1, 1, 1, 1};
    auto err = cuTensorMapEncodeTiled(tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
        rank,
        const_cast<uint8_t *>(scale_ptr),
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(err == CUDA_SUCCESS);
}

inline void init_3D_output_tmap(
    CUtensorMap *tmap,
    nv_bfloat16 *ptr,
    uint64_t global_height,
    uint64_t global_width,
    uint32_t shared_height,
    uint32_t shared_width) {

    constexpr uint32_t rank = 3;
    constexpr uint32_t kSwizzleElements = 32;
    assert(global_width % kSwizzleElements == 0);
    assert(shared_width % kSwizzleElements == 0);
    uint64_t global_dim[rank] = {kSwizzleElements, global_width / kSwizzleElements, global_height};
    uint64_t global_strides[rank - 1] = {kSwizzleElements * sizeof(nv_bfloat16),
        global_width * sizeof(nv_bfloat16)};
    uint32_t box_dim[rank] = {kSwizzleElements, shared_width / kSwizzleElements, shared_height};
    uint32_t element_strides[rank] = {1, 1, 1};

    cuTensorMapEncodeTiled(tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        ptr,
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline void init_2D_output_tmap(
    CUtensorMap *tmap,
    nv_bfloat16 *ptr,
    uint64_t global_height,
    uint64_t global_width,
    uint32_t shared_height,
    uint32_t shared_width) {

    constexpr uint32_t rank = 2;
    uint64_t global_dim[rank] = {global_width, global_height};
    uint64_t global_strides[rank - 1] = {global_width * sizeof(nv_bfloat16)};
    uint32_t box_dim[rank] = {shared_width, shared_height};
    uint32_t element_strides[rank] = {1, 1};

    cuTensorMapEncodeTiled(tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        ptr,
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template <
    int kTileM,
    int kTileN,
    int kTileK,
    int kClusterM,
    int kEpilogueBlockN,
    int kSplitK,
    bool SWAP_AB>
__global__
__cluster_dims__(kClusterM, kSplitK, 1)
__launch_bounds__(kBlockThreads)
void fp8_shared_fc1_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap SFA_tmap,
    const __grid_constant__ CUtensorMap SFB_tmap,
    int M,
    int N,
    int K,
    __nv_bfloat16 *fc1_out) {
    sm110a::pdl_trigger<SM110A_FP8_SHARED_FC1_PDL>();

    using Config = KernelConfig<kTileM, kTileN, kTileK, kClusterM, kEpilogueBlockN, kSplitK>;
    constexpr int kStageCount = Config::kStageCount;
    constexpr int kATmaBytes = Config::kATmaBytes;
    constexpr int kBTmaBytes = Config::kBTmaBytes;
    constexpr int kASmemBytes = Config::kASmemBytes;
    constexpr int kBSmemBytes = Config::kBSmemBytes;
    constexpr int kSFASmemBytes = Config::kSFASmemBytes;
    constexpr int kSFBSmemBytes = Config::kSFBSmemBytes;
    constexpr int kSFBScaleTileCount = Config::kSFBScaleTileCount;
    constexpr int kMmaStepsPerStage = Config::kMmaStepsPerStage;
    constexpr int kAccumulatorStageCount = Config::kAccumulatorStageCount;
    constexpr int kAccumulatorOverlapCols = Config::kAccumulatorOverlapCols;
    constexpr int kAccumulatorStageStride = Config::kAccumulatorStageStride;
    constexpr int kMainloopSmemBytes = Config::kMainloopSmemBytes;
    constexpr int kEpilogueStageBytes = Config::kEpilogueStageBytes;
    constexpr int kPerStageSmemBytes = Config::kPerStageSmemBytes;

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;

    const int grid_m = M / kTileM;
    const int grid_n = N / kTileN;
    const int grid_size = grid_m * grid_n;

    // set up cta rank
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int m_rank = cta_rank % kClusterM;
    const int split_rank = cta_rank / kClusterM;
    const int _bid = blockIdx.x;
    const int grid_x = gridDim.x;

    // set up smem
    extern __shared__ __align__(1024) char _smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(_smem_ptr));
    nv_bfloat16 *epilogue_smem = reinterpret_cast<nv_bfloat16 *>(_smem_ptr + kMainloopSmemBytes);
    float *reduction_smem =
        reinterpret_cast<float *>(_smem_ptr + kMainloopSmemBytes + Config::kEpilogueSmemBytes);

    constexpr int kClusterTileM = kClusterM * kTileM;
    constexpr uint32_t i_desc =
        // mxf8f6f4 with E4M3 on both sides (atype = btype = 0), scaled by
        // UE8M0.
        (1U << 23U) | ((uint32_t(kTileN) >> 3U) << 17U) |
        ((uint32_t(kClusterTileM) >> 7U) << 27U);
    static_assert(i_desc == (kClusterM == 1 ? (kTileN == 64       ? 0x08900000U
                                                  : kTileN == 128 ? 0x08A00000U
                                                                  : 0x08C00000U)
                                            : (kTileN == 64       ? 0x10900000U
                                                  : kTileN == 128 ? 0x10A00000U
                                                                  : 0x10C00000U)));

    // setup Mbarrier
    __shared__ int tmem_addr[1];
    __shared__ uint64_t tma_mbars[kStageCount];
    __shared__ uint64_t mma_mbars[kStageCount];
    __shared__ uint64_t mainlooop_mbars[kAccumulatorStageCount];
    __shared__ uint64_t epilogue_mbars[kAccumulatorStageCount];
    // Each CTA writes its overlap epoch to CTA rank 0 through DSMEM.  The
    // MMA issuer can then poll a local, cluster-complete dependency token.
    __shared__ volatile unsigned int overlap_ready[kClusterM][kAccumulatorStageCount];
    __shared__ unsigned int overlap_epoch_local[kAccumulatorStageCount];
    const int _base_tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(tma_mbars));
    const int _base_mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mma_mbars));
    const int _base_mainloop_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(mainlooop_mbars));
    const int _base_epilogue_mbar_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_mbars));

    // init mbarrier
    if (tid == 0) {
        for (int s = 0; s < kStageCount; ++s) {
            mbarrier_init(_base_tma_mbar_addr + 8 * s, kClusterM);
            mbarrier_init(_base_mma_mbar_addr + 8 * s, 1);
        }
        for (int s = 0; s < kAccumulatorStageCount; ++s) {
            mbarrier_init(_base_mainloop_mbar_addr + 8 * s, 1);
            mbarrier_init(_base_epilogue_mbar_addr + 8 * s, 4 * kClusterM);
            overlap_epoch_local[s] = 0;
            if (cta_rank == 0) {
#pragma unroll
                for (int rank = 0; rank < kClusterM; ++rank) {
                    overlap_ready[rank][s] = 0;
                }
            }
        }

        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    // alloc tmem
    if (warp_id == 1) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile(
            "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(addr),
            "r"(kTmemColumnCapacity),
            "n"(kClusterM));
    }
    // prefetch tma desc
    if constexpr (kPrefetchTmaDesc) {
        if (warp_id == 0 && elect_sync()) {
            prefetch_tensormap(&A_tmap);
            prefetch_tensormap(&B_tmap);
            prefetch_tensormap(&SFA_tmap);
            prefetch_tensormap(&SFB_tmap);
        }
    }

    if constexpr (kClusterM * kSplitK > 1) {
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
    } else {
        __syncthreads();
    }

    sm110a::pdl_wait<SM110A_FP8_SHARED_FC1_PDL>();
    const int taddr = tmem_addr[0];

    // Hint the exact tiles consumed by a future TMA load into L2.  This has
    // no completion barrier: it is only a cache residency hint, while the
    // corresponding regular TMA transfer remains responsible for SMEM
    // visibility and mbarrier completion.
    auto prefetch = [&](int iter_k, int off_m, int off_n) {
        const int off_k = (split_rank * (K / (kSplitK * kTileK)) + iter_k) * kTileK;
        const int off_k_32 = (split_rank * (K / (kSplitK * kTileK)) + iter_k) * kMmaStepsPerStage;
        tma_3d_prefetch_l2(&A_tmap, off_k, off_m, 0);
        tma_3d_prefetch_l2(&B_tmap, off_k, off_n + m_rank * (kTileN / kClusterM), 0);
#pragma unroll
        for (int scale_tile = 0; scale_tile < kSFBScaleTileCount; ++scale_tile) {
            // Match the regular multicast load: rank 0 is the sole cluster
            // producer for SFB, so avoid redundant L2 prefetch requests.
            if constexpr (kClusterM == 1) {
                tma_4d_prefetch_l2(&SFB_tmap, 0, 0, off_k_32, off_n / 128 + scale_tile);
            } else if (m_rank == 0) {
                tma_4d_prefetch_l2(&SFB_tmap, 0, 0, off_k_32, off_n / 128 + scale_tile);
            }
        }
    };

    // tma load
    auto load = [&](int iter_k, int mma_phase, int stage_id, int off_m, int off_n) {
        const int mma_mbar_addr = _base_mma_mbar_addr + 8 * stage_id;
        wait_mma_slot(mma_mbar_addr, mma_phase);

        const int A_smem = smem + stage_id * kPerStageSmemBytes;
        const int B_smem = A_smem + kASmemBytes;
        const int SFA_smem = B_smem + kBSmemBytes;
        const int SFB_smem = SFA_smem + kSFASmemBytes;
        const int tma_mbar_addr = kClusterM == 1
                                      ? _base_tma_mbar_addr + 8 * stage_id
                                      : (_base_tma_mbar_addr + 8 * stage_id) & 0xFEFFFFFF;

        // tma issue.
        constexpr int cp_size = kATmaBytes + kBTmaBytes + kSFASmemBytes + kSFBSmemBytes;

        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::"r"(
                         tma_mbar_addr),
            "r"(cp_size)
            : "memory");
        const int off_k = (split_rank * (K / (kSplitK * kTileK)) + iter_k) * kTileK;
        const int off_k_32 = (split_rank * (K / (kSplitK * kTileK)) + iter_k) * kMmaStepsPerStage;
        tma_3d_gmem2smem<kClusterM>(A_smem, &A_tmap, off_k, off_m, 0, tma_mbar_addr);
        tma_3d_gmem2smem<kClusterM>(B_smem,
            &B_tmap,
            off_k,
            off_n + m_rank * (kTileN / kClusterM),
            0,
            tma_mbar_addr);
        tma_4d_gmem2smem<kClusterM>(
            SFA_smem, &SFA_tmap, 0, 0, off_k_32, off_m / kTileM, tma_mbar_addr);
        // SFB is one 512-byte tile per K=32 MMA and 128 N rows.
#pragma unroll
        for (int scale_tile = 0; scale_tile < kSFBScaleTileCount; ++scale_tile) {
            const int SFB_tile_smem = SFB_smem + scale_tile * kMmaStepsPerStage * 512;
            if constexpr (kClusterM == 1) {
                tma_4d_gmem2smem<1>(SFB_tile_smem,
                    &SFB_tmap,
                    0,
                    0,
                    off_k_32,
                    off_n / 128 + scale_tile,
                    tma_mbar_addr);
            } else if (m_rank == 0) {
                const int16_t cta_mask = ((1 << kClusterM) - 1) << (split_rank * kClusterM);
                tma_4d_gmem2smem_multicast<kClusterM>(SFB_tile_smem,
                    &SFB_tmap,
                    0,
                    0,
                    off_k_32,
                    off_n / 128 + scale_tile,
                    tma_mbar_addr,
                    cta_mask);
            }
        }
    };

    // mma compute
    auto compute = [&](int iter_k, int mainloop_stage, int tma_phase, int stage_id, int off_n) {
        const int tma_mbar_addr = _base_tma_mbar_addr + 8 * stage_id;
        wait_tma(tma_mbar_addr, tma_phase);

        const int A_smem = smem + stage_id * kPerStageSmemBytes;
        const int B_smem = A_smem + kASmemBytes;
        const int SFA_smem = B_smem + kBSmemBytes;
        const int SFB_smem = SFA_smem + kSFASmemBytes;
        const int mma_mbar_addr = _base_mma_mbar_addr + 8 * stage_id;

        auto make_operand_desc = [](int addr) -> uint64_t {
            // The descriptor always describes one K=32 MMA slab.  Wider
            // TileK values advance the base address between MMA issues.
            constexpr int SBO = 8 * 128;
            const uint64_t base_offset = (addr >> 7) & 0x7;
            return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) |
                   (base_offset << 49ULL) | (2ULL << 61ULL);
        };
        auto make_scale_desc = [](int addr) -> uint64_t {
            // 32 rows x 16 bytes; the stride dimension advances from one
            // group of eight rows to the next group (8 * 16 bytes).
            constexpr int SBO = 128;
            return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
        };

        asm volatile("tcgen05.fence::after_thread_sync;");
        const int this_taddr = taddr + mainloop_stage * kAccumulatorStageStride;
        constexpr int kScaleTmemBase = Config::kAccumulatorTmemCols;
        const int scale_taddr = taddr + kScaleTmemBase;

        auto copy_mma_scales = [&](int mma_k, int mma_scale_taddr) {
            const int SFA_mma_smem = SFA_smem + mma_k * 512;
            tcgen05_cp_scale<kClusterM>(mma_scale_taddr, make_scale_desc(SFA_mma_smem));
#pragma unroll
            for (int scale_tile = 0; scale_tile < kSFBScaleTileCount; ++scale_tile) {
                const int SFB_mma_smem = SFB_smem + (scale_tile * kMmaStepsPerStage + mma_k) * 512;
                tcgen05_cp_scale<kClusterM>(mma_scale_taddr + 4 + scale_tile * 4,
                    make_scale_desc(SFB_mma_smem));
            }
        };

#pragma unroll
        for (int mma_k = 0; mma_k < kMmaStepsPerStage; ++mma_k) {
            const int A_mma_smem = A_smem + mma_k * kMmaK;
            const int B_mma_smem = B_smem + mma_k * kMmaK;
            const int mma_scale_taddr =
                scale_taddr + (kTileN < 256 ? (stage_id * kMmaStepsPerStage + mma_k) *
                                                  Config::kScaleTmemSlotStride
                                            : 0);

            // Pair each scale copy with the MMA that consumes it.  Narrow-N
            // uses a distinct slot per (SMEM stage, K=32 slab); TileN=256
            // reuses its sole capacity-constrained slot.  In both cases the
            // cp -> mma ordering supplies the required pipeline dependency.
            copy_mma_scales(mma_k, mma_scale_taddr);

            // In the scale_vec::1X SFB layout, each TMEM column covers 32 N values.
            // A TileN=64 tile may therefore start at column 0 or 2 of its
            // containing 128-row blocked scale tile.
            const int mma_sfb_taddr = mma_scale_taddr + 4 + (off_n % 128) / 32;
            tcgen05_mma_mxf8f6f4<kClusterM>(this_taddr,
                make_operand_desc(A_mma_smem),
                make_operand_desc(B_mma_smem),
                i_desc,
                mma_scale_taddr,
                mma_sfb_taddr,
                iter_k != 0 || mma_k != 0);
        }

        tcgen05_commit<kClusterM, kSplitK>(mma_mbar_addr, split_rank);
    };

    auto epilogue = [&](int mainloop_stage, int off_m, int off_n) {
        auto release_tmem = [&]() {
            if (elect_sync()) {
                const int mbar_addr = (_base_epilogue_mbar_addr + mainloop_stage * 8) & 0xFEFFFFFF;
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(mbar_addr)
                    : "memory");
            }
        };

        static_assert(kTileN == 128 && kSplitK == 1 && !SWAP_AB);
        // Each 128-column weight tile contains gate[64], then up[64].
        // No overlap between the two 128-column accumulator banks.
        for (int group = 0; group < 8; ++group) {
            float gate[8], up[8];
            const int addr = taddr + ((m_rank * 128 + warp_id * 32) << 16) +
                mainloop_stage * kAccumulatorStageStride + group * 8;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(gate[0]), "=f"(gate[1]), "=f"(gate[2]), "=f"(gate[3]),
                  "=f"(gate[4]), "=f"(gate[5]), "=f"(gate[6]), "=f"(gate[7]) : "r"(addr));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(up[0]), "=f"(up[1]), "=f"(up[2]), "=f"(up[3]),
                  "=f"(up[4]), "=f"(up[5]), "=f"(up[6]), "=f"(up[7]) : "r"(addr + 64));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            alignas(16) nv_bfloat16 result[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float g = __bfloat162float(__float2bfloat16_rn(gate[i]));
                float u = __bfloat162float(__float2bfloat16_rn(up[i]));
                constexpr float limit = SM110A_FP8_SHARED_FC1_SWIGLU_LIMIT;
                if constexpr (limit > 0) {
                    g = g > limit ? limit : g;
                    u = u > limit ? limit : (u < -limit ? -limit : u);
                }
                result[i] = __float2bfloat16_rn((g * (1.0f / (1.0f + expf(-g)))) * u);
            }
            const int row = off_m + tid;
            reinterpret_cast<int4 *>(fc1_out + row * (N / 2) + off_n / 2 + group * 8)[0] =
                reinterpret_cast<int4 *>(result)[0];
        }
        release_tmem();
    };

    auto compute_bid = [&](int bid) -> std::tuple<int, int> {
        constexpr int kMacroMSize = kTileM * kMacroSwizzleM;
        constexpr int kMacroNSize = kTileN * kMacroSwizzleN;

        const int macro_m_num = (M + kMacroMSize - 1) / kMacroMSize;
        const int macro_n_num = (N + kMacroNSize - 1) / kMacroNSize;

        auto _compute_bid_in_macro = [&](int macro_block_m_idx,
                                         int macro_block_n_idx,
                                         int macro_lane) -> std::tuple<int, int> {
            const int base_m = macro_block_m_idx * kMacroMSize;
            const int base_n = macro_block_n_idx * kMacroNSize;

            const int _real_macro_m = min(kMacroMSize, M - base_m) / kTileM;
            const int _real_macro_n = min(kMacroNSize, N - base_n) / kTileN;

            const int macro_bid_m =
                macro_lane / (_real_macro_n * kClusterM) * kClusterM + (macro_lane % kClusterM);
            const int macro_bid_n = (macro_lane / kClusterM) % _real_macro_n;

            const int off_m = base_m + macro_bid_m * kTileM;
            const int off_n = base_n + macro_bid_n * kTileN;

            return {off_m, off_n};
        };

        if constexpr (kMacroSwizzleMFast) {
            const int full_n_band_size = grid_m * kMacroSwizzleN;

            const int macro_block_n_idx = bid / full_n_band_size;
            const int bid_in_n_band = bid - macro_block_n_idx * full_n_band_size;

            const int real_macro_n =
                min(kMacroSwizzleN, grid_n - macro_block_n_idx * kMacroSwizzleN);

            const int macro_m_block_size = kMacroSwizzleM * real_macro_n;

            const int macro_block_m_idx = bid_in_n_band / macro_m_block_size;
            const int macro_lane = bid_in_n_band % macro_m_block_size;

            return _compute_bid_in_macro(macro_block_m_idx, macro_block_n_idx, macro_lane);
        } else {
            const int full_m_band_size = grid_n * kMacroSwizzleM;

            const int macro_block_m_idx = bid / full_m_band_size;
            const int bid_in_m_band = bid - macro_block_m_idx * full_m_band_size;

            const int real_macro_m =
                min(kMacroSwizzleM, grid_m - macro_block_m_idx * kMacroSwizzleM);

            const int macro_n_block_size = real_macro_m * kMacroSwizzleN;

            const int macro_block_n_idx = bid_in_m_band / macro_n_block_size;
            const int macro_lane = bid_in_m_band % macro_n_block_size;

            return _compute_bid_in_macro(macro_block_m_idx, macro_block_n_idx, macro_lane);
        }
    };

    // load warp
    if (warp_id == 4 && elect_sync()) {
        int mma_stage = 0;
        int mma_phase = 1;
        for (int bid = _bid; bid < grid_size; bid += grid_x) {
            auto [off_m, off_n] = compute_bid(bid);
            constexpr int kPrefetchDistance = kStageCount;
            const int k_tile_count = K / kTileK;

            if constexpr (kPrefetchTmaData) {
                // Prime a full pipeline-distance window.  The rolling issue
                // below visits every later K tile exactly once.
                for (int iter_k = 0; iter_k < min(kPrefetchDistance, k_tile_count / kSplitK);
                    ++iter_k) {
                    prefetch(iter_k, off_m, off_n);
                }
            }

            for (int iter_k = 0; iter_k < k_tile_count / kSplitK; ++iter_k) {
                load(iter_k, mma_phase, mma_stage, off_m, off_n);
                if constexpr (kPrefetchTmaData) {
                    const int future_k = iter_k + kPrefetchDistance;
                    if (future_k < k_tile_count / kSplitK) {
                        prefetch(future_k, off_m, off_n);
                    }
                }
                mma_stage = (mma_stage + 1) % kStageCount;
                if (mma_stage == 0) {
                    mma_phase ^= 1;
                }
            }
        }
    }

    // mma warp
    if (m_rank == 0 && warp_id == 5 && elect_sync()) {
        int tma_phase = 0;
        int tma_stage = 0;

        int mainloop_stage = 0;
        int epilogue_phase = 1;
        unsigned int overlap_epoch[kAccumulatorStageCount] = {};
        bool first_output_tile = true;

        for (int bid = _bid; bid < grid_size; bid += grid_x) {
            auto [off_m, off_n] = compute_bid(bid);
            wait_epilogue(_base_epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);
            if constexpr (kAccumulatorOverlapCols != 0) {
                if (!first_output_tile) {
                    const int previous_stage = 1 - mainloop_stage;
                    const unsigned int target = ++overlap_epoch[previous_stage];
#pragma unroll
                    for (int rank = 0; rank < kClusterM; ++rank) {
                        while (overlap_ready[rank][previous_stage] < target) {
                        }
                    }
                }
            }

            for (int iter_k = 0; iter_k < K / (kSplitK * kTileK); ++iter_k) {
                compute(iter_k, mainloop_stage, tma_phase, tma_stage, off_n);
                tma_stage = (tma_stage + 1) % kStageCount;
                if (tma_stage == 0) {
                    tma_phase ^= 1;
                }
            }

            tcgen05_commit<kClusterM, kSplitK>(_base_mainloop_mbar_addr + 8 * mainloop_stage,
                split_rank);

            mainloop_stage = (mainloop_stage + 1) % kAccumulatorStageCount;
            if (mainloop_stage == 0) {
                epilogue_phase ^= 1;
            }
            first_output_tile = false;
        }
    }

    if (warp_id < 4) {
        int mainloop_stage = 0;
        int mainloop_phase = 0;

        for (int bid = _bid; bid < grid_size; bid += grid_x) {
            wait_mainloop(_base_mainloop_mbar_addr + 8 * mainloop_stage, mainloop_phase);
            asm volatile("tcgen05.fence::after_thread_sync;");
            auto [off_m, off_n] = compute_bid(bid);
            epilogue(mainloop_stage, off_m, off_n);
            mainloop_stage = (mainloop_stage + 1) % kAccumulatorStageCount;
            if (mainloop_stage == 0) {
                mainloop_phase ^= 1;
            }
        }
    }

    if constexpr (kClusterM * kSplitK > 1) {
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
    } else {
        __syncthreads();
    }
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" ::"r"(taddr),
            "r"(kTmemColumnCapacity),
            "n"(kClusterM));
    }
}

} // namespace sm110a::fp8_shared_fc1

using namespace sm110a::fp8_shared_fc1;

template <int kTileM,
    int kTileN,
    int kTileK,
    int kClusterM,
    int kEpilogueBlockN,
    int kSplitK,
    bool SWAP_AB>
void launch_fp8_shared_fc1_impl(
    const uint8_t *A_ptr,
    const uint8_t *B_ptr,
    const uint8_t *SFA_ptr,
    const uint8_t *SFB_ptr,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K,
    cudaStream_t stream) {

    using Config = KernelConfig<kTileM, kTileN, kTileK, kClusterM, kEpilogueBlockN, kSplitK>;
    constexpr int kDynamicSmemBytes = Config::kDynamicSmemBytes;

    static_assert(!SWAP_AB && kSplitK == 1 && kTileN == 128);

    CUtensorMap A_tmap, B_tmap, SFA_tmap, SFB_tmap;

    static_assert(
        kMacroSwizzleM * kMacroSwizzleN <= (32 * 1024 * 1024) / ((kTileM + kTileN) * kTileK));
    assert(M % (kClusterM * kTileM) == 0);
    assert(N % kTileN == 0);
    assert(N % 128 == 0);
    assert(K % (kTileK * kSplitK) == 0);

    init_fp8_tmap(&A_tmap,
        A_ptr,
        M,
        K,
        kTileM,
        kTileK,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_fp8_tmap(&B_tmap,
        B_ptr,
        N,
        K,
        kTileN / kClusterM,
        kTileK,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    constexpr uint32_t kScaleBoxTiles = kTileK / kMmaK;
    init_scale_tmap(&SFA_tmap, SFA_ptr, M / kTileM, K / kMmaK, kScaleBoxTiles);
    init_scale_tmap(&SFB_tmap, SFB_ptr, N / 128, K / kMmaK, kScaleBoxTiles);
    const int smem_size = kDynamicSmemBytes + kMbarrierSmemBytes;

    dim3 grid(kSplitK == 1 ? kSmCount : (M / kTileM) * (N / kTileN), kSplitK, 1);

    auto kernel =
        fp8_shared_fc1_kernel<kTileM, kTileN, kTileK, kClusterM, kEpilogueBlockN, kSplitK, SWAP_AB>;

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(kBlockThreads);
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    sm110a::launch_pdl<SM110A_FP8_SHARED_FC1_PDL>(&config, kernel,
        A_tmap, B_tmap, SFA_tmap, SFB_tmap, M, N, K, C_ptr);
}

// Non-template wrapper for the pre-compiled path — instantiate with the #define defaults.
void launch_fp8_shared_fc1(
    const uint8_t *A_ptr,
    const uint8_t *B_ptr,
    const uint8_t *SFA_ptr,
    const uint8_t *SFB_ptr,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    launch_fp8_shared_fc1_impl<SM110A_FP8_SHARED_FC1_TILE_M,
        SM110A_FP8_SHARED_FC1_TILE_N,
        SM110A_FP8_SHARED_FC1_TILE_K,
        SM110A_FP8_SHARED_FC1_CLUSTER_M,
        SM110A_FP8_SHARED_FC1_EPILOGUE_BLOCK_N,
        SM110A_FP8_SHARED_FC1_SPLIT_K,
        false>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, M, N, K, stream);
}

extern "C" void sm110a_fp8_shared_fc1_launch(
    const uint8_t *a,
    const uint8_t *b,
    const uint8_t *sfa,
    const uint8_t *sfb,
    __nv_bfloat16 *c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
    launch_fp8_shared_fc1(a, b, sfa, sfb, c, m, n, k, stream);
}
