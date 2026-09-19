// =============================================================================
// SM110A BF16 GEMM — Overview
// =============================================================================
//
// Target hardware
// ---------------
//   NVIDIA Blackwell SM (SM110A, sm_110a). Assumes first-class support for
//   Tensor Memory (TMEM), distributed shared memory (DSMEM), thread-block
//   clusters and the Tensor Memory Accelerator (TMA) bulk-copy engine.
//
// Problem
// -------
//   C[M,N] = A[M,K] @ B[K,N]   with A, B, C stored as nv_bfloat16.
//   M, N, K must be multiples of (kTileM, kTileN, kTileK) respectively.
//   A "SWAP_AB" template variant reuses the kernel for the transposed
//   product B[N,K] @ A[K,M] by treating (B,A) as the operands on the host
//   side (see launch_bf16_gemm_impl).
//
// Bottleneck model
// ----------------
//   Back-of-the-envelope estimate for the default 128x256x64 tile with
//   kClusterM == 2 (2 CTAs cooperate along M):
//
//     Per-tile loads (one K tile = 64 elements):
//         A: 128 rows * 64 * sizeof(bf16)               = 16 KiB / CTA
//         B: 128 cols * 64 * sizeof(bf16)               = 16 KiB / CTA
//         Total per output tile: 2 * (128 + 128) * 64   = 32 KiB of K data
//
//     Over the full problem (M = 128 * Mtiles, N = 256 * Ntiles):
//         Total bytes moved = 2 * (128 + 128) * K * (Mtiles * Ntiles)
//                            = K * M * N / 64 bytes.
//         Memory-bound time = total_bytes / peak_L2_or_DRAM_bandwidth.
//         Compute-bound time = 2 * K * M * N / peak_BF16_MMA_rate.
//
//     For a 4K^3 problem the two limits are roughly:
//         * L2-hit TMA: K * M * N / 64 / ~1.281 TB/s  ~= 3.12 ms.
//         * MMA issue : 2 * K * M * N     / ~250 TF/s  ~= 0.55 ms.
//     (Effective rates depend on the deployed Blackwell part; treat these
//     numbers as a sanity check, not a contract.)
//
//     Per-stage latency model (single output tile, kClusterM = 2):
//         * MMA work        : 2 * 128 * 256 * 64 / 8K  ~= 512 cycles.
//         * DRAM -> L2 fetch: (2 * 128 * 64 + 2 * 128 * 64) / 8  ~= 4096 cycles.
//         * L2 hit on fetch : (2 * 128 * 64 + 2 * 128 * 64) / 40 ~= 819 cycles.
//
//   Reading the three numbers together explains the design pressure points
//   addressed below:
//     * The MMA path is only ~2x of an L2-hit fetch, so compute and memory
//       must overlap tightly -- that is the role of the multi-stage mainloop.
//     * An L2 miss inflates the fetch by ~5x, so the macro-swizzle and the
//       L2-friendly tile order are first-order optimizations, not niceties.
//     * TMA descriptor prefetch matters because every mainloop iteration
//       consumes a fresh A/B tile, so the very first load cannot afford to
//       stall on the descriptor fetch.
//

// Functional summary
// ------------------
//   * Persistent tiled kernel: launches a caller-selected number of clusters
//     (one CTA per SM in each cluster) and walks the output-tile grid in
//     round-robin.
//     and walks the (grid_m * grid_n) output-tile grid in round-robin so each
//     CTA accumulates multiple output tiles per launch.
//   * Each CTA executes a 6-warp pipeline (192 threads):
//       warp 4              – load warp (TMA gmem→smem)
//       warp 5 (cta_rank 0) – MMA warp   (tcgen05.mma on TMEM)
//       warps 0..3          – epilogue warps (tmem→smem→gmem via TMA)
//   * Clusters of size kClusterM along M cooperate via DSMEM and multicast:
//     every cluster CTA loads its own A row, and rank `cta_rank` loads its
//     slice of B; the MMA is issued once with `cta_group::kClusterM`.
//   * Output accumulator lives in TMEM (kTileN*2 columns, i.e. 2 banks for
//     mainloop double-buffering) and is drained to shared memory via
//     tcgen05.ld, then to global memory via 2-D/3-D TMA bulk stores.
//   * Two entry points:
//       launch_bf16_gemm          – regular, unprofiled path.
//       launch_bf16_gemm_profile  – same kernel with per-stage profiler.
//
// Optimizations
// -------------
//   1. TMA for all bulk data movement.
//        * Loads  : 3-D tensor maps for A and B, 128 B swizzle.
//        * Stores : 2-D map for the KEpilogueBlockN==32 fast path, 3-D map
//                   otherwise; 64 B swizzle. Register traffic on the memory
//                   side of the pipeline is zero.
//   2. Multi-stage software pipeline (kStageCount stages for A/B in smem).
//        Stage count is chosen at *compile time* from a 227 KiB smem budget
//        minus the epilogue double buffer (KernelConfig::kStageCount), so the
//        largest configuration that still fits is selected automatically.
//   3. Double-buffered TMEM accumulator (mainloop_stage ∈ {0,1}).
//        The epilogue owns one bank while the MMA owns the other. Release of
//        each bank is gated by an `epilogue_mbar` that the epilogue trips
//        only after its TMA store has committed, hiding the TMEM lifetime
//        from the mainloop.
//   4. Producer/consumer decoupling with two mbarrier rings:
//          tma_mbars[kStageCount]  – TMA completion -> MMA consumer.
//          mma_mbars[kStageCount]  – MMA commit     -> TMA producer.
//        A separate `mainlooop_mbars[2]` array — note the misspelling kept
//        on purpose — signals mainloop-stage completion to the epilogue.
//        Phase parity bits per ring track the 2-cycle wrap.
//   5. L2-friendly tile schedule (`compute_bid`).
//        The output grid is mapped through an 8x8 macro-swizzle. With
//        `kMacroSwizzleMFast=true`, M is iterated fastest inside a macro
//        band so consecutive CTAs read overlapping K-resident operands
//        and reuse L2 lines.
//   6. TMA descriptor prefetch (`prefetch.tensormap`) issued once at startup,
//        so the first TMA does not stall on the descriptor fetch.
//   7. Pipelined epilogue:
//          * 2 smem slots per mainloop_stage (`kEpilogueStageCount == 2`).
//          * `cp.async.bulk.wait_group.read 1` + `bar.sync 1, 128` keeps the
//            TMA store of slot N concurrent with the tcgen05.ld that fills
//            slot N+1.
//   8. `SWAP_AB` template variant reuses the same kernel binary for the
//      transposed product (no second hand-written kernel); only the host-side
//      launch swaps pointer/dimension arguments and selects the matching
//      C tensor-map encoder.
//   9. Per-stage profiler (DO_PROFILE): tag-based timing of Setup,
//      WaitMMA, IssueTMA, WaitTMA, IssueMMA, WaitMainloop, WaitEpilogue,
//      Epilogue via elected leaders, no impact on the unprofiled path.
//
// Detailed pipeline
// -----------------
//   Legend
//     L – load warp (warp_id == 4)
//     M – MMA  warp (warp_id == 5 on cta_rank == 0)
//     E – epilogue warps (warp_id in {0,1,2,3})
//
//   0. Setup (all warps, once per CTA)
//      ------------------------------------------------
//      tid==0   : mbarrier.init for tma_mbars[kStageCount] (count=kClusterM),
//                 mma_mbars[kStageCount] (count=1), mainlooop_mbars[2] (1),
//                 epilogue_mbars[2] (4*kClusterM); then
//                 `fence.mbarrier_init.release.cluster`.
//      warp 1   : `tcgen05.alloc.cta_group::kClusterM` reserves kTileN*2
//                 TMEM columns (returned in tmem_addr[0]).
//      warp 0*  : `prefetch.tensormap` for A_tmap / B_tmap / C_tmap.
//      barrier.cluster.arrive+wait (or __syncthreads when kClusterM==1).
//
//   1. Main loop — repeated `K / kTileK` iterations
//      ------------------------------------------------
//      ┌─── L (load warp) ──────────────────────────────────────────────┐
//      │  wait(mma_mbar[mma_stage], mma_phase)        // slot free?     │
//      │  mbar.arrive.expect_tx(tma_mbar[tma_stage], cp_size)           │
//      │  cp.async.bulk.tensor.3d A: smem[stage]<-A_tmap(m_off,k_off)   │
//      │  cp.async.bulk.tensor.3d B: smem[stage]+A_sz <-                │
//      │      B_tmap(n_off + rank*(TileN/Cluster), k_off)               │
//      │  mma_stage = (mma_stage+1)%kStageCount;                        │
//      │  mma_phase ^= 1   when mma_stage wraps to 0                    │
//      └─── (consumes mma_mbar; produces tma_mbar) ─────────────────────┘
//
//      ┌─── M (MMA warp, one issued per cluster) ───────────────────────┐
//      │  wait(tma_mbar[tma_stage], tma_phase)        // data ready?    │
//      │  tcgen05.fence::after_thread_sync                              │
//      │  taddr_use = taddr + mainloop_stage * kTileN                   │
//      │  for k1 in [0, kTileK/64):                                     │
//      │    for k2 in [0, 64/16):                                       │
//      │      tcgen05.mma.cta_group::kClusterM                          │
//      │          tmem[taddr_use] ^= A_sm^k1^k2 * B_sm^...              │
//      │  tcgen05.commit.cta_group::kClusterM                           │
//      │      .mbarrier::arrive::one.shared::cluster.multicast          │
//      │      mma_mbar[mma_stage]   -> tells L the slot is empty        │
//      └─── (consumes tma_mbar; produces mma_mbar) ─────────────────────┘
//
//   2. End-of-K – hand control to the epilogue
//      ------------------------------------------------
//      M issues one extra `tcgen05.commit.cta_group::kClusterM
//      .mbarrier.arrive::one.shared::cluster.multicast` on
//      mainlooop_mbar[mainloop_stage], then toggles mainloop_stage and the
//      epilogue_phase parity bit.
//
//   3. Epilogue (warps 0..3; double-buffered on TMEM)
//      ------------------------------------------------
//      ┌─── E ──────────────────────────────────────────────────────────┐
//      │  wait(mainloop_mbar[mainloop_stage], mainloop_phase)           │
//      │  tcgen05.fence::after_thread_sync                              │
//      │  For n_block in [0, kTileN/kEpilogueBlockN):                   │
//      │     smem_slot = n_block % kEpilogueStageCount                  │
//      │     [fill] tmem->smem of (kEpilogueBlockN x kTileM) block      │
//      │             via tcgen05.ld (x8, .32x32b) + swizzle write       │
//      │             bar.sync 1, 128                                    │
//      │     [push] tid==0: cp.async.bulk.tensor.{2|3}d                 │
//      │             smem_slot -> C_tmap (according to kEpilogueBlockN  │
//      │             and SWAP_AB)                                       │
//      │     if more blocks remain:                                     │
//      │         cp.async.bulk.wait_group.read 1;  bar.sync 1, 128      │
//      │         fill the *other* slot (next n_block)                   │
//      │         bar.sync 1, 128                                        │
//      │         if this is the last block:                             │
//      │             release_tmem -> epilogue_mbar[mainloop_stage]      │
//      │  cp.async.bulk.wait_group 0;   bar.sync 1, 128                 │
//      │  (loop) advance mainloop_stage, flip mainloop_phase            │
//      └─── (consumes mainloop_mbar; produces epilogue_mbar) ───────────┘
//
//   4. Tile cycling (`compute_bid`)
//      ------------------------------------------------
//      Linear `bid` is decomposed through the 8x8 macro-swizzle into an
//      (off_m, off_n) tile coordinate. With `kMacroSwizzleMFast = true` the
//      inner loop walks M, which empirically keeps the K-resident operand
//      hot in L2. The triple-nested loop in the launch
//          for bid = _bid; bid < grid_size; bid += grid_x
//      turns the grid persistent: all launched CTAs share the tile grid evenly.
//
//   5. Teardown
//      ------------------------------------------------
//      cluster barrier -> warp 0 `tcgen05.dealloc.cta_group::kClusterM`
//      returns kTileN*2 TMEM columns. If DO_PROFILE, the elected leader
//      flushes the profiler buffer.
//
// Notes
// -----
//   * The `__grid_constant__ CUtensorMap` parameters carry the host-built
//     TMA descriptors; we never rebuild them on the device.
//   * The `& 0xFEFFFFFF` mask on mbar addresses sets the "CTA-local" bit
//     expected by the `mbarrier` PTX (state<->phase packing convention).
//   * All TMEM addresses are 32-bit shared-memory offsets returned by
//     `tcgen05.alloc` and stored in `tmem_addr[0]`.
//   * `kClusterM == 1` falls back to `__syncthreads()` for setup/teardown;
//     otherwise `barrier.cluster.arrive/wait` is used.
//   * This kernel targets SM110A only. It will not assemble on older SMs
//     that lack the tcgen05.* / bulk.tensor.* opcodes.
// =============================================================================

#include <cassert>
#include <cstdint>
#include <cudaTypedefs.h> // required for CUtensorMap

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bf16_gemm.h"
#include "bf16_gemm_pipeline_delay.cuh"
#include "pipeline_delay.h"
#include "profiler.h"

#ifndef SM110A_BF16_TILE_M
#define SM110A_BF16_TILE_M 128
#endif
#ifndef SM110A_BF16_TILE_N
#define SM110A_BF16_TILE_N 256
#endif
#ifndef SM110A_BF16_TILE_K
#define SM110A_BF16_TILE_K 128
#endif
#ifndef SM110A_BF16_CLUSTER_M
#define SM110A_BF16_CLUSTER_M 2
#endif
#ifndef SM110A_BF16_EPILOGUE_BLOCK_N
#define SM110A_BF16_EPILOGUE_BLOCK_N 32
#endif

namespace {

constexpr int kMmaK = 16;
constexpr int kWarpSize = 32;
constexpr int kWarpCount = 6;
constexpr int kBlockThreads = kWarpSize * kWarpCount;
constexpr int kEpilogueStageCount = 2;
constexpr bool kPrefetchTmaDesc = true;
constexpr int kMacroSwizzleM = 8;
constexpr int kMacroSwizzleN = 8;
constexpr bool kMacroSwizzleMFast = true;

template <int kTileM, int kTileN, int kTileK, int kClusterM, int kEpilogueBlockN>
struct KernelConfig {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert(kTileM == 4 * kWarpSize);
    static_assert(kEpilogueBlockN == 32 || kEpilogueBlockN == 64 || kEpilogueBlockN == 128);
    static_assert(kTileN % kEpilogueBlockN == 0);
    static_assert(kTileN % kClusterM == 0);
    static_assert(kTileK % 64 == 0);
    static_assert(kTileK % kMmaK == 0);

    static constexpr int kClusterTileM = kClusterM * kTileM;
    static constexpr int kASmemBytes = kTileM * kTileK * static_cast<int>(sizeof(nv_bfloat16));
    static constexpr int kBSmemBytes =
        (kTileN / kClusterM) * kTileK * static_cast<int>(sizeof(nv_bfloat16));
    static constexpr int kPerStageSmemBytes = kASmemBytes + kBSmemBytes;
    static constexpr int kEpilogueStageBytes =
        kTileM * kEpilogueBlockN * static_cast<int>(sizeof(nv_bfloat16));
    static constexpr int kEpilogueSmemBytes = kEpilogueStageCount * kEpilogueStageBytes;
    static constexpr int kMaxSmemBytes = 227 * 1024;
    static constexpr int kStageCount = (kMaxSmemBytes - kEpilogueSmemBytes) / kPerStageSmemBytes;
    static_assert(kStageCount >= 2);
    static constexpr int kMainloopSmemBytes = kStageCount * kPerStageSmemBytes;
    static constexpr int kDynamicSmemBytes = kMainloopSmemBytes + kEpilogueSmemBytes;
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

template <int CTA_GROUP = 1>
__device__ inline void tcgen05_mma_f16(
    int taddr,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int enable_input_d) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t"
                 "}" ::"r"(taddr),
        "l"(a_desc),
        "l"(b_desc),
        "r"(i_desc),
        "r"(enable_input_d),
        "n"(CTA_GROUP));
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

inline void init_3D_tmap(
    CUtensorMap *tmap,
    const nv_bfloat16 *ptr,
    uint64_t global_height,
    uint64_t global_width,
    uint32_t shared_height,
    uint32_t shared_width,
    CUtensorMapSwizzle swizzle) {

    constexpr uint32_t rank = 3;

    uint64_t global_dim[rank] = {64, global_height, global_width / 64};

    uint64_t global_strides[rank - 1] = {global_width * sizeof(nv_bfloat16), 128};

    uint32_t box_dim[rank] = {64, shared_height, shared_width / 64};
    uint32_t element_strides[rank] = {1, 1, 1};

    auto err = cuTensorMapEncodeTiled(tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
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
    bool DO_PROFILE,
    bool SWAP_AB>
__global__
__cluster_dims__(kClusterM, 1, 1)
__launch_bounds__(kBlockThreads)
void bf16_gemm_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    int M,
    int N,
    int K,
    int64_t *profiler_ptr,
    int num_entries) {

    using Config = KernelConfig<kTileM, kTileN, kTileK, kClusterM, kEpilogueBlockN>;
    constexpr int kStageCount = Config::kStageCount;
    constexpr int kASmemBytes = Config::kASmemBytes;
    constexpr int kBSmemBytes = Config::kBSmemBytes;
    constexpr int kMainloopSmemBytes = Config::kMainloopSmemBytes;
    constexpr int kEpilogueStageBytes = Config::kEpilogueStageBytes;
    constexpr int kPerStageSmemBytes = Config::kPerStageSmemBytes;

    const int tid = threadIdx.x;
    const int _bid = blockIdx.x;
    const int grid_x = gridDim.x;
    const int warp_id = tid / kWarpSize;

    const int grid_m = M / kTileM;
    const int grid_n = N / kTileN;
    const int grid_size = grid_m * grid_n;

    Profiler profiler;
    if constexpr (DO_PROFILE) {
        if (elect_sync()) {
            profiler.init(num_entries, profiler_ptr, _bid * kWarpCount + warp_id);
            profiler.start(ProfilerTag::Setup);
        }
    }

    // set up cta rank
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // set up smem
    extern __shared__ __align__(1024) char _smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(_smem_ptr));
    nv_bfloat16 *epilogue_smem = reinterpret_cast<nv_bfloat16 *>(_smem_ptr + kMainloopSmemBytes);

    constexpr int kClusterTileM = kClusterM * kTileM;
    constexpr uint32_t i_desc = (1U << 4U)                               // dtype=FP32
                                | (1U << 7U)                             // atype=BF16
                                | (1U << 10U)                            // btype=BF16
                                | ((uint32_t)kTileN >> 3U << 17U)        // MMA_N
                                | ((uint32_t)kClusterTileM >> 4U << 24U) // MMA_M
        ;

    // setup Mbarrier
    __shared__ int tmem_addr[1];
    __shared__ uint64_t tma_mbars[kStageCount];
    __shared__ uint64_t mma_mbars[kStageCount];
    __shared__ uint64_t mainlooop_mbars[2];
    __shared__ uint64_t epilogue_mbars[2];
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
        for (int s = 0; s < 2; ++s) {
            mbarrier_init(_base_mainloop_mbar_addr + 8 * s, 1);
            mbarrier_init(_base_epilogue_mbar_addr + 8 * s, 4 * kClusterM);
        }

        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    // alloc tmem
    if (warp_id == 1) {
        static_assert(kTileN * 2 <= 512);
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile(
            "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(addr),
            "r"(kTileN * 2),
            "n"(kClusterM));
    }
    // prefetch tma desc
    if constexpr (kPrefetchTmaDesc) {
        if (warp_id == 0 && elect_sync()) {
            prefetch_tensormap(&A_tmap);
            prefetch_tensormap(&B_tmap);
            prefetch_tensormap(&C_tmap);
        }
    }

    if constexpr (kClusterM > 1) {
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
    } else {
        __syncthreads();
    }
    if constexpr (DO_PROFILE) {
        if (elect_sync()) {
            profiler.stop();
        }
    }

    const int taddr = tmem_addr[0];
    sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_SETUP>();

    // tma load
    auto load = [&](int iter_k, int mma_phase, int stage_id, int off_m, int off_n) {
        const int mma_mbar_addr = _base_mma_mbar_addr + 8 * stage_id;
        if constexpr (DO_PROFILE) {
            profiler.start(ProfilerTag::WaitMMA);
        }
        wait_mma_slot(mma_mbar_addr, mma_phase);
        sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_PRODUCER_WAIT_MMA>();
        if constexpr (DO_PROFILE) {
            profiler.stop();
        }

        const int A_smem = smem + stage_id * kPerStageSmemBytes;
        const int B_smem = A_smem + kASmemBytes;
        const int tma_mbar_addr = (_base_tma_mbar_addr + 8 * stage_id) & 0xFEFFFFFF;

        // tma issue
        constexpr int cp_size = kASmemBytes + kBSmemBytes;

        if constexpr (DO_PROFILE) {
            profiler.start(ProfilerTag::IssueTMA);
        }
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::"r"(
                         tma_mbar_addr),
            "r"(cp_size)
            : "memory");
        const int off_k_64 = iter_k * (kTileK / 64);
        tma_3d_gmem2smem<kClusterM>(A_smem, &A_tmap, 0, off_m, off_k_64, tma_mbar_addr);
        tma_3d_gmem2smem<kClusterM>(B_smem,
            &B_tmap,
            0,
            off_n + cta_rank * (kTileN / kClusterM),
            off_k_64,
            tma_mbar_addr);
        sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_PRODUCER_ISSUE_TMA>();
        if constexpr (DO_PROFILE) {
            profiler.stop();
        }
    };

    // mma compute
    auto compute = [&](int iter_k, int mainloop_stage, int tma_phase, int stage_id) {
        const int tma_mbar_addr = _base_tma_mbar_addr + 8 * stage_id;
        if constexpr (DO_PROFILE) {
            profiler.start(ProfilerTag::WaitTMA);
        }
        wait_tma(tma_mbar_addr, tma_phase);
        sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_CONSUMER_WAIT_TMA>();
        if constexpr (DO_PROFILE) {
            profiler.stop();
        }

        const int A_smem = smem + stage_id * kPerStageSmemBytes;
        const int B_smem = A_smem + kASmemBytes;
        const int mma_mbar_addr = _base_mma_mbar_addr + 8 * stage_id;

        auto make_desc = [](int addr) -> uint64_t {
            constexpr int SBO = 8 * 128;
            return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) |
                   (2ULL << 61ULL);
        };

        asm volatile("tcgen05.fence::after_thread_sync;");
        const int this_taddr = taddr + mainloop_stage * kTileN;

        if constexpr (DO_PROFILE) {
            profiler.start(ProfilerTag::IssueMMA);
        }
        tcgen05_mma_f16<kClusterM>(this_taddr,
            make_desc(A_smem),
            make_desc(B_smem),
            i_desc,
            iter_k);
        for (int k2 = 1; k2 < 64 / kMmaK; k2++) {
            uint64_t a_desc = make_desc(A_smem + k2 * 32);
            uint64_t b_desc = make_desc(B_smem + k2 * 32);
            tcgen05_mma_f16<kClusterM>(this_taddr, a_desc, b_desc, i_desc, 1);
        }
        for (int k1 = 1; k1 < kTileK / 64; k1++) {
            for (int k2 = 0; k2 < 64 / kMmaK; k2++) {
                uint64_t a_desc = make_desc(A_smem + k1 * kTileM * 128 + k2 * 32);
                uint64_t b_desc = make_desc(B_smem + k1 * (kTileN / kClusterM) * 128 + k2 * 32);
                tcgen05_mma_f16<kClusterM>(this_taddr, a_desc, b_desc, i_desc, 1);
            }
        }

        constexpr int16_t cta_mask = (1 << kClusterM) - 1;

        asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast:"
                     ":cluster.b64 [%0], %1;" ::"r"(mma_mbar_addr),
            "h"(cta_mask),
            "n"(kClusterM)
            : "memory");
        sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_CONSUMER_ISSUE_MMA>();
        if constexpr (DO_PROFILE) {
            profiler.stop();
        }
    };

    auto tmem_to_smem = [&](int mainloop_stage, int n_block, int smem_stage) {
        nv_bfloat16 *smem_out = epilogue_smem + smem_stage * kTileM * kEpilogueBlockN;

        for (int n = 0; n < kEpilogueBlockN / 8; n++) {
            float tmp[8];
            const int row = cta_rank * 128 + warp_id * 32;
            const int col = n_block * kEpilogueBlockN + n * 8 + mainloop_stage * kTileN;
            const int addr = taddr + (row << 16) + col;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=f"(tmp[0]),
                "=f"(tmp[1]),
                "=f"(tmp[2]),
                "=f"(tmp[3]),
                "=f"(tmp[4]),
                "=f"(tmp[5]),
                "=f"(tmp[6]),
                "=f"(tmp[7])
                : "r"(addr));

            // wait
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            if constexpr (!SWAP_AB) {
                nv_bfloat162 out[4];
                for (int i = 0; i < 4; i++) {
                    out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});
                }
                const int chunk = tid * (kEpilogueBlockN / 8) + n;
                const int swizzled_chunk = (chunk & ~7) + ((chunk & 7) ^ ((chunk >> 3) & 3));
                reinterpret_cast<int4 *>(smem_out + swizzled_chunk * 8)[0] =
                    reinterpret_cast<int4 *>(out)[0];
            } else {
                // Form one kEpilogueBlockN x 128 tile from the four
                // transposed TMEM panels.
                const int dst_row = n * 8;
                const int dst_col = warp_id * kWarpSize + tid % kWarpSize;
                for (int i = 0; i < 8; ++i) {
                    const int elem = (dst_row + i) * kTileM + dst_col;
                    const int chunk = elem / 8;
                    const int swizzled_chunk = (chunk & ~7) + ((chunk & 7) ^ ((chunk >> 3) & 3));
                    smem_out[swizzled_chunk * 8 + elem % 8] = __float2bfloat16_rn(tmp[i]);
                }
            }
        }
    };

    auto epilogue = [&](int mainloop_stage, int off_m, int off_n) {
        constexpr int n_block_num = kTileN / kEpilogueBlockN;
        static_assert(kTileN % kEpilogueBlockN == 0);
        static_assert(kEpilogueBlockN % 8 == 0);

        auto release_tmem = [&]() {
            if (elect_sync()) {
                const int mbar_addr = (_base_epilogue_mbar_addr + mainloop_stage * 8) & 0xFEFFFFFF;
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(mbar_addr)
                    : "memory");
            }
        };

        // Only the four epilogue warps participate in named barrier 1.
        tmem_to_smem(mainloop_stage, 0, 0);
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if constexpr (n_block_num == 1) {
            release_tmem();
        }

        uint64_t cache_policy = 0;
        if (tid == 0) {
            cache_policy = make_epilogue_cache_policy();
        }
        for (int n_block = 0; n_block < n_block_num; n_block++) {
            const int smem_stage = n_block % kEpilogueStageCount;
            const int stage_smem =
                smem + kMainloopSmemBytes +
                smem_stage * kTileM * kEpilogueBlockN * static_cast<int>(sizeof(nv_bfloat16));

            // TMA consumes the current stage while the epilogue warps fill the
            // other stage from tensor memory.
            if (tid == 0) {
                if constexpr (!SWAP_AB && kEpilogueBlockN == 32) {
                    tma_2d_smem2gmem(cache_policy,
                        stage_smem,
                        &C_tmap,
                        off_n + n_block * kEpilogueBlockN,
                        off_m);
                } else if constexpr (!SWAP_AB) {
                    tma_3d_smem2gmem(cache_policy,
                        stage_smem,
                        &C_tmap,
                        0,
                        (off_n + n_block * kEpilogueBlockN) / 32,
                        off_m);
                } else {
                    tma_3d_smem2gmem(cache_policy,
                        stage_smem,
                        &C_tmap,
                        0,
                        off_m / 32,
                        off_n + n_block * kEpilogueBlockN);
                }
            }

            if (n_block + 1 < n_block_num) {
                if (n_block > 0) {
                    if (tid == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                }

                const int next_stage = (n_block + 1) % kEpilogueStageCount;
                tmem_to_smem(mainloop_stage, n_block + 1, next_stage);
                asm volatile("bar.sync 1, 128;" ::: "memory");
                if (n_block + 2 == n_block_num) {
                    release_tmem();
                }
            }
        }

        if (tid == 0) {
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
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
            for (int iter_k = 0; iter_k < K / kTileK; ++iter_k) {
                load(iter_k, mma_phase, mma_stage, off_m, off_n);
                mma_stage = (mma_stage + 1) % kStageCount;
                if (mma_stage == 0) {
                    mma_phase ^= 1;
                }
            }
        }
    }

    // mma warp
    if (cta_rank == 0 && warp_id == 5 && elect_sync()) {
        int tma_phase = 0;
        int tma_stage = 0;

        int mainloop_stage = 0;
        int epilogue_phase = 1;

        for (int bid = _bid; bid < grid_size; bid += grid_x) {
            if constexpr (DO_PROFILE) {
                profiler.start(ProfilerTag::WaitEpilogue);
            }
            wait_epilogue(_base_epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);
            sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_CONSUMER_WAIT_EPILOGUE>();
            if constexpr (DO_PROFILE) {
                profiler.stop();
            }

            for (int iter_k = 0; iter_k < K / kTileK; ++iter_k) {
                compute(iter_k, mainloop_stage, tma_phase, tma_stage);
                tma_stage = (tma_stage + 1) % kStageCount;
                if (tma_stage == 0) {
                    tma_phase ^= 1;
                }
            }

            constexpr int16_t cta_mask = (1 << kClusterM) - 1;
            asm volatile(
                "tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast:"
                ":cluster.b64 [%0], %1;" ::"r"(_base_mainloop_mbar_addr + 8 * mainloop_stage),
                "h"(cta_mask),
                "n"(kClusterM)
                : "memory");

            mainloop_stage = (mainloop_stage + 1) % 2;
            if (mainloop_stage == 0) {
                epilogue_phase ^= 1;
            }
        }
    }

    // epilogue
    if (warp_id < 4) {
        int mainloop_stage = 0;
        int mainloop_phase = 0;

        for (int bid = _bid; bid < grid_size; bid += grid_x) {
            if constexpr (DO_PROFILE)
                if (elect_sync())
                    profiler.start(ProfilerTag::WaitMainloop);
            wait_mainloop(_base_mainloop_mbar_addr + 8 * mainloop_stage, mainloop_phase);
            sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_EPILOGUE_WAIT_MAINLOOP>();
            asm volatile("tcgen05.fence::after_thread_sync;");
            if constexpr (DO_PROFILE)
                if (elect_sync())
                    profiler.stop();
            auto [off_m, off_n] = compute_bid(bid);
            if constexpr (DO_PROFILE)
                if (elect_sync())
                    profiler.start(ProfilerTag::Epilogue);
            epilogue(mainloop_stage, off_m, off_n);
            sm110a::pipeline_delay<SM110A_PIPELINE_DELAY_EPILOGUE>();
            if constexpr (DO_PROFILE)
                if (elect_sync())
                    profiler.stop();
            mainloop_stage = (mainloop_stage + 1) % 2;
            if (mainloop_stage == 0) {
                mainloop_phase ^= 1;
            }
        }
    }

    if constexpr (kClusterM > 1) {
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
    } else {
        __syncthreads();
    }
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" ::"r"(taddr),
            "r"(kTileN * 2),
            "n"(kClusterM));
    }
    if constexpr (DO_PROFILE) {
        if (elect_sync()) {
            profiler.flush();
        }
    }
}

} // namespace

template <int kTileM,
    int kTileN,
    int kTileK,
    int kClusterM,
    int kEpilogueBlockN,
    bool DO_PROFILE,
    bool SWAP_AB>
void launch_bf16_gemm_impl(
    const __nv_bfloat16 *A_ptr,
    const __nv_bfloat16 *B_ptr,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K,
    int64_t *profiler_ptr,
    int num_entries,
    int max_active_clusters,
    cudaStream_t stream) {

    using Config = KernelConfig<kTileM, kTileN, kTileK, kClusterM, kEpilogueBlockN>;
    constexpr int kStageCount = Config::kStageCount;
    constexpr int kDynamicSmemBytes = Config::kDynamicSmemBytes;

    const int logic_M = M;
    const int logic_N = N;
    if constexpr (SWAP_AB) {
        const __nv_bfloat16 *tmp_ptr = A_ptr;
        A_ptr = B_ptr;
        B_ptr = tmp_ptr;
        const int tmp_dim = M;
        M = N;
        N = tmp_dim;
    }

    CUtensorMap A_tmap, B_tmap, C_tmap;

    static_assert(
        kMacroSwizzleM * kMacroSwizzleN <=
        (32 * 1024 * 1024) / (static_cast<int>(sizeof(nv_bfloat16)) * kTileM * kTileK +
                                 static_cast<int>(sizeof(nv_bfloat16)) * kTileN * kTileK));
    assert(M % (kClusterM * kTileM) == 0);
    assert(N % kTileN == 0);
    assert(K % kTileK == 0);

    init_3D_tmap(&A_tmap,
        A_ptr,
        M,
        K,
        kTileM,
        kTileK,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_3D_tmap(&B_tmap,
        B_ptr,
        N,
        K,
        kTileN / kClusterM,
        kTileK,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    if constexpr (SWAP_AB) {
        init_3D_output_tmap(&C_tmap, C_ptr, logic_M, logic_N, kEpilogueBlockN, kTileM);
    } else if constexpr (kEpilogueBlockN == 32) {
        init_2D_output_tmap(&C_tmap, C_ptr, logic_M, logic_N, kTileM, kEpilogueBlockN);
    } else {
        init_3D_output_tmap(&C_tmap, C_ptr, logic_M, logic_N, kTileM, kEpilogueBlockN);
    }

    constexpr int mbarrier_smem = 256;
    const int smem_size = kDynamicSmemBytes + mbarrier_smem;

    // The kernel has a fixed compile-time cluster size.  A grid with this many
    // clusters therefore contains max_active_clusters * kClusterM CTAs.
    assert(max_active_clusters > 0);
    const int grid = max_active_clusters * kClusterM;

    auto kernel =
        bf16_gemm_kernel<kTileM, kTileN, kTileK, kClusterM, kEpilogueBlockN, DO_PROFILE, SWAP_AB>;

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    kernel<<<grid, kBlockThreads, smem_size, stream>>>(A_tmap,
        B_tmap,
        C_tmap,
        M,
        N,
        K,
        profiler_ptr,
        num_entries);
}

// Non-template wrappers for pre-compiled path — instantiate with the #define defaults.
void launch_bf16_gemm(
    const __nv_bfloat16 *A_ptr,
    const __nv_bfloat16 *B_ptr,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K,
    bool swap_ab,
    cudaStream_t stream,
    int max_active_clusters) {
    if (max_active_clusters <= 0) {
        int device = 0;
        int sm_count = 0;
        assert(cudaGetDevice(&device) == cudaSuccess);
        assert(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                      device) == cudaSuccess);
        max_active_clusters = sm_count / SM110A_BF16_CLUSTER_M;
    }
    assert(max_active_clusters > 0);
    if (swap_ab) {
        launch_bf16_gemm_impl<SM110A_BF16_TILE_M,
            SM110A_BF16_TILE_N,
            SM110A_BF16_TILE_K,
            SM110A_BF16_CLUSTER_M,
            SM110A_BF16_EPILOGUE_BLOCK_N,
            false,
            true>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0,
                  max_active_clusters, stream);
    } else {
        launch_bf16_gemm_impl<SM110A_BF16_TILE_M,
            SM110A_BF16_TILE_N,
            SM110A_BF16_TILE_K,
            SM110A_BF16_CLUSTER_M,
            SM110A_BF16_EPILOGUE_BLOCK_N,
            false,
            false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0,
                   max_active_clusters, stream);
    }
}

extern "C" void sm110a_bf16_gemm_launch(
    const __nv_bfloat16* d_a, const __nv_bfloat16* d_b,
    __nv_bfloat16* d_c, int m, int n, int k, bool swap_ab,
    int max_active_clusters, cudaStream_t stream) {
  launch_bf16_gemm(d_a, d_b, d_c, m, n, k, swap_ab, stream,
                   max_active_clusters);
}

void launch_bf16_gemm_profile(
    const __nv_bfloat16 *A_ptr,
    const __nv_bfloat16 *B_ptr,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K,
    int64_t *profiler_ptr,
    int num_entries,
    bool swap_ab,
    cudaStream_t stream,
    int max_active_clusters) {
    if (max_active_clusters <= 0) {
        int device = 0;
        int sm_count = 0;
        assert(cudaGetDevice(&device) == cudaSuccess);
        assert(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                      device) == cudaSuccess);
        max_active_clusters = sm_count / SM110A_BF16_CLUSTER_M;
    }
    assert(max_active_clusters > 0);
    if (swap_ab) {
        launch_bf16_gemm_impl<SM110A_BF16_TILE_M,
            SM110A_BF16_TILE_N,
            SM110A_BF16_TILE_K,
            SM110A_BF16_CLUSTER_M,
            SM110A_BF16_EPILOGUE_BLOCK_N,
            true,
            true>(A_ptr, B_ptr, C_ptr, M, N, K, profiler_ptr, num_entries,
                  max_active_clusters, stream);
    } else {
        launch_bf16_gemm_impl<SM110A_BF16_TILE_M,
            SM110A_BF16_TILE_N,
            SM110A_BF16_TILE_K,
            SM110A_BF16_CLUSTER_M,
            SM110A_BF16_EPILOGUE_BLOCK_N,
            true,
            false>(A_ptr, B_ptr, C_ptr, M, N, K, profiler_ptr, num_entries,
                   max_active_clusters, stream);
    }
}
