// =============================================================================
// SM110A MXFP4 GEMM — Overview
// =============================================================================
//
// Target hardware
// ---------------
//   NVIDIA Blackwell SM (SM110A, sm_110a). Assumes first-class support for
//   Tensor Memory (TMEM), distributed shared memory (DSMEM), thread-block
//   clusters and the Tensor Memory Accelerator (TMA) bulk-copy engine. Uses
//   the block-scaled mxf4 tcgen05.mma instruction family introduced on
//   Blackwell, together with UE8M0 block-32 micro-scales.
//
// Problem
// -------
//   C[M,N] = scale(A[M,K]) @ scale(B[N,K])
//   with A, B stored as packed E2M1 (MXFP4), scales stored as UE8M0 with one
//   scale per 32 contiguous K elements, and C stored as nv_bfloat16.
//
//   - M must be a multiple of kClusterM * kTileM.
//   - N must be a multiple of 128 (scale-tile boundary) and kTileN.
//   - K must be a multiple of kTileK (one or more TileK stages per output tile).
//
//   A "SWAP_AB" template variant reuses the kernel for the transposed
//   product B[N,K] @ A[K,M] by treating (B,A) as the operands on the host
//   side and rebuilding the matching C tensor-map encoder
//   (see launch_mxfp4_gemm_impl).
//
//   MMA shape is (128*ClusterM) x TileN x 64 with TileN ∈ {64, 128, 256},
//   ClusterM ∈ {1, 2}, and TileK ∈ {64, 128, 256} realized as TileK/64 such
//   mma instructions:
//
//     tcgen05.mma.cta_group::ClusterM.kind::mxf4.block_scale.scale_vec::2X
//
//   Per CTA, each mainloop slot holds a 128B-swizzled FP4 A[128,TileK] tile,
//   a 128B-swizzled FP4 B[TileN/ClusterM,TileK] tile, and contiguous
//   scale-factor tiles covering the full TileK span.  With ClusterM=2 both
//   peer CTAs consume identical SFB data, so cluster rank 0 loads each SFB
//   tile once and TMA-multicasts it into the same shared-memory offset in
//   both CTAs.
//
// Bottleneck model
// ----------------
//   Back-of-the-envelope estimate for the default 128x256x64 tile with
//   kClusterM == 2 (2 CTAs cooperate along M):
//
//     Per-tile loads (one K tile = 64 elements):
//         A   : 128 rows * 64 elements * 0.5 B  (E2M1)         =   4 KiB / CTA
//         B   : 128 cols * 64 elements * 0.5 B  (E2M1)         =   4 KiB / CTA
//         SFA : 128 rows * 2 scales   * 1 B   (UE8M0, kTileK=64) = 256 B logical
//               (stored in a 512 B padded 128x4 scale tile for tcgen05.cp)
//         SFB : 2 tiles    * 512 B     (UE8M0 padded tiles, TileN=256) = 1 KiB
//         Total per output tile: ~ 9.5 KiB of K-resident data per CTA
//
//     Over the full problem (M = 128 * Mtiles, N = 256 * Ntiles):
//         Total bytes moved ~= K * M * N / 128  (operand + scale bytes,
//                                                count is approximate).
//         Memory-bound time = total_bytes / peak_L2_or_DRAM_bandwidth.
//         Compute-bound time = K * M * N / peak_MXFP4_MMA_rate (each mxf4
//                              MMA produces twice the elements of an
//                              equivalent BF16 MMA per cycle).
//
//     For a 4K^3 problem the two limits are roughly:
//         * L2-hit TMA: K * M * N / 128 / ~1.281 TB/s ~= 1.56 ms.
//         * MMA issue : K * M * N          / ~500 TF/s ~= 0.28 ms.
//     (Effective rates depend on the deployed Blackwell part; treat these
//     numbers as a sanity check, not a contract.)
//
//     Per-stage latency model (single output tile, kClusterM = 2):
//         * MMA work        : 128 * 256 * 64 / 16K  ~= 128 cycles.
//         * DRAM -> L2 fetch: (2 * 128 * 64 * 0.5 + 2 * 128 * 64 * 0.5) / 8 ~= 1024 cycles.
//         * L2 hit on fetch : (2 * 128 * 64 * 0.5 + 2 * 128 * 64 * 0.5) / 40 ~= 205 cycles.
//         * Scale cp+mma   : scales live in TMEM (tcgen05.cp) once per K=64
//                            slab, gating one mma and ~16 cycles per slab.
//
//   Reading the three numbers together explains the design pressure points
//   addressed below:
//     * The MMA path is comparable to an L2-hit fetch, so compute and memory
//       must overlap tightly -- that is the role of the multi-stage mainloop.
//     * An L2 miss inflates the fetch by ~5x, so the macro-swizzle and the
//       L2-friendly tile order are first-order optimizations, not niceties.
//     * Scale-factor copy into TMEM is on the critical path of every MMA,
//       so each stage must have a dedicated TMEM scale slot.  TileN=256 is
//       capacity constrained and reuses a single 12-column slot, forcing a
//       serialized cp->mma ordering that narrows the advantage of wider N.
//     * TMA descriptor prefetch matters because every mainloop iteration
//       consumes a fresh A/B/SFA/SFB tile, so the very first load cannot
//       afford to stall on the descriptor fetch.
//
// Functional summary
// ------------------
//   * Persistent tiled kernel: launches one CTA per SM (grid == kSmCount == 20)
//     and walks the (grid_m * grid_n) output-tile grid in round-robin so each
//     CTA accumulates multiple output tiles per launch.
//   * Each CTA executes a 6-warp pipeline (192 threads):
//       warp 4              – load warp (TMA gmem→smem for A, B, SFA, SFB)
//       warp 5 (cta_rank 0) – MMA warp   (tcgen05.mma mxf4 on TMEM)
//       warps 0..3          – epilogue warps (tmem→smem→gmem via TMA)
//   * Clusters of size kClusterM along M cooperate via DSMEM and multicast:
//     every cluster CTA loads its own A row and its slice of B; rank 0 also
//     loads SFB and multicasts it to peers; the MMA is issued once with
//     `cta_group::kClusterM`.
//   * FP32 accumulator lives in TMEM (kTileN or 2*kTileN columns depending
//     on TileN) and is drained to shared memory via tcgen05.ld, then to
//     global memory via 2-D/3-D TMA bulk stores.
//   * Scale-factor slot policy in TMEM:
//       - TileN ≤ 128: kStageCount * (TileK/64) distinct slots of 8 columns,
//         one per in-flight SMEM stage and K=64 slab.  All scales for a
//         TileK stage can be prefetched before its MMAs without aliasing any
//         in-flight stage.
//       - TileN == 256: a single 12-column slot is reused; cp() runs
//         immediately before each MMA to refresh the live slabs.
//   * Two entry points:
//       launch_mxfp4_gemm           – regular, unprofiled path.
//   * SWAP_AB template variant reuses the same kernel binary for the
//     transposed product (no second hand-written kernel); only the host-side
//     launch swaps pointer/dimension arguments and selects the matching
//     C tensor-map encoder.
//
// Optimizations
// -------------
//   1. TMA for all bulk data movement.
//        * Loads  : 3-D tensor maps for A and B (128 B swizzle); 4-D maps
//                   for SFA and SFB covering the full TileK scale span.
//        * Stores : 2-D map for the KEpilogueBlockN==32 fast path, 3-D map
//                   otherwise; 64 B swizzle. Register traffic on the memory
//                   side of the pipeline is zero.
//   2. Multi-stage software pipeline (kStageCount stages for A/B/SFA/SFB in smem).
//        Stage count is chosen at *compile time* from a 232 KiB smem budget
//        (cudaDevAttrMaxSharedMemoryPerBlockOptin) minus the epilogue
//        double buffer and the static/mbarrier reservations
//        (KernelConfig::kStageCount), capped at six stages.  The largest
//        configuration that still fits is selected automatically.
//   3. Double-buffered TMEM accumulator (kAccumulatorStageCount).
//        TileN ≤ 128 uses disjoint banks.  TileN=256 makes two logical
//        256-column banks overlap by 16 columns, leaving room for its
//        12-column scale slot within the 512-column TMEM allocation.  The
//        epilogue drains the shared edge first and releases the bank as soon
//        as that TMEM load has completed, letting MMA overwrite the shared
//        columns while the epilogue stores the disjoint remainder.
//   4. Scale-factor TMEM slot policy (kScaleTmemSlotCount).
//        Narrow-N keeps every in-flight SMEM stage and K=64 slab's scales in
//        a distinct TMEM slot, allowing the load warp to prefetch the entire
//        TileK stage's scales ahead of the MMA consumer.  The mma_mbar
//        handshake prevents a stage's SMEM and TMEM scale slots from being
//        reused until its asynchronous MMA has completed.
//        TileN=256 retains one 12-column scale slot and refreshes it
//        immediately before each MMA, trading prefetch depth for footprint.
//   5. SFB cluster multicast (ClusterM=2 only).
//        Both peer CTAs read identical SFB data, so rank 0 issues a single
//        TMA with a cta_mask and shares the resulting SMEM bytes with rank 1.
//        This halves SFB traffic from DRAM.
//   6. Producer/consumer decoupling with two mbarrier rings:
//          tma_mbars[kStageCount]   – TMA completion -> MMA consumer.
//          mma_mbars[kStageCount]   – MMA commit     -> TMA producer.
//        A separate `mainlooop_mbars[2]` array — note the misspelling kept
//        on purpose — signals mainloop-stage completion to the epilogue;
//        `epilogue_mbars[2]` releases accumulator banks back to the MMA
//        warp.  Phase parity bits per ring track the 2-cycle wrap.
//   7. L2-friendly tile schedule (`compute_bid`).
//        The output grid is mapped through an 8x8 macro-swizzle. With
//        `kMacroSwizzleMFast=true`, M is iterated fastest inside a macro
//        band so consecutive CTAs read overlapping K-resident operands
//        and reuse L2 lines.
//   8. TMA descriptor prefetch (`prefetch.tensormap`) issued once at startup,
//        so the first TMA does not stall on the descriptor fetch.
//   9. Pipelined epilogue:
//          * 2 smem slots per mainloop_stage (`kEpilogueStageCount == 2`).
//          * `cp.async.bulk.wait_group.read 1` + `bar.sync 1, 128` keeps the
//            TMA store of slot N concurrent with the tcgen05.ld that fills
//            slot N+1.
//  10. `SWAP_AB` template variant reuses the same kernel binary for the
//      transposed product (no second hand-written kernel); only the host-side
//      launch swaps pointer/dimension arguments and selects the matching
//      C tensor-map encoder.
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
//                 mma_mbars[kStageCount] (count=1), mainlooop_mbars[kAccumulatorStageCount] (1),
//                 epilogue_mbars[kAccumulatorStageCount] (4*kClusterM); then
//                 `fence.mbarrier_init.release.cluster`.
//      warp 1   : `tcgen05.alloc.cta_group::kClusterM` reserves kTmemColumnCapacity
//                 (= 512) TMEM columns (returned in tmem_addr[0]).
//      warp 0*  : `prefetch.tensormap` for A_tmap / B_tmap / SFA_tmap / SFB_tmap
//                 / C_tmap.
//      barrier.cluster.arrive+wait (or __syncthreads when kClusterM==1).
//
//   1. Main loop — repeated `K / kTileK` iterations
//      ------------------------------------------------
//      ┌─── L (load warp) ──────────────────────────────────────────────┐
//      │  wait(mma_mbar[mma_stage], mma_phase)        // slot free?     │
//      │  mbar.arrive.expect_tx(tma_mbar[tma_stage], cp_size)           │
//      │  cp.async.bulk.tensor.3d A: smem[stage]<-A_tmap(k_off,m_off,0) │
//      │  cp.async.bulk.tensor.3d B: smem[stage]+A_sz <-                │
//      │      B_tmap(k_off, n_off + rank*(TileN/Cluster), 0)            │
//      │  cp.async.bulk.tensor.4d SFA: ... <- SFA_tmap covering all     │
//      │      (TileK/64) K=64 slabs in this TileK stage (one box)       │
//      │  for scale_tile in [0, kSFBScaleTileCount):                    │
//      │      cp.async.bulk.tensor.4d SFB(scale_tile) <- SFA_tmap(...)  │
//      │      ClusterM==2: rank==0 only, with multicast cta_mask        │
//      │  mma_stage = (mma_stage+1)%kStageCount;                        │
//      │  mma_phase ^= 1   when mma_stage wraps to 0                    │
//      └─── (consumes mma_mbar; produces tma_mbar) ─────────────────────┘
//
//      ┌─── M (MMA warp, one issued per cluster) ───────────────────────┐
//      │  wait(tma_mbar[tma_stage], tma_phase)        // data ready?    │
//      │  tcgen05.fence::after_thread_sync                              │
//      │  taddr_use = taddr + mainloop_stage * kAccumulatorStageStride  │
//      │  for k1 in [0, kTileK/64):                                     │
//      │    TileN ≤ 128:                                                │
//      │      tcgen05.cp SFA -> TMEM scale slot                         │
//      │          (stage_id * TileK/64 + k1) * kScaleColsPerMma         │
//      │      tcgen05.cp SFB(scale_tile) -> TMEM scale slot             │
//      │      The cp() chains can be batched ahead of the MMA.          │
//      │    TileN == 256:                                               │
//      │      cp() runs immediately before each MMA in slot 0.          │
//      │    tcgen05.mma.cta_group::kClusterM                            │
//      │        .kind::mxf4.block_scale.scale_vec::2X                     │
//      │        tmem[taddr_use] ^= A_sm^k1 * B_sm^k1                    │
//      │        scale slots as above                                    │
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
//   3. Epilogue (warps 0..3; FP32 in TMEM, BF16 to gmem)
//      ------------------------------------------------
//      ┌─── E ──────────────────────────────────────────────────────────┐
//      │  wait(mainloop_mbar[mainloop_stage], mainloop_phase)           │
//      │  tcgen05.fence::after_thread_sync                              │
//      │  For n_block in [0, kTileN/kEpilogueBlockN):                   │
//      │     smem_slot = n_block % kEpilogueStageCount                  │
//      │     [fill] tmem->smem of (kEpilogueBlockN x kTileM) block      │
//      │             via tcgen05.ld (x8, .32x32b) + swizzle write,      │
//      │             fp32 -> bf16 conversion per element                │
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
//      turns the grid persistent: kSmCount CTAs share the tile grid evenly.
//
//   5. Teardown
//      ------------------------------------------------
//      cluster barrier -> warp 0 `tcgen05.dealloc.cta_group::kClusterM`
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
//   * A 128 B swizzle always reserves a full 128 B span per row, even when
//     a TileK=64/128 FP4 transfer only populates 32 B/64 B of that span.
//   * Scale tensors use the Blackwell SWIZZLE_32_4_4 physical format
//     accepted by torch._scaled_mm/cuBLASLt.  A logical [128,4] tile is
//     stored as 32 rows of 16 bytes: rows r, r+32, r+64, r+96 are
//     concatenated for each r.  This is exactly the layout consumed by
//     `tcgen05.cp.cta_group::C.32x128b.warpx4`.
//   * TileN=64 is realized by selecting one half of a containing 128-row
//     blocked scale tile in TMEM; the SFB TMEM column is computed as
//     `(off_n % 128) / 32`.
//   * This kernel targets SM110A only. It will not assemble on older SMs
//     that lack the tcgen05.* / bulk.tensor.* opcodes or the mxf4 MMA.
// =============================================================================

#include <cassert>
#include <cstdint>
#include <cudaTypedefs.h> // required for CUtensorMap

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include "mxfp4_gemm.h"

#ifndef SM110A_MXFP4_TILE_M
#define SM110A_MXFP4_TILE_M 128
#endif
#ifndef SM110A_MXFP4_TILE_N
#define SM110A_MXFP4_TILE_N 256
#endif
#ifndef SM110A_MXFP4_TILE_K
#define SM110A_MXFP4_TILE_K 64
#endif
#ifndef SM110A_MXFP4_CLUSTER_M
#define SM110A_MXFP4_CLUSTER_M 2
#endif
#ifndef SM110A_MXFP4_EPILOGUE_BLOCK_N
#define SM110A_MXFP4_EPILOGUE_BLOCK_N 32
#endif
#ifndef SM110A_MXFP4_SPLIT_K
#define SM110A_MXFP4_SPLIT_K 1
#endif
#ifndef SM110A_MXFP4_USE_TMA_PREFETCH
#define SM110A_MXFP4_USE_TMA_PREFETCH 0
#endif

namespace {

constexpr int kMmaK = 64;
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
constexpr bool kPrefetchTmaData = SM110A_MXFP4_USE_TMA_PREFETCH != 0;
constexpr int kSmCount = 20;

constexpr int kMacroSwizzleM = 8;
constexpr int kMacroSwizzleN = 8;
constexpr bool kMacroSwizzleMFast = true;

template <int kTileM, int kTileN, int kTileK, int kClusterM, int kEpilogueBlockN, int kSplitK>
struct KernelConfig {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert((kSplitK == 1 || kSplitK == 2 || kSplitK == 4 || kSplitK == 8) &&
                  kClusterM * kSplitK <= 8);
    static_assert(kTileM == 128);
    static_assert(kTileN == 64 || kTileN == 128 || kTileN == 256);
    static_assert(kTileK == 64 || kTileK == 128 || kTileK == 256);
    static_assert(kEpilogueBlockN == 32 || kEpilogueBlockN == 64 || kEpilogueBlockN == 128);
    static_assert(kTileN % kEpilogueBlockN == 0);
    static_assert(kTileN % kClusterM == 0);
    static_assert(kTileK % 64 == 0);
    static_assert(kTileK % kMmaK == 0);

    static constexpr int kClusterTileM = kClusterM * kTileM;
    static constexpr int kATmaBytes = kTileM * kTileK / 2;
    static constexpr int kBTmaBytes = (kTileN / kClusterM) * kTileK / 2;
    // A 128B swizzle always reserves a full 128B span per row, even when a
    // TileK=64/128 transfer only populates 32B/64B of that span.
    static constexpr int kASmemBytes = kTileM * 128;
    static constexpr int kBSmemBytes = (kTileN / kClusterM) * 128;
    static constexpr int kMmaStepsPerStage = kTileK / kMmaK;
    // K=64 MXFP4 supplies 128x2 live E8M0 scales.  Keep the 128x4 physical
    // tile (512 B) required by tcgen05.cp.32x128b.warpx4.
    static constexpr int kSFASmemBytes = kTileM * (kTileK / 16);
    // Scale storage is blocked in 128-row, 512-byte units.  TileN=64 loads
    // its containing 128-row tile and selects the appropriate half in TMEM.
    static constexpr int kSFBScaleTileCount = (kTileN + 127) / 128;
    static constexpr int kSFBSmemBytes = kMmaStepsPerStage * kSFBScaleTileCount * 512;
    static constexpr int kPerStagePayloadBytes =
        kASmemBytes + kBSmemBytes + kSFASmemBytes + kSFBSmemBytes;
    static constexpr int kPerStageSmemBytes =
        (kPerStagePayloadBytes + 1023) & ~1023;
    static constexpr int kEpilogueStageBytes =
        kTileM * kEpilogueBlockN * static_cast<int>(sizeof(nv_bfloat16));
    static constexpr int kEpilogueSmemBytes = kEpilogueStageCount * kEpilogueStageBytes;
    static constexpr int kReductionSmemBytes =
        kSplitK == 1 ? 0 : kTileM * kEpilogueBlockN * sizeof(float);
    // Cap the ring at six stages, while reducing it for wider K tiles so the
    // mainloop and epilogue remain within Blackwell's per-CTA SMEM budget.
    static constexpr int kSmemLimitedStageCount =
        (kMaxOptInSmemBytes - kStaticBarrierSmemBytes -
         kMbarrierSmemBytes - kEpilogueSmemBytes - kReductionSmemBytes) /
        kPerStageSmemBytes;
    static constexpr int kStageCount = kSmemLimitedStageCount < 6 ? kSmemLimitedStageCount : 6;
    static_assert(kStageCount >= 2);
    // A uses two live E8M0 subcolumns but its 32x128b transfer occupies a
    // four-column TMEM tile.  Keep the two padding columns before B so both
    // tcgen05.cp destinations remain 4-column aligned.
    static constexpr int kScaleColsPerMma = 4 + 4 * kSFBScaleTileCount;
    // Narrow-N kernels have ample TMEM after their two accumulator stages.
    // Keep every in-flight SMEM stage and K=64 slab's scales in a distinct
    // slot. The mma_mbar handshake prevents a stage's SMEM and TMEM scale
    // slots from being reused until its asynchronous MMA has completed.
    // TileN=256 is capacity constrained and reuses a single scale slot.
    static constexpr int kScaleTmemSlotCount =
        kTileN < 256 ? kStageCount * kMmaStepsPerStage : 1;
    static constexpr int kScaleTmemSlotStride = kScaleColsPerMma;
    static constexpr int kScaleTmemCols =
        kScaleTmemSlotStride * kScaleTmemSlotCount;
    // A 16-column overlap is aligned for tcgen05.ld (.x8 values per thread)
    // and fits in every supported epilogue block size.  At TileN=256 this
    // changes the footprint from 2*256 + 12 (impossible) to
    // 2*256 - 16 + 12 = 508 columns.
    static constexpr int kAccumulatorOverlapCols = kTileN == 256 ? 16 : 0;
    static constexpr int kAccumulatorStageCount = 2;
    static constexpr int kAccumulatorStageStride =
        kTileN - kAccumulatorOverlapCols;
    static constexpr int kAccumulatorTmemCols =
        kAccumulatorStageCount * kTileN - kAccumulatorOverlapCols;
    static_assert(kAccumulatorTmemCols + kScaleTmemCols <= kTmemColumnCapacity);
    static_assert(kAccumulatorOverlapCols == 0 ||
                  (kAccumulatorOverlapCols == 16 &&
                   kEpilogueBlockN >= kAccumulatorOverlapCols));
    static constexpr int kMainloopSmemBytes = kStageCount * kPerStageSmemBytes;
    static constexpr int kDynamicSmemBytes =
        kMainloopSmemBytes + kEpilogueSmemBytes + kReductionSmemBytes;
    static_assert(kStaticBarrierSmemBytes + kDynamicSmemBytes +
                      kMbarrierSmemBytes <=
                  kMaxOptInSmemBytes);
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
    int dst, const void *tmap_ptr, int x, int y, int z, int w,
    int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%7 "
        "[%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr), "n"(CTA_GROUP) : "memory");
}

template <int CTA_GROUP = 1>
__device__ inline void tma_4d_gmem2smem_multicast(
    int dst, const void *tmap_ptr, int x, int y, int z, int w,
    int mbar_addr, int16_t cta_mask) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes."
        "multicast::cluster.cta_group::%8 "
        "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr), "h"(cta_mask), "n"(CTA_GROUP) : "memory");
}

__device__ inline void tma_3d_prefetch_l2(
    const void *tmap_ptr, int x, int y, int z) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];"
        :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(z) : "memory");
}

__device__ inline void tma_4d_prefetch_l2(
    const void *tmap_ptr, int x, int y, int z, int w) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];"
        :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
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
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(taddr), "l"(smem_desc) : "memory");
    } else {
        asm volatile(
            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
            :: "r"(taddr), "l"(smem_desc) : "memory");
    }
}

template <int kClusterM>
__device__ inline void tcgen05_mma_mxf4(
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
                     "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
                     "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
                     "}" ::"r"(taddr),
            "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr),
            "r"(sfb_taddr), "r"(enable_input_d) : "memory");
    } else {
        asm volatile("{\n\t"
                     ".reg .pred p;\n\t"
                     "setp.ne.b32 p, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X "
                     "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
                     "}" ::"r"(taddr),
            "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr),
            "r"(sfb_taddr), "r"(enable_input_d) : "memory");
    }
}

template <int kClusterM, int kSplitK = 1>
__device__ inline void tcgen05_commit(int mbar_addr, int split_rank = 0) {
    if constexpr (kClusterM == 1) {
        asm volatile("tcgen05.commit.cta_group::1."
                     "mbarrier::arrive::one.shared::cluster.b64 [%0];" ::
                         "r"(mbar_addr)
            : "memory");
    } else {
        if constexpr (kSplitK == 1) {
            constexpr int16_t cta_mask = (1 << kClusterM) - 1;
            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
                         "multicast::cluster.b64 [%0], %1;" ::
                             "r"(mbar_addr), "h"(cta_mask)
                : "memory");
        } else {
            const int16_t cta_mask = ((1 << kClusterM) - 1) << (split_rank * kClusterM);
            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
                         "multicast::cluster.b64 [%0], %1;" ::
                             "r"(mbar_addr), "h"(cta_mask)
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

inline void init_fp4_tmap(
    CUtensorMap *tmap,
    const uint8_t *ptr,
    uint64_t global_height,
    uint64_t global_width,
    uint32_t shared_height,
    uint32_t shared_width,
    CUtensorMapSwizzle swizzle) {

    constexpr uint32_t rank = 3;

    uint64_t global_dim[rank] = {global_width, global_height, 1};

    uint64_t global_strides[rank - 1] = {
        global_width / 2, global_width * global_height / 2};

    uint32_t box_dim[rank] = {shared_width, shared_height, 1};
    uint32_t element_strides[rank] = {1, 1, 1};

    auto err = cuTensorMapEncodeTiled(tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
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
    // K=64 scale tiles are contiguous in both global and shared memory.  Make
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
void mxfp4_gemm_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap SFA_tmap,
    const __grid_constant__ CUtensorMap SFB_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    int M,
    int N,
    int K) {

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
    float *reduction_smem = reinterpret_cast<float *>(
        _smem_ptr + kMainloopSmemBytes + Config::kEpilogueSmemBytes);

    constexpr int kClusterTileM = kClusterM * kTileM;
    constexpr uint32_t i_desc =
        (1U << 7U) |
        (1U << 10U) |
        // The MXFP4 descriptor requires UE8M0 scale type (NVFP4's UE4M3
        // form leaves this bit clear).
        (1U << 23U) |
        ((uint32_t(kTileN) >> 3U) << 17U) |
        ((uint32_t(kClusterTileM) >> 7U) << 27U);
    static_assert(i_desc ==
        (kClusterM == 1 ?
                (kTileN == 64 ? 0x08900480U :
                 kTileN == 128 ? 0x08A00480U : 0x08C00480U) :
                (kTileN == 64 ? 0x10900480U :
                 kTileN == 128 ? 0x10A00480U : 0x10C00480U)));

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
            prefetch_tensormap(&C_tmap);
        }
    }

    if constexpr (kClusterM * kSplitK > 1) {
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
    } else {
        __syncthreads();
    }

    const int taddr = tmem_addr[0];

    // Hint the exact tiles consumed by a future TMA load into L2.  This has
    // no completion barrier: it is only a cache residency hint, while the
    // corresponding regular TMA transfer remains responsible for SMEM
    // visibility and mbarrier completion.
    auto prefetch = [&](int iter_k, int off_m, int off_n) {
        const int off_k = (split_rank * (K / (kSplitK * kTileK)) + iter_k) * kTileK;
        const int off_k_64 = (split_rank * (K / (kSplitK * kTileK)) + iter_k) * kMmaStepsPerStage;
        tma_3d_prefetch_l2(&A_tmap, off_k, off_m, 0);
        tma_3d_prefetch_l2(&B_tmap,
            off_k,
            off_n + m_rank * (kTileN / kClusterM),
            0);
        tma_4d_prefetch_l2(&SFA_tmap,
            0,
            0,
            off_k_64,
            off_m / kTileM);
        #pragma unroll
        for (int scale_tile = 0; scale_tile < kSFBScaleTileCount; ++scale_tile) {
            // Match the regular multicast load: rank 0 is the sole cluster
            // producer for SFB, so avoid redundant L2 prefetch requests.
            if constexpr (kClusterM == 1) {
                tma_4d_prefetch_l2(&SFB_tmap,
                    0, 0, off_k_64, off_n / 128 + scale_tile);
            } else if (m_rank == 0) {
                tma_4d_prefetch_l2(&SFB_tmap,
                    0, 0, off_k_64, off_n / 128 + scale_tile);
            }
        }
    };

    // tma load
    auto load = [&](int iter_k, int mma_phase, int stage_id, int off_m, int off_n) {
#ifdef SM110A_MXFP4_DEBUG
        if (iter_k == 0 && _bid < 2) {
            printf("mxfp4 load bid=%d rank=%d m=%d n=%d b_row=%d sf_row=%d\\n",
                _bid, cta_rank, off_m, off_n,
                off_n + cta_rank * (kTileN / kClusterM),
                (off_n + cta_rank * (kTileN / kClusterM)) /
                    (kTileN / kClusterM));
        }
#endif
        const int mma_mbar_addr = _base_mma_mbar_addr + 8 * stage_id;
                wait_mma_slot(mma_mbar_addr, mma_phase);

        const int A_smem = smem + stage_id * kPerStageSmemBytes;
        const int B_smem = A_smem + kASmemBytes;
        const int SFA_smem = B_smem + kBSmemBytes;
        const int SFB_smem = SFA_smem + kSFASmemBytes;
        const int tma_mbar_addr = kClusterM == 1
            ? _base_tma_mbar_addr + 8 * stage_id
            : (_base_tma_mbar_addr + 8 * stage_id) & 0xFEFFFFFF;

        // tma issue
        constexpr int cp_size =
            kATmaBytes + kBTmaBytes + kSFASmemBytes + kSFBSmemBytes;

                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::"r"(
                         tma_mbar_addr),
            "r"(cp_size)
            : "memory");
        const int off_k = (split_rank * (K / (kSplitK * kTileK)) + iter_k) * kTileK;
        const int off_k_64 = (split_rank * (K / (kSplitK * kTileK)) + iter_k) * kMmaStepsPerStage;
        tma_3d_gmem2smem<kClusterM>(A_smem, &A_tmap, off_k, off_m, 0, tma_mbar_addr);
        tma_3d_gmem2smem<kClusterM>(B_smem,
            &B_tmap,
            off_k,
            off_n + m_rank * (kTileN / kClusterM),
            0,
            tma_mbar_addr);
        // The scale tensor map spans all K=64 slabs in this TileK.  SFA is
        // therefore one contiguous transfer, and SFB needs only one transfer
        // per distinct 128-row scale tile (one or two, depending on TileN).
        tma_4d_gmem2smem<kClusterM>(SFA_smem,
            &SFA_tmap,
            0,
            0,
            off_k_64,
            off_m / kTileM,
            tma_mbar_addr);
        #pragma unroll
        for (int scale_tile = 0; scale_tile < kSFBScaleTileCount; ++scale_tile) {
            const int SFB_tile_smem =
                SFB_smem + scale_tile * kMmaStepsPerStage * 512;
            if constexpr (kClusterM == 1) {
                tma_4d_gmem2smem<1>(SFB_tile_smem,
                    &SFB_tmap,
                    0,
                    0,
                    off_k_64,
                    off_n / 128 + scale_tile,
                    tma_mbar_addr);
            } else if (m_rank == 0) {
                const int16_t cta_mask = ((1 << kClusterM) - 1) << (split_rank * kClusterM);
                tma_4d_gmem2smem_multicast<kClusterM>(SFB_tile_smem,
                    &SFB_tmap,
                    0,
                    0,
                    off_k_64,
                    off_n / 128 + scale_tile,
                    tma_mbar_addr,
                    cta_mask);
            }
        }
            };

    // mma compute
    auto compute = [&](int iter_k,
                       int mainloop_stage,
                       int tma_phase,
                       int stage_id,
                       int off_n) {
        const int tma_mbar_addr = _base_tma_mbar_addr + 8 * stage_id;
                wait_tma(tma_mbar_addr, tma_phase);

        const int A_smem = smem + stage_id * kPerStageSmemBytes;
        const int B_smem = A_smem + kASmemBytes;
        const int SFA_smem = B_smem + kBSmemBytes;
        const int SFB_smem = SFA_smem + kSFASmemBytes;
        const int mma_mbar_addr = _base_mma_mbar_addr + 8 * stage_id;

        auto make_fp4_desc = [](int addr) -> uint64_t {
            // The descriptor always describes one K=64 MMA slab.  Wider
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
            return desc_encode(addr) | (desc_encode(SBO) << 32ULL) |
                   (1ULL << 46ULL);
        };

        asm volatile("tcgen05.fence::after_thread_sync;");
        const int this_taddr = taddr + mainloop_stage * kAccumulatorStageStride;
        constexpr int kScaleTmemBase = Config::kAccumulatorTmemCols;
        const int scale_taddr = taddr + kScaleTmemBase;

                auto copy_mma_scales = [&](int mma_k, int mma_scale_taddr) {
            const int SFA_mma_smem = SFA_smem + mma_k * 512;
            tcgen05_cp_scale<kClusterM>(
                mma_scale_taddr, make_scale_desc(SFA_mma_smem));
            #pragma unroll
            for (int scale_tile = 0; scale_tile < kSFBScaleTileCount; ++scale_tile) {
                const int SFB_mma_smem =
                    SFB_smem + (scale_tile * kMmaStepsPerStage + mma_k) * 512;
                tcgen05_cp_scale<kClusterM>(
                    mma_scale_taddr + 4 + scale_tile * 4,
                    make_scale_desc(SFB_mma_smem));
            }
        };

        #pragma unroll
        for (int mma_k = 0; mma_k < kMmaStepsPerStage; ++mma_k) {
            const int A_mma_smem = A_smem + mma_k * (kMmaK / 2);
            const int B_mma_smem = B_smem + mma_k * (kMmaK / 2);
            const int mma_scale_taddr = scale_taddr +
                (kTileN < 256 ?
                        (stage_id * kMmaStepsPerStage + mma_k) *
                            Config::kScaleTmemSlotStride :
                        0);

            // Pair each scale copy with the MMA that consumes it.  Narrow-N
            // uses a distinct slot per (SMEM stage, K=64 slab); TileN=256
            // reuses its sole capacity-constrained slot.  In both cases the
            // cp -> mma ordering supplies the required pipeline dependency.
            copy_mma_scales(mma_k, mma_scale_taddr);

            // In the scale_vec::2X SFB layout, each TMEM column covers 32 N values.
            // A TileN=64 tile may therefore start at column 0 or 2 of its
            // containing 128-row blocked scale tile.
            const int mma_sfb_taddr =
                mma_scale_taddr + 4 + (off_n % 128) / 32;
            tcgen05_mma_mxf4<kClusterM>(this_taddr,
                make_fp4_desc(A_mma_smem),
                make_fp4_desc(B_mma_smem),
                i_desc,
                mma_scale_taddr,
                mma_sfb_taddr,
                iter_k != 0 || mma_k != 0);
        }

        tcgen05_commit<kClusterM, kSplitK>(mma_mbar_addr, split_rank);
            };

    auto tmem_to_smem = [&](int mainloop_stage,
                            int n_block,
                            int smem_stage,
                            int n_group_begin,
                            int n_group_end) {
        nv_bfloat16 *smem_out = epilogue_smem + smem_stage * kTileM * kEpilogueBlockN;

        for (int n = n_group_begin; n < n_group_end; n++) {
            float tmp[8];
            const int row = m_rank * 128 + warp_id * 32;
            const int col =
                mainloop_stage * kAccumulatorStageStride +
                n_block * kEpilogueBlockN + n * 8;
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

        constexpr int n_group_num = kEpilogueBlockN / 8;

        if constexpr (kAccumulatorOverlapCols != 0) {
            static_assert(kAccumulatorOverlapCols == 16);
            static_assert(kEpilogueBlockN >= kAccumulatorOverlapCols);

            // Stage 0 shares its last 16 columns with stage 1's first 16.
            // Drain that half-block first.  Once all eight epilogue warps
            // have arrived, the next MMA may overwrite the shared columns
            // while this epilogue continues reading the disjoint remainder.
            const int overlap_block = mainloop_stage == 0 ? n_block_num - 1 : 0;
            const int overlap_group_begin =
                mainloop_stage == 0 ? n_group_num - kAccumulatorOverlapCols / 8 : 0;
            const int overlap_group_end =
                overlap_group_begin + kAccumulatorOverlapCols / 8;

            tmem_to_smem(mainloop_stage,
                overlap_block,
                0,
                overlap_group_begin,
                overlap_group_end);
            asm volatile("bar.sync 1, 128;" ::: "memory");
            if (tid == 0) {
                const unsigned int epoch = ++overlap_epoch_local[mainloop_stage];
                const int local_addr = static_cast<int>(__cvta_generic_to_shared(
                    const_cast<unsigned int *>(
                        &overlap_ready[cta_rank][mainloop_stage])));
                int rank_zero_addr;
                asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                             : "=r"(rank_zero_addr) : "r"(local_addr), "r"(0));
                asm volatile("st.shared::cluster.u32 [%0], %1;" ::
                             "r"(rank_zero_addr), "r"(epoch) : "memory");
            }
            tmem_to_smem(mainloop_stage, overlap_block, 0, 0, overlap_group_begin);
            tmem_to_smem(mainloop_stage,
                overlap_block,
                0,
                overlap_group_end,
                n_group_num);
            asm volatile("bar.sync 1, 128;" ::: "memory");

            uint64_t cache_policy = 0;
            if (tid == 0) {
                cache_policy = make_epilogue_cache_policy();
            }
            for (int block_idx = 0; block_idx < n_block_num; ++block_idx) {
                const int n_block = (overlap_block + block_idx) % n_block_num;
                const int smem_stage = block_idx % kEpilogueStageCount;
                const int stage_smem =
                    smem + kMainloopSmemBytes +
                    smem_stage * kTileM * kEpilogueBlockN *
                        static_cast<int>(sizeof(nv_bfloat16));

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

                if (block_idx + 1 < n_block_num) {
                    if (block_idx > 0) {
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
                        }
                        asm volatile("bar.sync 1, 128;" ::: "memory");
                    }

                    const int next_block =
                        (overlap_block + block_idx + 1) % n_block_num;
                    const int next_stage = (block_idx + 1) % kEpilogueStageCount;
                    tmem_to_smem(
                        mainloop_stage, next_block, next_stage, 0, n_group_num);
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    if (block_idx + 2 == n_block_num) {
                        release_tmem();
                    }
                }
            }

            if (tid == 0) {
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            }
            asm volatile("bar.sync 1, 128;" ::: "memory");
            return;
        }

        // Only the four epilogue warps participate in named barrier 1.
        tmem_to_smem(mainloop_stage, 0, 0, 0, n_group_num);
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
                tmem_to_smem(mainloop_stage, n_block + 1, next_stage, 0, n_group_num);
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
            constexpr int kPrefetchDistance = kStageCount;
            const int k_tile_count = K / kTileK;

            if constexpr (kPrefetchTmaData) {
                // Prime a full pipeline-distance window.  The rolling issue
                // below visits every later K tile exactly once.
                for (int iter_k = 0;
                     iter_k < min(kPrefetchDistance, k_tile_count / kSplitK);
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
                        while (overlap_ready[rank][previous_stage] < target) {}
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

            tcgen05_commit<kClusterM, kSplitK>(
                _base_mainloop_mbar_addr + 8 * mainloop_stage, split_rank);

            mainloop_stage = (mainloop_stage + 1) % kAccumulatorStageCount;
            if (mainloop_stage == 0) {
                epilogue_phase ^= 1;
            }
            first_output_tile = false;
        }
    }

    // Split-K publishes FP32 partials in each CTA's shared memory.  Cluster-N
    // ranks reduce different M-row ranges through DSMEM and own their stores.
    if constexpr (kSplitK > 1) {
        static_assert(!SWAP_AB, "split-K and swap_ab are mutually exclusive");
        static_assert(kTileM % kSplitK == 0);
        constexpr int kRowsPerSplit = kTileM / kSplitK;
        constexpr int kBlocks = kTileN / kEpilogueBlockN;
        constexpr int kVectorsPerRow = kEpilogueBlockN / 8;
        constexpr int kOutputVectors = (kRowsPerSplit * kEpilogueBlockN) / 8;
        const auto [off_m, off_n] = compute_bid(_bid);
        uint64_t cache_policy = 0;
        if (tid == 0) cache_policy = make_epilogue_cache_policy();
        if (warp_id < 4) {
            wait_mainloop(_base_mainloop_mbar_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
        for (int n_block = 0; n_block < kBlocks; ++n_block) {
            const int smem_stage = n_block % kEpilogueStageCount;
            if (n_block >= kEpilogueStageCount) {
                if (tid == 0) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
                __syncthreads();
            }
            if (warp_id < 4) {
                const int row = warp_id * 32 + (tid & 31);
                for (int n = 0; n < kEpilogueBlockN / 8; ++n) {
                    float tmp[8];
                    const int addr = taddr + ((m_rank * kTileM + warp_id * 32) << 16) +
                                     n_block * kEpilogueBlockN + n * 8;
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                          "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7]) : "r"(addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    for (int i = 0; i < 8; ++i)
                        reduction_smem[row * kEpilogueBlockN + n * 8 + i] = tmp[i];
                }
            }
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            nv_bfloat16 *out_smem = epilogue_smem + smem_stage * kTileM * kEpilogueBlockN;
            for (int vector = tid; vector < kOutputVectors; vector += kBlockThreads) {
                const int local_row = vector / kVectorsPerRow;
                const int col = (vector % kVectorsPerRow) * 8;
                const int source_elem = (split_rank * kRowsPerSplit + local_row) *
                                        kEpilogueBlockN + col;
                const int local_addr = static_cast<int>(__cvta_generic_to_shared(
                    reduction_smem + source_elem));
                float sum[8] = {};
                for (int s = 0; s < kSplitK; ++s) {
                    int remote_addr;
                    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(remote_addr) : "r"(local_addr), "r"(s * kClusterM + m_rank));
                    const int remote_addr_hi = remote_addr + 4 * static_cast<int>(sizeof(float));
                    float value[8];
                    asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%8];\n\t"
                                 "ld.shared::cluster.v4.f32 {%4, %5, %6, %7}, [%9];"
                        : "=f"(value[0]), "=f"(value[1]), "=f"(value[2]), "=f"(value[3]),
                          "=f"(value[4]), "=f"(value[5]), "=f"(value[6]), "=f"(value[7])
                        : "r"(remote_addr), "r"(remote_addr_hi) : "memory");
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) sum[i] += value[i];
                }
                const int swizzled_vector = (vector & ~7) + ((vector & 7) ^ ((vector >> 3) & 3));
                nv_bfloat162 out[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) out[i] = __float22bfloat162_rn({sum[2*i], sum[2*i+1]});
                reinterpret_cast<int4 *>(out_smem + swizzled_vector * 8)[0] =
                    reinterpret_cast<int4 *>(out)[0];
            }
            __syncthreads();
            if (tid == 0) {
                const int stage_smem = static_cast<int>(__cvta_generic_to_shared(out_smem));
                if constexpr (kEpilogueBlockN == 32)
                    tma_2d_smem2gmem(cache_policy, stage_smem, &C_tmap,
                        off_n + n_block * kEpilogueBlockN, off_m + split_rank * kRowsPerSplit);
                else
                    tma_3d_smem2gmem(cache_policy, stage_smem, &C_tmap, 0,
                        (off_n + n_block * kEpilogueBlockN) / 32,
                        off_m + split_rank * kRowsPerSplit);
            }
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
        if (tid == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        __syncthreads();
    // Normal persistent epilogue.
    } else if (warp_id < 4) {
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

} // namespace

template <int kTileM,
    int kTileN,
    int kTileK,
    int kClusterM,
    int kEpilogueBlockN,
    int kSplitK,
    bool SWAP_AB>
void launch_mxfp4_gemm_impl(
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
    constexpr int kStageCount = Config::kStageCount;
    constexpr int kDynamicSmemBytes = Config::kDynamicSmemBytes;

    const int logic_M = M;
    const int logic_N = N;
    if constexpr (SWAP_AB) {
        const uint8_t *tmp_ptr = A_ptr;
        A_ptr = B_ptr;
        B_ptr = tmp_ptr;
        const int tmp_dim = M;
        M = N;
        N = tmp_dim;
    }

    CUtensorMap A_tmap, B_tmap, SFA_tmap, SFB_tmap, C_tmap;

    static_assert(
        kMacroSwizzleM * kMacroSwizzleN <=
        (32 * 1024 * 1024) / ((kTileM + kTileN) * kTileK / 2));
    assert(M % (kClusterM * kTileM) == 0);
    assert(N % kTileN == 0);
    assert(N % 128 == 0);
    assert(K % (kTileK * kSplitK) == 0);

    init_fp4_tmap(&A_tmap,
        A_ptr,
        M,
        K,
        kTileM,
        kTileK,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_fp4_tmap(&B_tmap,
        B_ptr,
        N,
        K,
        kTileN / kClusterM,
        kTileK,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    constexpr uint32_t kScaleBoxTiles = kTileK / kMmaK;
    init_scale_tmap(&SFA_tmap, SFA_ptr, M / kTileM, K / kMmaK, kScaleBoxTiles);
    init_scale_tmap(&SFB_tmap, SFB_ptr, N / 128, K / kMmaK, kScaleBoxTiles);
    if constexpr (SWAP_AB) {
        init_3D_output_tmap(&C_tmap, C_ptr, logic_M, logic_N, kEpilogueBlockN, kTileM);
    } else if constexpr (kEpilogueBlockN == 32) {
        init_2D_output_tmap(&C_tmap, C_ptr, logic_M, logic_N,
                            kTileM / kSplitK, kEpilogueBlockN);
    } else {
        init_3D_output_tmap(&C_tmap, C_ptr, logic_M, logic_N,
                            kTileM / kSplitK, kEpilogueBlockN);
    }

    const int smem_size = kDynamicSmemBytes + kMbarrierSmemBytes;

    dim3 grid(kSplitK == 1 ? kSmCount : (M / kTileM) * (N / kTileN), kSplitK, 1);

    auto kernel =
        mxfp4_gemm_kernel<kTileM, kTileN, kTileK, kClusterM, kEpilogueBlockN, kSplitK, SWAP_AB>;

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    kernel<<<grid, kBlockThreads, smem_size, stream>>>(A_tmap,
        B_tmap,
        SFA_tmap,
        SFB_tmap,
        C_tmap,
        M,
        N,
        K);
}

// Non-template wrappers for pre-compiled path — instantiate with the #define defaults.
void launch_mxfp4_gemm(
    const uint8_t *A_ptr,
    const uint8_t *B_ptr,
    const uint8_t *SFA_ptr,
    const uint8_t *SFB_ptr,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    launch_mxfp4_gemm_impl<SM110A_MXFP4_TILE_M,
        SM110A_MXFP4_TILE_N,
        SM110A_MXFP4_TILE_K,
        SM110A_MXFP4_CLUSTER_M,
        SM110A_MXFP4_EPILOGUE_BLOCK_N,
        SM110A_MXFP4_SPLIT_K,
        false>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, M, N, K, stream);
}

extern "C" void sm110a_mxfp4_gemm_launch(
    const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
    const uint8_t* sfb, __nv_bfloat16* c, int m, int n, int k,
    cudaStream_t stream) {
  launch_mxfp4_gemm(a, b, sfa, sfb, c, m, n, k, stream);
}
