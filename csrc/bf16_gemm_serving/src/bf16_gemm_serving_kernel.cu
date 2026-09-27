// =============================================================================
// SM110A BF16 persistent serving GEMM — design and execution pipeline
// =============================================================================
//
// Target hardware
// ---------------
//   NVIDIA Blackwell SM (SM110A, sm_110a). Assumes first-class support for
//   Tensor Memory (TMEM), distributed shared memory (DSMEM), thread-block
//   clusters and the Tensor Memory Accelerator (TMA) bulk-copy engine.
//
// Scope and implementation status
// -------------------------------
//   The target operation is C[M,N] = A[M,K] @ B[K,N] in BF16.  The online
//   serving variant keeps N, K and B fixed for one long-lived launch and
//   receives only M rows at runtime.  Its physical work unit is
//   kClusterM*kTileM rows by kTileN columns; N and K must be divisible by
//   kTileN and kTileK.  Padding makes an explicitly flushed M tail physical-
//   tile aligned even when its logical valid_rows is not.
//
//   This file contains both the original finite-launch validation entry and
//   the low-level persistent ring entry.  They share the complete
//   scheduler->TMA->MMA->epilogue data path; the scheduler selects either a
//   finite grid or ServingQueueControl as its task source.  Queue allocation,
//   production/consumption helpers and the public Python lifetime handle are
//   layered above these low-level CUDA entry points.
//
// Online-serving contract
// -----------------------
//   The final kernel is launched once, remains resident, and
//   consumes M rows published online by a separate GPU producer.  N, K and B
//   are fixed for the lifetime of a launch; only the amount of available M
//   work changes.  The producer is responsible for obeying the release/acquire
//   queue protocol described here.  How it is co-scheduled (for example with
//   CUDA Green Contexts) is deliberately outside this kernel's contract.
//
//   The first supported serving tactic is fixed rather than autotuned:
//
//       kTileM              = 128
//       kTileN              = 256
//       kTileK              = 128
//       kClusterM           = 2
//       kEpilogueBlockN     = 32
//       SWAP_AB             = false
//       physical B layout   = B_nt[N,K]
//
//   This is the winner stored by the existing 4096x4096x4096 autotuning runs.
//   It gives one 256-row cluster-M group, sixteen independent N tasks when
//   N=4096, and three A/B shared-memory stages under the 227 KiB budget.  The
//   protocol is intentionally tactic-independent, so a later launch API can
//   select another compiled tactic without changing the ring representation.
//
//   1. Physical input/output rings
//      ---------------------------
//      The launch owns X row slots in each of two fixed-address arenas:
//
//          A_ring[X][kTileM][K]
//          C_ring[X][kTileM][N]
//
//      A and C use the same physical slot number.  X must be divisible by
//      kClusterM.  The externally visible queue is expressed in cluster-M
//      groups, not individual CTAs:
//
//          group_capacity = X / kClusterM
//          group ticket g -> group_index = g % group_capacity
//          base_slot       = group_index * kClusterM
//
//      Consequently a cluster group never straddles the physical end of the
//      ring.  CTA rank r reads A_ring[base_slot+r] and writes the matching
//      C_ring[base_slot+r].  The persistent launch can therefore keep one
//      immutable A tensor map, one immutable C tensor map, and one immutable B
//      tensor map; tasks carry coordinates/tickets rather than arbitrary
//      pointers or dynamically rebuilt tensor maps.
//
//      Logical requests may have any positive M.  The producer writes them in
//      strict arrival order and packs their rows consecutively into physical
//      slots.  Request-boundary segment metadata is owned by the producer and
//      downstream output consumer; GEMM sees only packed rows.  A normal
//      publication contains kClusterM*kTileM valid rows.  To bound latency, the
//      producer explicitly FLUSHes a partial group: it zero-fills every unused
//      A row and publishes valid_rows for downstream unpacking.  GEMM still
//      computes and stores the full padded physical group.
//
//   2. Monotonic queue positions
//      -------------------------
//      Three cache-line-separated uint64_t counters describe queue progress:
//
//          published_tail  number of completely published M groups
//          dispatch_head   next flattened N-tile work item to claim
//          recycle_head    number of in-order output groups released for reuse
//
//      A published group creates grid_n = N/kTileN independently claimable
//      work items.  Work items are claimed in N-tile-major epochs so that the
//      whole cluster fleet shares one B N-tile at a time.  Epoch width
//      W = min(group_capacity, launched_clusters): W consecutive claims map to
//      W different groups at the same n_tile, so each 2 MiB B tile is fetched
//      from DRAM once per epoch instead of once per group (an 8x reduction at
//      W=8), keeping B L2-resident instead of re-streaming all of B per group.
//      Capping W at the fleet size also bounds the L2 A/C working set to the
//      one active epoch; deeper rings only add waiting groups, which stay
//      untouched in DRAM instead of thrashing L2:
//
//          work_tail    = published_tail * grid_n
//          work_id      = atomic claim in [dispatch_head, work_tail)
//          epoch        = work_id / W            (one B N-tile per epoch)
//          n_tile       = epoch % grid_n
//          group_ticket = (epoch / grid_n) * W + work_id % W
//          off_m        = (group_ticket % group_capacity)
//                         * kClusterM * kTileM
//          off_n        = n_tile * kTileN
//
//      A cluster scheduler claims one (group_ticket,n_tile) at a time and feeds
//      it into that cluster's existing scheduler->TMA->MMA->epilogue pipeline.
//      group_ticket stays monotonic along claim order, so in-order
//      OUTPUT_READY, consumption, and slot recycling are unchanged, and
//      W == 1 degenerates to the old group-major order.  Epochs are a claim
//      ordering, not a barrier: clusters flow across epoch boundaries.  The
//      cost of the reuse is completion latency - a group finishes only after
//      grid_n epochs (~W x the group-major latency at full occupancy) - which
//      is the intended trade for serving throughput.  Claiming should use a
//      bounded CAS loop (claim only while the mapped group_ticket is below the
//      published frontier), never an unconditional atomicAdd that can run past
//      the published frontier.
//
//      The producer may reuse ticket g's physical storage only after the
//      downstream consumer has released it and advanced recycle_head.  In
//      particular, GEMM completion does not make the C slot writable by the
//      producer: C must remain live until the output consumer acknowledges it.
//
//   3. Per-group state-word encoding
//      -----------------------------
//      Each physical group has one atomically accessed 64-bit state word.  The
//      v1 encoding keeps publication, completion and ABA protection in
//      a single naturally atomic value:
//
//          bits  0..1   state: 0 FREE, 1 READY, 2 OUTPUT_READY
//          bit      2   FLUSH (the producer explicitly closed a partial group)
//          bit      3   reserved (zero in ABI v1)
//          bits  4..15  valid_rows in [1, kClusterM*kTileM]
//          bits 16..63  low 48 bits of the monotonic group generation/ticket
//
//      The 48-bit generation makes a wrapped physical slot distinguishable
//      from its previous occupant.  State transitions preserve the generation,
//      FLUSH and valid_rows fields:
//
//          FREE(g) -> READY(g) -> OUTPUT_READY(g) -> FREE(g+capacity)
//
//      The producer initializes A data, completion_count=0 and all immutable
//      metadata before a release-store of READY(g), then release-advances
//      published_tail after the complete cluster-M group is visible.  A
//      scheduler acquire-loads published_tail/state before issuing TMA.  The
//      There is deliberately no ACTIVE state: dispatch_head is the unique work
//      allocator, and all N tasks for a group must continue to observe READY
//      concurrently. CUDA atomic accesses use device scope while producer,
//      GEMM and output consumer are GPU kernels in one context.
//
//   4. N-tile completion and output publication
//      ----------------------------------------
//      Each group has a uint32_t completion_count reset by the producer before
//      READY publication.  An epilogue increments it only after its C TMA store
//      for one n_tile is globally complete.  The thread observing
//      completion_count == grid_n performs a release-store of OUTPUT_READY.
//      Therefore an acquire-load of OUTPUT_READY by the downstream consumer
//      makes the entire C_ring group visible.  N tasks and whole groups may
//      complete out of order; request order is recovered from group_ticket and
//      the producer-owned segment metadata.  recycle_head advances only across
//      the contiguous prefix of groups that the output consumer has released.
//
//   5. Empty queue, backpressure and shutdown
//      --------------------------------------
//      An empty scheduler first spins briefly, then uses progressively longer
//      nanosleeps so an idle persistent launch does not continuously saturate
//      GMEM/atomic fabric.  The input ring is full when
//      published_tail-recycle_head == group_capacity; the producer must wait
//      rather than overwrite live A or C.  Ring size X is a runtime launch
//      property and should initially be at least twice the resident cluster
//      count times kClusterM slots, then tuned from measured burst depth.
//
//      Shutdown is draining, never an independent per-warp branch.  After its
//      final release-publication, the producer sets DRAIN_AND_STOP.  A cluster
//      scheduler that can claim no remaining work injects a typed STOP task
//      into its local shared-memory task ring.  STOP travels through
//      scheduler->TMA->MMA->epilogue, preserving every mbarrier phase and
//      allowing outstanding stores to finish before cluster synchronization
//      and TMEM deallocation.  The final cluster publishes STOPPED.  A future
//      immediate-cancel mode, if needed, must be a separate protocol.
//
//   6. Boundary between the global and local pipelines
//      ------------------------------------------------
//      Only the elected scheduler lane on cluster rank 0 reads or atomically
//      modifies the GMEM queue.  It converts a claimed global work item into an
//      OutputTileTask stored in the existing two-entry SMEM ring.  From that
//      point onward metadata follows exactly one ordered path:
//
//          GMEM work ring -> cluster scheduler -> DSMEM task multicast
//                         -> TMA -> MMA -> epilogue -> GMEM completion
//
//      Load, MMA and epilogue warps never poll global queue state.  This keeps
//      queue latency/atomics off the data path and makes STOP/FLUSH ordinary
//      typed pipeline messages rather than out-of-band synchronization.
//
// Shared finite/persistent execution core
// ---------------------------------------
//   * The finite entry drains one output grid; the persistent entry claims one
//     (group_ticket,n_tile) at a time until DRAIN_AND_STOP is drained.
//   * Each CTA executes a 7-warp pipeline (224 threads):
//       warp 4              – load warp (TMA gmem→smem)
//       warp 5 (cta_rank 0) – MMA warp   (tcgen05.mma on TMEM)
//       warp 6 (cta_rank 0) – scheduler warp (cluster tile-task producer)
//       warps 0..3          – epilogue warps (tmem→smem→gmem via TMA)
//   * Clusters of size kClusterM along M cooperate via DSMEM and multicast:
//     every cluster CTA loads its own A row, and rank `cta_rank` loads its
//     slice of B; the MMA is issued once with `cta_group::kClusterM`.
//   * Output accumulator lives in TMEM (kTileN*2 columns, i.e. 2 banks for
//     mainloop double-buffering) and is drained to shared memory via
//     tcgen05.ld, then to global memory via 2-D/3-D TMA bulk stores.
//   * Low-level entry points:
//       launch_bf16_gemm_serving          – regular, unprofiled path.
//       launch_bf16_gemm_persistent_serving – long-lived ring consumer.
//
// Implemented execution mechanisms
// --------------------------------
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
//   4. Producer/consumer decoupling with task and data mbarrier rings:
//          scheduler_mbars[2]      – rank-0 scheduler -> every TMA warp.
//          scheduler_free_mbars[2] – all epilogues    -> rank-0 reuse.
//          tma_mbars[kStageCount]  – TMA completion -> MMA consumer.
//          mma_mbars[kStageCount]  – MMA commit     -> TMA producer.
//        `mainlooop_mbars[2]` signals mainloop-stage completion to the
//        epilogue, and `completion_mbars[2]` joins both cluster CTAs after
//        their C-store completion before the group completion atomic.
//        Phase parity bits per ring track the 2-cycle wrap.
//   5. TMA descriptor prefetch (`prefetch.tensormap`) issued once at startup,
//        so the first TMA does not stall on the descriptor fetch.
//   6. Pipelined epilogue:
//          * 2 smem slots per mainloop_stage (`kEpilogueStageCount == 2`).
//          * `cp.async.bulk.wait_group.read 1` + `bar.sync 1, 128` keeps the
//            TMA store of slot N concurrent with the tcgen05.ld that fills
//            slot N+1.
//   7. The finite scaffold retains a `SWAP_AB` template variant for its
//      existing benchmark API.  Online v1 fixes SWAP_AB=false and B_nt[N,K].
//      WaitMMA, IssueTMA, WaitTMA, IssueMMA, WaitMainloop, WaitEpilogue,
//      Epilogue via elected leaders, no impact on the unprofiled path.
//
// Current cluster-local pipeline (retained by the online design)
// --------------------------------------------------------------
//   Legend
//     S – scheduler warp (warp_id == 6 on cta_rank == 0)
//     L – load warp (warp_id == 4)
//     M – MMA  warp (warp_id == 5 on cta_rank == 0)
//     E – epilogue warps (warp_id in {0,1,2,3})
//
//   0. Setup (all warps, once per CTA)
//      ------------------------------------------------
//      tid==0   : mbarrier.init for scheduler_mbars[2] (count=1),
//                 scheduler_free_mbars[2] (count=4*kClusterM),
//                 tma_mbars[kStageCount] (count=kClusterM),
//                 mma_mbars[kStageCount] (count=1), mainlooop_mbars[2] (1),
//                 epilogue_mbars[2] (4*kClusterM); then
//                 `fence.mbarrier_init.release.cluster`.
//      warp 1   : `tcgen05.alloc.cta_group::kClusterM` reserves kTileN*2
//                 TMEM columns (returned in tmem_addr[0]).
//      warp 0*  : `prefetch.tensormap` for A_tmap / B_tmap / C_tmap.
//      barrier.cluster.arrive+wait (or __syncthreads when kClusterM==1).
//
//   1. Scheduler task ring — once per output tile
//      Rank-0 S waits for scheduler_free_mbar[task_stage], writes a cluster-base
//      task to output_tasks[task_stage], then remotely arrives on every
//      CTA's scheduler_mbar[task_stage].  Each TMA warp loads the rank-0 task
//      through DSMEM and derives its local M coordinate.  The finite entry uses
//      compute_bid(); the persistent entry uses a bounded-CAS claim of
//      (group_ticket,n_tile), without changing the downstream interface.
//
//   2. Main loop — repeated `K / kTileK` iterations
//      ------------------------------------------------
//      ┌─── L (load warp) ──────────────────────────────────────────────┐
//      │  wait(scheduler_mbar[task_stage], task_phase) // task ready?  │
//      │  read output_tasks[task_stage]                                 │
//      │  wait(mma_mbar[mma_stage], mma_phase)        // slot free?     │
//      │  write mainloop_tasks[mma_stage]                               │
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
//      │  read mainloop_tasks[tma_stage]                               │
//      │  first K task waits for the TMEM/epilogue slot                │
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
//   3. End-of-K – hand control to the epilogue
//      ------------------------------------------------
//      M issues one extra `tcgen05.commit.cta_group::kClusterM
//      .mbarrier.arrive::one.shared::cluster.multicast` on
//      mainlooop_mbar[mainloop_stage], then toggles mainloop_stage and the
//      epilogue_phase parity bit.
//
//   4. Epilogue (warps 0..3; double-buffered on TMEM)
//      ------------------------------------------------
//      ┌─── E ──────────────────────────────────────────────────────────┐
//      │  wait(mainloop_mbar[mainloop_stage], mainloop_phase)           │
//      │  read output_tasks[mainloop_stage]; signal scheduler_free     │
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
//   5. Teardown
//      ------------------------------------------------
//      cluster barrier -> warp 0 `tcgen05.dealloc.cta_group::kClusterM`
//
// Notes
// -----
//   * The `__grid_constant__ CUtensorMap` parameters carry the host-built
//     TMA descriptors; we never rebuild them on the device.
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

#include "bf16_gemm_serving.h"

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
constexpr int kWarpCount = 7;
constexpr int kBlockThreads = kWarpSize * kWarpCount;
constexpr int kEpilogueStageCount = 2;
constexpr bool kPrefetchTmaDesc = true;
constexpr int kMacroSwizzleM = 8;
constexpr int kMacroSwizzleN = 8;
constexpr bool kMacroSwizzleMFast = true;

template <int kTileM, int kTileN, int kTileK, int kClusterM, int kEpilogueBlockN>
struct KernelConfig {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert(kMacroSwizzleM % kClusterM == 0);
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

enum class TaskKind : uint32_t {
    kCompute = 0,
    kStop = 1,
};

struct OutputTileTask {
    int off_m;
    int off_n;
    uint64_t group_ticket;
    uint32_t group_index;
    uint32_t n_tile;
    uint32_t valid_rows;
    TaskKind kind;
};

struct MainloopTask {
    int off_m;
    int off_n;
    int iter_k;
    int mainloop_stage;
    uint64_t group_ticket;
    uint32_t group_index;
    uint32_t n_tile;
    uint32_t valid_rows;
    TaskKind kind;
};

static_assert(sizeof(OutputTileTask) == 32);
static_assert(sizeof(MainloopTask) == 40);
static_assert(std::is_trivially_copyable_v<OutputTileTask>);
static_assert(std::is_trivially_copyable_v<MainloopTask>);

using sm110a::bf16_gemm_serving::ServingGroupMeta;
using sm110a::bf16_gemm_serving::ServingGroupState;
using sm110a::bf16_gemm_serving::ServingLifecycle;
using sm110a::bf16_gemm_serving::ServingQueueControl;
using sm110a::bf16_gemm_serving::decode_serving_group_state;
using sm110a::bf16_gemm_serving::decode_serving_group_ticket;
using sm110a::bf16_gemm_serving::decode_serving_group_valid_rows;
using sm110a::bf16_gemm_serving::decode_serving_group_flush;
using sm110a::bf16_gemm_serving::encode_serving_group_state;
using sm110a::bf16_gemm_serving::serving_group_meta;
using sm110a::bf16_gemm_serving::serving_compare_exchange_weak_relaxed;
using sm110a::bf16_gemm_serving::serving_fetch_add_acq_rel;
using sm110a::bf16_gemm_serving::serving_load_acquire;
using sm110a::bf16_gemm_serving::serving_load_relaxed;
using sm110a::bf16_gemm_serving::serving_store_relaxed;
using sm110a::bf16_gemm_serving::serving_store_release;

__global__ void initialize_serving_queue_kernel(
    ServingQueueControl *control,
    uint32_t slot_count,
    uint32_t group_capacity,
    uint32_t n,
    uint32_t k,
    uint32_t grid_n,
    uint32_t launched_clusters) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < group_capacity) {
        ServingGroupMeta *meta = serving_group_meta(control) + tid;
        serving_store_relaxed(
            &meta->state_word,
            encode_serving_group_state(ServingGroupState::kFree, tid, 0, false));
        serving_store_relaxed(&meta->completion_count, uint32_t{0});
        serving_store_relaxed(&meta->io_arrival_count, uint32_t{0});
    }
    if (tid == 0) {
        control->config.magic = sm110a::bf16_gemm_serving::kServingQueueMagic;
        control->config.abi_version =
            sm110a::bf16_gemm_serving::kServingQueueAbiVersion;
        control->config.header_bytes = sizeof(ServingQueueControl);
        control->config.group_meta_bytes = sizeof(ServingGroupMeta);
        control->config.slot_count = slot_count;
        control->config.group_capacity = group_capacity;
        control->config.n = n;
        control->config.k = k;
        control->config.grid_n = grid_n;
        control->config.tile_m = SM110A_BF16_TILE_M;
        control->config.tile_n = SM110A_BF16_TILE_N;
        control->config.tile_k = SM110A_BF16_TILE_K;
        control->config.cluster_m = SM110A_BF16_CLUSTER_M;
        control->config.epilogue_block_n = SM110A_BF16_EPILOGUE_BLOCK_N;
        serving_store_relaxed(&control->published_tail.value, uint64_t{0});
        serving_store_relaxed(&control->dispatch_head.value, uint64_t{0});
        serving_store_relaxed(&control->recycle_head.value, uint64_t{0});
        serving_store_relaxed(
            &control->lifecycle.state,
            static_cast<uint32_t>(ServingLifecycle::kRunning));
        serving_store_relaxed(&control->lifecycle.stopped_clusters, uint32_t{0});
        control->lifecycle.launched_clusters = launched_clusters;
    }
}

__global__ void request_serving_drain_kernel(ServingQueueControl *control) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint32_t state = serving_load_acquire(&control->lifecycle.state);
        if (state == static_cast<uint32_t>(ServingLifecycle::kRunning)) {
            serving_store_release(&control->lifecycle.state,
                static_cast<uint32_t>(ServingLifecycle::kDrainAndStop));
        }
    }
}

__device__ inline void copy_uint4_grid_stride(
    const uint4 *__restrict__ src,
    uint4 *__restrict__ dst,
    size_t count) {
    constexpr size_t kUnroll = 8;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; index + (kUnroll - 1) * stride < count;
         index += kUnroll * stride) {
        uint4 values[kUnroll];
#pragma unroll
        for (size_t item = 0; item < kUnroll; ++item) {
            values[item] = src[index + item * stride];
        }
#pragma unroll
        for (size_t item = 0; item < kUnroll; ++item) {
            dst[index + item * stride] = values[item];
        }
    }
    for (; index < count; index += stride) {
        dst[index] = src[index];
    }
}

__device__ inline void zero_uint4_grid_stride(
    uint4 *__restrict__ dst,
    size_t begin,
    size_t end) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t index = begin +
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    for (; index < end; index += stride) {
        dst[index] = zero;
    }
}

__global__ void serving_test_producer_kernel(
    const nv_bfloat16 *__restrict__ rows,
    nv_bfloat16 *__restrict__ a_ring,
    ServingQueueControl *control,
    uint64_t group_ticket,
    uint32_t valid_rows,
    bool flush,
    uint64_t delay_cycles,
    bool bypass_payload) {
    const uint32_t k = control->config.k;
    const uint32_t physical_rows =
        control->config.cluster_m * control->config.tile_m;
    const uint32_t group_index = static_cast<uint32_t>(
        group_ticket % control->config.group_capacity);
    ServingGroupMeta *meta = serving_group_meta(control) + group_index;
    if (blockIdx.x == 0 && threadIdx.x == 0 && delay_cycles != 0) {
        const uint64_t start = clock64();
        while (clock64() - start < delay_cycles) {
        }
    }
    __syncthreads();
    // Backpressure is resolved before any A-ring write. This is essential at
    // capacity=1: ticket g+1 must not overwrite the rows still used by g.
    if (threadIdx.x == 0) {
        uint32_t backoff_step = 0;
        while (true) {
            const uint64_t free_word = serving_load_acquire(&meta->state_word);
            const uint64_t tail = serving_load_acquire(&control->published_tail.value);
            if (decode_serving_group_state(free_word) == ServingGroupState::kFree &&
                decode_serving_group_ticket(free_word) ==
                    (group_ticket &
                        sm110a::bf16_gemm_serving::kServingTicketValueMask) &&
                tail == group_ticket) {
                break;
            }
            const uint32_t shift = min(backoff_step, uint32_t{10});
            __nanosleep(32u << shift);
            backoff_step = min(backoff_step + 1, uint32_t{10});
        }
    }
    __syncthreads();
    if (!bypass_payload) {
        // Vectorized row-major copy: k % 8 == 0 (queue validates k % 128),
        // so every row is a whole number of 16-byte vectors. Rows beyond
        // valid_rows are zero-padded to the physical group height.
        constexpr int kVecElems = 8;
        const uint32_t vec_per_row = k / kVecElems;
        const size_t copy_vecs = static_cast<size_t>(valid_rows) * vec_per_row;
        const size_t pad_vecs =
            static_cast<size_t>(physical_rows) * vec_per_row;
        const size_t dst_base =
            static_cast<size_t>(group_index) * physical_rows * k;
        uint4 *dst = reinterpret_cast<uint4 *>(a_ring + dst_base);
        const uint4 *src = reinterpret_cast<const uint4 *>(rows);
        copy_uint4_grid_stride(src, dst, copy_vecs);
        zero_uint4_grid_stride(dst, copy_vecs, pad_vecs);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // Copy blocks arrive on io_arrival_count; whoever arrives last
        // publishes, so the release-store of kReady covers every block's
        // payload writes and re-zeroes the tally for the consumer phase.
        __threadfence();
        const uint32_t arrived = serving_fetch_add_acq_rel(
            &meta->io_arrival_count, uint32_t{1});
        if (arrived + 1 == gridDim.x) {
            serving_store_relaxed(&meta->io_arrival_count, uint32_t{0});
            // completion_count belongs to the GEMM phase of this slot
            // generation.  Reset it only after every producer block has
            // finished the A-ring copy, and before READY makes the new
            // generation claimable.  Without this reset a wrapped slot starts
            // from the previous generation's grid_n count and can never hit
            // the exact OUTPUT_READY publication threshold again.
            serving_store_relaxed(&meta->completion_count, uint32_t{0});
            serving_store_release(&meta->state_word,
                encode_serving_group_state(
                    ServingGroupState::kReady, group_ticket, valid_rows, flush));
            serving_store_release(&control->published_tail.value, group_ticket + 1);
        }
    }
}

__global__ void serving_test_consumer_kernel(
    const nv_bfloat16 *__restrict__ c_ring,
    nv_bfloat16 *__restrict__ rows,
    ServingQueueControl *control,
    uint64_t group_ticket,
    uint32_t expected_valid_rows,
    bool bypass_payload) {
    const uint32_t group_capacity = control->config.group_capacity;
    const uint32_t group_index =
        static_cast<uint32_t>(group_ticket % group_capacity);
    ServingGroupMeta *meta = serving_group_meta(control) + group_index;
    uint64_t state_word;
    uint32_t backoff_step = 0;
    do {
        state_word = serving_load_acquire(&meta->state_word);
        if (decode_serving_group_state(state_word) == ServingGroupState::kOutputReady &&
            decode_serving_group_ticket(state_word) ==
                (group_ticket & sm110a::bf16_gemm_serving::kServingTicketValueMask)) {
            break;
        }
        const uint32_t shift = min(backoff_step, uint32_t{10});
        __nanosleep(32u << shift);
        backoff_step = min(backoff_step + 1, uint32_t{10});
    } while (true);

    const uint32_t valid_rows = decode_serving_group_valid_rows(state_word);
    if (valid_rows != expected_valid_rows) {
        if (threadIdx.x == 0) {
            control->lifecycle.reserved0 = 2;
        }
        return;
    }
    if (!bypass_payload) {
        const uint32_t n = control->config.n;
        const uint32_t physical_rows =
            control->config.cluster_m * control->config.tile_m;
        constexpr int kVecElems = 8;
        const size_t src_base =
            static_cast<size_t>(group_index) * physical_rows * n;
        const size_t copy_vecs =
            static_cast<size_t>(valid_rows) * (n / kVecElems);
        const uint4 *src = reinterpret_cast<const uint4 *>(c_ring + src_base);
        uint4 *dst = reinterpret_cast<uint4 *>(rows);
        copy_uint4_grid_stride(src, dst, copy_vecs);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // Same last-arriver discipline as the producer: the release-store of
        // kFree orders every copy block's reads/writes before the recycle and
        // re-zeroes the arrival tally for the next producer phase.
        __threadfence();
        const uint32_t arrived = serving_fetch_add_acq_rel(
            &meta->io_arrival_count, uint32_t{1});
        if (arrived + 1 == gridDim.x) {
            serving_store_relaxed(&meta->io_arrival_count, uint32_t{0});
            const uint64_t recycle =
                serving_load_acquire(&control->recycle_head.value);
            if (recycle != group_ticket) {
                control->lifecycle.reserved0 = 3;
                return;
            }
            serving_store_release(&meta->state_word,
                encode_serving_group_state(ServingGroupState::kFree,
                    group_ticket + group_capacity,
                    0,
                    false));
            serving_store_release(&control->recycle_head.value, group_ticket + 1);
        }
    }
}

enum class ServingClaimStatus : uint32_t {
    kClaimed,
    kEmpty,
    kContended,
    kStop,
};

// Called only by the elected rank-0 scheduler lane. A successful CAS advances
// dispatch_head by exactly one N tile; no claim spans or speculatively skips a
// task. The producer publishes group state before published_tail, so observing
// the latter with acquire makes the input payload and state word visible.
__device__ inline ServingClaimStatus try_claim_serving_work(
    ServingQueueControl *control,
    OutputTileTask *task) {
    constexpr int kMaxCasAttempts = 16;
    const uint32_t grid_n = control->config.grid_n;
    const uint32_t group_capacity = control->config.group_capacity;
    const uint64_t window_width = static_cast<uint64_t>(
        min(group_capacity, control->lifecycle.launched_clusters));

#pragma unroll 1
    for (int attempt = 0; attempt < kMaxCasAttempts; ++attempt) {
        uint64_t head = serving_load_relaxed(&control->dispatch_head.value);
        const uint64_t published_tail =
            serving_load_acquire(&control->published_tail.value);
        const uint64_t epoch = head / window_width;
        const uint32_t n_tile = static_cast<uint32_t>(epoch % grid_n);
        const uint64_t group_ticket =
            (epoch / grid_n) * window_width + head % window_width;
        if (group_ticket >= published_tail) {
            const auto lifecycle = static_cast<ServingLifecycle>(
                serving_load_acquire(&control->lifecycle.state));
            return lifecycle == ServingLifecycle::kRunning
                ? ServingClaimStatus::kEmpty
                : ServingClaimStatus::kStop;
        }

        const uint32_t group_index =
            static_cast<uint32_t>(group_ticket % group_capacity);
        ServingGroupMeta *meta = serving_group_meta(control) + group_index;
        const uint64_t state_word = serving_load_acquire(&meta->state_word);
        if (decode_serving_group_state(state_word) != ServingGroupState::kReady ||
            decode_serving_group_ticket(state_word) !=
                (group_ticket & sm110a::bf16_gemm_serving::kServingTicketValueMask)) {
            // published_tail is authoritative. A mismatch means this claimant
            // raced producer publication visibility or slot recycling; retry
            // without consuming the flattened task.
            continue;
        }

        const uint64_t desired = head + 1;
        if (serving_compare_exchange_weak_relaxed(
                &control->dispatch_head.value, head, desired)) {
            const uint32_t base_slot = group_index * SM110A_BF16_CLUSTER_M;
            *task = OutputTileTask{
                static_cast<int>(base_slot * SM110A_BF16_TILE_M),
                static_cast<int>(n_tile * SM110A_BF16_TILE_N),
                group_ticket,
                group_index,
                n_tile,
                decode_serving_group_valid_rows(state_word),
                TaskKind::kCompute};
            return ServingClaimStatus::kClaimed;
        }
    }
    return ServingClaimStatus::kContended;
}

__device__ inline void serving_queue_backoff(uint32_t *backoff_step) {
    const uint32_t shift = min(*backoff_step, uint32_t{10});
    __nanosleep(32u << shift);
    *backoff_step = min(*backoff_step + 1, uint32_t{10});
}

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
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::
            "r"(remote_mbar_addr)
        : "memory");
}

__device__ inline OutputTileTask load_output_task_from_cluster_cta(
    const OutputTileTask *tasks,
    int stage,
    int target_cta_rank) {
    const int local_addr =
        static_cast<int>(__cvta_generic_to_shared(tasks + stage));
    const int remote_addr = map_shared_to_cluster_cta(local_addr, target_cta_rank);
    OutputTileTask task{};
    auto *words = reinterpret_cast<uint32_t *>(&task);
#pragma unroll
    for (int i = 0; i < static_cast<int>(sizeof(OutputTileTask) / sizeof(uint32_t)); ++i) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
            : "=r"(words[i])
            : "r"(remote_addr + i * static_cast<int>(sizeof(uint32_t)))
            : "memory");
    }
    return task;
}

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

__device__ inline void wait_scheduler_task(
    int mbar_addr,
    int phase) {
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni DONE;\n\t"
                 "bra.uni LAB_WAIT;\n\t"
                 "DONE:\n\t"
                 "}" :: "r"(mbar_addr),
        "r"(phase),
        "r"(ticks)
        : "memory");
}

__device__ inline void wait_scheduler_slot(
    int mbar_addr,
    int phase) {
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni DONE;\n\t"
                 "bra.uni LAB_WAIT;\n\t"
                 "DONE:\n\t"
                 "}" :: "r"(mbar_addr),
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
    bool PERSISTENT,
    bool SWAP_AB>
__global__
__cluster_dims__(kClusterM, 1, 1)
__launch_bounds__(kBlockThreads)
void bf16_gemm_serving_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    int M,
    int N,
    int K,
    ServingQueueControl *queue_control) {

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
    __shared__ uint64_t scheduler_mbars[2];
    __shared__ uint64_t scheduler_free_mbars[2];
    __shared__ uint64_t mainlooop_mbars[2];
    __shared__ uint64_t epilogue_mbars[2];
    __shared__ uint64_t completion_mbars[2];
    __shared__ OutputTileTask output_tasks[2];
    __shared__ MainloopTask mainloop_tasks[kStageCount];
    const int _base_tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(tma_mbars));
    const int _base_mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mma_mbars));
    const int _base_scheduler_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(scheduler_mbars));
    const int _base_scheduler_free_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(scheduler_free_mbars));
    const int _base_mainloop_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(mainlooop_mbars));
    const int _base_epilogue_mbar_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_mbars));
    const int _base_completion_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(completion_mbars));

    // init mbarrier
    if (tid == 0) {
        for (int s = 0; s < kStageCount; ++s) {
            mbarrier_init(_base_tma_mbar_addr + 8 * s, kClusterM);
            mbarrier_init(_base_mma_mbar_addr + 8 * s, 1);
        }
        for (int s = 0; s < 2; ++s) {
            mbarrier_init(_base_scheduler_mbar_addr + 8 * s, 1);
            mbarrier_init(_base_scheduler_free_mbar_addr + 8 * s, 4 * kClusterM);
            mbarrier_init(_base_mainloop_mbar_addr + 8 * s, 1);
            mbarrier_init(_base_epilogue_mbar_addr + 8 * s, 4 * kClusterM);
            mbarrier_init(_base_completion_mbar_addr + 8 * s, kClusterM);
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

    const int taddr = tmem_addr[0];

    // tma load
    auto load = [&](const MainloopTask &task, int mma_phase, int stage_id) {
        const int mma_mbar_addr = _base_mma_mbar_addr + 8 * stage_id;
                wait_mma_slot(mma_mbar_addr, mma_phase);

        const int A_smem = smem + stage_id * kPerStageSmemBytes;
        const int B_smem = A_smem + kASmemBytes;
        const int tma_mbar_addr = (_base_tma_mbar_addr + 8 * stage_id) & 0xFEFFFFFF;

        // tma issue
        constexpr int cp_size = kASmemBytes + kBSmemBytes;

                // The TMA completion barrier publishes both the task metadata and
        // the asynchronously copied A/B payload to the MMA consumer.
        mainloop_tasks[stage_id] = task;
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::"r"(
                         tma_mbar_addr),
            "r"(cp_size)
            : "memory");
        const int off_k_64 = task.iter_k * (kTileK / 64);
        tma_3d_gmem2smem<kClusterM>(A_smem, &A_tmap, 0, task.off_m, off_k_64, tma_mbar_addr);
        tma_3d_gmem2smem<kClusterM>(B_smem,
            &B_tmap,
            0,
            task.off_n + cta_rank * (kTileN / kClusterM),
            off_k_64,
            tma_mbar_addr);
            };

    // mma compute
    auto compute =
        [&](int tma_phase, int stage_id, int epilogue_phase) -> MainloopTask {
        const int tma_mbar_addr = _base_tma_mbar_addr + 8 * stage_id;
                wait_tma(tma_mbar_addr, tma_phase);

        const MainloopTask task = mainloop_tasks[stage_id];
        if (task.kind == TaskKind::kStop) {
            return task;
        }
        if (task.iter_k == 0) {
                        wait_epilogue(
                _base_epilogue_mbar_addr + task.mainloop_stage * 8, epilogue_phase);
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
        const int this_taddr = taddr + task.mainloop_stage * kTileN;

                tcgen05_mma_f16<kClusterM>(this_taddr,
            make_desc(A_smem),
            make_desc(B_smem),
            i_desc,
            task.iter_k);
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
                return task;
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

    // One scheduler per cluster owns output-tile work.  It publishes the
    // cluster's rank-0 coordinate and signals each CTA's local ready barrier.
    // Consumer CTAs derive their M coordinate from the cluster base task.
    if (cta_rank == 0 && warp_id == 6 && elect_sync()) {
        int task_stage = 0;
        int task_free_phase = 1;
        auto publish_task = [&](const OutputTileTask &task) {
            wait_scheduler_slot(_base_scheduler_free_mbar_addr + task_stage * 8,
                task_free_phase);
            output_tasks[task_stage] = task;
            for (int target_rank = 0; target_rank < kClusterM; ++target_rank) {
                mbarrier_arrive_on_cluster_cta(
                    _base_scheduler_mbar_addr + task_stage * 8, target_rank);
            }

            task_stage = (task_stage + 1) % 2;
            if (task_stage == 0) {
                task_free_phase ^= 1;
            }
        };

        if constexpr (PERSISTENT) {
            uint32_t backoff_step = 0;
            while (true) {
                OutputTileTask task{};
                const ServingClaimStatus status =
                    try_claim_serving_work(queue_control, &task);
                if (status == ServingClaimStatus::kClaimed) {
                    publish_task(task);
                    backoff_step = 0;
                } else if (status == ServingClaimStatus::kStop) {
                    publish_task(OutputTileTask{
                        0, 0, 0, 0, 0, 0, TaskKind::kStop});
                    break;
                } else {
                    serving_queue_backoff(&backoff_step);
                }
            }
        } else {
            for (int bid = _bid; bid < grid_size; bid += grid_x) {
                auto [off_m, off_n] = compute_bid(bid);
                publish_task(OutputTileTask{
                    off_m, off_n, 0, 0, 0, kClusterTileM, TaskKind::kCompute});
            }
            publish_task(OutputTileTask{
                0, 0, 0, 0, 0, 0, TaskKind::kStop});
        }
    }

    // TMA warp: consume output tasks, expand each one into K-stage tasks, and
    // pass those tasks to MMA alongside the A/B stage payload.
    if (warp_id == 4 && elect_sync()) {
        int mma_stage = 0;
        int mma_phase = 1;
        int task_stage = 0;
        int task_phase = 0;
        while (true) {
            wait_scheduler_task(_base_scheduler_mbar_addr + task_stage * 8, task_phase);
            OutputTileTask output_task = cta_rank == 0
                ? output_tasks[task_stage]
                : load_output_task_from_cluster_cta(output_tasks, task_stage, 0);
            output_task.off_m += cta_rank * kTileM;
            // Keep a rank-local copy for the epilogue.  It is protected by
            // the mainloop completion chain before epilogue consumption.
            output_tasks[task_stage] = output_task;
            if (output_task.kind == TaskKind::kStop) {
                const int mma_mbar_addr = _base_mma_mbar_addr + 8 * mma_stage;
                wait_mma_slot(mma_mbar_addr, mma_phase);
                mainloop_tasks[mma_stage] = MainloopTask{
                    output_task.off_m,
                    output_task.off_n,
                    0,
                    task_stage,
                    output_task.group_ticket,
                    output_task.group_index,
                    output_task.n_tile,
                    output_task.valid_rows,
                    TaskKind::kStop};
                for (int target_rank = 0; target_rank < kClusterM; ++target_rank) {
                    mbarrier_arrive_on_cluster_cta(
                        _base_tma_mbar_addr + 8 * mma_stage, target_rank);
                }
                break;
            }
            for (int iter_k = 0; iter_k < K / kTileK; ++iter_k) {
                const MainloopTask task{
                    output_task.off_m,
                    output_task.off_n,
                    iter_k,
                    task_stage,
                    output_task.group_ticket,
                    output_task.group_index,
                    output_task.n_tile,
                    output_task.valid_rows,
                    output_task.kind};
                load(task, mma_phase, mma_stage);
                mma_stage = (mma_stage + 1) % kStageCount;
                if (mma_stage == 0) {
                    mma_phase ^= 1;
                }
            }
            task_stage = (task_stage + 1) % 2;
            if (task_stage == 0) {
                task_phase ^= 1;
            }
        }
    }

    // MMA warp: its loop bounds only describe how many tasks will arrive;
    // task metadata itself comes exclusively from the TMA stage ring.
    if (cta_rank == 0 && warp_id == 5 && elect_sync()) {
        int tma_phase = 0;
        int tma_stage = 0;
        int epilogue_phase = 1;

        while (true) {
            const MainloopTask task = compute(tma_phase, tma_stage, epilogue_phase);
            if (task.kind == TaskKind::kStop) {
                for (int target_rank = 0; target_rank < kClusterM; ++target_rank) {
                    mbarrier_arrive_on_cluster_cta(
                        _base_mma_mbar_addr + 8 * tma_stage, target_rank);
                    mbarrier_arrive_on_cluster_cta(
                        _base_mainloop_mbar_addr + 8 * task.mainloop_stage, target_rank);
                }
                break;
            }
            tma_stage = (tma_stage + 1) % kStageCount;
            if (tma_stage == 0) {
                tma_phase ^= 1;
            }

            if (task.iter_k == K / kTileK - 1) {
                constexpr int16_t cta_mask = (1 << kClusterM) - 1;
                asm volatile(
                    "tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast:"
                    ":cluster.b64 [%0], %1;" ::"r"(
                        _base_mainloop_mbar_addr + 8 * task.mainloop_stage),
                    "h"(cta_mask),
                    "n"(kClusterM)
                    : "memory");
                if (task.mainloop_stage == 1) {
                    epilogue_phase ^= 1;
                }
            }
        }
    }

    // epilogue
    if (warp_id < 4) {
        int mainloop_stage = 0;
        int mainloop_phase = 0;
        int completion_phase = 0;

        while (true) {
            wait_mainloop(_base_mainloop_mbar_addr + 8 * mainloop_stage, mainloop_phase);
            asm volatile("tcgen05.fence::after_thread_sync;");
            const OutputTileTask task = output_tasks[mainloop_stage];
            if (task.kind == TaskKind::kStop) {
                break;
            }
            if (elect_sync()) {
                mbarrier_arrive_on_cluster_cta(
                    _base_scheduler_free_mbar_addr + mainloop_stage * 8, 0);
            }
            epilogue(mainloop_stage, task.off_m, task.off_n);
            if constexpr (PERSISTENT) {
                if (tid == 0) {
                    mbarrier_arrive_on_cluster_cta(
                        _base_completion_mbar_addr + 8 * mainloop_stage, 0);
                    if (cta_rank == 0) {
                        wait_scheduler_task(
                            _base_completion_mbar_addr + 8 * mainloop_stage,
                            completion_phase);
                        ServingGroupMeta *meta =
                            serving_group_meta(queue_control) + task.group_index;
                        const uint32_t completed = serving_fetch_add_acq_rel(
                            &meta->completion_count, uint32_t{1}) + 1;
                        if (completed == queue_control->config.grid_n) {
                            const uint64_t ready_word =
                                serving_load_acquire(&meta->state_word);
                            serving_store_release(&meta->state_word,
                                encode_serving_group_state(
                                    ServingGroupState::kOutputReady,
                                    task.group_ticket,
                                    task.valid_rows,
                                    decode_serving_group_flush(ready_word)));
                        }
                    }
                }
            }
            mainloop_stage = (mainloop_stage + 1) % 2;
            if (mainloop_stage == 0) {
                mainloop_phase ^= 1;
                completion_phase ^= 1;
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
    if constexpr (kClusterM > 1) {
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
    } else {
        __syncthreads();
    }
    if constexpr (PERSISTENT) {
        if (cta_rank == 0 && warp_id == 6 && elect_sync()) {
            const uint32_t stopped = serving_fetch_add_acq_rel(
                &queue_control->lifecycle.stopped_clusters, uint32_t{1}) + 1;
            if (stopped == queue_control->lifecycle.launched_clusters) {
                serving_store_release(&queue_control->lifecycle.state,
                    static_cast<uint32_t>(ServingLifecycle::kStopped));
            }
        }
    }
    }

} // namespace

cudaError_t initialize_bf16_gemm_serving_queue(
    sm110a::bf16_gemm_serving::ServingQueueControl *control,
    size_t control_bytes,
    uint32_t slot_count,
    uint32_t n,
    uint32_t k,
    uint32_t launched_clusters,
    cudaStream_t stream) {
    constexpr uint32_t kTileM = SM110A_BF16_TILE_M;
    constexpr uint32_t kTileN = SM110A_BF16_TILE_N;
    constexpr uint32_t kTileK = SM110A_BF16_TILE_K;
    constexpr uint32_t kClusterM = SM110A_BF16_CLUSTER_M;
    constexpr uint32_t kEpilogueBlockN = SM110A_BF16_EPILOGUE_BLOCK_N;

    if (control == nullptr ||
        reinterpret_cast<uintptr_t>(control) % alignof(
            sm110a::bf16_gemm_serving::ServingQueueControl) != 0 ||
        slot_count == 0 || slot_count % kClusterM != 0 || n == 0 ||
        n % kTileN != 0 || k == 0 || k % kTileK != 0 ||
        launched_clusters == 0) {
        return cudaErrorInvalidValue;
    }
    if constexpr (kTileM != 128 || kTileN != 256 || kTileK != 128 ||
        kClusterM != 2 || kEpilogueBlockN != 32) {
        return cudaErrorInvalidConfiguration;
    }

    const uint32_t group_capacity = slot_count / kClusterM;
    const size_t required_bytes =
        sm110a::bf16_gemm_serving::serving_queue_control_bytes(group_capacity);
    if (control_bytes < required_bytes) {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = cudaMemsetAsync(control, 0, required_bytes, stream);
    if (status != cudaSuccess) {
        return status;
    }
    constexpr uint32_t kThreads = 256;
    const uint32_t blocks = (group_capacity + kThreads - 1) / kThreads;
    initialize_serving_queue_kernel<<<blocks, kThreads, 0, stream>>>(control,
        slot_count,
        group_capacity,
        n,
        k,
        n / kTileN,
        launched_clusters);
    return cudaGetLastError();
}

extern "C" cudaError_t sm110a_bf16_gemm_serving_queue_initialize(
    sm110a::bf16_gemm_serving::ServingQueueControl *control,
    size_t control_bytes,
    uint32_t slot_count,
    uint32_t n,
    uint32_t k,
    uint32_t launched_clusters,
    cudaStream_t stream) {
    return initialize_bf16_gemm_serving_queue(control,
        control_bytes,
        slot_count,
        n,
        k,
        launched_clusters,
        stream);
}

cudaError_t request_bf16_gemm_serving_drain(
    sm110a::bf16_gemm_serving::ServingQueueControl *control,
    cudaStream_t stream) {
    if (control == nullptr) {
        return cudaErrorInvalidValue;
    }
    request_serving_drain_kernel<<<1, 1, 0, stream>>>(control);
    return cudaGetLastError();
}

extern "C" cudaError_t sm110a_bf16_gemm_serving_request_drain(
    sm110a::bf16_gemm_serving::ServingQueueControl *control,
    cudaStream_t stream) {
    return request_bf16_gemm_serving_drain(control, stream);
}

cudaError_t launch_bf16_gemm_serving_test_producer(
    const __nv_bfloat16 *rows,
    __nv_bfloat16 *a_ring,
    sm110a::bf16_gemm_serving::ServingQueueControl *control,
    uint64_t group_ticket,
    uint32_t valid_rows,
    bool flush,
    uint64_t delay_cycles,
    bool bypass_payload,
    cudaStream_t stream) {
    constexpr uint32_t kPhysicalRows =
        SM110A_BF16_CLUSTER_M * SM110A_BF16_TILE_M;
    if (rows == nullptr || a_ring == nullptr || control == nullptr ||
        valid_rows == 0 || valid_rows > kPhysicalRows ||
        (valid_rows != kPhysicalRows && !flush)) {
        return cudaErrorInvalidValue;
    }
    serving_test_producer_kernel<<<2, 512, 0, stream>>>(
        rows, a_ring, control, group_ticket, valid_rows, flush, delay_cycles,
        bypass_payload);
    return cudaGetLastError();
}

extern "C" cudaError_t sm110a_bf16_gemm_serving_test_publish(
    const __nv_bfloat16 *rows,
    __nv_bfloat16 *a_ring,
    sm110a::bf16_gemm_serving::ServingQueueControl *control,
    uint64_t group_ticket,
    uint32_t valid_rows,
    bool flush,
    uint64_t delay_cycles,
    bool bypass_payload,
    cudaStream_t stream) {
    return launch_bf16_gemm_serving_test_producer(
        rows,
        a_ring,
        control,
        group_ticket,
        valid_rows,
        flush,
        delay_cycles,
        bypass_payload,
        stream);
}

cudaError_t launch_bf16_gemm_serving_test_consumer(
    const __nv_bfloat16 *c_ring,
    __nv_bfloat16 *rows,
    sm110a::bf16_gemm_serving::ServingQueueControl *control,
    uint64_t group_ticket,
    uint32_t expected_valid_rows,
    bool bypass_payload,
    cudaStream_t stream) {
    constexpr uint32_t kPhysicalRows =
        SM110A_BF16_CLUSTER_M * SM110A_BF16_TILE_M;
    if (c_ring == nullptr || rows == nullptr || control == nullptr ||
        expected_valid_rows == 0 || expected_valid_rows > kPhysicalRows) {
        return cudaErrorInvalidValue;
    }
    serving_test_consumer_kernel<<<2, 512, 0, stream>>>(
        c_ring, rows, control, group_ticket, expected_valid_rows,
        bypass_payload);
    return cudaGetLastError();
}

extern "C" cudaError_t sm110a_bf16_gemm_serving_test_consume(
    const __nv_bfloat16 *c_ring,
    __nv_bfloat16 *rows,
    sm110a::bf16_gemm_serving::ServingQueueControl *control,
    uint64_t group_ticket,
    uint32_t expected_valid_rows,
    bool bypass_payload,
    cudaStream_t stream) {
    return launch_bf16_gemm_serving_test_consumer(
        c_ring, rows, control, group_ticket, expected_valid_rows,
        bypass_payload, stream);
}

cudaError_t launch_bf16_gemm_persistent_serving(
    const __nv_bfloat16 *a_ring,
    const __nv_bfloat16 *b_nt,
    __nv_bfloat16 *c_ring,
    sm110a::bf16_gemm_serving::ServingQueueControl *control,
    uint32_t slot_count,
    uint32_t n,
    uint32_t k,
    uint32_t launched_clusters,
    cudaStream_t stream) {
    constexpr uint32_t kTileM = SM110A_BF16_TILE_M;
    constexpr uint32_t kTileN = SM110A_BF16_TILE_N;
    constexpr uint32_t kTileK = SM110A_BF16_TILE_K;
    constexpr uint32_t kClusterM = SM110A_BF16_CLUSTER_M;
    constexpr uint32_t kEpilogueBlockN = SM110A_BF16_EPILOGUE_BLOCK_N;
    if constexpr (kTileM != 128 || kTileN != 256 || kTileK != 128 ||
        kClusterM != 2 || kEpilogueBlockN != 32) {
        return cudaErrorInvalidConfiguration;
    }
    if (a_ring == nullptr || b_nt == nullptr || c_ring == nullptr ||
        control == nullptr || slot_count == 0 || slot_count % kClusterM != 0 ||
        n == 0 || n % kTileN != 0 || k == 0 || k % kTileK != 0 ||
        launched_clusters == 0) {
        return cudaErrorInvalidValue;
    }

    CUtensorMap a_tmap, b_tmap, c_tmap;
    const uint64_t ring_rows = static_cast<uint64_t>(slot_count) * kTileM;
    init_3D_tmap(&a_tmap,
        a_ring,
        ring_rows,
        k,
        kTileM,
        kTileK,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_3D_tmap(&b_tmap,
        b_nt,
        n,
        k,
        kTileN / kClusterM,
        kTileK,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_2D_output_tmap(&c_tmap, c_ring, ring_rows, n, kTileM, kEpilogueBlockN);

    using Config = KernelConfig<kTileM, kTileN, kTileK, kClusterM, kEpilogueBlockN>;
    constexpr int kSmemBytes = Config::kDynamicSmemBytes + 256;
    auto kernel = bf16_gemm_serving_kernel<
        kTileM, kTileN, kTileK, kClusterM, kEpilogueBlockN, true, false>;
    cudaError_t status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    if (status != cudaSuccess) {
        return status;
    }
    kernel<<<launched_clusters * kClusterM, kBlockThreads, kSmemBytes, stream>>>(
        a_tmap,
        b_tmap,
        c_tmap,
        static_cast<int>(ring_rows),
        static_cast<int>(n),
        static_cast<int>(k),
        control);
    return cudaGetLastError();
}

extern "C" cudaError_t sm110a_bf16_gemm_persistent_serving_launch(
    const __nv_bfloat16 *a_ring,
    const __nv_bfloat16 *b_nt,
    __nv_bfloat16 *c_ring,
    sm110a::bf16_gemm_serving::ServingQueueControl *control,
    uint32_t slot_count,
    uint32_t n,
    uint32_t k,
    uint32_t launched_clusters,
    cudaStream_t stream) {
    return launch_bf16_gemm_persistent_serving(a_ring,
        b_nt,
        c_ring,
        control,
        slot_count,
        n,
        k,
        launched_clusters,
        stream);
}

template <int kTileM,
    int kTileN,
    int kTileK,
    int kClusterM,
    int kEpilogueBlockN,
    bool SWAP_AB>
void launch_bf16_gemm_serving_impl(
    const __nv_bfloat16 *A_ptr,
    const __nv_bfloat16 *B_ptr,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K,
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
        bf16_gemm_serving_kernel<
            kTileM, kTileN, kTileK, kClusterM, kEpilogueBlockN, false, SWAP_AB>;

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    kernel<<<grid, kBlockThreads, smem_size, stream>>>(A_tmap,
        B_tmap,
        C_tmap,
        M,
        N,
        K,
        nullptr);
}

// Non-template wrappers for pre-compiled path — instantiate with the #define defaults.
void launch_bf16_gemm_serving(
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
        launch_bf16_gemm_serving_impl<SM110A_BF16_TILE_M,
            SM110A_BF16_TILE_N,
            SM110A_BF16_TILE_K,
            SM110A_BF16_CLUSTER_M,
            SM110A_BF16_EPILOGUE_BLOCK_N,
            true>(A_ptr, B_ptr, C_ptr, M, N, K,
                  max_active_clusters, stream);
    } else {
        launch_bf16_gemm_serving_impl<SM110A_BF16_TILE_M,
            SM110A_BF16_TILE_N,
            SM110A_BF16_TILE_K,
            SM110A_BF16_CLUSTER_M,
            SM110A_BF16_EPILOGUE_BLOCK_N,
            false>(A_ptr, B_ptr, C_ptr, M, N, K,
                   max_active_clusters, stream);
    }
}

extern "C" void sm110a_bf16_gemm_serving_launch(
    const __nv_bfloat16* d_a, const __nv_bfloat16* d_b,
    __nv_bfloat16* d_c, int m, int n, int k, bool swap_ab,
    int max_active_clusters, cudaStream_t stream) {
  launch_bf16_gemm_serving(d_a, d_b, d_c, m, n, k, swap_ab, stream,
                   max_active_clusters);
}
