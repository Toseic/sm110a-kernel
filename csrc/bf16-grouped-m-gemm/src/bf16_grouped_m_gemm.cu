/*
 * bf16_grouped_m_gemm.cu — SM110A / Blackwell BF16 grouped-M GEMM kernel.
 *
 * What it does
 * ------------
 * Computes  C = A · B^T  in BF16, where rows of A are routed to per-expert
 * weight slices B[expert, :, :] via `grouped_layout[row] -> expert`. Different
 * rows may select different experts, so the kernel is responsible for fusing
 * these otherwise fragmented accesses into a single high-throughput kernel
 * that still reuses each expert's B tile as much as possible.
 *
 * The kernel runs entirely on Blackwell/SM110A, using the cluster/TMA/tcgen05
 * execution model. It is exposed as `sm110a_bf16_grouped_m_gemm_launch`
 * (extern "C") and as `launch_bf16_grouped_m_gemm` from the host wrapper.
 *
 * Key optimizations
 * -----------------
 * 1. Expert coalescing on the host. Rows belonging to the same expert are
 *    assumed to live in one contiguous segment of `grouped_layout`. Two
 *    device kernels, `collect_expert_metadata` and
 *    `build_expert_tile_tasks`, compress that layout into a compact task
 *    queue of `ExpertTileTask{ row_start, row_count, tile_row, expert,
 *    tile_n }`. From that queue the GEMM kernel never has to re-parse the
 *    original row-to-expert table.
 *
 * 2. Cluster cooperation along M. The kernel uses one cluster as the basic
 *    scheduling unit. A cluster contains `kClusterM` CTAs, each owning one
 *    `kTileM` sub-tile along M, so one cluster covers `kClusterTileM =
 *    kClusterM * kTileM` rows. Along N the kernel processes `kTileN`
 *    columns at a time. `cta_group::1` and `cta_group::2` variants of the
 *    TMA / tcgen05 instructions are selected by the `kClusterM` template
 *    parameter, so the same kernel body supports both cluster sizes.
 *
 * 3. Macro-level swizzle. In persistent mode (`SM110A_GROUPED_MODE == 1`),
 *    tasks are issued in an order that walks M first inside a band of
 *    `kMacroSwizzleM` tile-rows and N second inside a band of
 *    `kMacroSwizzleN` N-tiles. The chosen N band (8) keeps the combined A+B
 *    footprint below Thor's 32 MiB L2 so concurrent clusters can still hit
 *    in-cache B tiles.
 *
 * 4. Deep, SMEM-bound mainloop pipeline. `KernelConfig` packs as many
 *    mainloop stages as the per-CTA 228 KiB SMEM budget (minus the epilogue
 *    regions) allows. Each stage holds the A and B tiles for one
 *    `kTileK`-sized K slab and exchanges ownership through `tma_mbars` /
 *    `mma_mbars` pairs so the producer and consumer overlap perfectly.
 *
 * 5. TMEM accumulators + tcgen05.mma. The consumer warp issues
 *    `tcgen05.mma.kind::f16` directly from shared-memory descriptors into
 *    TMEM accumulators. Two TMEM stages (`kTmemStageCount = 2`) let the
 *    epilogue consume the previous N block while the consumer is still
 *    producing the next one. `mainloop_mbars`/`epilogue_mbars` ensure the
 *    epilogue never races the consumer on a TMEM stage.
 *
 * 6. TMA-store epilogue. By default the epilogue reshapes each `kTileM x
 *    kEpilogueTileN` TMEM slice into a swizzled SMEM staging buffer and
 *    issues a `cp.async.bulk.tensor` store. A second swizzle/prefetch
 *    pipeline inside the epilogue (`kEpilogueStageCount`) hides TMEM→SMEM
 *    latency behind the GMEM stores. Boundary tiles whose valid row count
 *    is smaller than `kTileM` / `kEpilogueTileN` fall back to per-thread
 *    stores to avoid out-of-bounds writes.
 *
 * 7. Optional direct-GMEM epilogue. Setting `SM110A_GROUPED_DIRECT_GMEM`
 *    skips the SMEM/TMA path on the non-`swap_ab` branch and writes
 *    BF16-packed TMEM values straight to global memory. Useful when the
 *    SMEM/TMA path is bandwidth-bound on small shapes.
 *
 * 8. Persistent scheduling. With `SM110A_GROUPED_MODE == 1`, clusters pull
 *    work dynamically through `next_cluster_tile`, claiming
 *    `kPersistentTileCount = 2` tiles per fetch. This amortizes the cost of
 *    the atomic work claim while preventing workers from drifting across
 *    unrelated experts and destroying expert-local reuse.
 *
 * 9. `swap_ab` view. When `swap_ab == true` the A/B tensor maps and the
 *    C tensor map are swapped (using `init_swapped_c_tensor_map`), the task
 *    granularity changes to slice N rather than M, and the epilogue writes
 *    via `stmatrix.trans` into a swizzled SMEM atom layout. The intent is
 *    to reuse the same tcgen05/TMA skeleton while flipping the originally
 *    less-contiguous dimension into the tensor-map-friendly read direction.
 *
 * Pipeline in detail
 * ------------------
 * The host path performs the following steps per launch:
 *
 *     launch_bf16_grouped_m_gemm_entry
 *       └── launch_bf16_grouped_m_gemm_impl<...>
 *             ├── Build CUtensorMaps for A, B, C (init_a/b/c/swapped_*).
 *             ├── cudaMallocAsync a workspace laid out as
 *             │     [expert_offsets | expert_counts | task_count
 *             │      | next_cluster_tile | tasks[]].
 *             ├── cudaMemsetAsync expert_offsets to INT32_MAX (so atomicMin
 *             │     yields the first row of each expert) and the counts to 0.
 *             ├── collect_expert_metadata<<<>>>:
 *             │     one thread per grouped row, performs
 *             │       atomicMin(expert_offsets + expert, row)
 *             │       atomicAdd(expert_counts + expert, 1).
 *             ├── build_expert_tile_tasks<<<1,1>>>:
 *             │     single-threaded pass that emits one ExpertTileTask per
 *             │     (M chunk within an expert, tile_n) pair, ordered to
 *             │     match the persistent swizzle and writes task_count.
 *             └── Launch bf16_grouped_m_gemm_kernel in either:
 *                   * one cluster per output tile (non-persistent), or
 *                   * `resident_clusters` persistent clusters that
 *                     steal through next_cluster_tile.
 *
 * The device-side kernel (`bf16_grouped_m_gemm_kernel`) is organized as a
 * three-stage pipeline plus a small persistent-stealing outer loop:
 *
 *   ┌────────────────────────────────────────────────────────────────────┐
 *   │ outer loop (persistent mode only):                                 │
 *   │   rank-0 CTA thread 0 does atomicAdd(next_cluster_tile, 2)         │
 *   │   to claim the next two cluster tiles. The other ranks in the      │
 *   │   cluster fetch the cursor via cluster.shared::cluster and exit    │
 *   │   when the cursor has reached `task_count`.                        │
 *   └────────────────────────────────────────────────────────────────────┘
 *
 *   ┌────────────────────────────────────────────────────────────────────┐
 *   │ Producer (warp 4, one elected thread):                             │
 *   │   for each tile_k in 0..k_tiles:                                   │
 *   │     mbarrier_wait(mma_mbars[stage], producer_phase)                │
 *   │       # MMA has released this SMEM stage for reuse.                │
 *   │     arrive.expect_tx on tma_mbar for kASmemBytes + kBSmemBytes     │
 *   │     tma_load_4d<kClusterM>(a_smem, A tensor map, ...)              │
 *   │     tma_load_4d<kClusterM>(b_smem, B tensor map, ...)              │
 *   │       # On cta_group::2 the B load is multicast so both cluster    │
 *   │       # ranks share one TMA transaction.                           │
 *   │     stage = (stage + 1) % kStageCount; phase ^= when wraps         │
 *   └────────────────────────────────────────────────────────────────────┘
 *
 *   ┌────────────────────────────────────────────────────────────────────┐
 *   │ Consumer (warp 5, cluster rank 0, one elected thread):             │
 *   │   In persistent mode, mbarrier_wait(epilogue_mbars[stage], phase)  │
 *   │     before touching TMEM so the epilogue has released the stage.   │
 *   │   for each tile_k in 0..k_tiles:                                   │
 *   │     mbarrier_wait(tma_mbars[stage], consumer_phase)                │
 *   │       # TMA has filled this SMEM stage.                            │
 *   │     tcgen05.fence::after_thread_sync                               │
 *   │     for mma_k in 0..kTileK/kMmaK:                                  │
 *   │       build k-major 128B shared-memory descriptors for the         │
 *   │       current k_slab / k_in_slab and issue                         │
 *   │       tcgen05.mma.kind::f16 with accumulate=(tile_k|mma_k != 0)    │
 *   │     tcgen05.commit.cta_group::N → mma_mbars[stage]                 │
 *   │     stage = (stage + 1) % kStageCount; phase ^= when wraps         │
 *   │   tcgen05.commit.cta_group::N → mainloop_mbars[tmem_stage]         │
 *   │     signals the epilogue that one full kTileN-wide TMEM stage is   │
 *   │     ready to read.                                                 │
 *   └────────────────────────────────────────────────────────────────────┘
 *
 *   ┌────────────────────────────────────────────────────────────────────┐
 *   │ Epilogue (warps 0..3):                                             │
 *   │   mbarrier_wait(mainloop_mbars[stage], epilogue_phase)             │
 *   │   tcgen05.fence::after_thread_sync                                 │
 *   │   for n_block in 0..kTileN/kEpilogueTileN:                         │
 *   │     tmem_to_smem(stage, n_block, smem_stage%kEpilogueStageCount):  │
 *   │       * non-swap: tcgen05.ld.32x32b.x8 → float[8] → __nv_bfloat162 │
 *   │         → SMEM at the swizzled chunk index for this tid.           │
 *   │       * swap_ab: tcgen05.ld.16x256b.x1 → STSM transposed atom      │
 *   │         into the swizzled SMEM staging region.                     │
 *   │       * SM110A_GROUPED_DIRECT_GMEM (non-swap only): write the      │
 *   │         packed BF16 values straight to GMEM.                       │
 *   │     bar.sync 1, 128 (keeps the four epilogue warps in lockstep).   │
 *   │     if full tile: tid 0 issues tma_store_3d for each store tile    │
 *   │                   + cp.async.bulk.commit_group;                    │
 *   │     else:       per-thread fallback stores from the swizzled SMEM  │
 *   │                   to GMEM to avoid OOB writes.                     │
 *   │     rotate to the next smem_stage (epilogue prefetch pipeline).    │
 *   │   release_tmem(): in persistent mode, arrive on epilogue_mbar so   │
 *   │     the consumer may overwrite this TMEM stage next iteration.     │
 *   │   cp.async.bulk.wait_group 0; bar.sync 1, 128.                     │
 *   └────────────────────────────────────────────────────────────────────┘
 *
 *   ┌────────────────────────────────────────────────────────────────────┐
 *   │ End-of-kernel: barrier.cluster + tcgen05.dealloc (warp 0).         │
 *   └────────────────────────────────────────────────────────────────────┘
 *
 * Thread roles
 * ------------
 *   warp 0..3   epilogue warps (TMEM → SMEM/GMEM, TMA store, fallbacks).
 *   warp 4      mainloop producer (TMA loads for A and B).
 *   warp 5      mainloop consumer / TMEM producer (cluster rank 0 only).
 *
 * Synchronization objects (all in dynamic SMEM)
 * ---------------------------------------------
 *   tma_mbars[stage]      TMA → MMA      : stage is filled and may be read.
 *   mma_mbars[stage]      MMA → TMA      : stage may be overwritten by TMA.
 *   mainloop_mbars[stage] MMA → Epilogue : TMEM output stage is ready.
 *   epilogue_mbars[stage] Epilogue → MMA : TMEM output stage is consumed
 *                                          (persistent mode only).
 *
 * In one sentence
 * ---------------
 *   "Coalesce grouped rows into expert-local tile tasks, then drive a
 *    cluster-persistent Blackwell pipeline of TMA + tcgen05 + TMEM →
 *    swizzled SMEM → TMA-store epilogue, balancing expert-local reuse
 *    against L2 working-set pressure."
 */

#include <cstdint>
#include <cstdlib>
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
#ifndef SM110A_GROUPED_MODE
#define SM110A_GROUPED_MODE 0
#endif
#ifndef SM110A_GROUPED_DIRECT_GMEM
#define SM110A_GROUPED_DIRECT_GMEM 0
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
// Keep the persistent working set below Thor's 32 MiB L2.  A 16-wide N band
// needs 25 MiB of B plus up to 12 MiB of A for four 256-row tasks, which
// evicts the B tiles before the next M task can reuse them.  An 8-wide band
// keeps the combined A/B footprint at or below 25 MiB for the large-M cases.
constexpr int kMacroSwizzleN = 8;

template <int kTileM, int kTileN, int kTileK, int kEpilogueTileN, int kClusterM>
struct KernelConfig {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert(kTileM == 4 * kWarpSize);
    // tcgen05.mma.kind::f16 supports N in steps of 8 for cta_group::1
    // and steps of 16 for cta_group::2, up to N=256. The instruction
    // descriptor encodes this dimension as N >> 3 in bits 17-22.
    static_assert(kTileN >= 8 * kClusterM);
    static_assert(kTileN <= 256);
    static_assert(kTileN % (8 * kClusterM) == 0);
    static_assert(kEpilogueTileN == 32 || kEpilogueTileN == 64 || kEpilogueTileN == 128);
    static_assert(kTileN % kEpilogueTileN == 0);
    static_assert(kTileN % kClusterM == 0);
    static_assert(kTileK % 64 == 0);
    static_assert(kTileK % kMmaK == 0);
    static constexpr int kClusterTileM = kClusterM * kTileM;
    static constexpr int kASmemBytes = kTileM * kTileK * sizeof(nv_bfloat16);
    static constexpr int kBSmemBytes = (kTileN / kClusterM) * kTileK * sizeof(nv_bfloat16);
    static constexpr int kPerStageSmemBytes = kASmemBytes + kBSmemBytes;
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
    static constexpr int kTmemColumnsNeeded = kTileN * kTmemStageCount;
    // tcgen05.alloc requires a power-of-two column count. In particular,
    // two N=192 accumulator stages need 384 columns but must allocate 512.
    static constexpr int kTmemAllocationColumns = kTmemColumnsNeeded <= 32    ? 32
                                                  : kTmemColumnsNeeded <= 64  ? 64
                                                  : kTmemColumnsNeeded <= 128 ? 128
                                                  : kTmemColumnsNeeded <= 256 ? 256
                                                                              : 512;
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

template <int kTaskTileRows>
__global__ void build_expert_tile_tasks(
    const int *expert_offsets,
    const int *expert_counts,
    int experts,
    int grid_n,
    ExpertTileTask *tasks,
    int *task_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int next_task = 0;
    for (int expert = 0; expert < experts; ++expert) {
        const int row_count = expert_counts[expert];
        if (row_count == 0) {
            continue;
        }
        const int row_start = expert_offsets[expert];
        for (int group_row = 0; group_row < row_count;
            group_row += kMacroSwizzleM * kTaskTileRows) {
            const int group_row_end =
                group_row + kMacroSwizzleM * kTaskTileRows < row_count
                    ? group_row + kMacroSwizzleM * kTaskTileRows
                    : row_count;
            for (int macro_n = 0; macro_n < grid_n; macro_n += kMacroSwizzleN) {
                const int band_end =
                    macro_n + kMacroSwizzleN < grid_n ? macro_n + kMacroSwizzleN : grid_n;
                for (int tile_row = group_row; tile_row < group_row_end;
                    tile_row += kTaskTileRows) {
                    for (int tile_n = macro_n; tile_n < band_end; ++tile_n) {
                        tasks[next_task++] = {row_start, row_count, tile_row, expert, tile_n};
                    }
                }
            }
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

__device__ inline constexpr uint64_t matrix_desc_encode(
    uint64_t value) {
    return (value & 0x3FFFFULL) >> 4ULL;
}

__device__ inline uint64_t make_k_major_128b_desc(
    int smem_addr) {
    constexpr int kStrideByteOffset = 8 * 64 * sizeof(nv_bfloat16);
    return matrix_desc_encode(smem_addr) | (matrix_desc_encode(kStrideByteOffset) << 32ULL) |
           (1ULL << 46ULL) | (2ULL << 61ULL);
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

template <int kTileM, int kTileK>
inline bool init_a_tensor_map(
    CUtensorMap *tensor_map,
    const nv_bfloat16 *ptr,
    uint64_t m,
    uint64_t k) {
    constexpr uint32_t rank = 4;
    uint64_t global_dim[rank] = {64, m, k / 64, 1};
    uint64_t global_stride[rank - 1] = {k * sizeof(nv_bfloat16), 128, m * k * sizeof(nv_bfloat16)};
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
    uint64_t global_stride[rank - 1] = {k * sizeof(nv_bfloat16), 128, n * k * sizeof(nv_bfloat16)};
    uint32_t box_dim[rank] = {64, kTileN / kClusterM, kTileK / 64, 1};
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
    bool kSwapAB = false,
    bool kDirectGmem = false>
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
    using Config = KernelConfig<kTileM, kTileN, kTileK, kEpilogueTileN, kClusterM>;
    constexpr int kClusterTileM = Config::kClusterTileM;
    constexpr int kStageCount = Config::kStageCount;
    constexpr int kASmemBytes = Config::kASmemBytes;
    constexpr int kBSmemBytes = Config::kBSmemBytes;
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
    constexpr int kWorkerTileCount = kPersistent ? kPersistentTileCount : 1;
    const int static_first_cluster_bid = worker_id * kWorkerTileCount;
    if constexpr (!kPersistent) {
        if (static_first_cluster_bid >= cluster_tiles) {
            return;
        }
    }

    __shared__ int tmem_base[1];
    __shared__ int next_work_smem[2];
    __shared__ uint64_t tma_mbars[kStageCount];
    __shared__ uint64_t mma_mbars[kStageCount];
    __shared__ uint64_t mainloop_mbars[kTmemStageCount];
    __shared__ uint64_t epilogue_mbars[kTmemStageCount];

    extern __shared__ __align__(1024) char dynamic_smem[];
    const int mainloop_smem = static_cast<int>(__cvta_generic_to_shared(dynamic_smem));
    const int epilogue_smem = mainloop_smem + kMainloopSmemBytes;
    const int base_tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(tma_mbars));
    const int base_mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mma_mbars));
    const int base_mainloop_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mainloop_mbars));
    const int base_epilogue_mbar_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_mbars));

    if (tid == 0) {
        for (int stage = 0; stage < kStageCount; ++stage) {
            mbarrier_init(base_tma_mbar_addr + stage * sizeof(uint64_t), kClusterM);
            mbarrier_init(base_mma_mbar_addr + stage * sizeof(uint64_t), 1);
        }
        for (int stage = 0; stage < kTmemStageCount; ++stage) {
            mbarrier_init(base_mainloop_mbar_addr + stage * sizeof(uint64_t), 1);
            if constexpr (kPersistent) {
                mbarrier_init(base_epilogue_mbar_addr + stage * sizeof(uint64_t), 4 * kClusterM);
            }
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp_id == 5) {
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
        prefetch_tensor_map(&c_tensor_map);
    }
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    const int tmem_addr = tmem_base[0];
    constexpr uint32_t instruction_desc = (1U << 4U) | (1U << 7U) | (1U << 10U) |
                                          (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
                                          (static_cast<uint32_t>(kClusterTileM) >> 4U << 24U);

    const int k_tiles = k / kTileK;

    auto tmem_to_smem = [&](int mainloop_stage, int n_block, int smem_stage,
                             int cta_row = 0, int tile_col = 0,
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
                        const int gmem_col =
                            tile_col + n_block * kEpilogueTileN + n_group * 8;
                        reinterpret_cast<int4 *>(
                            c + gmem_row * n + gmem_col)[0] =
                            reinterpret_cast<int4 *>(packed)[0];
                    }
                } else {
                    const int store_tile = n_group / kChunksPerStoreTile;
                    const int chunk_in_store =
                        tid * kChunksPerStoreTile + n_group % kChunksPerStoreTile;
                    const int swizzled_chunk =
                        (chunk_in_store & ~7) +
                        ((chunk_in_store & 7) ^
                            ((chunk_in_store >> 3) & kSwizzlePhaseMask));
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
    int consumer_tma_phase = 0;
    int consumer_mainloop_stage = 0;
    int consumer_epilogue_phase = 1;
    int epilogue_mainloop_stage = 0;
    int epilogue_mainloop_phase = 0;

    while (true) {
        int first_cluster_bid;
        if constexpr (kPersistent) {
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
        if (first_cluster_bid >= cluster_tiles) {
            break;
        }
        const bool has_work = first_cluster_bid < cluster_tiles;
        const int last_cluster_bid = has_work
                                         ? (first_cluster_bid + kWorkerTileCount < cluster_tiles
                                                   ? first_cluster_bid + kWorkerTileCount
                                                   : cluster_tiles)
                                         : first_cluster_bid;

        // The producer waits until MMA has released a stage before TMA overwrites it.
        if (warp_id == 4 && elect_sync()) {
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

                    mbarrier_wait(mma_mbar_addr, producer_mma_phase);
                    const int tma_transaction_mbar_addr = tma_mbar_addr & 0xFEFFFFFF;
                    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
                                 "_, [%0], %1;" ::"r"(tma_transaction_mbar_addr),
                        "r"(kASmemBytes + kBSmemBytes)
                        : "memory");
                    const int k_slice = tile_k * (kTileK / 64);
                    if constexpr (!kSwapAB) {
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
                    } else {
                        const int swapped_tile_col = tile_n * kClusterTileM;
                        tma_load_4d<kClusterM>(a_smem,
                            &a_tensor_map,
                            0,
                            swapped_tile_col + cta_rank * kTileM,
                            k_slice,
                            task.expert,
                            tma_transaction_mbar_addr);
                        tma_load_4d<kClusterM>(b_smem,
                            &b_tensor_map,
                            0,
                            task.row_start + task.tile_row + cta_rank * (kTileN / kClusterM),
                            k_slice,
                            0,
                            tma_transaction_mbar_addr);
                    }

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
            for (int cluster_bid = first_cluster_bid; cluster_bid < last_cluster_bid;
                ++cluster_bid) {
                if constexpr (kPersistent) {
                    mbarrier_wait(
                        base_epilogue_mbar_addr + consumer_mainloop_stage * sizeof(uint64_t),
                        consumer_epilogue_phase);
                }

                const int stage_tmem_addr = tmem_addr + consumer_mainloop_stage * kTileN;
                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    const int tma_mbar_addr =
                        base_tma_mbar_addr + consumer_stage * sizeof(uint64_t);
                    const int mma_mbar_addr =
                        base_mma_mbar_addr + consumer_stage * sizeof(uint64_t);

                    mbarrier_wait(tma_mbar_addr, consumer_tma_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    {
                        const int a_smem = mainloop_smem + consumer_stage * kStageSmemBytes;
                        const int b_smem = a_smem + kASmemBytes;
                        for (int mma_k = 0; mma_k < kMmaStepsPerStage; ++mma_k) {
                            const int k_slab = mma_k / (64 / kMmaK);
                            const int k_in_slab = mma_k % (64 / kMmaK);
                            const int a_byte_offset = k_slab * kTileM * 64 * sizeof(nv_bfloat16) +
                                                      k_in_slab * kMmaK * sizeof(nv_bfloat16);
                            const int b_byte_offset =
                                k_slab * (kTileN / kClusterM) * 64 * sizeof(nv_bfloat16) +
                                k_in_slab * kMmaK * sizeof(nv_bfloat16);
                            const int accumulate = tile_k != 0 || mma_k != 0;
                            tcgen05_mma_bf16<kClusterM>(stage_tmem_addr,
                                make_k_major_128b_desc(a_smem + a_byte_offset),
                                make_k_major_128b_desc(b_smem + b_byte_offset),
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
                    base_mainloop_mbar_addr + consumer_mainloop_stage * sizeof(uint64_t);
                tcgen05_commit<kClusterM>(mainloop_mbar_addr);

                consumer_mainloop_stage = (consumer_mainloop_stage + 1) % kTmemStageCount;
                if (consumer_mainloop_stage == 0) {
                    if constexpr (kPersistent) {
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
                    if constexpr (kPersistent) {
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
                                            store_tile *
                                                kStoreTileBytes)
                                            [swizzled_chunk];
                                    const int output_offset =
                                        (cta_row + tid) * n + block_col +
                                        n_group * 8;
                                    reinterpret_cast<int4 *>(
                                        c + output_offset)[0] = value;
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
                                : (remaining_block_rows < kEpilogueTileN
                                          ? remaining_block_rows
                                          : kEpilogueTileN);
                        const int block_col = tile_col + cta_rank * kTileM;
                        if (valid_block_rows == kEpilogueTileN) {
                            if (tid == 0) {
                                constexpr int kSwappedStoreColumns = 64;
                                constexpr int kSwappedStoreBytes =
                                    kEpilogueTileN * kSwappedStoreColumns *
                                    sizeof(nv_bfloat16);
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
                                reinterpret_cast<nv_bfloat16 *>(
                                    dynamic_smem + kMainloopSmemBytes +
                                    smem_stage * kEpilogueStageBytes);
                            const int atom = tid / kSwappedStoreColumns;
                            const int col_in_atom = tid % kSwappedStoreColumns;
                            for (int row = 0; row < valid_block_rows; ++row) {
                                const int bank_group = col_in_atom / kBankGroupElements;
                                const int value_in_group = col_in_atom % kBankGroupElements;
                                const int atom_elements =
                                    kEpilogueTileN * kSwappedStoreColumns;
                                const int swizzled_index =
                                    atom * atom_elements +
                                    row * (kSwizzleAtomBytes / sizeof(nv_bfloat16)) +
                                    (bank_group ^ (row & 7)) * kBankGroupElements +
                                    value_in_group;
                                c[(block_row + row) * n + block_col + tid] =
                                    stage_values[swizzled_index];
                            }
                        }
                    }

                    if (n_block + 1 < kEpilogueBlockCount) {
                        if (kEpilogueStageCount == 1 || n_block > 0) {
                            if constexpr (!kDirectGmem || kSwapAB) {
                                if (tid == 0) {
                                    asm volatile(
                                        "cp.async.bulk.wait_group.read %0;"
                                        :
                                        : "n"(kEpilogueStageCount - 1)
                                        : "memory");
                                }
                            }
                            asm volatile("bar.sync 1, 128;" ::: "memory");
                        }

                        const int next_stage = (n_block + 1) % kEpilogueStageCount;
                        tmem_to_smem(epilogue_mainloop_stage, n_block + 1, next_stage,
                                     cta_row, tile_col, valid_rows);
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

        if constexpr (!kPersistent) {
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

template <
    int kTileM,
    int kTileN,
    int kTileK,
    int kEpilogueTileN,
    int kClusterM,
    bool kSwapAB>
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
    using Config = KernelConfig<kTileM, kTileN, kTileK, kEpilogueTileN, kClusterM>;
    constexpr int kClusterTileM = Config::kClusterTileM;
    constexpr int kDynamicSmemBytes = Config::kDynamicSmemBytes;
    constexpr int kTaskTileRows = kSwapAB ? kTileN : kClusterTileM;
    constexpr int kLogicalNTile = kSwapAB ? kClusterTileM : kTileN;
    if (n % kLogicalNTile != 0 || k % kTileK != 0) {
        return;
    }

    CUtensorMap a_tensor_map;
    CUtensorMap b_tensor_map;
    CUtensorMap c_tensor_map;
    bool tensor_maps_ok;
    if constexpr (!kSwapAB) {
        tensor_maps_ok =
            init_a_tensor_map<kTileM, kTileK>(&a_tensor_map, d_a, m, k) &&
            init_b_tensor_map<kTileN, kTileK, kClusterM>(&b_tensor_map, d_b, experts, n, k) &&
            init_c_tensor_map<kTileM, kEpilogueTileN>(&c_tensor_map, d_c, m, n);
    } else {
        tensor_maps_ok =
            init_b_tensor_map<kClusterTileM, kTileK, kClusterM>(&a_tensor_map,
                d_b,
                experts,
                n,
                k) &&
            init_a_tensor_map<kTileN / kClusterM, kTileK>(&b_tensor_map, d_a, m, k) &&
            init_swapped_c_tensor_map<kTileM, kEpilogueTileN>(&c_tensor_map, d_c, m, n);
    }
    if (!tensor_maps_ok) {
        return;
    }

    // ── Workspace layout ──────────────────────────────────────────
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
    //  │  next_cluster_tile          int32        │  4 B
    //  │    global work-stealing cursor           │
    //  ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -  ─┤  ← task_offset
    //  │  tasks[max_tasks]        ExpertTileTask  │  max_tasks × 16 B
    //  │    {row_start, row_count,                │
    //  │     tile_row, expert}                    │
    //  └──────────────────────────────────────────┘
    //
    // Each row bundle produces one task per logical N tile.  The extra expert
    // slots account for the per-expert ceil when row counts are uneven.
    const int grid_n = n / kLogicalNTile;
    const int max_m_tasks = (m + kTaskTileRows - 1) / kTaskTileRows + experts;
    const int max_tasks = max_m_tasks * grid_n;
    const size_t metadata_bytes = (2 * experts + 2) * sizeof(int);
    const size_t task_offset =
        (metadata_bytes + alignof(ExpertTileTask) - 1) & ~(alignof(ExpertTileTask) - 1);
    const size_t workspace_bytes = task_offset + max_tasks * sizeof(ExpertTileTask);
    void *workspace = nullptr;
    if (cudaMallocAsync(&workspace, workspace_bytes, stream) != cudaSuccess) {
        return;
    }

    auto *expert_offsets = static_cast<int *>(workspace);
    int *expert_counts = expert_offsets + experts;
    int *task_count = expert_counts + experts;
    int *next_cluster_tile = task_count + 1;
    auto *tasks = reinterpret_cast<ExpertTileTask *>(static_cast<char *>(workspace) + task_offset);
    cudaMemsetAsync(expert_offsets, 0x7f, experts * sizeof(int), stream);
    cudaMemsetAsync(expert_counts, 0, experts * sizeof(int), stream);
    cudaMemsetAsync(task_count, 0, 2 * sizeof(int), stream);
    constexpr int kMetadataThreads = 256;
    collect_expert_metadata<<<(m + kMetadataThreads - 1) / kMetadataThreads,
        kMetadataThreads,
        0,
        stream>>>(d_grouped_layout, m, experts, expert_offsets, expert_counts);
    build_expert_tile_tasks<kTaskTileRows>
        <<<1, 1, 0, stream>>>(expert_offsets, expert_counts, experts, grid_n, tasks, task_count);

    const int max_cluster_tiles = max_tasks;

    if constexpr (SM110A_GROUPED_MODE == 1) {
        const int persistent_clusters =
            (max_cluster_tiles + kPersistentTileCount - 1) / kPersistentTileCount;
        auto kernel = bf16_grouped_m_gemm_kernel<kTileM,
            kTileN,
            kTileK,
            kEpilogueTileN,
            kClusterM,
            true,
            kSwapAB,
            (SM110A_GROUPED_DIRECT_GMEM != 0)>;
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
            a_tensor_map,
            b_tensor_map,
            c_tensor_map,
            tasks,
            task_count,
            next_cluster_tile,
            d_c,
            n,
            k);
    } else {
        auto kernel = bf16_grouped_m_gemm_kernel<kTileM,
            kTileN,
            kTileK,
            kEpilogueTileN,
            kClusterM,
            false,
            kSwapAB,
            (SM110A_GROUPED_DIRECT_GMEM != 0)>;
        if constexpr (kDynamicSmemBytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kDynamicSmemBytes);
        }
        kernel<<<kClusterM * max_cluster_tiles, kBlockThreads, kDynamicSmemBytes, stream>>>(
            a_tensor_map,
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

void launch_bf16_grouped_m_gemm_entry(
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b,
    const int32_t *d_grouped_layout,
    __nv_bfloat16 *d_c,
    int m,
    int experts,
    int n,
    int k,
    bool swap_ab,
    cudaStream_t stream) {
    if (d_a == nullptr || d_b == nullptr || d_grouped_layout == nullptr || d_c == nullptr) {
        return;
    }
    if (m <= 0 || experts <= 0 || n <= 0 || k <= 0) {
        return;
    }

    if (swap_ab) {
        launch_bf16_grouped_m_gemm_impl<SM110A_GROUPED_TILE_M,
            SM110A_GROUPED_TILE_N,
            SM110A_GROUPED_TILE_K,
            SM110A_GROUPED_EPILOGUE_TILE_N,
            SM110A_GROUPED_CLUSTER_M,
            true>(d_a, d_b, d_grouped_layout, d_c, m, experts, n, k, stream);
    } else {
        launch_bf16_grouped_m_gemm_impl<SM110A_GROUPED_TILE_M,
            SM110A_GROUPED_TILE_N,
            SM110A_GROUPED_TILE_K,
            SM110A_GROUPED_EPILOGUE_TILE_N,
            SM110A_GROUPED_CLUSTER_M,
            false>(d_a, d_b, d_grouped_layout, d_c, m, experts, n, k, stream);
    }
}

} // namespace

#if SM110A_GROUPED_MODE != 2
void launch_bf16_grouped_m_gemm(
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b,
    const int32_t *d_grouped_layout,
    __nv_bfloat16 *d_c,
    int m,
    int experts,
    int n,
    int k,
    bool swap_ab,
    cudaStream_t stream) {
    launch_bf16_grouped_m_gemm_entry(
        d_a, d_b, d_grouped_layout, d_c, m, experts, n, k, swap_ab, stream);
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
    bool swap_ab,
    cudaStream_t stream) {
    launch_bf16_grouped_m_gemm(d_a, d_b, d_grouped_layout, d_c, m, experts, n, k, swap_ab, stream);
}
#endif
