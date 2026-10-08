#include <cassert>
#include <cstdint>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// One KV stage holds three 128-row MMA tiles. The three epilogue groups drain
// independent TMEM results while the producer walks an eight-stage KV ring.

namespace {

constexpr int kHeads = 64;
constexpr int kDim = 128;
constexpr int kTileQ = 2;
constexpr int kTileKV = 128;
constexpr int kQStages = 2;
constexpr int kKVStages = 8;
constexpr int kComputeGroups = 3;
constexpr int kSplitKV = kComputeGroups * kTileKV;
constexpr int kPersistentCtasPerSm = 1;
constexpr int kTileM = kTileQ * kHeads;
constexpr int kEpilogueWarps = kTileKV / 32;
constexpr int kThreads = (4 + kComputeGroups * kEpilogueWarps) * 32;
constexpr int kABytes = kTileM * (kDim / 2);
constexpr int kBBytes = kSplitKV * (kDim / 2);
constexpr int kQScaleBytes = kTileM * 4;
constexpr int kKScaleBytes = kSplitKV * 4;
constexpr int kQSmemBytes = kABytes + kQScaleBytes;
constexpr int kKVStageSmemBytes = kBBytes + kKScaleBytes;
constexpr int kMainloopSmemBytes =
    kQStages * kQSmemBytes + kKVStages * kKVStageSmemBytes;
constexpr int kSmemBytes = kMainloopSmemBytes;

// KV supplies the 128 MMA rows and Q/head supplies the 128 columns.
// Three accumulator slots overlap MMA with three independent reductions.
// Keep scales in stage-specific TMEM slots until their Q or KV stage is free.
constexpr int kAccumulatorTmemCols = kComputeGroups * kTileM;
constexpr int kQScaleTmemCol = kAccumulatorTmemCols;
constexpr int kKScaleTmemCol = kQScaleTmemCol + kQStages * 4;
constexpr int kScaleTmemCols = kQStages * 4 + kKVStages * kComputeGroups * 4;
constexpr int kTmemCols = 512;
static_assert(kAccumulatorTmemCols + kScaleTmemCols <= kTmemCols);

// Q and K are row-major packed E2M1; scales arrive in the repository's
// blocked 128x4 UE8M0 layout and are compacted by the producer warps.
void encode_fp4(CUtensorMap* map, const uint8_t* ptr, int rows, int tile_rows) {
    const uint64_t dims[3] = {kDim, static_cast<uint64_t>(rows), 1};
    const uint64_t strides[2] = {kDim / 2, static_cast<uint64_t>(rows) * (kDim / 2)};
    const uint32_t box[3] = {kDim, static_cast<uint32_t>(tile_rows), 1};
    const uint32_t elem[3] = {1, 1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
        const_cast<uint8_t*>(ptr), dims, strides, box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(result == CUDA_SUCCESS);
}

__device__ __forceinline__ uint64_t fp4_desc(int addr) {
    const uint64_t base = (static_cast<uint64_t>(addr) & 0x3ffffULL) >> 4;
    const uint64_t stride = ((8 * 64) >> 4);
    return base | (stride << 32) | (1ULL << 46) | (4ULL << 61);
}

__device__ __forceinline__ uint64_t scale_desc(int addr) {
    return ((static_cast<uint64_t>(addr) & 0x3ffffULL) >> 4) |
           (8ULL << 32) | (1ULL << 46);
}

// Blocked input stores two useful E8M0 bytes in each four-byte K=64 slab.
// The destination is the 32x4-word layout expected by tcgen05.cp.
__device__ __forceinline__ void pack_scale_tile(
    const uint8_t* src, uint32_t* dst, int lane) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int offset = lane * 16 + i * 4;
        const uint32_t lo = *reinterpret_cast<const uint16_t*>(src + offset);
        const uint32_t hi = *reinterpret_cast<const uint16_t*>(src + 512 + offset);
        dst[lane * 4 + i] = lo | (hi << 16);
    }
}

// Result stages occupy columns 0..383. KV and Q scale slots follow.
__device__ __forceinline__ int result_col(int stage) {
    return stage * kTileM;
}

__device__ __forceinline__ void load_tmem_x16(float* value, int address) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(value[0]), "=f"(value[1]), "=f"(value[2]), "=f"(value[3]),
          "=f"(value[4]), "=f"(value[5]), "=f"(value[6]), "=f"(value[7]),
          "=f"(value[8]), "=f"(value[9]), "=f"(value[10]), "=f"(value[11]),
          "=f"(value[12]), "=f"(value[13]), "=f"(value[14]), "=f"(value[15])
        : "r"(address));
}

__device__ __forceinline__ __nv_bfloat162 relu_bf16x2(float x, float y) {
    uint32_t packed;
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;"
                 : "=r"(packed) : "f"(y), "f"(x));
    return *reinterpret_cast<__nv_bfloat162*>(&packed);
}

__device__ __forceinline__ void wait_barrier(int address, int phase) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "L_wait:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, 10000000;\n\t"
                 "@!p bra.uni L_wait;\n\t"
                 "}" :: "r"(address), "r"(phase) : "memory");
}

template <int kClusterM>
__global__ __launch_bounds__(kThreads, 1) __cluster_dims__(kClusterM, 1, 1)
void indexer_score_prefill_kernel(
    const __grid_constant__ CUtensorMap q_map,
    const __grid_constant__ CUtensorMap k_map,
    const uint8_t* __restrict__ q_scales,
    const uint8_t* __restrict__ k_scales,
    const __nv_bfloat16* __restrict__ head_weight,
    __nv_bfloat16* __restrict__ out, int query_count, int kv_count,
    int batch_count, int page_size, int block_table_stride,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ kv_seqlens,
    bool causal) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int batch = blockIdx.y;
    const int q_offset = cu_seqlens_q ? cu_seqlens_q[batch] : batch * query_count;
    const int local_query_len = cu_seqlens_q
        ? cu_seqlens_q[batch + 1] - q_offset : query_count;
    const int local_kv_len = kv_seqlens ? kv_seqlens[batch] : kv_count;
    const int q_tiles = (local_query_len + kTileQ - 1) / kTileQ;
    const int kv_tiles = (local_kv_len + kSplitKV - 1) / kSplitKV;
    const int padded_q_tiles = ((q_tiles + kClusterM - 1) / kClusterM) * kClusterM;
    const int cta_rank = blockIdx.x % kClusterM;
    if (blockIdx.x >= padded_q_tiles || local_kv_len <= 0) {
        return;
    }

    extern __shared__ __align__(1024) uint8_t smem[];
    const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem));
    const int kv_smem_base = smem_base + kQStages * kQSmemBytes;
    __shared__ uint64_t q_barrier[kQStages];
    __shared__ uint64_t q_done_barrier[kQStages];
    __shared__ uint64_t load_barrier[kKVStages];
    __shared__ uint64_t mma_barrier[kKVStages];
    __shared__ uint64_t result_barrier[kComputeGroups];
    __shared__ uint64_t epilogue_barrier[kComputeGroups];
    __shared__ int tmem_base;
    if (tid == 0) {
        #pragma unroll
        for (int stage = 0; stage < kQStages; ++stage) {
            const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier[stage]));
            const int q_done_bar = static_cast<int>(__cvta_generic_to_shared(&q_done_barrier[stage]));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(q_bar));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         :: "r"(q_done_bar), "r"(kComputeGroups));
        }
        #pragma unroll
        for (int stage = 0; stage < kKVStages; ++stage) {
            const int load_addr = static_cast<int>(__cvta_generic_to_shared(&load_barrier[stage]));
            const int mma_addr = static_cast<int>(__cvta_generic_to_shared(&mma_barrier[stage]));
            // The scale producer and the TMA producer each arrive once.
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" :: "r"(load_addr));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         :: "r"(mma_addr), "r"(1));
        }
        #pragma unroll
        for (int group = 0; group < kComputeGroups; ++group) {
            const int result_addr = static_cast<int>(__cvta_generic_to_shared(&result_barrier[group]));
            const int epi_addr = static_cast<int>(__cvta_generic_to_shared(&epilogue_barrier[group]));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(result_addr));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         :: "r"(epi_addr), "r"(kEpilogueWarps));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 2) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(&tmem_base));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(addr), "r"(kTmemCols));
    }
    __syncthreads();
    if constexpr (kClusterM == 2) {
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }

    // Every CTA owns a static strided subset of query tiles. Q and its scales
    // remain in shared/TMEM while that CTA traverses visible KV tiles.
    int next_task = 0;
    for (int q_tile = blockIdx.x, q_round = 0; q_tile < padded_q_tiles;
         q_tile += gridDim.x, ++q_round) {
        const int q_base = q_tile * kTileQ;
        // Cluster peers multicast KV into each other's shared memory, so both
        // CTAs must run the same number of KV tasks even at a tile boundary.
        const int cluster_q_end =
            (q_tile - cta_rank + kClusterM) * kTileQ;
        const int active_kv_tiles = causal
            ? min(kv_tiles, (cluster_q_end + kSplitKV - 1) / kSplitKV)
            : kv_tiles;
        const int first_task = next_task;
        next_task += active_kv_tiles;
        const int prev_q_tile = q_tile - gridDim.x;
        const int prev_cluster_q_end =
            (prev_q_tile - cta_rank + kClusterM) * kTileQ;
        const int prev_active_kv_tiles = causal
            ? min(kv_tiles, (prev_cluster_q_end + kSplitKV - 1) / kSplitKV)
            : kv_tiles;
        // When consecutive Q rounds traverse the same complete KV ring,
        // its data and scales remain resident in shared memory and TMEM.
        const bool reuse_kv_ring = q_round > 0 &&
            prev_active_kv_tiles == kKVStages &&
            active_kv_tiles == kKVStages;

        if (warp == 1) {
            const int zero = 0;
            const int q_stage = q_round % kQStages;
            const int q_addr = smem_base + q_stage * kQSmemBytes;
            const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier[q_stage]));
            const uint64_t qptr = reinterpret_cast<uint64_t>(&q_map);
            if (q_round >= kQStages) {
                const int q_done = static_cast<int>(
                    __cvta_generic_to_shared(&q_done_barrier[q_stage]));
                wait_barrier(q_done, ((q_round - kQStages) / kQStages) & 1);
            }
            const int scale_tile = batch * (((query_count + 1) & ~1) / 2) + q_base / 2;
            auto* compact_q_scales = reinterpret_cast<uint32_t*>(
                smem + q_stage * kQSmemBytes + kABytes);
            if (q_base < local_query_len) {
                pack_scale_tile(q_scales + scale_tile * 1024, compact_q_scales, lane);
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    compact_q_scales[lane * 4 + i] = 0;
            }
            if (q_base >= local_query_len) {
                auto* packed_q = reinterpret_cast<uint32_t*>(
                    smem + q_stage * kQSmemBytes);
                for (int i = lane; i < kABytes / 4; i += 32)
                    packed_q[i] = 0;
            }
            __syncwarp();
            if (lane == 0) {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (q_base < local_query_len) {
                    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                 :: "r"(q_bar), "r"(kABytes) : "memory");
                    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                                 "[%0], [%1, {%2, %3, %4}], [%5];"
                                 :: "r"(q_addr), "l"(qptr), "r"(zero), "r"((q_offset + q_base) * kHeads),
                                    "r"(zero), "r"(q_bar) : "memory");
                } else {
                    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                 :: "r"(q_bar) : "memory");
                }
            }
        }

        if (warp == 0) {
            const int zero = 0;
            const uint64_t kptr = reinterpret_cast<uint64_t>(&k_map);
            auto load_kv = [&](int kv_tile) {
                const int task = first_task + kv_tile;
                // Initial fills can multicast safely. Refilled stages use
                // CTA-local TMA, so each CTA can recycle its stage as soon as
                // its own MMA has finished reading that shared memory.
                const bool multicast = kClusterM == 2 && task < kKVStages;
                const bool owns_tma = !multicast || (kv_tile & 1) == cta_rank;
                const int stage = task % kKVStages;
                const int kv_base = kv_tile * kSplitKV;
                if (task >= kKVStages) {
                    // MMA is the final reader of this shared KV stage. Its
                    // completion releases the stage for the next TMA load;
                    // epilogue only consumes the independent TMEM result.
                    const int old_mma = static_cast<int>(
                        __cvta_generic_to_shared(&mma_barrier[stage]));
                    wait_barrier(old_mma, ((task - kKVStages) / kKVStages) & 1);
                }
                const int k_addr = kv_smem_base + stage * kKVStageSmemBytes;
                const int load_bar = static_cast<int>(
                    __cvta_generic_to_shared(&load_barrier[stage]));
                if (reuse_kv_ring) {
                    if (lane != 0) return;
                    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                 :: "r"(load_bar) : "memory");
                    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                 :: "r"(load_bar) : "memory");
                    return;
                }
                #pragma unroll
                for (int subtile = 0; subtile < kComputeGroups; ++subtile) {
                    const int logical_row = min(kv_base + subtile * kTileKV,
                                                ((local_kv_len - 1) / 128) * 128);
                    const int logical_page = page_size
                        ? min(logical_row / page_size, (local_kv_len - 1) / page_size) : 0;
                    const int physical_page = page_size
                        ? block_table[batch * block_table_stride + logical_page] : 0;
                    const int scale_row = page_size
                        ? physical_page * (page_size / 128) +
                          (logical_row / 128) % (page_size / 128)
                        : batch * ((kv_count + 127) / 128) + logical_row / 128;
                    pack_scale_tile(k_scales + scale_row * 1024,
                                    reinterpret_cast<uint32_t*>(
                                        smem + kQStages * kQSmemBytes + stage * kKVStageSmemBytes +
                                        kBBytes + subtile * 512), lane);
                }
                __syncwarp();
                if (lane != 0) return;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                             :: "r"(load_bar) : "memory");
                if (!owns_tma) return;
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                             :: "r"(load_bar), "r"(kBBytes) : "memory");
                if constexpr (kClusterM == 2) {
                    if (multicast) {
                        int peer_bar;
                        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                                     : "=r"(peer_bar) : "r"(load_bar), "r"(1 - cta_rank));
                        asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;"
                                     :: "r"(peer_bar), "r"(kBBytes) : "memory");
                    }
                }
                #pragma unroll
                for (int tile_page = 0; tile_page < kComputeGroups; ++tile_page) {
                    const int logical_row = kv_base + tile_page * kTileKV;
                    const int logical_page = page_size
                        ? min(logical_row / page_size, (local_kv_len - 1) / page_size) : 0;
                    const int physical_page = page_size
                        ? block_table[batch * block_table_stride + logical_page] : 0;
                    const int k_row = page_size ? physical_page * page_size +
                        logical_row % page_size
                        : batch * kv_count + logical_row;
                    const int dst = k_addr + tile_page * kTileKV * (kDim / 2);
                    if constexpr (kClusterM == 2) {
                        if (multicast) {
                            asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1 "
                                         "[%0], [%1, {%2, %3, %4}], [%5], %6;"
                                         :: "r"(dst), "l"(kptr), "r"(zero), "r"(k_row),
                                            "r"(zero), "r"(load_bar), "h"(int16_t{3}) : "memory");
                            continue;
                        }
                    }
                    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                                 "[%0], [%1, {%2, %3, %4}], [%5];"
                                 :: "r"(dst), "l"(kptr), "r"(zero), "r"(k_row),
                                    "r"(zero), "r"(load_bar) : "memory");
                }
            };
            for (int kv_tile = 0;
                 kv_tile < (active_kv_tiles < kKVStages
                                ? active_kv_tiles : kKVStages);
                 ++kv_tile)
                load_kv(kv_tile);
            for (int kv_tile = 0; kv_tile < active_kv_tiles; ++kv_tile) {
                if (kv_tile + kKVStages < active_kv_tiles)
                    load_kv(kv_tile + kKVStages);
            }
        }

        if (warp == 2 && lane == 0) {
            const int q_stage = q_round % kQStages;
            const int q_addr = smem_base + q_stage * kQSmemBytes;
            const int qs_addr = q_addr + kABytes;
            const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier[q_stage]));
            wait_barrier(q_bar, (q_round / kQStages) & 1);
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         :: "r"(tmem_base + kQScaleTmemCol + q_stage * 4),
                            "l"(scale_desc(qs_addr)) : "memory");
            // M=128 KV rows, N=128 Q/head columns, K=128 per slab.
            constexpr uint32_t instruction = 0x08A00480U;
            for (int kv_tile = 0; kv_tile < active_kv_tiles; ++kv_tile) {
                const int task = first_task + kv_tile;
                const int stage = task % kKVStages;
                const int phase = (task / kKVStages) & 1;
                const int load_bar = static_cast<int>(
                    __cvta_generic_to_shared(&load_barrier[stage]));
                wait_barrier(load_bar, phase);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const int k_addr = kv_smem_base + stage * kKVStageSmemBytes;
                const int ks_addr = k_addr + kBBytes;
                if (!reuse_kv_ring) {
                    #pragma unroll
                    for (int subtile = 0; subtile < kComputeGroups; ++subtile)
                        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                     :: "r"(tmem_base + kKScaleTmemCol +
                                            stage * kComputeGroups * 4 + subtile * 4),
                                        "l"(scale_desc(ks_addr + subtile * 512)) : "memory");
                }
                #pragma unroll
                for (int subtile = 0; subtile < kComputeGroups; ++subtile) {
                    const int result_group = subtile;
                    if (task > 0) {
                        const int old_epi = static_cast<int>(
                            __cvta_generic_to_shared(&epilogue_barrier[result_group]));
                        wait_barrier(old_epi, (task - 1) & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    }
                    const int result_addr = tmem_base + result_col(result_group);
                    #pragma unroll
                    for (int slab = 0; slab < 2; ++slab) {
                        const uint64_t adesc = fp4_desc(k_addr + subtile * kTileKV * (kDim / 2) + slab * 32);
                        const uint64_t bdesc = fp4_desc(q_addr + slab * 32);
                        const uint32_t slab_instruction = instruction |
                            (slab ? ((2U << 29) | (2U << 4)) : 0U);
                        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
                                     "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
                                     "[%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                                     :: "r"(result_addr), "l"(adesc), "l"(bdesc), "r"(slab_instruction),
                                        "r"(tmem_base + kKScaleTmemCol + stage * kComputeGroups * 4 + subtile * 4),
                                        "r"(tmem_base + kQScaleTmemCol + q_stage * 4),
                                        "r"(slab) : "memory");
                    }
                    const int ready = static_cast<int>(
                        __cvta_generic_to_shared(&result_barrier[result_group]));
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                 :: "r"(ready) : "memory");
                }
                const int mma_bar = static_cast<int>(
                    __cvta_generic_to_shared(&mma_barrier[stage]));
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                             :: "r"(mma_bar) : "memory");
            }
        }

        if (warp >= 4) {
            const int compute_group = (warp - 4) / kEpilogueWarps;
            const int local_kv = ((warp - 4) % kEpilogueWarps) * 32 + lane;
            const int tmem_row = ((warp - 4) % 4) * 32;
            // One lane owns one KV candidate. Keep both queries' head weights
            // in registers while this CTA walks its visible KV tiles.
            __nv_bfloat162 weights[kTileQ][kHeads / 2];
            #pragma unroll
            for (int q = 0; q < kTileQ; ++q) {
                #pragma unroll
                for (int i = 0; i < kHeads / 2; ++i) {
                    weights[q][i] = q_base + q < local_query_len
                        ? reinterpret_cast<const __nv_bfloat162*>(head_weight +
                            (q_offset + q_base + q) * kHeads)[i]
                        : __floats2bfloat162_rn(0.0f, 0.0f);
                }
            }
            for (int kv_tile = 0; kv_tile < active_kv_tiles; ++kv_tile) {
                const int task = first_task + kv_tile;
                const int stage = task % kKVStages;
                const int ready = static_cast<int>(
                    __cvta_generic_to_shared(&result_barrier[compute_group]));
                wait_barrier(ready, task & 1);
                #pragma unroll
                for (int local_q = 0; local_q < kTileQ; ++local_q) {
                    __nv_bfloat162 sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                    __nv_bfloat162 sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                    #pragma unroll
                    for (int head_base = 0; head_base < kHeads; head_base += 16) {
                        float value[16];
                        const int address = tmem_base + result_col(compute_group) +
                            local_q * kHeads + head_base + (tmem_row << 16);
                        load_tmem_x16(value, address);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (local_q == kTileQ - 1 && head_base == kHeads - 16) {
                            // All TMEM values are in registers. Release the
                            // result slot while reduction and store continue.
                            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                            if (lane == 0) {
                                const int epi_bar = static_cast<int>(
                                    __cvta_generic_to_shared(&epilogue_barrier[compute_group]));
                                asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                                             :: "r"(epi_bar) : "memory");
                            }
                        }
                        #pragma unroll
                        for (int i = 0; i < 16; i += 4) {
                            sum_0 = __hfma2(relu_bf16x2(value[i], value[i + 1]),
                                            weights[local_q][(head_base + i) / 2], sum_0);
                            sum_1 = __hfma2(relu_bf16x2(value[i + 2], value[i + 3]),
                                            weights[local_q][(head_base + i) / 2 + 1], sum_1);
                        }
                    }
                    const __nv_bfloat162 sum = __hadd2_rn(sum_0, sum_1);
                    const __nv_bfloat162 reduced = __hadd2_rn(
                        __low2bfloat162(sum), __high2bfloat162(sum));
                    const int global_kv = kv_tile * kSplitKV +
                        compute_group * kTileKV + local_kv;
                    if (q_base + local_q < local_query_len &&
                        global_kv < local_kv_len &&
                        (!causal || global_kv <= q_base + local_q))
                        out[static_cast<int64_t>(q_offset + q_base + local_q) * kv_count +
                            global_kv] = __low2bfloat16(reduced);
                }
            }
            // Every group has finished reading the MMAs for this Q tile.
            // Release the Q stage without stalling the MMA issuer at the
            // end of the tile; groups with no KV task arrive as well.
            if (lane == 0 && (warp - 4) % kEpilogueWarps == 0) {
                const int q_done_bar = static_cast<int>(__cvta_generic_to_shared(
                    &q_done_barrier[q_round % kQStages]));
                asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                             :: "r"(q_done_bar) : "memory");
            }
        }
    }

    if (warp == 2) {
        if (next_task > 0) {
            const int task = next_task - 1;
            #pragma unroll
            for (int group = 0; group < kComputeGroups; ++group) {
            const int last_epi = static_cast<int>(
                __cvta_generic_to_shared(&epilogue_barrier[group]));
            wait_barrier(last_epi, task & 1);
            }
        }
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tmem_base), "r"(kTmemCols));
    }
}

}  // namespace

extern "C" void sm110a_indexer_score_prefill_launch(
    const uint8_t* q, const uint8_t* k,
    const uint8_t* q_scales, const uint8_t* k_scales,
    const __nv_bfloat16* head_weight, __nv_bfloat16* out,
    int query_count, int kv_count, int batch_count, int total_q,
    int num_pages, int page_size, const int32_t* block_table,
    int block_table_stride, const int32_t* cu_seqlens_q,
    const int32_t* kv_seqlens, int cluster_m, bool causal,
    cudaStream_t stream) {
    CUtensorMap q_map, k_map;
    encode_fp4(&q_map, q, total_q * kHeads, kTileM);
    encode_fp4(&k_map, k, page_size ? num_pages * page_size : batch_count * kv_count,
               kTileKV);
    int device = 0;
    int sm_count = 0;
    assert(cudaGetDevice(&device) == cudaSuccess);
    assert(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                  device) == cudaSuccess);
    const int q_tiles = (query_count + kTileQ - 1) / kTileQ;
    const int max_ctas = sm_count * kPersistentCtasPerSm;
    assert(cluster_m == 1 || cluster_m == 2);
    const bool use_cluster = cluster_m == 2 && q_tiles > 1;
    cluster_m = use_cluster ? 2 : 1;
    const int padded_q_tiles = ((q_tiles + cluster_m - 1) / cluster_m) * cluster_m;
    const int max_cluster_ctas = (max_ctas / cluster_m) * cluster_m;
    const dim3 grid(padded_q_tiles < max_cluster_ctas ? padded_q_tiles : max_cluster_ctas,
                    batch_count);
    const auto kernel = use_cluster ? indexer_score_prefill_kernel<2>
                                    : indexer_score_prefill_kernel<1>;
    cudaFuncSetAttribute(kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    kernel<<<grid, kThreads, kSmemBytes, stream>>>(
        q_map, k_map, q_scales, k_scales, head_weight, out,
        query_count, kv_count, batch_count, page_size, block_table_stride,
        block_table, cu_seqlens_q, kv_seqlens, causal);
}
