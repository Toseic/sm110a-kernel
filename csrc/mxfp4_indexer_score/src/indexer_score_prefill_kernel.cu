#include <cassert>
#include <cstdint>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeads = 64;
constexpr int kDim = 128;
constexpr int kTileQ = 2;
constexpr int kTileKV = 128;
constexpr int kPipelineStages = 2;
constexpr int kPersistentCtasPerSm = 1;
constexpr int kTileM = kTileQ * kHeads;
constexpr int kEpilogueWarps = kTileKV / 32;
constexpr int kThreads = (4 + kEpilogueWarps) * 32;
constexpr int kABytes = kTileM * 128;
constexpr int kBBytes = kTileKV * 128;
constexpr int kQScaleBytes = 2 * 512;
constexpr int kKScaleTiles = kTileKV / 128;
constexpr int kKScaleBytes = kKScaleTiles * 2 * 512;
constexpr int kQSmemBytes = kABytes + kQScaleBytes;
constexpr int kKVStageSmemBytes = kBBytes + kKScaleBytes;
constexpr int kMainloopSmemBytes =
    kQSmemBytes + kPipelineStages * kKVStageSmemBytes;
constexpr int kSmemBytes = kMainloopSmemBytes;

// KV supplies the 128 MMA rows and Q/head supplies the 128 columns. Each
// pipeline stage has its own accumulator columns; scales use 8 scratch columns.
constexpr int kAccumulatorTmemCols = kPipelineStages * kTileM;
constexpr int kScaleTmemCol = kAccumulatorTmemCols;
constexpr int kScaleTmemCols = 8;
constexpr int kTmemCols = 512;
static_assert(kAccumulatorTmemCols + kScaleTmemCols <= kTmemCols);

// Q and K are row-major packed E2M1; their scales use the same 128x4
// blocked UE8M0 layout as the repository's MXFP4 GEMM.
void encode_fp4(CUtensorMap* map, const uint8_t* ptr, int rows, int tile_rows) {
    const uint64_t dims[3] = {kDim, static_cast<uint64_t>(rows), 1};
    const uint64_t strides[2] = {kDim / 2, static_cast<uint64_t>(rows) * (kDim / 2)};
    const uint32_t box[3] = {kDim, static_cast<uint32_t>(tile_rows), 1};
    const uint32_t elem[3] = {1, 1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
        const_cast<uint8_t*>(ptr), dims, strides, box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(result == CUDA_SUCCESS);
}

void encode_scales(CUtensorMap* map, const uint8_t* ptr, int row_tiles) {
    const uint64_t dims[4] = {256, 2, 2, static_cast<uint64_t>(row_tiles)};
    const uint64_t strides[3] = {256, 512, 1024};
    const uint32_t box[4] = {256, 2, 2, 1};
    const uint32_t elem[4] = {1, 1, 1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
        const_cast<uint8_t*>(ptr), dims, strides, box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(result == CUDA_SUCCESS);
}

// Each batch's scale rows are padded to a two-query (128-row) block. The two
// 512-byte halves are copied separately, but stay within the same blocked tile.
void encode_q_scale_rows(CUtensorMap* map, const uint8_t* ptr, int rows) {
    const uint64_t dims[3] = {256, 2, static_cast<uint64_t>(rows)};
    const uint64_t strides[2] = {256, 512};
    const uint32_t box[3] = {256, 2, 1};
    const uint32_t elem[3] = {1, 1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
        const_cast<uint8_t*>(ptr), dims, strides, box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(result == CUDA_SUCCESS);
}

__device__ __forceinline__ uint64_t fp4_desc(int addr) {
    const uint64_t base = (static_cast<uint64_t>(addr) & 0x3ffffULL) >> 4;
    const uint64_t stride = ((8 * 128) >> 4);
    return base | (stride << 32) | (1ULL << 46) |
           (static_cast<uint64_t>((addr >> 7) & 7) << 49) | (2ULL << 61);
}

__device__ __forceinline__ uint64_t scale_desc(int addr) {
    return ((static_cast<uint64_t>(addr) & 0x3ffffULL) >> 4) |
           (8ULL << 32) | (1ULL << 46);
}

// Result stages occupy columns 0..127 and 128..255. Scales occupy 256..263.
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
__global__ __cluster_dims__(kClusterM, 1, 1) void indexer_score_prefill_kernel(
    const __grid_constant__ CUtensorMap q_map,
    const __grid_constant__ CUtensorMap k_map,
    const __grid_constant__ CUtensorMap qs_map,
    const __grid_constant__ CUtensorMap ks_map,
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
    const int kv_tiles = (local_kv_len + kTileKV - 1) / kTileKV;
    const int padded_q_tiles = ((q_tiles + kClusterM - 1) / kClusterM) * kClusterM;
    const int cta_rank = blockIdx.x % kClusterM;
    if (blockIdx.x >= padded_q_tiles || local_kv_len <= 0) {
        return;
    }

    extern __shared__ __align__(1024) uint8_t smem[];
    const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem));
    const int q_addr = smem_base;
    const int qs_addr = q_addr + kABytes;
    const int kv_smem_base = smem_base + kQSmemBytes;
    __shared__ uint64_t q_barrier;
    __shared__ uint64_t q_done_barrier;
    __shared__ uint64_t load_barrier[kPipelineStages];
    __shared__ uint64_t mma_barrier[kPipelineStages];
    __shared__ uint64_t epilogue_barrier[kPipelineStages];
    __shared__ int tmem_base;
    if (tid == 0) {
        const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier));
        const int q_done_bar = static_cast<int>(__cvta_generic_to_shared(&q_done_barrier));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(q_bar));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(q_done_bar));
        #pragma unroll
        for (int stage = 0; stage < kPipelineStages; ++stage) {
            const int load_addr = static_cast<int>(__cvta_generic_to_shared(&load_barrier[stage]));
            const int mma_addr = static_cast<int>(__cvta_generic_to_shared(&mma_barrier[stage]));
            const int epi_addr = static_cast<int>(__cvta_generic_to_shared(&epilogue_barrier[stage]));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(load_addr));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         :: "r"(mma_addr), "r"(kClusterM));
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
            ? min(kv_tiles, (cluster_q_end + kTileKV - 1) / kTileKV)
            : kv_tiles;
        const int first_task = next_task;
        next_task += active_kv_tiles;

        if (warp == 0 && lane == 0) {
            const int zero = 0;
            const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier));
            const uint64_t qptr = reinterpret_cast<uint64_t>(&q_map);
            const uint64_t qsptr = reinterpret_cast<uint64_t>(&qs_map);
            if (q_round > 0)
                wait_barrier(q_bar, (q_round - 1) & 1);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                         :: "r"(q_bar), "r"(kTileM * kDim / 2 + kQScaleBytes) : "memory");
            asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                         "[%0], [%1, {%2, %3, %4}], [%5];"
                         :: "r"(q_addr), "l"(qptr), "r"(zero), "r"((q_offset + q_base) * kHeads),
                            "r"(zero), "r"(q_bar) : "memory");
            const int qs_row = batch * ((query_count + 1) & ~1) + q_base;
            #pragma unroll
            for (int row = 0; row < kTileQ; ++row)
                asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                             "[%0], [%1, {%2, %3, %4}], [%5];"
                             :: "r"(qs_addr + row * 512), "l"(qsptr), "r"(zero),
                                "r"(zero), "r"(qs_row + row), "r"(q_bar) : "memory");

            const uint64_t kptr = reinterpret_cast<uint64_t>(&k_map);
            const uint64_t ksptr = reinterpret_cast<uint64_t>(&ks_map);
            auto load_kv = [&](int kv_tile) {
                if constexpr (kClusterM == 2) {
                    if ((kv_tile & 1) != cta_rank) return;
                }
                const int task = first_task + kv_tile;
                const int stage = task % kPipelineStages;
                const int kv_base = kv_tile * kTileKV;
                if (task >= kPipelineStages) {
                    const int old_mma = static_cast<int>(
                        __cvta_generic_to_shared(&mma_barrier[stage]));
                    wait_barrier(old_mma, ((task - kPipelineStages) / kPipelineStages) & 1);
                }
                const int k_addr = kv_smem_base + stage * kKVStageSmemBytes;
                const int ks_addr = k_addr + kBBytes;
                const int load_bar = static_cast<int>(
                    __cvta_generic_to_shared(&load_barrier[stage]));
                if (task >= kPipelineStages)
                    wait_barrier(load_bar, ((task - kPipelineStages) / kPipelineStages) & 1);
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                             :: "r"(load_bar), "r"(kTileKV * kDim / 2 + kKScaleBytes) : "memory");
                if constexpr (kClusterM == 2) {
                    int peer_bar;
                    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                                 : "=r"(peer_bar) : "r"(load_bar), "r"(1 - cta_rank));
                    asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;"
                                 :: "r"(peer_bar), "r"(kTileKV * kDim / 2 + kKScaleBytes) : "memory");
                }
                const int pages_per_tile = page_size && page_size < kTileKV
                    ? kTileKV / page_size : 1;
                for (int tile_page = 0; tile_page < pages_per_tile; ++tile_page) {
                    const int logical_row = kv_base + tile_page * (page_size ? page_size : kTileKV);
                    const int logical_page = page_size
                        ? min(logical_row / page_size, (local_kv_len - 1) / page_size) : 0;
                    const int physical_page = page_size
                        ? block_table[batch * block_table_stride + logical_page] : 0;
                    const int k_row = page_size ? physical_page * page_size +
                        logical_row % page_size
                        : batch * kv_count + kv_base;
                    // TMA's shared FP4 tile uses 128 bytes per K row even
                    // though each packed global row occupies 64 bytes.
                    const int dst = k_addr + tile_page *
                        (page_size ? page_size : kTileKV) * kDim;
                    if constexpr (kClusterM == 2) {
                        asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1 "
                                     "[%0], [%1, {%2, %3, %4}], [%5], %6;"
                                     :: "r"(dst), "l"(kptr), "r"(zero), "r"(k_row),
                                        "r"(zero), "r"(load_bar), "h"(int16_t{3}) : "memory");
                    } else {
                        asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                                     "[%0], [%1, {%2, %3, %4}], [%5];"
                                     :: "r"(dst), "l"(kptr), "r"(zero), "r"(k_row),
                                        "r"(zero), "r"(load_bar) : "memory");
                    }
                }
                #pragma unroll
                for (int scale_tile = 0; scale_tile < kKScaleTiles; ++scale_tile) {
                    const int logical_row = kv_base + scale_tile * 128;
                    const int logical_page = page_size
                        ? min(logical_row / page_size, (local_kv_len - 1) / page_size) : 0;
                    const int physical_page = page_size
                        ? block_table[batch * block_table_stride + logical_page] : 0;
                    const int scale_row = page_size
                        ? physical_page * (page_size / 128) +
                              (logical_row / 128) % (page_size / 128)
                        : batch * ((kv_count + 127) / 128) + logical_row / 128;
                    if constexpr (kClusterM == 2) {
                        asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1 "
                                     "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                     :: "r"(ks_addr + scale_tile * 1024), "l"(ksptr),
                                        "r"(zero), "r"(zero), "r"(zero),
                                        "r"(scale_row), "r"(load_bar),
                                        "h"(int16_t{3}) : "memory");
                    } else {
                        asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                                     "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                                     :: "r"(ks_addr + scale_tile * 1024), "l"(ksptr),
                                        "r"(zero), "r"(zero), "r"(zero),
                                        "r"(scale_row), "r"(load_bar) : "memory");
                    }
                }
            };
            for (int kv_tile = 0;
                 kv_tile < (active_kv_tiles < kPipelineStages
                                ? active_kv_tiles : kPipelineStages);
                 ++kv_tile)
                load_kv(kv_tile);
            for (int kv_tile = 0; kv_tile < active_kv_tiles; ++kv_tile) {
                if (kv_tile + kPipelineStages < active_kv_tiles)
                    load_kv(kv_tile + kPipelineStages);
            }
            const int q_done_bar = static_cast<int>(
                __cvta_generic_to_shared(&q_done_barrier));
            wait_barrier(q_done_bar, q_round & 1);
        }

        if (warp == 2 && lane == 0) {
            const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier));
            wait_barrier(q_bar, q_round & 1);
            // M=128 KV rows, N=128 Q/head columns, K=128 per slab.
            constexpr uint32_t instruction = 0x08A00480U;
            for (int kv_tile = 0; kv_tile < active_kv_tiles; ++kv_tile) {
                const int task = first_task + kv_tile;
                const int stage = task % kPipelineStages;
                const int phase = (task / kPipelineStages) & 1;
                const int load_bar = static_cast<int>(
                    __cvta_generic_to_shared(&load_barrier[stage]));
                wait_barrier(load_bar, phase);
                if (task >= kPipelineStages) {
                    const int old_epi = static_cast<int>(
                        __cvta_generic_to_shared(&epilogue_barrier[stage]));
                    wait_barrier(old_epi, ((task - kPipelineStages) / kPipelineStages) & 1);
                }
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const int k_addr = kv_smem_base + stage * kKVStageSmemBytes;
                const int ks_addr = k_addr + kBBytes;
                const int taddr = tmem_base + result_col(stage);
                #pragma unroll
                for (int slab = 0; slab < 2; ++slab) {
                    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                 :: "r"(tmem_base + kScaleTmemCol + 4),
                                    "l"(scale_desc(qs_addr + slab * 512)) : "memory");
                    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                 :: "r"(tmem_base + kScaleTmemCol),
                                    "l"(scale_desc(ks_addr + slab * 512)) : "memory");
                    const uint64_t adesc = fp4_desc(k_addr + slab * 32);
                    const uint64_t bdesc = fp4_desc(q_addr + slab * 32);
                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
                                 "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
                                 "[%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                                 :: "r"(taddr), "l"(adesc), "l"(bdesc), "r"(instruction),
                                    "r"(tmem_base + kScaleTmemCol),
                                    "r"(tmem_base + kScaleTmemCol + 4),
                                    "r"(slab) : "memory");
                }
                const int mma_bar = static_cast<int>(
                    __cvta_generic_to_shared(&mma_barrier[stage]));
                if constexpr (kClusterM == 2) {
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                                 :: "r"(mma_bar), "h"(int16_t{3}) : "memory");
                } else {
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                 :: "r"(mma_bar) : "memory");
                }
            }
            const int last_task = next_task - 1;
            const int last_mma = static_cast<int>(__cvta_generic_to_shared(
                &mma_barrier[last_task % kPipelineStages]));
            wait_barrier(last_mma, (last_task / kPipelineStages) & 1);
            const int q_done_bar = static_cast<int>(
                __cvta_generic_to_shared(&q_done_barrier));
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                         :: "r"(q_done_bar) : "memory");
        }

        if (warp >= 4) {
            const int local_kv = (warp - 4) * 32 + lane;
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
                const int stage = task % kPipelineStages;
                const int phase = (task / kPipelineStages) & 1;
                const int mma_bar = static_cast<int>(
                    __cvta_generic_to_shared(&mma_barrier[stage]));
                wait_barrier(mma_bar, phase);
                #pragma unroll
                for (int local_q = 0; local_q < kTileQ; ++local_q) {
                    __nv_bfloat162 sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                    __nv_bfloat162 sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                    #pragma unroll
                    for (int head_base = 0; head_base < kHeads; head_base += 16) {
                        float value[16];
                        const int address = tmem_base + result_col(stage) +
                            local_q * kHeads + head_base + (tmem_row << 16);
                        load_tmem_x16(value, address);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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
                    const int global_kv = kv_tile * kTileKV + local_kv;
                    if (q_base + local_q < local_query_len &&
                        global_kv < local_kv_len &&
                        (!causal || global_kv <= q_base + local_q))
                        out[static_cast<int64_t>(q_offset + q_base + local_q) * kv_count +
                            global_kv] = __low2bfloat16(reduced);
                }
                if (lane == 0) {
                    const int epi_bar = static_cast<int>(
                        __cvta_generic_to_shared(&epilogue_barrier[stage]));
                    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                                 :: "r"(epi_bar) : "memory");
                }
            }
        }
    }

    if (warp == 2) {
        const int last_task = next_task - 1;
        const int last_epi = static_cast<int>(
            __cvta_generic_to_shared(&epilogue_barrier[last_task % kPipelineStages]));
        wait_barrier(last_epi, (last_task / kPipelineStages) & 1);
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
    CUtensorMap q_map, k_map, qs_map, ks_map;
    encode_fp4(&q_map, q, total_q * kHeads, kTileM);
    encode_fp4(&k_map, k, page_size ? num_pages * page_size : batch_count * kv_count,
               kTileKV);
    encode_q_scale_rows(&qs_map, q_scales,
        batch_count * ((query_count + 1) & ~1));
    encode_scales(&ks_map, k_scales, page_size
        ? num_pages * (page_size / 128)
        : batch_count * ((kv_count + 127) / 128));
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
        q_map, k_map, qs_map, ks_map, head_weight, out,
        query_count, kv_count, batch_count, page_size, block_table_stride,
        block_table, cu_seqlens_q, kv_seqlens, causal);
}
