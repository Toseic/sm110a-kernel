#include <cassert>
#include <cstdint>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeads = 64;
constexpr int kDim = 128;
constexpr int kTileQ = 2;
constexpr int kTileKV = 256;
constexpr int kPipelineStages = 2;
constexpr int kPersistentCtasPerSm = 1;
constexpr int kReductionLanes = 2;
constexpr int kHeadStride = kHeads + kReductionLanes;
constexpr int kTileM = kTileQ * kHeads;
constexpr int kEpilogueLoadWarps = 4;
constexpr int kReductionWarpsPerQuery = 2 * kReductionLanes;
constexpr int kEpilogueWarps = kTileQ * kReductionWarpsPerQuery;
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
constexpr int kReluSmemBytes =
    kTileQ * kTileKV * kHeadStride * sizeof(float);
constexpr int kOutputSmemBytes =
    kPipelineStages * kTileQ * kTileKV * sizeof(float);
constexpr int kSmemBytes =
    kMainloopSmemBytes + kReluSmemBytes + kOutputSmemBytes;

// The two 256-column accumulator stages overlap by 16 columns. The epilogue
// drains that range first and signals the MMA warp before the adjacent stage
// may overwrite it. Q and K scales share one 12-column scratch slot and are
// recopied immediately before each MMA slab, matching the capacity-constrained
// TileN=256 path in the repository's MXFP4 GEMM.
constexpr int kAccumulatorOverlapCols = 16;
constexpr int kAccumulatorStageStride = kTileKV - kAccumulatorOverlapCols;
constexpr int kAccumulatorTmemCols =
    kPipelineStages * kTileKV - kAccumulatorOverlapCols;
constexpr int kScaleTmemCol = kAccumulatorTmemCols;
constexpr int kScaleTmemCols = 4 + kKScaleTiles * 4;
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

void encode_output(CUtensorMap* map, float* ptr, int query_count, int kv_count) {
    const uint64_t dims[2] = {static_cast<uint64_t>(kv_count),
                              static_cast<uint64_t>(query_count)};
    const uint64_t strides[1] = {static_cast<uint64_t>(kv_count) * sizeof(float)};
    const uint32_t box[2] = {kTileKV, 1};
    const uint32_t elem[2] = {1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, ptr, dims, strides,
        box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
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

// Result stages overlap in columns 240..255. Scale scratch occupies 496..507.
__device__ __forceinline__ int result_col(int stage) {
    return stage * kAccumulatorStageStride;
}

__device__ __forceinline__ void load_tmem_x64(float* value, int address) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
        : "=f"(value[0]),
          "=f"(value[1]),
          "=f"(value[2]),
          "=f"(value[3]),
          "=f"(value[4]),
          "=f"(value[5]),
          "=f"(value[6]),
          "=f"(value[7]),
          "=f"(value[8]),
          "=f"(value[9]),
          "=f"(value[10]),
          "=f"(value[11]),
          "=f"(value[12]),
          "=f"(value[13]),
          "=f"(value[14]),
          "=f"(value[15]),
          "=f"(value[16]),
          "=f"(value[17]),
          "=f"(value[18]),
          "=f"(value[19]),
          "=f"(value[20]),
          "=f"(value[21]),
          "=f"(value[22]),
          "=f"(value[23]),
          "=f"(value[24]),
          "=f"(value[25]),
          "=f"(value[26]),
          "=f"(value[27]),
          "=f"(value[28]),
          "=f"(value[29]),
          "=f"(value[30]),
          "=f"(value[31]),
          "=f"(value[32]),
          "=f"(value[33]),
          "=f"(value[34]),
          "=f"(value[35]),
          "=f"(value[36]),
          "=f"(value[37]),
          "=f"(value[38]),
          "=f"(value[39]),
          "=f"(value[40]),
          "=f"(value[41]),
          "=f"(value[42]),
          "=f"(value[43]),
          "=f"(value[44]),
          "=f"(value[45]),
          "=f"(value[46]),
          "=f"(value[47]),
          "=f"(value[48]),
          "=f"(value[49]),
          "=f"(value[50]),
          "=f"(value[51]),
          "=f"(value[52]),
          "=f"(value[53]),
          "=f"(value[54]),
          "=f"(value[55]),
          "=f"(value[56]),
          "=f"(value[57]),
          "=f"(value[58]),
          "=f"(value[59]),
          "=f"(value[60]),
          "=f"(value[61]),
          "=f"(value[62]),
          "=f"(value[63])
        : "r"(address));
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
    const __grid_constant__ CUtensorMap out_map,
    const half* __restrict__ head_weight,
    float* __restrict__ out, int query_count, int kv_count,
    int batch_count, int page_size, int block_table_stride,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ kv_seqlens,
    bool use_tma_store, bool causal) {
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
        if (local_kv_len <= 0) {
            for (int q = blockIdx.x * kTileQ; q < local_query_len;
                 q += gridDim.x * kTileQ)
                for (int kv = tid; kv < kv_count; kv += blockDim.x)
                    out[(static_cast<int64_t>(q_offset + q) * kv_count) + kv] =
                        -__int_as_float(0x7f800000);
        }
        return;
    }

    extern __shared__ __align__(1024) uint8_t smem[];
    const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem));
    const int q_addr = smem_base;
    const int qs_addr = q_addr + kABytes;
    const int kv_smem_base = smem_base + kQSmemBytes;
    float (*relu_values)[kTileKV][kHeadStride] =
        reinterpret_cast<float (*)[kTileKV][kHeadStride]>(
            smem + kMainloopSmemBytes);
    float (*output_tile)[kTileQ][kTileKV] =
        reinterpret_cast<float (*)[kTileQ][kTileKV]>(
            smem + kMainloopSmemBytes + kReluSmemBytes);
    __shared__ uint64_t q_barrier;
    __shared__ uint64_t q_done_barrier;
    __shared__ uint64_t load_barrier[kPipelineStages];
    __shared__ uint64_t mma_barrier[kPipelineStages];
    __shared__ uint64_t overlap_barrier[kPipelineStages];
    __shared__ uint64_t epilogue_barrier[kPipelineStages];
    __shared__ uint64_t store_barrier[kPipelineStages];
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
            const int overlap_addr = static_cast<int>(
                __cvta_generic_to_shared(&overlap_barrier[stage]));
            const int epi_addr = static_cast<int>(__cvta_generic_to_shared(&epilogue_barrier[stage]));
            const int store_addr = static_cast<int>(__cvta_generic_to_shared(&store_barrier[stage]));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(load_addr));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         :: "r"(mma_addr), "r"(kClusterM));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         :: "r"(overlap_addr), "r"(kEpilogueLoadWarps));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         :: "r"(epi_addr), "r"(kEpilogueWarps));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" :: "r"(store_addr));
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
                const int pages_per_tile = page_size ? kTileKV / page_size : 1;
                for (int tile_page = 0; tile_page < pages_per_tile; ++tile_page) {
                    const int logical_row = kv_base + tile_page * (page_size ? page_size : kTileKV);
                    const int logical_page = page_size
                        ? min(logical_row / page_size, (local_kv_len - 1) / page_size) : 0;
                    const int physical_page = page_size
                        ? block_table[batch * block_table_stride + logical_page] : 0;
                    const int k_row = page_size ? physical_page * page_size
                        : batch * kv_count + kv_base;
                    const int dst = k_addr + tile_page * (page_size ? page_size : kTileKV) * (kDim / 2);
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

        // Keep stores off the TMA producer warp so it can maintain the KV
        // load window. Warp 3, otherwise idle, stores the first query row.
        if (warp == 1 || warp == 3) {
            const int store_q = warp == 3 ? 0 : 1;
            for (int kv_tile = 0; kv_tile < active_kv_tiles; ++kv_tile) {
                const int task = first_task + kv_tile;
                const int stage = task % kPipelineStages;
                const int epi_bar = static_cast<int>(
                    __cvta_generic_to_shared(&epilogue_barrier[stage]));
                wait_barrier(epi_bar, (task / kPipelineStages) & 1);
                if (q_base + store_q < local_query_len) {
                    const int kv_base = kv_tile * kTileKV;
                    if (use_tma_store) {
                        if (lane == 0) {
                            const int src = static_cast<int>(
                                __cvta_generic_to_shared(output_tile[stage][store_q]));
                            const uint64_t map = reinterpret_cast<uint64_t>(&out_map);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
                                         "[%0, {%1, %2}], [%3];"
                                         :: "l"(map), "r"(kv_base),
                                            "r"(q_base + store_q),
                                            "r"(src) : "memory");
                            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                        }
                    } else {
                        for (int col = lane; col < kTileKV; col += 32)
                            if (kv_base + col < kv_count)
                                out[static_cast<int64_t>(q_offset + q_base + store_q) * kv_count +
                                    kv_base + col] = output_tile[stage][store_q][col];
                    }
                }
                __syncwarp();
                if (lane == 0) {
                    const int store_bar = static_cast<int>(
                        __cvta_generic_to_shared(&store_barrier[stage]));
                    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                                 :: "r"(store_bar) : "memory");
                }
            }
            if (q_base + store_q < local_query_len) {
                for (int kv = active_kv_tiles * kTileKV + lane;
                     kv < kv_count; kv += 32)
                    out[static_cast<int64_t>(q_offset + q_base + store_q) * kv_count + kv] =
                        -__int_as_float(0x7f800000);
            }
        }

        if (warp == 2 && lane == 0) {
            const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier));
            wait_barrier(q_bar, q_round & 1);
            constexpr uint32_t instruction = 0x08C00480U;
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
                if (task > 0) {
                    const int previous_stage = 1 - stage;
                    const int overlap_addr = static_cast<int>(
                        __cvta_generic_to_shared(&overlap_barrier[previous_stage]));
                    wait_barrier(overlap_addr,
                                 ((task - 1) / kPipelineStages) & 1);
                }
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const int k_addr = kv_smem_base + stage * kKVStageSmemBytes;
                const int ks_addr = k_addr + kBBytes;
                const int taddr = tmem_base + result_col(stage);
                #pragma unroll
                for (int slab = 0; slab < 2; ++slab) {
                    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                 :: "r"(tmem_base + kScaleTmemCol),
                                    "l"(scale_desc(qs_addr + slab * 512)) : "memory");
                    #pragma unroll
                    for (int scale_tile = 0; scale_tile < kKScaleTiles; ++scale_tile)
                        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                     :: "r"(tmem_base + kScaleTmemCol + 4 + scale_tile * 4),
                                        "l"(scale_desc(ks_addr + scale_tile * 1024 +
                                                       slab * 512)) : "memory");
                    const uint64_t adesc = fp4_desc(q_addr + slab * 32);
                    const uint64_t bdesc = fp4_desc(k_addr + slab * 32);
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
            const int head_lane = lane % kReductionLanes;
            const int candidate_lane = lane / kReductionLanes;
            const int reducer_warp = warp - 4;
            const int local_q = reducer_warp / kReductionWarpsPerQuery;
            // Reuse each query's weights across all KV tiles in this CTA.
            float weight[kHeads / kReductionLanes];
            #pragma unroll
            for (int i = 0; i < kHeads / kReductionLanes; ++i) {
                const int head = i * kReductionLanes + head_lane;
                weight[i] = q_base + local_q < local_query_len
                    ? __half2float(head_weight[(q_offset + q_base + local_q) * kHeads + head])
                    : 0.0f;
            }
            for (int kv_tile = 0; kv_tile < active_kv_tiles; ++kv_tile) {
                const int task = first_task + kv_tile;
                const int stage = task % kPipelineStages;
                const int phase = (task / kPipelineStages) & 1;
                const int mma_bar = static_cast<int>(
                    __cvta_generic_to_shared(&mma_barrier[stage]));
                wait_barrier(mma_bar, phase);
                if (warp < 4 + kEpilogueLoadWarps) {
                    const int row = (warp - 4) * 32 + lane;
                    const int overlap_block = stage == 0 ? kTileKV / 64 - 1 : 0;
                    #pragma unroll
                    for (int block_idx = 0; block_idx < kTileKV / 64; ++block_idx) {
                        const int col_block =
                            (overlap_block + block_idx) % (kTileKV / 64);
                        float value[64];
                        const int address = tmem_base + result_col(stage) +
                                            col_block * 64 + ((row & ~31) << 16);
                        load_tmem_x64(value, address);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int i = 0; i < 64; ++i)
                            relu_values[row / kHeads][col_block * 64 + i]
                                       [row % kHeads] = fmaxf(value[i], 0.0f);
                        if (block_idx == 0 && lane == 0) {
                            const int overlap_addr = static_cast<int>(
                                __cvta_generic_to_shared(&overlap_barrier[stage]));
                            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                                         :: "r"(overlap_addr) : "memory");
                        }
                    }
                }
                asm volatile("bar.sync 1, %0;"
                             :: "n"(kEpilogueWarps * 32) : "memory");
                if (task >= kPipelineStages) {
                    const int old_store = static_cast<int>(
                        __cvta_generic_to_shared(&store_barrier[stage]));
                    wait_barrier(old_store, ((task - kPipelineStages) / kPipelineStages) & 1);
                }
                #pragma unroll
                for (int col_block = 0; col_block < kTileKV / 64; ++col_block) {
                    const int local_kv =
                        col_block * 64 +
                        (reducer_warp % kReductionWarpsPerQuery) *
                            (32 / kReductionLanes) + candidate_lane;
                    float sum = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < kHeads / kReductionLanes; ++i) {
                        const int head = i * kReductionLanes + head_lane;
                        sum += weight[i] * relu_values[local_q][local_kv][head];
                    }
                    #pragma unroll
                    for (int offset = kReductionLanes / 2; offset > 0; offset /= 2)
                        sum += __shfl_down_sync(0xffffffffU, sum, offset);
                    if (head_lane == 0)
                        output_tile[stage][local_q][local_kv] =
                            (kv_tile * kTileKV + local_kv >= local_kv_len ||
                             (causal && kv_tile * kTileKV + local_kv > q_base + local_q))
                                ? -__int_as_float(0x7f800000) : sum;
                }
                asm volatile("bar.sync 1, %0;"
                             :: "n"(kEpilogueWarps * 32) : "memory");
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
    const half* head_weight, float* out,
    int query_count, int kv_count, int batch_count, int total_q,
    int num_pages, int page_size, const int32_t* block_table,
    int block_table_stride, const int32_t* cu_seqlens_q,
    const int32_t* kv_seqlens, int cluster_m, bool causal,
    cudaStream_t stream) {
    CUtensorMap q_map, k_map, qs_map, ks_map, out_map{};
    encode_fp4(&q_map, q, total_q * kHeads, kTileM);
    encode_fp4(&k_map, k, page_size ? num_pages * page_size : batch_count * kv_count,
               page_size ? page_size : kTileKV);
    encode_q_scale_rows(&qs_map, q_scales,
        batch_count * ((query_count + 1) & ~1));
    encode_scales(&ks_map, k_scales, page_size
        ? num_pages * (page_size / 128)
        : batch_count * ((kv_count + 127) / 128));
    const bool use_tma_store = batch_count == 1 && !cu_seqlens_q &&
        !kv_seqlens && kv_count % 4 == 0;
    if (use_tma_store)
        encode_output(&out_map, out, query_count, kv_count);
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
        q_map, k_map, qs_map, ks_map, out_map, head_weight, out,
        query_count, kv_count, batch_count, page_size, block_table_stride,
        block_table, cu_seqlens_q, kv_seqlens, use_tma_store, causal);
}
