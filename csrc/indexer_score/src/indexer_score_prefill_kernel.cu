#include <cassert>
#include <cstdint>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeads = 64;
constexpr int kDim = 128;
constexpr int kTileQ = 2;
constexpr int kTileKV = 64;
constexpr int kPipelineStages = 2;
constexpr int kPersistentCtasPerSm = 2;
constexpr int kReductionLanes = 1;
constexpr int kHeadStride = kHeads + kReductionLanes;
constexpr int kTileM = kTileQ * kHeads;
constexpr int kThreads = 8 * 32;
constexpr int kEpilogueWarps = 4;
constexpr int kABytes = kTileM * 128;
constexpr int kBBytes = kTileKV * 128;
constexpr int kScaleBytes = 2 * 512;
constexpr int kQSmemBytes = kABytes + kScaleBytes;
constexpr int kKVStageSmemBytes = kBBytes + kScaleBytes;
constexpr int kSmemBytes = kQSmemBytes + kPipelineStages * kKVStageSmemBytes;

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

// Columns 128..151 hold Q/K scales; result stages occupy 0..127.
__device__ __forceinline__ int result_col(int stage) {
    return stage * kTileKV;
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

__global__ void indexer_score_prefill_kernel(
    const __grid_constant__ CUtensorMap q_map,
    const __grid_constant__ CUtensorMap k_map,
    const __grid_constant__ CUtensorMap qs_map,
    const __grid_constant__ CUtensorMap ks_map,
    const __grid_constant__ CUtensorMap out_map,
    const half* __restrict__ head_weight,
    float* __restrict__ out, int query_count, int kv_count,
    bool use_tma_store) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int q_tiles = (query_count + kTileQ - 1) / kTileQ;
    const int kv_tiles = (kv_count + kTileKV - 1) / kTileKV;

    extern __shared__ __align__(1024) uint8_t smem[];
    const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem));
    const int q_addr = smem_base;
    const int qs_addr = q_addr + kABytes;
    const int kv_smem_base = smem_base + kQSmemBytes;
    __shared__ uint64_t q_barrier;
    __shared__ uint64_t load_barrier[kPipelineStages];
    __shared__ uint64_t mma_barrier[kPipelineStages];
    __shared__ uint64_t epilogue_barrier[kPipelineStages];
    __shared__ uint64_t store_barrier[kPipelineStages];
    __shared__ int tmem_base;
    // Consecutive heads for one candidate occupy separate banks. The extra
    // float shifts adjacent candidates to different bank groups.
    __shared__ float relu_values[kTileQ][kTileKV][kHeadStride];
    __shared__ __align__(128) float output_tile[kPipelineStages][kTileQ][kTileKV];

    if (tid == 0) {
        const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(q_bar));
        #pragma unroll
        for (int stage = 0; stage < kPipelineStages; ++stage) {
            const int load_addr = static_cast<int>(__cvta_generic_to_shared(&load_barrier[stage]));
            const int mma_addr = static_cast<int>(__cvta_generic_to_shared(&mma_barrier[stage]));
            const int epi_addr = static_cast<int>(__cvta_generic_to_shared(&epilogue_barrier[stage]));
            const int store_addr = static_cast<int>(__cvta_generic_to_shared(&store_barrier[stage]));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(load_addr));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mma_addr));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         :: "r"(epi_addr), "r"(kEpilogueWarps));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" :: "r"(store_addr));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 2) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(&tmem_base));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
                     :: "r"(addr));
    }
    __syncthreads();

    // Every CTA owns a static strided subset of query tiles. Q and its scales
    // remain in shared/TMEM while that CTA traverses all KV tiles.
    for (int q_tile = blockIdx.x, q_round = 0; q_tile < q_tiles;
         q_tile += gridDim.x, ++q_round) {
        const int q_base = q_tile * kTileQ;
        const int first_task = q_round * kv_tiles;

        if (warp == 0 && lane == 0) {
            const int zero = 0;
            const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier));
            const uint64_t qptr = reinterpret_cast<uint64_t>(&q_map);
            const uint64_t qsptr = reinterpret_cast<uint64_t>(&qs_map);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                         :: "r"(q_bar), "r"(kTileM * kDim / 2 + kScaleBytes) : "memory");
            asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                         "[%0], [%1, {%2, %3, %4}], [%5];"
                         :: "r"(q_addr), "l"(qptr), "r"(zero), "r"(q_base * kHeads),
                            "r"(zero), "r"(q_bar) : "memory");
            asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                         "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                         :: "r"(qs_addr), "l"(qsptr), "r"(zero), "r"(zero),
                            "r"(zero), "r"(q_tile), "r"(q_bar) : "memory");

            const uint64_t kptr = reinterpret_cast<uint64_t>(&k_map);
            const uint64_t ksptr = reinterpret_cast<uint64_t>(&ks_map);
            auto load_kv = [&](int kv_tile) {
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
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                             :: "r"(load_bar), "r"(kTileKV * kDim / 2 + kScaleBytes) : "memory");
                asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                             "[%0], [%1, {%2, %3, %4}], [%5];"
                             :: "r"(k_addr), "l"(kptr), "r"(zero), "r"(kv_base),
                                "r"(zero), "r"(load_bar) : "memory");
                asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                             "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                             :: "r"(ks_addr), "l"(ksptr), "r"(zero), "r"(zero),
                                "r"(zero), "r"(kv_base / 128), "r"(load_bar) : "memory");
            };
            for (int kv_tile = 0;
                 kv_tile < (kv_tiles < kPipelineStages ? kv_tiles : kPipelineStages);
                 ++kv_tile)
                load_kv(kv_tile);
            for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
                if (kv_tile + kPipelineStages < kv_tiles)
                    load_kv(kv_tile + kPipelineStages);
                const int task = first_task + kv_tile;
                const int stage = task % kPipelineStages;
                const int epi_bar = static_cast<int>(
                    __cvta_generic_to_shared(&epilogue_barrier[stage]));
                wait_barrier(epi_bar, (task / kPipelineStages) & 1);
                const int src = static_cast<int>(
                    __cvta_generic_to_shared(output_tile[stage][0]));
                const int kv_base = kv_tile * kTileKV;
                if (use_tma_store) {
                    const uint64_t map = reinterpret_cast<uint64_t>(&out_map);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
                                 "[%0, {%1, %2}], [%3];"
                                 :: "l"(map), "r"(kv_base), "r"(q_base), "r"(src) : "memory");
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                } else {
                    for (int col = 0; col < kTileKV; ++col)
                        if (kv_base + col < kv_count)
                            out[static_cast<int64_t>(q_base) * kv_count + kv_base + col] =
                                output_tile[stage][0][col];
                }
                const int store_bar = static_cast<int>(
                    __cvta_generic_to_shared(&store_barrier[stage]));
                asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                             :: "r"(store_bar) : "memory");
            }
        }

        if (warp == 1) {
            for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
                const int task = first_task + kv_tile;
                const int stage = task % kPipelineStages;
                const int epi_bar = static_cast<int>(
                    __cvta_generic_to_shared(&epilogue_barrier[stage]));
                wait_barrier(epi_bar, (task / kPipelineStages) & 1);
                if (q_base + 1 < query_count) {
                    const int kv_base = kv_tile * kTileKV;
                    if (use_tma_store) {
                        if (lane == 0) {
                            const int src = static_cast<int>(
                                __cvta_generic_to_shared(output_tile[stage][1]));
                            const uint64_t map = reinterpret_cast<uint64_t>(&out_map);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
                                         "[%0, {%1, %2}], [%3];"
                                         :: "l"(map), "r"(kv_base), "r"(q_base + 1),
                                            "r"(src) : "memory");
                            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                        }
                    } else {
                        for (int col = lane; col < kTileKV; col += 32)
                            if (kv_base + col < kv_count)
                                out[static_cast<int64_t>(q_base + 1) * kv_count +
                                    kv_base + col] = output_tile[stage][1][col];
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
        }

        if (warp == 2 && lane == 0) {
            const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier));
            wait_barrier(q_bar, q_round & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            // Q scales are invariant across all KV tiles of this query tile.
            #pragma unroll
            for (int slab = 0; slab < 2; ++slab)
                asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                             :: "r"(tmem_base + 128 + slab * 4),
                                "l"(scale_desc(qs_addr + slab * 512)) : "memory");

            constexpr uint32_t instruction = 0x08900480U;
            for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
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
                const int kv_base = kv_tile * kTileKV;
                #pragma unroll
                for (int slab = 0; slab < 2; ++slab) {
                    const int ks_taddr = tmem_base + 136 + stage * 8 + slab * 4;
                    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                 :: "r"(ks_taddr),
                                    "l"(scale_desc(ks_addr + slab * 512)) : "memory");
                    const int sfb_taddr = ks_taddr + (kv_base % 128) / 32;
                    const uint64_t adesc = fp4_desc(q_addr + slab * 32);
                    const uint64_t bdesc = fp4_desc(k_addr + slab * 32);
                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
                                 "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
                                 "[%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                                 :: "r"(taddr), "l"(adesc), "l"(bdesc), "r"(instruction),
                                    "r"(tmem_base + 128 + slab * 4), "r"(sfb_taddr),
                                    "r"(slab) : "memory");
                }
                const int mma_bar = static_cast<int>(
                    __cvta_generic_to_shared(&mma_barrier[stage]));
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                             :: "r"(mma_bar) : "memory");
            }
        }

        if (warp >= 4) {
            const int head_lane = lane % kReductionLanes;
            const int local_q = (warp - 4) / 2;
            // Reuse each query's weights across all KV tiles in this CTA.
            float weight[kHeads / kReductionLanes];
            #pragma unroll
            for (int i = 0; i < kHeads / kReductionLanes; ++i) {
                const int head = i * kReductionLanes + head_lane;
                weight[i] = q_base + local_q < query_count
                    ? __half2float(head_weight[(q_base + local_q) * kHeads + head])
                    : 0.0f;
            }
            for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
                const int task = first_task + kv_tile;
                const int stage = task % kPipelineStages;
                const int phase = (task / kPipelineStages) & 1;
                const int mma_bar = static_cast<int>(
                    __cvta_generic_to_shared(&mma_barrier[stage]));
                wait_barrier(mma_bar, phase);
                const int row = (warp - 4) * 32 + lane;
                float value[64];
                const int address = tmem_base + result_col(stage) +
                                    ((row & ~31) << 16);
                load_tmem_x64(value, address);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int i = 0; i < kTileKV; ++i)
                    relu_values[row / kHeads][i][row % kHeads] =
                        fmaxf(value[i], 0.0f);
                asm volatile("bar.sync 1, 128;" ::: "memory");
                if (task >= kPipelineStages) {
                    const int old_store = static_cast<int>(
                        __cvta_generic_to_shared(&store_barrier[stage]));
                    wait_barrier(old_store, ((task - kPipelineStages) / kPipelineStages) & 1);
                }
                const int output_index = (warp - 4) * 32 + lane;
                const int local_kv = output_index % kTileKV;
                float sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < kHeads / kReductionLanes; ++i) {
                    const int head = i * kReductionLanes + head_lane;
                    sum += weight[i] * relu_values[local_q][local_kv][head];
                }
                output_tile[stage][local_q][local_kv] = sum;
                asm volatile("bar.sync 1, 128;" ::: "memory");
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
        const int rounds = (q_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
        const int last_task = rounds * kv_tiles - 1;
        const int last_epi = static_cast<int>(
            __cvta_generic_to_shared(&epilogue_barrier[last_task % kPipelineStages]));
        wait_barrier(last_epi, (last_task / kPipelineStages) & 1);
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;"
                     :: "r"(tmem_base));
    }
}

}  // namespace

extern "C" void sm110a_indexer_score_prefill_launch(
    const uint8_t* q, const uint8_t* k,
    const uint8_t* q_scales, const uint8_t* k_scales,
    const half* head_weight, float* out,
    int query_count, int kv_count, cudaStream_t stream) {
    CUtensorMap q_map, k_map, qs_map, ks_map, out_map{};
    encode_fp4(&q_map, q, query_count * kHeads, kTileM);
    encode_fp4(&k_map, k, kv_count, kTileKV);
    encode_scales(&qs_map, q_scales, (query_count + 1) / 2);
    encode_scales(&ks_map, k_scales, (kv_count + 127) / 128);
    const bool use_tma_store = kv_count % 4 == 0;
    if (use_tma_store)
        encode_output(&out_map, out, query_count, kv_count);
    int device = 0;
    int sm_count = 0;
    assert(cudaGetDevice(&device) == cudaSuccess);
    assert(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                  device) == cudaSuccess);
    const int q_tiles = (query_count + kTileQ - 1) / kTileQ;
    const int max_ctas = sm_count * kPersistentCtasPerSm;
    const dim3 grid(q_tiles < max_ctas ? q_tiles : max_ctas);
    cudaFuncSetAttribute(indexer_score_prefill_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    indexer_score_prefill_kernel<<<grid, kThreads, kSmemBytes, stream>>>(
        q_map, k_map, qs_map, ks_map, out_map, head_weight, out,
        query_count, kv_count, use_tma_store);
}
