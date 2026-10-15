#include <cassert>
#include <cstdint>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeads = 64;
constexpr int kDim = 128;
constexpr int kTileQ = 1;
constexpr int kTileKV = 128;
constexpr int kTileM = 128;  // 64 query heads and 64 zero-fill columns.
constexpr int kEpilogueWarps = kTileKV / 32;
constexpr int kThreads = (4 + kEpilogueWarps) * 32;
constexpr int kABytes = kTileM * 128;
constexpr int kBBytes = kTileKV * 128;
constexpr int kQScaleBytes = 2 * 512;
constexpr int kKScaleTiles = kTileKV / 128;
constexpr int kKScaleBytes = kKScaleTiles * 2 * 512;
constexpr int kHeadWeightBytes = kHeads * sizeof(__nv_bfloat16);
constexpr int kQSmemBytes = kABytes + kQScaleBytes;
constexpr int kKVStageSmemBytes = kBBytes + kKScaleBytes;
// KV supplies the MMA rows and Q/head supplies the columns. The pipeline
// result slots have disjoint accumulator columns and share eight scale columns.
constexpr int kScaleTmemCols = 8;
constexpr int kTmemCols = 512;

// Q and K are row-major packed E2M1; their scales use the same 128x4
// blocked UE8M0 layout as the repository's MXFP4 GEMM.
void encode_fp4(CUtensorMap* map, const uint8_t* ptr, int rows,
                int tile_rows) {
    const uint64_t dims[3] = {kDim, static_cast<uint64_t>(rows), 1};
    const uint64_t strides[2] = {kDim / 2,
                                  static_cast<uint64_t>(rows) * (kDim / 2)};
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

// Each result slot occupies 128 columns; scales follow all result slots.
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

template <int Offset>
__device__ __forceinline__ void load_tmem_x16_uniform(float* value, int address) {
    // Every epilogue lane uses the same TMEM row/column base. Keep the
    // warp-uniform reduction and load in one PTX block for uniform lowering.
    asm volatile(
        "{ .reg .u32 a; redux.sync.min.u32 a, %16, 0xffffffff; "
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [a+%17]; }"
        : "=f"(value[0]), "=f"(value[1]), "=f"(value[2]), "=f"(value[3]),
          "=f"(value[4]), "=f"(value[5]), "=f"(value[6]), "=f"(value[7]),
          "=f"(value[8]), "=f"(value[9]), "=f"(value[10]), "=f"(value[11]),
          "=f"(value[12]), "=f"(value[13]), "=f"(value[14]), "=f"(value[15])
        : "r"(address), "n"(Offset));
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

__device__ __forceinline__ bool elect_one() {
    uint32_t elected;
    asm volatile("{ .reg .pred p; elect.sync _|p, 0xffffffff; "
                 "selp.u32 %0, 1, 0, p; }" : "=r"(elected));
    return elected != 0;
}

template <int SplitKV, bool kDense = false, int kPageSize = 0,
          int kKVStages = 2, int kResultStages = 2>
__global__ void indexer_score_decode_kernel(
    const __grid_constant__ CUtensorMap q_map,
    const __grid_constant__ CUtensorMap k_map,
    const __grid_constant__ CUtensorMap qs_map,
    const __grid_constant__ CUtensorMap ks_map,
    const __nv_bfloat16* __restrict__ head_weight,
    __nv_bfloat16* __restrict__ out, int kv_count, int page_size,
    const int32_t* __restrict__ block_table, int block_table_stride,
    const int32_t* __restrict__ kv_seqlens) {
    constexpr int kPipelineStages = kKVStages;
    constexpr int kMainloopSmemBytes = kQSmemBytes + kKVStages * kKVStageSmemBytes;
    constexpr int kScaleTmemCol = kResultStages * kTileM;
    static_assert(kKVStages > 0 && kResultStages > 0);
    static_assert(kScaleTmemCol + kScaleTmemCols <= kTmemCols);
    // Specialize address generation, but keep per-batch KV lengths dynamic.
    if constexpr (kDense) {
        page_size = 0;
        block_table = nullptr;
    }
    static_assert(kDense || kPageSize == 128 || kPageSize == 256);
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int batch = blockIdx.y;
    const int split = blockIdx.x;
    const int local_kv_len = kv_seqlens
        ? max(0, min(kv_count, kv_seqlens[batch])) : kv_count;
    const int total_kv_tiles = (kv_count + kTileKV - 1) / kTileKV;
    const int kv_tile_start = total_kv_tiles * split / SplitKV;
    const int split_tile_end = total_kv_tiles * (split + 1) / SplitKV;
    const int kv_tile_end = min(split_tile_end,
                                (local_kv_len + kTileKV - 1) / kTileKV);
    const int kv_tiles = kv_tile_end - kv_tile_start;

    // Cover inactive splits and the invalid tail of the final active tile.
    if (kv_seqlens) {
        const int invalid_start = max(local_kv_len, kv_tile_start * kTileKV);
        const int invalid_end = min(kv_count, split_tile_end * kTileKV);
        for (int col = invalid_start + tid; col < invalid_end; col += blockDim.x)
            out[static_cast<int64_t>(batch) * kv_count + col] = __float2bfloat16_rn(-INFINITY);
    }
    if (kv_tiles <= 0) return;

    extern __shared__ __align__(1024) uint8_t smem[];
    const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem));
    const int q_addr = smem_base;
    const int qs_addr = q_addr + kABytes;
    const int kv_smem_base = smem_base + kQSmemBytes;
    auto* shared_weights = reinterpret_cast<__nv_bfloat162*>(
        smem + kMainloopSmemBytes);
    __shared__ uint64_t q_barrier;
    __shared__ uint64_t load_barrier[kPipelineStages];
    // KV reuse waits for MMA consumption, while result reuse waits for the
    // epilogue. Independent rings let TMA prefetch past occupied result slots.
    __shared__ uint64_t mma_barrier[kPipelineStages];
    __shared__ uint64_t epilogue_barrier[kResultStages];
    __shared__ uint64_t result_barrier[kKVStages != kResultStages ? kResultStages : 1];
    __shared__ int tmem_base;
    if (tid == 0) {
        const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier));
        // Q TMA and the weight producer each arrive once.
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" :: "r"(q_bar));
        #pragma unroll
        for (int stage = 0; stage < kPipelineStages; ++stage) {
            const int load_addr = static_cast<int>(__cvta_generic_to_shared(&load_barrier[stage]));
            const int mma_addr = static_cast<int>(__cvta_generic_to_shared(&mma_barrier[stage]));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(load_addr));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mma_addr));
            if constexpr (kKVStages == kResultStages) {
                const int epi_addr = static_cast<int>(__cvta_generic_to_shared(&epilogue_barrier[stage]));
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                             :: "r"(epi_addr), "r"(kEpilogueWarps));
            }
        }
        if constexpr (kKVStages != kResultStages) {
            #pragma unroll
            for (int stage = 0; stage < kResultStages; ++stage) {
                const int epi_addr = static_cast<int>(__cvta_generic_to_shared(&epilogue_barrier[stage]));
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                             :: "r"(epi_addr), "r"(kEpilogueWarps));
                const int result_addr = static_cast<int>(__cvta_generic_to_shared(&result_barrier[stage]));
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(result_addr));
            }
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 2) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(&tmem_base));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(addr), "r"(kTmemCols));
    }
    __syncthreads();

    if (warp == 1) {
        const auto* global_weights = reinterpret_cast<const __nv_bfloat162*>(
            head_weight + batch * kHeads);
        shared_weights[lane] = global_weights[lane];
        __syncwarp();
        if (lane == 0) {
            const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier));
            asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                         :: "r"(q_bar) : "memory");
        }
    }

    // One cluster per batch; each CTA owns a disjoint contiguous KV slice.
    // A 128-column MMA tile holds the 64 valid query heads and 64 zero-fill columns.
    {
        constexpr int first_task = 0;

        if (warp == 0 && lane == 0) {
            const int zero = 0;
            const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier));
            const uint64_t qptr = reinterpret_cast<uint64_t>(&q_map);
            const uint64_t qsptr = reinterpret_cast<uint64_t>(&qs_map);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                         :: "r"(q_bar), "r"(kTileM * kDim / 2 + kQScaleBytes) : "memory");
            asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                         "[%0], [%1, {%2, %3, %4}], [%5];"
                         :: "r"(q_addr), "l"(qptr), "r"(zero), "r"(batch * kHeads),
                            "r"(zero), "r"(q_bar) : "memory");
            asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                         "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                         :: "r"(qs_addr), "l"(qsptr), "r"(zero), "r"(zero),
                            "r"(zero), "r"(batch), "r"(q_bar) : "memory");

            const uint64_t kptr = reinterpret_cast<uint64_t>(&k_map);
            const uint64_t ksptr = reinterpret_cast<uint64_t>(&ks_map);
            auto load_kv = [&](int kv_tile) {
                const int task = first_task + kv_tile;
                const int stage = task % kPipelineStages;
                const int kv_base = (kv_tile_start + kv_tile) * kTileKV;
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
                             :: "r"(load_bar), "r"(kTileKV * kDim / 2 + kKScaleBytes) : "memory");
                if constexpr (!kDense) {
                    static_assert(kTileKV == 128 && kKScaleTiles == 1);
                    // Active tiles start before local_kv_len and fit in one
                    // aligned page. K and its scales share the same lookup.
                    const int physical_page = block_table[
                        batch * block_table_stride + kv_base / kPageSize];
                    const int k_row = physical_page * kPageSize + kv_base % kPageSize;
                    const int scale_row = k_row / 128;
                    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                                 "[%0], [%1, {%2, %3, %4}], [%5];"
                                 :: "r"(k_addr), "l"(kptr), "r"(zero), "r"(k_row),
                                    "r"(zero), "r"(load_bar) : "memory");
                    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                                 "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                                 :: "r"(ks_addr), "l"(ksptr), "r"(zero), "r"(zero),
                                    "r"(zero), "r"(scale_row), "r"(load_bar) : "memory");
                } else {
                    const int k_row = batch * kv_count + kv_base;
                    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                                 "[%0], [%1, {%2, %3, %4}], [%5];"
                                 :: "r"(k_addr), "l"(kptr), "r"(zero), "r"(k_row),
                                    "r"(zero), "r"(load_bar) : "memory");
                    #pragma unroll
                    for (int scale_tile = 0; scale_tile < kKScaleTiles; ++scale_tile) {
                        const int scale_row = batch * ((kv_count + 127) / 128) +
                                              kv_base / 128 + scale_tile;
                        asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
                                     "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                                     :: "r"(ks_addr + scale_tile * 1024), "l"(ksptr),
                                        "r"(zero), "r"(zero), "r"(zero),
                                        "r"(scale_row), "r"(load_bar) : "memory");
                    }
                }
            };
            for (int kv_tile = 0;
                 kv_tile < (kv_tiles < kPipelineStages ? kv_tiles : kPipelineStages);
                 ++kv_tile)
                load_kv(kv_tile);
            for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
                if (kv_tile + kPipelineStages < kv_tiles)
                    load_kv(kv_tile + kPipelineStages);
            }
            const int last_task = first_task + kv_tiles - 1;
            const int last_stage = last_task % kPipelineStages;
            const int last_mma = static_cast<int>(
                __cvta_generic_to_shared(&mma_barrier[last_stage]));
            wait_barrier(last_mma, (last_task / kPipelineStages) & 1);
        }

        // Elect for paged KV too, so ptxas can reuse one issuer across the
        // tcgen05 instructions instead of inserting repeated election loops.
        if (warp == 2 && elect_one()) {
            const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier));
            wait_barrier(q_bar, 0);
            constexpr uint32_t instruction = 0x08A00480U;
            for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
                const int task = first_task + kv_tile;
                const int stage = task % kPipelineStages;
                const int phase = (task / kPipelineStages) & 1;
                const int load_bar = static_cast<int>(
                    __cvta_generic_to_shared(&load_barrier[stage]));
                wait_barrier(load_bar, phase);
                const int result_stage = task % kResultStages;
                if (task >= kResultStages) {
                    const int old_epi = static_cast<int>(
                        __cvta_generic_to_shared(&epilogue_barrier[result_stage]));
                    wait_barrier(old_epi, ((task - kResultStages) / kResultStages) & 1);
                }
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const int k_addr = kv_smem_base + stage * kKVStageSmemBytes;
                const int ks_addr = k_addr + kBBytes;
                const int taddr = tmem_base + result_col(result_stage);
                #pragma unroll
                for (int slab = 0; slab < 2; ++slab) {
                    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                 :: "r"(tmem_base + kScaleTmemCol),
                                    "l"(scale_desc(ks_addr + slab * 512)) : "memory");
                    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                 :: "r"(tmem_base + kScaleTmemCol + 4),
                                    "l"(scale_desc(qs_addr + slab * 512)) : "memory");
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
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                             :: "r"(mma_bar) : "memory");
                if constexpr (kKVStages != kResultStages) {
                    // Signal result readiness separately from KV-buffer reuse.
                    const int result_bar = static_cast<int>(
                        __cvta_generic_to_shared(&result_barrier[result_stage]));
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                 :: "r"(result_bar) : "memory");
                }
            }
        }

        if (warp >= 4) {
            const int local_kv = (warp - 4) * 32 + lane;
            const int tmem_row = (warp - 4) * 32;
            const int q_bar = static_cast<int>(__cvta_generic_to_shared(&q_barrier));
            wait_barrier(q_bar, 0);
            // Warp 1 loaded the weights once; all epilogue lanes reuse them.
            __nv_bfloat162 weights[kHeads / 2];
            #pragma unroll
            for (int i = 0; i < kHeads / 2; ++i)
                weights[i] = shared_weights[i];
            // Allocation is complete and its base is immutable until dealloc.
            // Cache it across paged TMEM loads despite inline-PTX memory clobbers.
            const int paged_tmem_base = kDense ? 0 : tmem_base;
            for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
                const int task = first_task + kv_tile;
                const int stage = task % kResultStages;
                const int phase = (task / kResultStages) & 1;
                const int mma_bar = static_cast<int>(
                    __cvta_generic_to_shared(kKVStages == kResultStages
                        ? &mma_barrier[stage] : &result_barrier[stage]));
                wait_barrier(mma_bar, phase);
                __nv_bfloat162 sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                __nv_bfloat162 sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                #pragma unroll
                for (int head_base = 0; head_base < kHeads; head_base += 16) {
                    float value[16];
                    const int address = (kDense ? tmem_base : paged_tmem_base) + result_col(stage) +
                        head_base + (tmem_row << 16);
                    if constexpr (kDense)
                        load_tmem_x16(value, address);
                    else {
                        // The loop is unrolled, but use explicit compile-time
                        // offsets to keep the base uniform through lowering.
                        if (head_base == 0) load_tmem_x16_uniform<0>(value, address);
                        else if (head_base == 16) load_tmem_x16_uniform<16>(value, address - 16);
                        else if (head_base == 32) load_tmem_x16_uniform<32>(value, address - 32);
                        else load_tmem_x16_uniform<48>(value, address - 48);
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int i = 0; i < 16; i += 4) {
                        sum_0 = __hfma2(relu_bf16x2(value[i], value[i + 1]),
                                        weights[(head_base + i) / 2], sum_0);
                        sum_1 = __hfma2(relu_bf16x2(value[i + 2], value[i + 3]),
                                        weights[(head_base + i) / 2 + 1], sum_1);
                    }
                }
                const __nv_bfloat162 sum = __hadd2_rn(sum_0, sum_1);
                const __nv_bfloat162 reduced = __hadd2_rn(
                    __low2bfloat162(sum), __high2bfloat162(sum));
                const int global_kv = (kv_tile_start + kv_tile) * kTileKV + local_kv;
                if (global_kv < local_kv_len)
                    out[static_cast<int64_t>(batch) * kv_count + global_kv] =
                        __low2bfloat16(reduced);
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
        const int last_task = kv_tiles - 1;
        const int last_epi = static_cast<int>(
            __cvta_generic_to_shared(&epilogue_barrier[last_task % kResultStages]));
        wait_barrier(last_epi, (last_task / kResultStages) & 1);
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tmem_base), "r"(kTmemCols));
    }
}

}  // namespace

// Measured KV prefetch depths are 8 for dense and 4 for paged; result slots
// remain independently double-buffered in both paths.
template <int SplitKV, bool kDense = false, int kPageSize = 0,
          int kKVStages = kDense ? 8 : 4, int kResultStages = 2>
void launch_decode(const CUtensorMap& q_map, const CUtensorMap& k_map,
                   const CUtensorMap& qs_map, const CUtensorMap& ks_map,
                   const __nv_bfloat16* head_weight,
                   __nv_bfloat16* out, int batch_count, int kv_count, int page_size,
                   const int32_t* block_table, int block_table_stride,
                   const int32_t* kv_seqlens, cudaStream_t stream) {
    constexpr int kSmemBytes = kQSmemBytes + kKVStages * kKVStageSmemBytes + kHeadWeightBytes;
    cudaFuncSetAttribute(indexer_score_decode_kernel<SplitKV, kDense, kPageSize, kKVStages, kResultStages>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(SplitKV, batch_count);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = kSmemBytes;
    config.stream = stream;
    cudaLaunchAttribute cluster_attr{};
    cluster_attr.id = cudaLaunchAttributeClusterDimension;
    cluster_attr.val.clusterDim.x = SplitKV;
    cluster_attr.val.clusterDim.y = 1;
    cluster_attr.val.clusterDim.z = 1;
    config.attrs = &cluster_attr;
    config.numAttrs = 1;
    cudaLaunchKernelEx(&config, indexer_score_decode_kernel<SplitKV, kDense, kPageSize, kKVStages, kResultStages>,
                       q_map, k_map, qs_map, ks_map,
                       head_weight, out, kv_count, page_size, block_table,
                       block_table_stride, kv_seqlens);
}

extern "C" void sm110a_indexer_score_decode_launch(
    const uint8_t* q, const uint8_t* k,
    const uint8_t* q_scales, const uint8_t* k_scales,
    const __nv_bfloat16* head_weight, __nv_bfloat16* out,
    int batch_count, int kv_count, int num_pages, int page_size,
    const int32_t* block_table, int block_table_stride,
    const int32_t* kv_seqlens, int split_kv, cudaStream_t stream) {
    CUtensorMap q_map, k_map, qs_map, ks_map;
    encode_fp4(&q_map, q, batch_count * kHeads, kTileM);
    encode_fp4(&k_map, k, page_size ? num_pages * page_size : batch_count * kv_count,
               kTileKV);
    encode_scales(&qs_map, q_scales, batch_count);
    encode_scales(&ks_map, k_scales, page_size
        ? num_pages * (page_size / 128) : batch_count * ((kv_count + 127) / 128));
#define LAUNCH(S, DENSE, PAGE) launch_decode<S, DENSE, PAGE>(q_map, k_map, qs_map, ks_map, \
                                       head_weight, out, batch_count, kv_count, \
                                       page_size, block_table, block_table_stride, \
                                       kv_seqlens, stream)
#define DISPATCH(DENSE, PAGE) \
    switch (split_kv) { \
        case 1: LAUNCH(1, DENSE, PAGE); break; \
        case 2: LAUNCH(2, DENSE, PAGE); break; \
        case 4: LAUNCH(4, DENSE, PAGE); break; \
        case 8: LAUNCH(8, DENSE, PAGE); break; \
        default: assert(false && "split_kv must be 1, 2, 4, or 8"); \
    }
    if (page_size == 0) {
        DISPATCH(true, 0);
    } else if (page_size == 128) {
        DISPATCH(false, 128);
    } else {
        assert(page_size == 256);
        DISPATCH(false, 256);
    }
#undef DISPATCH
#undef LAUNCH
}
