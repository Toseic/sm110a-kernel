#include <cassert>
#include <cstdint>
#include <cudaTypedefs.h> // required for CUtensorMap
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bf16_gemm.h"
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
constexpr int kSmCount = 20;

constexpr int kMacroSwizzleM = 8;
constexpr int kMacroSwizzleN = 8;
constexpr bool kMacroSwizzleMFast = true;

template <int kTileM, int kTileN, int kTileK, int kClusterM, int kEpilogueBlockN>
struct KernelConfig {
    static_assert(kClusterM == 1 || kClusterM == 2);
    static_assert(kTileM == 4 * kWarpSize);
    static_assert(
        kEpilogueBlockN == 32 || kEpilogueBlockN == 64 || kEpilogueBlockN == 128);
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
    static constexpr int kStageCount =
        (kMaxSmemBytes - kEpilogueSmemBytes) / kPerStageSmemBytes;
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

__device__ inline void tma_3d_smem2gmem(
    int src,
    const void *tmap_ptr,
    int x,
    int y,
    int z) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
        "[%0, {%1, %2, %3}], [%4];" ::"l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(z),
        "r"(src)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ inline void tma_2d_smem2gmem(
    int src,
    const void *tmap_ptr,
    int x,
    int y) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
        "[%0, {%1, %2}], [%3];" ::"l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(src)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ inline void mbarrier_wait(
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
    uint64_t global_dim[rank] = {
        kSwizzleElements, global_width / kSwizzleElements, global_height};
    uint64_t global_strides[rank - 1] = {
        kSwizzleElements * sizeof(nv_bfloat16), global_width * sizeof(nv_bfloat16)};
    uint32_t box_dim[rank] = {
        kSwizzleElements, shared_width / kSwizzleElements, shared_height};
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
    nv_bfloat16 *epilogue_smem =
        reinterpret_cast<nv_bfloat16 *>(_smem_ptr + kMainloopSmemBytes);

    constexpr int kClusterTileM = kClusterM * kTileM;
    constexpr uint32_t i_desc = (1U << 4U)                           // dtype=FP32
                                | (1U << 7U)                         // atype=BF16
                                | (1U << 10U)                        // btype=BF16
                                | ((uint32_t)kTileN >> 3U << 17U)   // MMA_N
                                | ((uint32_t)kClusterTileM >> 4U << 24U)  // MMA_M
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

    // tma load
    auto load = [&](int iter_k, int mma_phase, int stage_id, int off_m, int off_n) {
        const int mma_mbar_addr = _base_mma_mbar_addr + 8 * stage_id;
        if constexpr (DO_PROFILE) {
            profiler.start(ProfilerTag::WaitMMA);
        }
        mbarrier_wait(mma_mbar_addr, mma_phase);
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
        tma_3d_gmem2smem<kClusterM>(
            A_smem, &A_tmap, 0, off_m, off_k_64, tma_mbar_addr);
        tma_3d_gmem2smem<kClusterM>(B_smem,
            &B_tmap,
            0,
            off_n + cta_rank * (kTileN / kClusterM),
            off_k_64,
            tma_mbar_addr);
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
        mbarrier_wait(tma_mbar_addr, tma_phase);
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
        if constexpr (DO_PROFILE) {
            profiler.stop();
        }
    };

    auto tmem_to_smem = [&](int mainloop_stage, int n_block, int smem_stage) {
        nv_bfloat16 *smem_out =
            epilogue_smem + smem_stage * kTileM * kEpilogueBlockN;

        for (int n = 0; n < kEpilogueBlockN / 8; n++) {
            float tmp[8];
            const int row = cta_rank * 128 + warp_id * 32;
            const int col =
                n_block * kEpilogueBlockN + n * 8 + mainloop_stage * kTileN;
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
                const int swizzled_chunk =
                    (chunk & ~7) + ((chunk & 7) ^ ((chunk >> 3) & 3));
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
                    const int swizzled_chunk =
                        (chunk & ~7) + ((chunk & 7) ^ ((chunk >> 3) & 3));
                    smem_out[swizzled_chunk * 8 + elem % 8] =
                        __float2bfloat16_rn(tmp[i]);
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
                const int mbar_addr =
                    (_base_epilogue_mbar_addr + mainloop_stage * 8) & 0xFEFFFFFF;
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

        for (int n_block = 0; n_block < n_block_num; n_block++) {
            const int smem_stage = n_block % kEpilogueStageCount;
            const int stage_smem =
                smem + kMainloopSmemBytes +
                smem_stage * kTileM * kEpilogueBlockN * static_cast<int>(sizeof(nv_bfloat16));

            // TMA consumes the current stage while the epilogue warps fill the
            // other stage from tensor memory.
            if (tid == 0) {
                if constexpr (!SWAP_AB && kEpilogueBlockN == 32) {
                    tma_2d_smem2gmem(stage_smem,
                        &C_tmap,
                        off_n + n_block * kEpilogueBlockN,
                        off_m);
                } else if constexpr (!SWAP_AB) {
                    tma_3d_smem2gmem(stage_smem,
                        &C_tmap,
                        0,
                        (off_n + n_block * kEpilogueBlockN) / 32,
                        off_m);
                } else {
                    tma_3d_smem2gmem(stage_smem,
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
            mbarrier_wait(_base_epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);
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
            mbarrier_wait(_base_mainloop_mbar_addr + 8 * mainloop_stage, mainloop_phase);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if constexpr (DO_PROFILE)
                if (elect_sync())
                    profiler.stop();
            auto [off_m, off_n] = compute_bid(bid);
            if constexpr (DO_PROFILE)
                if (elect_sync())
                    profiler.start(ProfilerTag::Epilogue);
            epilogue(mainloop_stage, off_m, off_n);
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

template <
    int kTileM,
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

    static_assert(kMacroSwizzleM * kMacroSwizzleN <=
                  (32 * 1024 * 1024) /
                      (static_cast<int>(sizeof(nv_bfloat16)) * kTileM * kTileK +
                       static_cast<int>(sizeof(nv_bfloat16)) * kTileN * kTileK));
    assert(M % (kClusterM * kTileM) == 0);
    assert(N % kTileN == 0);
    assert(K % kTileK == 0);

    init_3D_tmap(
        &A_tmap, A_ptr, M, K, kTileM, kTileK, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_3D_tmap(&B_tmap,
        B_ptr,
        N,
        K,
        kTileN / kClusterM,
        kTileK,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    if constexpr (SWAP_AB) {
        init_3D_output_tmap(
            &C_tmap, C_ptr, logic_M, logic_N, kEpilogueBlockN, kTileM);
    } else if constexpr (kEpilogueBlockN == 32) {
        init_2D_output_tmap(
            &C_tmap, C_ptr, logic_M, logic_N, kTileM, kEpilogueBlockN);
    } else {
        init_3D_output_tmap(
            &C_tmap, C_ptr, logic_M, logic_N, kTileM, kEpilogueBlockN);
    }

    constexpr int mbarrier_smem = 256;
    const int smem_size = kDynamicSmemBytes + mbarrier_smem;

    int grid = kSmCount;

    auto kernel =
        bf16_gemm_kernel<
            kTileM,
            kTileN,
            kTileK,
            kClusterM,
            kEpilogueBlockN,
            DO_PROFILE,
            SWAP_AB>;

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
    cudaStream_t stream) {
    if (swap_ab) {
        launch_bf16_gemm_impl<
            SM110A_BF16_TILE_M,
            SM110A_BF16_TILE_N,
            SM110A_BF16_TILE_K,
            SM110A_BF16_CLUSTER_M,
            SM110A_BF16_EPILOGUE_BLOCK_N,
            false,
            true>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0, stream);
    } else {
        launch_bf16_gemm_impl<
            SM110A_BF16_TILE_M,
            SM110A_BF16_TILE_N,
            SM110A_BF16_TILE_K,
            SM110A_BF16_CLUSTER_M,
            SM110A_BF16_EPILOGUE_BLOCK_N,
            false,
            false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0, stream);
    }
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
    cudaStream_t stream) {
    if (swap_ab) {
        launch_bf16_gemm_impl<
            SM110A_BF16_TILE_M,
            SM110A_BF16_TILE_N,
            SM110A_BF16_TILE_K,
            SM110A_BF16_CLUSTER_M,
            SM110A_BF16_EPILOGUE_BLOCK_N,
            true,
            true>(A_ptr, B_ptr, C_ptr, M, N, K, profiler_ptr, num_entries, stream);
    } else {
        launch_bf16_gemm_impl<
            SM110A_BF16_TILE_M,
            SM110A_BF16_TILE_N,
            SM110A_BF16_TILE_K,
            SM110A_BF16_CLUSTER_M,
            SM110A_BF16_EPILOGUE_BLOCK_N,
            true,
            false>(A_ptr, B_ptr, C_ptr, M, N, K, profiler_ptr, num_entries, stream);
    }
}
