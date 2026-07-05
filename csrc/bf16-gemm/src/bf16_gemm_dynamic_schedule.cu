#include <cassert>
#include <cstdint>
#include <cudaTypedefs.h> // required for CUtensorMap
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bf16_gemm.h"
#include "profiler.h"

// TODO: 2026-07-05 will add clc dynamic schedule logic into this


namespace {

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 128;
constexpr int WARP_SIZE = 32;
constexpr int WARP_NUM = 6;
constexpr int STAGE_NUM = 3;
constexpr int MMA_K = 16;
constexpr int EPILOGUE_BLOCK_N = 32;
constexpr int EPILOGUE_STAGE_NUM = 2;
constexpr int CLUSTER_M = 2;
constexpr bool PREFETCH_TMA_DESC = true;
constexpr int TB_SIZE = WARP_SIZE * WARP_NUM;

constexpr int _dtype_bytes = 2;
constexpr int _L2_per_tilem = _dtype_bytes * BLOCK_M * BLOCK_K;
constexpr int _L2_per_tilen = _dtype_bytes * BLOCK_N * BLOCK_K;
constexpr int _thor_L2_size = 32 * 1024 * 1024;
constexpr int _MACRO_SWIZZLE_SUM = _thor_L2_size / (_L2_per_tilem + _L2_per_tilen);

constexpr int MACRO_SWIZZLE_M = 8;
constexpr int MACRO_SWIZZLE_N = 8;
constexpr int MACRO_SWIZZLE_SIZE = MACRO_SWIZZLE_M * MACRO_SWIZZLE_N;
constexpr bool MACRO_SWIZZLE_M_FAST = true;

template <int CTA_GROUP = 1>
__device__ inline void tma_3d_gmem2smem(
    int dst,
    const void *tmap_ptr,
    int x,
    int y,
    int z,
    int _base_tma_mbar_addr) {
    // when CTA_GROUP=1, we can use .shared::cta instead.
    // but .shared::cluster doesn't seem to be slower, so always use it unconditionally here.
    // .cta_group::2 allows _base_tma_mbar_addr and dst to be in different CTA's smem.
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

// https://github.com/NVIDIA/cutlass/blob/v4.3.1/include/cute/arch/mma_sm100_umma.hpp#L86
template <int CTA_GROUP = 1>
__device__ inline void tcgen05_mma_f16(
    int taddr,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int enable_input_d) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t" // predicate register enable-input-d
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t"
                 "}" ::"r"(taddr),
        "l"(a_desc),
        "l"(b_desc),
        "r"(i_desc),
        "r"(enable_input_d),
        "n"(CTA_GROUP));
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/cluster_sm90.hpp#L180
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

inline void init_2D_tmap(
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
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template <bool DO_PROFILE>
__global__
__cluster_dims__(CLUSTER_M, 1, 1) 
__launch_bounds__(TB_SIZE) 
void bf16_gemm_dynamic_schedule_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    int M,
    int N,
    int K,
    int64_t *profiler_ptr,
    int num_entries) {

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int cluster_bid = blockIdx.x / CLUSTER_M;

    const int grid_m = M / (BLOCK_M * CLUSTER_M);
    const int grid_n = N / BLOCK_N;

    Profiler profiler;
    if constexpr (DO_PROFILE) {
        if (elect_sync()) {
            profiler.init(num_entries, profiler_ptr, cluster_bid * WARP_NUM + warp_id);
            profiler.start(ProfilerTag::Setup);
        }
    }

    // set up cta rank
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // set up smem
    // A: [BLOCK_M, BLOCK_K]  B: [BLOCK_N, BLOCK_K]
    extern __shared__ __align__(1024) char _smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(_smem_ptr));
    constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
    constexpr int B_size = (BLOCK_N / CLUSTER_M) * BLOCK_K * sizeof(nv_bfloat16);
    constexpr int mainloop_smem_size = STAGE_NUM * (A_size + B_size);
    nv_bfloat16 *epilogue_smem =
        reinterpret_cast<nv_bfloat16 *>(_smem_ptr + mainloop_smem_size);

    constexpr int MMA_M = BLOCK_M * CLUSTER_M;
    constexpr uint32_t i_desc = (1U << 4U)                         // dtype=FP32
                                | (1U << 7U)                       // atype=BF16
                                | (1U << 10U)                      // btype=BF16
                                | ((uint32_t)BLOCK_N >> 3U << 17U) // MMA_N
                                | ((uint32_t)MMA_M >> 4U << 24U)   // MMA_M
        ;

    // setup Mbarrier
    __shared__ int tmem_addr[1];
    __shared__ uint64_t tma_mbars[STAGE_NUM];
    __shared__ uint64_t mma_mbars[STAGE_NUM];
    __shared__ uint64_t mainlooop_mbars[2];
    __shared__ uint64_t epilogue_mbars[2];
    const int _base_tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(tma_mbars));
    const int _base_mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mma_mbars));
    const int _base_mainloop_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(mainlooop_mbars));
    const int _base_epilogue_mbar_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_mbars));

    // init mbarrier
    if (tid == 0) {
        for (int s = 0; s < STAGE_NUM; ++s) {
            mbarrier_init(_base_tma_mbar_addr + 8 * s, CLUSTER_M);
            mbarrier_init(_base_mma_mbar_addr + 8 * s, 1);
        }
        for (int s = 0; s < 2; ++s) {
            mbarrier_init(_base_mainloop_mbar_addr + 8 * s, 1);
            mbarrier_init(_base_epilogue_mbar_addr + 8 * s, 4 * CLUSTER_M);
        }

        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    // alloc tmem
    if (warp_id == 1) {
        static_assert(BLOCK_N * 2 <= 512);
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile(
            "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(addr),
            "r"(BLOCK_N * 2),
            "n"(CLUSTER_M));
    }
    // prefetch tma desc
    if constexpr (PREFETCH_TMA_DESC) {
        if (warp_id == 0 && elect_sync()) {
            prefetch_tensormap(&A_tmap);
            prefetch_tensormap(&B_tmap);
            prefetch_tensormap(&C_tmap);
        }
    }

    if constexpr (CLUSTER_M > 1) {
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
        // const int stage_id = iter_k % STAGE_NUM;
        const int mma_mbar_addr = _base_mma_mbar_addr + 8 * stage_id;
        if constexpr (DO_PROFILE) {
            profiler.start(ProfilerTag::WaitMMA);
        }
        mbarrier_wait(mma_mbar_addr, mma_phase);
        if constexpr (DO_PROFILE) {
            profiler.stop();
        }

        const int A_smem = smem + stage_id * (A_size + B_size);
        const int B_smem = A_smem + A_size;
        const int tma_mbar_addr = (_base_tma_mbar_addr + 8 * stage_id) & 0xFEFFFFFF;

        // tma issue
        constexpr int cp_size = A_size + B_size;

        if constexpr (DO_PROFILE) {
            profiler.start(ProfilerTag::IssueTMA);
        }
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::"r"(
                         tma_mbar_addr),
            "r"(cp_size)
            : "memory");
        const int off_k_64 = iter_k * (BLOCK_K / 64);
        tma_3d_gmem2smem<CLUSTER_M>(
            A_smem, &A_tmap, 0, off_m, off_k_64, tma_mbar_addr);
        tma_3d_gmem2smem<CLUSTER_M>(B_smem,
            &B_tmap,
            0,
            off_n + cta_rank * (BLOCK_N / CLUSTER_M),
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

        const int A_smem = smem + stage_id * (A_size + B_size);
        const int B_smem = A_smem + A_size;
        const int mma_mbar_addr = _base_mma_mbar_addr + 8 * stage_id;

        auto make_desc = [](int addr) -> uint64_t {
            constexpr int SBO = 8 * 128;
            return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) |
                   (2ULL << 61ULL);
        };

        // https://github.com/deepseek-ai/DeepGEMM/blob/9b680f42/deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh#L289
        asm volatile("tcgen05.fence::after_thread_sync;");
        const int this_taddr = taddr + mainloop_stage * BLOCK_N;

        if constexpr (DO_PROFILE) {
            profiler.start(ProfilerTag::IssueMMA);
        }
        tcgen05_mma_f16<CLUSTER_M>(this_taddr,
            make_desc(A_smem),
            make_desc(B_smem),
            i_desc,
            iter_k);
        for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
            uint64_t a_desc = make_desc(A_smem + k2 * 32);
            uint64_t b_desc = make_desc(B_smem + k2 * 32);
            tcgen05_mma_f16<CLUSTER_M>(this_taddr, a_desc, b_desc, i_desc, 1);
        }
        for (int k1 = 1; k1 < BLOCK_K / 64; k1++) {
            for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
                uint64_t a_desc = make_desc(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
                uint64_t b_desc = make_desc(B_smem + k1 * (BLOCK_N / CLUSTER_M) * 128 + k2 * 32);
                tcgen05_mma_f16<CLUSTER_M>(this_taddr, a_desc, b_desc, i_desc, 1);
            }
        }

        constexpr int16_t cta_mask = (1 << CLUSTER_M) - 1;

        asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast:"
                     ":cluster.b64 [%0], %1;" ::"r"(mma_mbar_addr),
            "h"(cta_mask),
            "n"(CLUSTER_M)
            : "memory");
        if constexpr (DO_PROFILE) {
            profiler.stop();
        }
    };

    auto tmem_to_smem = [&](int mainloop_stage, int n_block, int smem_stage) {
        nv_bfloat16 *smem_out =
            epilogue_smem + smem_stage * BLOCK_M * EPILOGUE_BLOCK_N;

        for (int n = 0; n < EPILOGUE_BLOCK_N / 8; n++) {
            float tmp[8];
            const int row = cta_rank * 128 + warp_id * 32;
            const int col =
                n_block * EPILOGUE_BLOCK_N + n * 8 + mainloop_stage * BLOCK_N;
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

            nv_bfloat162 out[4];
            for (int i = 0; i < 4; i++) {
                out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});
            }
            // The 64B swizzle permutes each consecutive group of eight 16B
            // chunks.  Since a logical output row is 64B, one swizzle group
            // spans two adjacent rows.  Both stage bases are 512B-aligned.
            const int chunk = tid * (EPILOGUE_BLOCK_N / 8) + n;
            const int swizzled_chunk =
                (chunk & ~7) + ((chunk & 7) ^ ((chunk >> 3) & 3));
            reinterpret_cast<int4 *>(smem_out + swizzled_chunk * 8)[0] =
                reinterpret_cast<int4 *>(out)[0];
        }
    };

    auto epilogue = [&](int mainloop_stage, int off_m, int off_n) {
        constexpr int n_block_num = BLOCK_N / EPILOGUE_BLOCK_N;
        static_assert(BLOCK_N % EPILOGUE_BLOCK_N == 0);
        static_assert(EPILOGUE_BLOCK_N % 8 == 0);

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
            const int smem_stage = n_block % EPILOGUE_STAGE_NUM;
            const int stage_smem =
                smem + mainloop_smem_size +
                smem_stage * BLOCK_M * EPILOGUE_BLOCK_N * sizeof(nv_bfloat16);

            // TMA consumes the current stage while the epilogue warps fill the
            // other stage from tensor memory.
            if (tid == 0) {
                tma_2d_smem2gmem(
                    stage_smem, &C_tmap, off_n + n_block * EPILOGUE_BLOCK_N, off_m);
            }

            if (n_block + 1 < n_block_num) {
                // Once both stages are in flight, wait until the oldest TMA
                // store has finished reading before reusing its smem stage.
                if (n_block > 0) {
                    if (tid == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                }

                const int next_stage = (n_block + 1) % EPILOGUE_STAGE_NUM;
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
        constexpr int CLUSTER_BLOCK_M = BLOCK_M * CLUSTER_M;
        constexpr int MACRO_M_SIZE = CLUSTER_BLOCK_M * MACRO_SWIZZLE_M;
        constexpr int MACRO_N_SIZE = BLOCK_N * MACRO_SWIZZLE_N;

        auto _compute_bid_in_macro = [&](int macro_block_m_idx,
                                         int macro_block_n_idx,
                                         int macro_lane) -> std::tuple<int, int> {
            const int base_m = macro_block_m_idx * MACRO_M_SIZE;
            const int base_n = macro_block_n_idx * MACRO_N_SIZE;

            const int _real_macro_n = min(MACRO_N_SIZE, N - base_n) / BLOCK_N;

            const int macro_bid_m = macro_lane / _real_macro_n;
            const int macro_bid_n = macro_lane % _real_macro_n;

            const int off_m = base_m + macro_bid_m * CLUSTER_BLOCK_M;
            const int off_n = base_n + macro_bid_n * BLOCK_N;

            return {off_m, off_n};
        };

        if constexpr (MACRO_SWIZZLE_M_FAST) {
            const int full_n_band_size = grid_m * MACRO_SWIZZLE_N;

            const int macro_block_n_idx = bid / full_n_band_size;
            const int bid_in_n_band = bid - macro_block_n_idx * full_n_band_size;

            const int real_macro_n =
                min(MACRO_SWIZZLE_N, grid_n - macro_block_n_idx * MACRO_SWIZZLE_N);

            const int macro_m_block_size = MACRO_SWIZZLE_M * real_macro_n;

            const int macro_block_m_idx = bid_in_n_band / macro_m_block_size;
            const int macro_lane = bid_in_n_band % macro_m_block_size;

            return _compute_bid_in_macro(macro_block_m_idx, macro_block_n_idx, macro_lane);
        } else {
            const int full_m_band_size = grid_n * MACRO_SWIZZLE_M;

            const int macro_block_m_idx = bid / full_m_band_size;
            const int bid_in_m_band = bid - macro_block_m_idx * full_m_band_size;

            const int real_macro_m =
                min(MACRO_SWIZZLE_M, grid_m - macro_block_m_idx * MACRO_SWIZZLE_M);

            const int macro_n_block_size = real_macro_m * MACRO_SWIZZLE_N;

            const int macro_block_n_idx = bid_in_m_band / macro_n_block_size;
            const int macro_lane = bid_in_m_band % macro_n_block_size;

            return _compute_bid_in_macro(macro_block_m_idx, macro_block_n_idx, macro_lane);
        }
    };

    // load warp
    if (warp_id == 4 && elect_sync()) {
        int mma_stage = 0;
        int mma_phase = 1;
        auto [off_m, off_n] = compute_bid(cluster_bid);
        for (int iter_k = 0; iter_k < K / BLOCK_K; ++iter_k) {
            load(iter_k, mma_phase, mma_stage, off_m, off_n);
            mma_stage = (mma_stage + 1) % STAGE_NUM;
            if (mma_stage == 0) {
                mma_phase ^= 1;
            }
        }
    }

    // mma warp
    if (cta_rank == 0 && warp_id == 5 && elect_sync()) {
        int tma_phase = 0;
        int tma_stage = 0;

        int mainloop_stage = 0;
        int epilogue_phase = 1;

        if constexpr (DO_PROFILE) {
            profiler.start(ProfilerTag::WaitEpilogue);
        }
        mbarrier_wait(_base_epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);
        if constexpr (DO_PROFILE) {
            profiler.stop();
        }

        for (int iter_k = 0; iter_k < K / BLOCK_K; ++iter_k) {
            compute(iter_k, mainloop_stage, tma_phase, tma_stage);
            tma_stage = (tma_stage + 1) % STAGE_NUM;
            if (tma_stage == 0) {
                tma_phase ^= 1;
            }
        }

        constexpr int16_t cta_mask = (1 << CLUSTER_M) - 1;
        asm volatile(
            "tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast:"
            ":cluster.b64 [%0], %1;" ::"r"(_base_mainloop_mbar_addr + 8 * mainloop_stage),
            "h"(cta_mask),
            "n"(CLUSTER_M)
            : "memory");

        mainloop_stage = (mainloop_stage + 1) % 2;
        if (mainloop_stage == 0) {
            epilogue_phase ^= 1;
        }
    }

    // epilogue

    if (warp_id < 4) {
        int mainloop_stage = 0;
        int mainloop_phase = 0;

        if constexpr (DO_PROFILE)
            if (elect_sync())
                profiler.start(ProfilerTag::WaitMainloop);
        mbarrier_wait(_base_mainloop_mbar_addr + 8 * mainloop_stage, mainloop_phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if constexpr (DO_PROFILE)
            if (elect_sync())
                profiler.stop();
        auto [off_m, off_n] = compute_bid(cluster_bid);
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

    if constexpr (CLUSTER_M > 1) {
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
    } else {
        __syncthreads();
    }
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" ::"r"(taddr),
            "r"(BLOCK_N * 2),
            "n"(CLUSTER_M));
    }
    if constexpr (DO_PROFILE) {
        if (elect_sync()) {
            profiler.flush();
        }
    }
}

} // namespace

template <bool DO_PROFILE>
void launch_bf16_gemm_dynamic_schedule_impl(
    const __nv_bfloat16 *A_ptr,
    const __nv_bfloat16 *B_ptr,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K,
    int64_t *profiler_ptr,
    int num_entries,
    cudaStream_t stream) {

    CUtensorMap A_tmap, B_tmap, C_tmap;

    static_assert(MACRO_SWIZZLE_M * MACRO_SWIZZLE_N <= _MACRO_SWIZZLE_SUM);
    assert(M % (CLUSTER_M * BLOCK_M) == 0);
    assert(N % BLOCK_N == 0);
    assert(K % BLOCK_K == 0);

    init_3D_tmap(
        &A_tmap, A_ptr, M, K, BLOCK_M, BLOCK_K, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_3D_tmap(&B_tmap,
        B_ptr,
        N,
        K,
        BLOCK_N / CLUSTER_M,
        BLOCK_K,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_2D_tmap(&C_tmap, C_ptr, M, N, BLOCK_M, EPILOGUE_BLOCK_N);

    constexpr int mbarrier_smem = 256;
    constexpr int epilogue_smem_size =
        EPILOGUE_STAGE_NUM * BLOCK_M * EPILOGUE_BLOCK_N * sizeof(nv_bfloat16);
    const int smem_size =
        (BLOCK_M + BLOCK_N / CLUSTER_M) * BLOCK_K * STAGE_NUM * sizeof(nv_bfloat16) +
        epilogue_smem_size + mbarrier_smem;

    const int grid_m = M / (BLOCK_M * CLUSTER_M);
    const int grid_n = N / BLOCK_N;
    const int grid = grid_m * grid_n * CLUSTER_M;

    auto kernel = bf16_gemm_dynamic_schedule_kernel<DO_PROFILE>;

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    kernel<<<grid, TB_SIZE, smem_size, stream>>>(A_tmap,
        B_tmap,
        C_tmap,
        M,
        N,
        K,
        profiler_ptr,
        num_entries);
}

void launch_bf16_gemm_dynamic_schedule(
    const __nv_bfloat16 *A_ptr,
    const __nv_bfloat16 *B_ptr,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    launch_bf16_gemm_dynamic_schedule_impl<false>(
        A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0, stream);
}

void launch_bf16_gemm_dynamic_schedule_profile(
    const __nv_bfloat16 *A_ptr,
    const __nv_bfloat16 *B_ptr,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K,
    int64_t *profiler_ptr,
    int num_entries,
    cudaStream_t stream) {
    launch_bf16_gemm_dynamic_schedule_impl<true>(
        A_ptr, B_ptr, C_ptr, M, N, K, profiler_ptr, num_entries, stream);
}
