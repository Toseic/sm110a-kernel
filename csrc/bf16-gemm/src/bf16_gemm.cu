#include <cassert>
#include <cstdint>
#include <cudaTypedefs.h> // required for CUtensorMap
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bf16_gemm.h"
#include "profiler.h"

namespace {

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 128;
constexpr int WARP_SIZE = 32;
constexpr int WARP_NUM = 6;
constexpr int STAGE_NUM = 3;
constexpr int MMA_K = 16;
constexpr int SM_NUM = 20;
constexpr int CLUSTER_M = 2;
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

template <bool DO_PROFILE>
__global__
__cluster_dims__(CLUSTER_M, 1, 1) 
__launch_bounds__(TB_SIZE) 
void bf16_gemm_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K,
    int64_t *profiler_ptr,
    int num_entries) {

    const int tid = threadIdx.x;
    const int _bid = blockIdx.x;
    const int grid_x = gridDim.x;
    const int warp_id = tid / WARP_SIZE;

    const int grid_m = M / BLOCK_M;
    const int grid_n = N / BLOCK_N;
    const int grid_size = grid_m * grid_n;

    Profiler profiler;
    if constexpr (DO_PROFILE) {
        if (elect_sync()) {
            profiler.init(num_entries, profiler_ptr, _bid * WARP_NUM + warp_id);
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
        for (int k = 0; k < BLOCK_K / 64; k++) {
            const int off_k = iter_k * BLOCK_K + k * 64;
            tma_3d_gmem2smem<CLUSTER_M>(A_smem + k * BLOCK_M * 64 * sizeof(nv_bfloat16),
                &A_tmap,
                0,
                off_m,
                off_k / 64,
                tma_mbar_addr);
            tma_3d_gmem2smem<CLUSTER_M>(B_smem + k * (BLOCK_N / CLUSTER_M) * 64 * sizeof(nv_bfloat16),
                &B_tmap,
                0,
                off_n + cta_rank * (BLOCK_N / CLUSTER_M),
                off_k / 64,
                tma_mbar_addr);
        }
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

    auto epilogue = [&](int mainloop_stage, int off_m, int off_n) {
        for (int n = 0; n < BLOCK_N / 8; n++) {
            float tmp[8];
            const int row = cta_rank * 128 + warp_id * 32;
            const int col = n * 8 + mainloop_stage * BLOCK_N;
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
            nv_bfloat16 *out_ptr = C_ptr + (off_m + tid) * N + (off_n + n * 8);

            reinterpret_cast<int4 *>(out_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
        }
    };

    auto compute_bid = [&](int bid) -> std::tuple< int, int> {
        constexpr int MACRO_M_SIZE = BLOCK_M * MACRO_SWIZZLE_M;
        constexpr int MACRO_N_SIZE = BLOCK_N * MACRO_SWIZZLE_N;

        const int macro_m_num = (M + MACRO_M_SIZE - 1) / MACRO_M_SIZE;
        const int macro_n_num = (N + MACRO_N_SIZE - 1) / MACRO_N_SIZE;

        auto _compute_bid_in_macro = [&](int macro_block_m_idx,
                                         int macro_block_n_idx,
                                         int macro_lane) -> std::tuple< int, int> {
            const int base_m = macro_block_m_idx * MACRO_M_SIZE;
            const int base_n = macro_block_n_idx * MACRO_N_SIZE;

            const int _real_macro_m = min(MACRO_M_SIZE, M - base_m) / BLOCK_M;
            const int _real_macro_n = min(MACRO_N_SIZE, N - base_n) / BLOCK_N;

            const int macro_bid_m =
                macro_lane / (_real_macro_n * CLUSTER_M) * CLUSTER_M + (macro_lane % CLUSTER_M);
            const int macro_bid_n = (macro_lane / CLUSTER_M) % _real_macro_n;

            const int off_m = base_m + macro_bid_m * BLOCK_M;
            const int off_n = base_n + macro_bid_n * BLOCK_N;

            return { off_m, off_n};
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

        // const int bid_m = bid / (grid_n * CLUSTER_M) * CLUSTER_M + (bid % CLUSTER_M);
        // const int bid_n = (bid / CLUSTER_M) % grid_n;

        // const int off_m = bid_m * BLOCK_M;
        // const int off_n = bid_n * BLOCK_N;
        // return {bid_m, bid_n, off_m, off_n};
    };

    // load warp
    if (warp_id == 4 && elect_sync()) {
        int mma_stage = 0;
        int mma_phase = 1;
        for (int bid = _bid; bid < grid_size; bid += grid_x) {
            auto [ off_m, off_n] = compute_bid(bid);
            for (int iter_k = 0; iter_k < K / BLOCK_K; ++iter_k) {
                load(iter_k, mma_phase, mma_stage, off_m, off_n);
                mma_stage = (mma_stage + 1) % STAGE_NUM;
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

            if (elect_sync()) {
                const int mbar_addr = (_base_epilogue_mbar_addr + mainloop_stage * 8) & 0xFEFFFFFF;
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(mbar_addr)
                    : "memory");
            }
            mainloop_stage = (mainloop_stage + 1) % 2;
            if (mainloop_stage == 0) {
                mainloop_phase ^= 1;
            }
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

    CUtensorMap A_tmap, B_tmap;

    static_assert(MACRO_SWIZZLE_M * MACRO_SWIZZLE_N <= _MACRO_SWIZZLE_SUM);
    assert(M % (CLUSTER_M * BLOCK_M) == 0);
    assert(N % BLOCK_N == 0);

    init_3D_tmap(&A_tmap, A_ptr, M, K, BLOCK_M, 64, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_3D_tmap(&B_tmap,
        B_ptr,
        N,
        K,
        BLOCK_N / CLUSTER_M,
        64,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    constexpr int mbarrier_smem = 256;
    const int smem_size =
        (BLOCK_M + BLOCK_N / CLUSTER_M) * BLOCK_K * STAGE_NUM * sizeof(nv_bfloat16) + mbarrier_smem;

    int grid = SM_NUM;

    auto kernel = bf16_gemm_kernel<DO_PROFILE>;

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    kernel<<<grid, TB_SIZE, smem_size, stream>>>(A_tmap,
        B_tmap,
        C_ptr,
        M,
        N,
        K,
        profiler_ptr,
        num_entries);
}

void launch_bf16_gemm(
    const __nv_bfloat16 *A_ptr,
    const __nv_bfloat16 *B_ptr,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    launch_bf16_gemm_impl<false>(A_ptr, B_ptr, C_ptr, M, N, K, nullptr, 0, stream);
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
    cudaStream_t stream) {
    launch_bf16_gemm_impl<true>(A_ptr, B_ptr, C_ptr, M, N, K, profiler_ptr, num_entries, stream);
}
