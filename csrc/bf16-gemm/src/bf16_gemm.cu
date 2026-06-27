#include <cstdint>
#include <cudaTypedefs.h> // required for CUtensorMap
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bf16_gemm.h"

namespace {

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 64;
constexpr int WARP_SIZE = 32;

constexpr int MMA_K = 16;

constexpr int TB_SIZE = 128;

template <int CTA_GROUP = 1>
__device__ inline void tma_3d_gmem2smem(
    int dst,
    const void *tmap_ptr,
    int x,
    int y,
    int z,
    int mbar_addr) {
    // when CTA_GROUP=1, we can use .shared::cta instead.
    // but .shared::cluster doesn't seem to be slower, so always use it unconditionally here.
    // .cta_group::2 allows mbar_addr and dst to be in different CTA's smem.
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6 "
        "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
        "l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(z),
        "r"(mbar_addr),
        "n"(CTA_GROUP)
        : "memory");
}

__device__ inline void tma_2d_gmem2smem(
    int dst,
    const void *tmap_ptr,
    int x,
    int y,
    int mbar_addr) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], "
                 "[%1, {%2, %3}], [%4];" ::"r"(dst),
        "l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(mbar_addr)
        : "memory");
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/arch/barrier.h#L408
__device__ inline void mbarrier_wait(
    int mbar_addr,
    int phase) {
    uint32_t ticks = 0x989680; // this is optional
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni DONE;\n\t"
                 "bra.uni LAB_WAIT;\n\t"
                 "DONE:\n\t"
                 "}" ::"r"(mbar_addr),
        "r"(phase),
        "r"(ticks));
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


inline void init_2D_tmap(
    CUtensorMap *tmap,
    const nv_bfloat16 *ptr,
    uint64_t global_height,
    uint64_t global_width,
    uint32_t shared_height,
    uint32_t shared_width,
    CUtensorMapSwizzle swizzle) {

    constexpr uint32_t rank = 2;
    uint64_t globalDim[rank] = {global_width, global_height};
    uint64_t globalStrides[rank - 1] = {global_width * sizeof(nv_bfloat16)}; // in bytes
    uint32_t boxDim[rank] = {shared_width, shared_height};
    uint32_t elementStrides[rank] = {1, 1};

    auto err = cuTensorMapEncodeTiled(tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        (void *)ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

__global__ void bf16_gemm_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K) {

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;

    const int grid_n = N / BLOCK_N;
    const int bid_m = bid / grid_n;
    const int bid_n = bid % grid_n;

    const int off_m = bid_m * BLOCK_M;
    const int off_n = bid_n * BLOCK_N;

    // set up smem
    extern __shared__ __align__(1024) char smem[];
    // A: [BLOCK_M, BLOCK_K]  B: [BLOCK_N, BLOCK_K]
    const int A_smem = static_cast<int>(__cvta_generic_to_shared(smem));
    const int B_smem = A_smem + BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);

    constexpr uint32_t i_desc = (1U << 4U)                         // dtype=FP32
                                | (1U << 7U)                       // atype=BF16
                                | (1U << 10U)                      // btype=BF16
                                | ((uint32_t)BLOCK_N >> 3U << 17U) // MMA_N
                                | ((uint32_t)BLOCK_M >> 4U << 24U) // MMA_M
        ;

    // setup Mbarrier
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbars[1];
    __shared__ int tmem_addr[1];
    const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));

    if (tid == 0) {
        // init mbarrier
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(addr),
            "r"(BLOCK_N));
    }
    __syncthreads();

    int phase = 0;
    const int taddr = tmem_addr[0];

    // main loop
    for (int iter_k = 0; iter_k < K / BLOCK_K; ++iter_k) {
        if (warp_id == 0 && elect_sync()) {
            // tma issue
            constexpr int cp_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(
                             mbar_addr),
                "r"(cp_size)
                : "memory");
            for (int k = 0; k < BLOCK_K / 8; k++) {
                const int off_k = iter_k * BLOCK_K + k * 8;
                tma_2d_gmem2smem(A_smem + k * BLOCK_M * 16, &A_tmap, off_k, off_m, mbar_addr);
                tma_2d_gmem2smem(B_smem + k * BLOCK_N * 16, &B_tmap, off_k, off_n, mbar_addr);
            }
        }

        mbarrier_wait(mbar_addr, phase);
        phase ^= 1;

        // https://github.com/deepseek-ai/DeepGEMM/blob/9b680f42/deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh#L289
        asm volatile("tcgen05.fence::after_thread_sync;");

        // tcgen05 mma
        if (warp_id == 0 && elect_sync()) {
            auto make_desc = [](int addr, int height) -> uint64_t {
                const int LBO = height * 16;
                constexpr int SBO = 8 * 16;
                return desc_encode(addr) | (desc_encode(LBO) << 16ULL) |
                       (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
            };
            tcgen05_mma_f16<1>(taddr,
                make_desc(A_smem, BLOCK_M),
                make_desc(B_smem, BLOCK_N),
                i_desc,
                iter_k);
            for (int k = 1; k < BLOCK_K / MMA_K; k++) {
                uint64_t a_desc = make_desc(A_smem + k * BLOCK_M * 32, BLOCK_M);
                uint64_t b_desc = make_desc(B_smem + k * BLOCK_N * 32, BLOCK_N);
                tcgen05_mma_f16<1>(taddr, a_desc, b_desc, i_desc, 1);
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
                    mbar_addr)
                : "memory");
        }
        mbarrier_wait(mbar_addr, phase);
        phase ^= 1; // flip the phase
    }

    // epilogue
    asm volatile("tcgen05.fence::after_thread_sync;");

    for (int n = 0; n < BLOCK_N / 8; n++) {
        float tmp[8];
        const int row = warp_id * 32;
        const int col = n * 8;
        const int addr = taddr + (row << 16) + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
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

    __syncthreads();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
            "r"(BLOCK_N));
    }
}

} // namespace

void launch_bf16_gemm(
    const __nv_bfloat16 *A_ptr,
    const __nv_bfloat16 *B_ptr,
    __nv_bfloat16 *C_ptr,
    int M,
    int N,
    int K) {

    CUtensorMap A_tmap, B_tmap;

    init_2D_tmap(&A_tmap, A_ptr, M, K, BLOCK_M, 8, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    init_2D_tmap(&B_tmap, B_ptr, M, K, BLOCK_N, 8, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);

    const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);

    int grid = (M / BLOCK_M) * (N / BLOCK_N);

    auto kernel = bf16_gemm_kernel;

    // if (smem_size > 48'000)
    //     cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<grid, TB_SIZE, smem_size>>>(A_tmap, B_tmap, C_ptr, M, N, K);
}
