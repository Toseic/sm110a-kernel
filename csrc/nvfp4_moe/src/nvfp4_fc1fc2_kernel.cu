/*
 * Two-CTA time-switched NVFP4 FC1 + FC2 kernel.
 *
 * The fixed-shape implementation uses two independent cta_group::2 pairs in
 * one cta_group::2 pair, persistent row-task workers, TMA mainloop staging,
 * a direct FC1-epilogue SMEM layout, and a configurable-I SMEM streaming
 * buffer. The shared memory released by smaller buffers is used for deeper
 * FC1 and FC2 pipelines.
 */

#include <cmath>
#include <cstdint>

#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "nvfp4_fc1fc2.h"

namespace {

constexpr int kH = 6144;
constexpr int kI = 2048;
constexpr int kPhysicalFc1N = 2 * kI;
#ifndef SM110A_NVFP4_FC1FC2_BUFFER_I
#define SM110A_NVFP4_FC1FC2_BUFFER_I 2048
#endif
constexpr int kBufferI = SM110A_NVFP4_FC1FC2_BUFFER_I;
constexpr int kTileM = 128;
constexpr int kClusterM = 2;
constexpr int kPairTileM = kClusterM * kTileM;
constexpr int kClusterPairs = 1;
constexpr int kClusterRows = kClusterPairs * kPairTileM;
constexpr int kTileN = 128;
constexpr int kStageK = 256;
constexpr int kMmaK = 64;
constexpr int kMmaSteps = kStageK / kMmaK;
constexpr int kWarpSize = 32;
constexpr int kEpilogueWarps = 4;
constexpr int kMmaWarp = 4;
constexpr int kTmaWarp = 5;
constexpr int kTmaReadyWarp = 6;
constexpr int kWarpCount = 7;

constexpr int kMatrixSmemBytes = kTileM * 128;
constexpr int kWeightSmemBytes = (kTileN / kClusterM) * 128;
constexpr int kScaleStageBytes = kMmaSteps * 512;
constexpr int kIntermediateAtoms = kBufferI / kStageK;
constexpr int kIntermediateValueBytes = kIntermediateAtoms * kMatrixSmemBytes;
constexpr int kIntermediateScaleBytes = (kBufferI / kMmaK) * 512;
constexpr int kIntermediateBytes =
    kIntermediateValueBytes + kIntermediateScaleBytes;
constexpr int kWorkspaceBytes =
    kMatrixSmemBytes + kWeightSmemBytes + 2 * kScaleStageBytes;
constexpr int kFc2WorkspaceBytes = kWeightSmemBytes + kScaleStageBytes;
constexpr int kFc1StageCount = kBufferI >= 2048 ? 2 :
    (kBufferI >= 1024 ? 4 : 5);
constexpr int kFc2StageCount =
    (kFc1StageCount * kWorkspaceBytes) / kFc2WorkspaceBytes;
constexpr int kBarrierStageCount = kFc2StageCount;
constexpr int kReduceN = 64;
constexpr int kFinalizeSmemBytes =
    kTileM * kReduceN * sizeof(__nv_bfloat16);
constexpr int kDynamicSmemBytes =
    kIntermediateBytes + kFc1StageCount * kWorkspaceBytes + kFinalizeSmemBytes;
constexpr int kTmemColumns = 512;
constexpr int kAccumulatorStages = 2;
constexpr int kScaleTmemCol = kAccumulatorStages * kTileN;

constexpr int kMaxOptInSmemBytes = 232448;
static_assert(kDynamicSmemBytes <= kMaxOptInSmemBytes);
static_assert(kI % kBufferI == 0);
static_assert(kBufferI % kStageK == 0);

__device__ __forceinline__ float silu(float x) {
    constexpr float kLog2E = 1.4426950408889634f;
    float exponential;
    float reciprocal;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(-kLog2E * x));
    asm("rcp.approx.ftz.f32 %0, %1;"
        : "=f"(reciprocal) : "f"(1.0f + exponential));
    return x * reciprocal;
}

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t elected = 0;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "elect.sync _|p, %1;\n\t"
                 "@p mov.s32 %0, 1;\n\t}"
        : "+r"(elected) : "r"(0xffffffffu));
    return elected;
}

__device__ __forceinline__ constexpr uint64_t matrix_desc_encode(uint64_t value) {
    return (value & 0x3ffffULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t make_k_major_128b_desc(int smem_addr) {
    constexpr int kStrideByteOffset = 8 * 128;
    const uint64_t base_offset = (smem_addr >> 7) & 7;
    return matrix_desc_encode(smem_addr) |
           (matrix_desc_encode(kStrideByteOffset) << 32ULL) |
           (1ULL << 46ULL) | (base_offset << 49ULL) | (2ULL << 61ULL);
}

__device__ __forceinline__ uint64_t make_scale_desc(int smem_addr) {
    return matrix_desc_encode(smem_addr) |
           (matrix_desc_encode(128) << 32ULL) | (1ULL << 46ULL);
}

__device__ __forceinline__ void mbarrier_init(int addr) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(addr));
}

__device__ __forceinline__ void mbarrier_wait(int addr, int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t.reg .pred done;\n\t"
                 "WAIT_FUSED_MMA:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE_FUSED_MMA;\n\t"
                 "bra.uni WAIT_FUSED_MMA;\n\t"
                 "COMPLETE_FUSED_MMA:\n\t}"
        :: "r"(addr), "r"(phase), "r"(kSuspendTicks) : "memory");
}

__device__ __forceinline__ int swizzled_vector(int row, int vector) {
    const int linear = row * 8 + vector;
    return (linear & ~7) + ((linear & 7) ^ ((linear >> 3) & 7));
}

__device__ __forceinline__ int scale_offset(int row, int group) {
    return (row & 31) * 16 + ((row >> 5) & 3) * 4 + group;
}

__device__ __forceinline__ void cluster_barrier() {
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}

__device__ __forceinline__ int map_shared_to_cluster_cta(
    int local_smem_addr, int target_cta_rank) {
    int remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote_addr)
        : "r"(local_smem_addr), "r"(target_cta_rank));
    return remote_addr;
}

__device__ __forceinline__ void mbarrier_arrive_on_cluster_cta(
    int local_mbar_addr, int target_cta_rank) {
    const int remote_addr =
        map_shared_to_cluster_cta(local_mbar_addr, target_cta_rank);
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
        :: "r"(remote_addr) : "memory");
}

__device__ __forceinline__ void pair_barrier(
    int gather_addr, int release_addr, int pair_root,
    int phase, int tid) {
    __syncthreads();
    if (tid == 0) {
        mbarrier_arrive_on_cluster_cta(gather_addr, pair_root);
        int cta_rank;
        asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
        if (cta_rank == pair_root) {
            mbarrier_wait(gather_addr, phase);
            mbarrier_arrive_on_cluster_cta(release_addr, pair_root);
            mbarrier_arrive_on_cluster_cta(release_addr, pair_root + 1);
        }
        mbarrier_wait(release_addr, phase);
    }
    __syncthreads();
}

__device__ __forceinline__ void mbarrier_expect_tx(int addr, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
                 "_, [%0], %1;" :: "r"(addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void tma_load_4d(
    int destination, const void* tensor_map,
    int x, int y, int z, int w, int mbar_addr) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global."
                 "mbarrier::complete_tx::bytes.cta_group::1 "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(destination), "l"(tensor_map), "r"(x), "r"(y),
           "r"(z), "r"(w), "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void bulk_store_bf16(
    __nv_bfloat16* destination, const __nv_bfloat16* source, int bytes) {
    const uint64_t destination_addr = reinterpret_cast<uint64_t>(destination);
    const uint32_t source_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(source));
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group "
                 "[%0], [%1], %2;"
        :: "l"(destination_addr), "r"(source_addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void bulk_reduce_add_bf16(
    __nv_bfloat16* destination, const __nv_bfloat16* source, int bytes) {
    const uint64_t destination_addr = reinterpret_cast<uint64_t>(destination);
    const uint32_t source_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(source));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
                 "[%0], [%1], %2;"
        :: "l"(destination_addr), "r"(source_addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void bulk_commit_and_wait_read() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

__device__ __forceinline__ void issue_mma_step(
    int tmem_addr, int scale_addr, int a_smem_addr, int b_smem_addr,
    int sfa_smem_addr, int sfb_smem_addr, int mma_k, int accumulate) {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(scale_addr),
           "l"(make_scale_desc(sfa_smem_addr + mma_k * 512)) : "memory");
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(scale_addr + 4),
           "l"(make_scale_desc(sfb_smem_addr + mma_k * 512)) : "memory");
    constexpr uint32_t instruction_desc =
        (1U << 7U) | (1U << 10U) |
        (static_cast<uint32_t>(kTileN) >> 3U << 17U) |
        (static_cast<uint32_t>(kPairTileM) >> 7U << 27U);
    const uint64_t a_desc =
        make_k_major_128b_desc(a_smem_addr + mma_k * 32);
    const uint64_t b_desc =
        make_k_major_128b_desc(b_smem_addr + mma_k * 32);
    asm volatile("{\n\t.reg .pred input_d;\n\t"
                 "setp.ne.b32 input_d, %6, 0;\n\t"
                 "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 "
                 "[%0], %1, %2, %3, [%4], [%5], input_d;\n\t}"
        :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc),
           "r"(instruction_desc), "r"(scale_addr), "r"(scale_addr + 4),
           "r"(accumulate) : "memory");
}

__device__ __forceinline__ void commit_pair_mma(
    int done_mbar_addr, int pair_id) {
    const int16_t mask = static_cast<int16_t>(3 << (pair_id * 2));
    asm volatile("tcgen05.commit.cta_group::2."
                 "mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
                 "[%0], %1;" :: "r"(done_mbar_addr), "h"(mask) : "memory");
}

__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(kWarpCount * kWarpSize, 1)
void nvfp4_fc1fc2_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap sfa_map,
    const __grid_constant__ CUtensorMap sfw1_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap sfw2_map,
    __nv_bfloat16* output, int task_count) {
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int pair_id = cta_rank >> 1;
    const int pair_rank = cta_rank & 1;
    const int worker_id = blockIdx.x >> 1;
    const int worker_count = gridDim.x >> 1;

    __shared__ int tmem_base[1];
    __shared__ uint64_t pair_ready[kBarrierStageCount];
    __shared__ uint64_t tma_done[kBarrierStageCount];
    __shared__ uint64_t stage_free[kBarrierStageCount];
    __shared__ uint64_t output_done[kAccumulatorStages];
    __shared__ uint64_t epilogue_free[kAccumulatorStages];
    __shared__ uint64_t pair_gather[1];
    __shared__ uint64_t pair_release[1];
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* intermediate_values = smem;
    uint8_t* intermediate_scales =
        intermediate_values + kIntermediateValueBytes;
    uint8_t* workspace = smem + kIntermediateBytes;
    auto* finalize_smem = reinterpret_cast<__nv_bfloat16*>(
        workspace + kFc1StageCount * kWorkspaceBytes);
    const int workspace_addr =
        static_cast<int>(__cvta_generic_to_shared(workspace));
    const int intermediate_values_addr =
        static_cast<int>(__cvta_generic_to_shared(intermediate_values));
    const int intermediate_scales_addr =
        static_cast<int>(__cvta_generic_to_shared(intermediate_scales));
    const int pair_ready_addr =
        static_cast<int>(__cvta_generic_to_shared(pair_ready));
    const int tma_done_addr =
        static_cast<int>(__cvta_generic_to_shared(tma_done));
    const int stage_free_addr =
        static_cast<int>(__cvta_generic_to_shared(stage_free));
    const int output_done_addr =
        static_cast<int>(__cvta_generic_to_shared(output_done));
    const int epilogue_free_addr =
        static_cast<int>(__cvta_generic_to_shared(epilogue_free));
    const int pair_gather_addr =
        static_cast<int>(__cvta_generic_to_shared(pair_gather));
    const int pair_release_addr =
        static_cast<int>(__cvta_generic_to_shared(pair_release));

    if (warp == kMmaWarp) {
        const int tmem_base_addr =
            static_cast<int>(__cvta_generic_to_shared(tmem_base));
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 "
                     "[%0], %1;" :: "r"(tmem_base_addr), "r"(kTmemColumns));
    }
    cluster_barrier();
    const int tmem_addr = tmem_base[0];

    for (int task = worker_id; task < task_count; task += worker_count) {
        const int pair_row = task * kClusterRows + pair_id * kPairTileM;
        const int cta_row = pair_row + pair_rank * kTileM;
        if (tid == 0) {
            for (int stage = 0; stage < kBarrierStageCount; ++stage) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;"
                    :: "r"(pair_ready_addr +
                        stage * static_cast<int>(sizeof(uint64_t))));
                mbarrier_init(tma_done_addr + stage * sizeof(uint64_t));
                mbarrier_init(stage_free_addr + stage * sizeof(uint64_t));
            }
            for (int stage = 0; stage < kAccumulatorStages; ++stage) {
                mbarrier_init(output_done_addr + stage * sizeof(uint64_t));
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;"
                    :: "r"(epilogue_free_addr +
                        stage * static_cast<int>(sizeof(uint64_t))));
            }
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;"
                :: "r"(pair_gather_addr));
            mbarrier_init(pair_release_addr);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        __syncthreads();
        cluster_barrier();
        int tma_stage = 0;
        int tma_free_phase = 1;
        int ready_tma_stage = 0;
        int ready_tma_phase = 0;
        int mma_stage = 0;
        int mma_ready_phase = 0;
        int output_phase = 0;
        int pair_sync_phase = 0;

        for (int chunk = 0; chunk < kI / kBufferI; ++chunk) {
        // FC1 mode. Each physical 128-column tile produces 64 logical I
        // columns; 32 tiles fill the complete-I persistent SMEM buffer.
        if (warp == kTmaWarp && elect_sync()) {
            for (int i64 = 0; i64 < kBufferI / kMmaK; ++i64) {
                const int physical_n =
                    (chunk * (kBufferI / kMmaK) + i64) * kTileN;
                for (int h_stage = 0; h_stage < kH / kStageK; ++h_stage) {
                    const int h = h_stage * kStageK;
                    const int stage_base = workspace_addr + tma_stage * kWorkspaceBytes;
                    const int stage_a = stage_base;
                    const int stage_b = stage_a + kMatrixSmemBytes;
                    const int stage_sfa = stage_b + kWeightSmemBytes;
                    const int stage_sfb = stage_sfa + kScaleStageBytes;
                    const int local_tma_done =
                        tma_done_addr + tma_stage * sizeof(uint64_t);
                    mbarrier_wait(
                        stage_free_addr + tma_stage * sizeof(uint64_t),
                        tma_free_phase);
                    mbarrier_expect_tx(local_tma_done, kWorkspaceBytes);
                    tma_load_4d(stage_a, &a_map,
                        h, cta_row, 0, 0, local_tma_done);
                    tma_load_4d(stage_b, &w1_map,
                        h, physical_n + pair_rank * (kTileN / kClusterM),
                        0, 0, local_tma_done);
                    tma_load_4d(stage_sfa, &sfa_map,
                        0, 0, h / kMmaK, cta_row / kTileM,
                        local_tma_done);
                    tma_load_4d(stage_sfb, &sfw1_map,
                        0, 0, h / kMmaK, physical_n / kTileN,
                        local_tma_done);
                    tma_stage = (tma_stage + 1) % kFc1StageCount;
                    if (tma_stage == 0) {
                        tma_free_phase ^= 1;
                    }
                }
            }
        }
        if (warp == kTmaReadyWarp && elect_sync()) {
            for (int i64 = 0; i64 < kBufferI / kMmaK; ++i64) {
                for (int h_stage = 0; h_stage < kH / kStageK; ++h_stage) {
                    mbarrier_wait(
                        tma_done_addr + ready_tma_stage * sizeof(uint64_t),
                        ready_tma_phase);
                    mbarrier_arrive_on_cluster_cta(
                        pair_ready_addr + ready_tma_stage * sizeof(uint64_t),
                        pair_id * kClusterM);
                    ready_tma_stage =
                        (ready_tma_stage + 1) % kFc1StageCount;
                    if (ready_tma_stage == 0) ready_tma_phase ^= 1;
                }
            }
        }
        if (pair_rank == 0 && warp == kMmaWarp && elect_sync()) {
            int accumulator_stage = 0;
            int accumulator_free_phase = 1;
            for (int i64 = 0; i64 < kBufferI / kMmaK; ++i64) {
                mbarrier_wait(
                    epilogue_free_addr + accumulator_stage * sizeof(uint64_t),
                    accumulator_free_phase);
                const int accumulator_addr =
                    tmem_addr + accumulator_stage * kTileN;
                for (int h_stage = 0; h_stage < kH / kStageK; ++h_stage) {
                    const int stage_base = workspace_addr + mma_stage * kWorkspaceBytes;
                    mbarrier_wait(
                        pair_ready_addr + mma_stage * sizeof(uint64_t),
                        mma_ready_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
                    for (int mma_k = 0; mma_k < kMmaSteps; ++mma_k) {
                        issue_mma_step(accumulator_addr,
                            tmem_addr + kScaleTmemCol,
                            stage_base,
                            stage_base + kMatrixSmemBytes,
                            stage_base + kMatrixSmemBytes + kWeightSmemBytes,
                            stage_base + kMatrixSmemBytes + kWeightSmemBytes +
                                kScaleStageBytes,
                            mma_k,
                            h_stage != 0 || mma_k != 0);
                    }
                    commit_pair_mma(
                        stage_free_addr + mma_stage * sizeof(uint64_t), pair_id);
                    mma_stage = (mma_stage + 1) % kFc1StageCount;
                    if (mma_stage == 0) mma_ready_phase ^= 1;
                }
                commit_pair_mma(
                    output_done_addr + accumulator_stage * sizeof(uint64_t),
                    pair_id);
                accumulator_stage =
                    (accumulator_stage + 1) % kAccumulatorStages;
                if (accumulator_stage == 0) accumulator_free_phase ^= 1;
            }
        }

        if (warp < kEpilogueWarps) {
            int accumulator_stage = 0;
            for (int i64 = 0; i64 < kBufferI / kMmaK; ++i64) {
                const int accumulator_addr =
                    tmem_addr + accumulator_stage * kTileN;
                mbarrier_wait(
                    output_done_addr + accumulator_stage * sizeof(uint64_t),
                    output_phase);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const int local_row = warp * kWarpSize + (tid & 31);
                const int tmem_row = pair_rank * kTileM + warp * kWarpSize;
                uint8_t row_packed[kMmaK / 2];
#pragma unroll
                for (int group = 0; group < kMmaK / 16; ++group) {
                    float gate_values[16];
                    float up_values[16];
                    const int gate_addr =
                        accumulator_addr + (tmem_row << 16) + group * 16;
                    const int up_addr = gate_addr + kMmaK;
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                        : "=f"(gate_values[0]), "=f"(gate_values[1]),
                          "=f"(gate_values[2]), "=f"(gate_values[3]),
                          "=f"(gate_values[4]), "=f"(gate_values[5]),
                          "=f"(gate_values[6]), "=f"(gate_values[7]),
                          "=f"(gate_values[8]), "=f"(gate_values[9]),
                          "=f"(gate_values[10]), "=f"(gate_values[11]),
                          "=f"(gate_values[12]), "=f"(gate_values[13]),
                          "=f"(gate_values[14]), "=f"(gate_values[15])
                        : "r"(gate_addr));
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                        : "=f"(up_values[0]), "=f"(up_values[1]),
                          "=f"(up_values[2]), "=f"(up_values[3]),
                          "=f"(up_values[4]), "=f"(up_values[5]),
                          "=f"(up_values[6]), "=f"(up_values[7]),
                          "=f"(up_values[8]), "=f"(up_values[9]),
                          "=f"(up_values[10]), "=f"(up_values[11]),
                          "=f"(up_values[12]), "=f"(up_values[13]),
                          "=f"(up_values[14]), "=f"(up_values[15])
                        : "r"(up_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float activated[16];
                    float abs_max = 0.0f;
#pragma unroll
                    for (int col = 0; col < 16; ++col) {
                        activated[col] = silu(gate_values[col]) * up_values[col];
                        abs_max = fmaxf(abs_max, fabsf(activated[col]));
                    }
                    const uint8_t scale_raw = __nv_cvt_float_to_fp8(
                        abs_max / 6.0f, __NV_SATFINITE, __NV_E4M3);
                    const __half_raw scale_half_raw =
                        __nv_cvt_fp8_to_halfraw(scale_raw, __NV_E4M3);
                    const float scale = abs_max == 0.0f
                        ? 1.0f
                        : __half2float(
                            *reinterpret_cast<const __half*>(&scale_half_raw));
                    const float inverse_scale = 1.0f / scale;
#pragma unroll
                    for (int pair = 0; pair < 8; ++pair) {
                        row_packed[group * 8 + pair] =
                            __nv_cvt_float2_to_fp4x2(
                                {activated[pair * 2] * inverse_scale,
                                 activated[pair * 2 + 1] * inverse_scale},
                                __NV_E2M1, cudaRoundNearest);
                    }
                    intermediate_scales[i64 * 512 +
                        scale_offset(local_row, group)] = scale_raw;
                }
                const int atom = i64 / kMmaSteps;
                const int in_atom = i64 % kMmaSteps;
#pragma unroll
                for (int vector = 0; vector < 2; ++vector) {
                    const auto value =
                        reinterpret_cast<const uint4*>(row_packed)[vector];
                    reinterpret_cast<uint4*>(intermediate_values +
                        atom * kMatrixSmemBytes)[swizzled_vector(
                            local_row, in_atom * 2 + vector)] = value;
                }
                asm volatile("bar.sync 2, 128;" ::: "memory");
                if (warp == 0 && elect_sync()) {
                    mbarrier_arrive_on_cluster_cta(
                        epilogue_free_addr +
                            accumulator_stage * sizeof(uint64_t),
                        pair_id * kClusterM);
                }
                accumulator_stage =
                    (accumulator_stage + 1) % kAccumulatorStages;
                if (accumulator_stage == 0) output_phase ^= 1;
            }
        }
        pair_barrier(pair_gather_addr, pair_release_addr,
                     pair_id * kClusterM, pair_sync_phase, tid);
        pair_sync_phase ^= 1;

        // FC1 and FC2 use different ring depths in the same shared-memory
        // workspace. Both roles have drained, so reset the stage barriers
        // before switching modes instead of relying on a particular loop
        // count to land on the initial slot/parity.
        if constexpr (kBufferI < kI) {
            if (tid == 0) {
                for (int stage = 0; stage < kBarrierStageCount; ++stage) {
                    asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;"
                        :: "r"(pair_ready_addr +
                            stage * static_cast<int>(sizeof(uint64_t))));
                    mbarrier_init(tma_done_addr + stage * sizeof(uint64_t));
                    mbarrier_init(stage_free_addr + stage * sizeof(uint64_t));
                }
                asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            }
            __syncthreads();
            cluster_barrier();
        }

        // Reset the role-local cursors before reinterpreting the same
        // workspace as compact FC2 B/SFB stages.
        tma_stage = 0;
        tma_free_phase = 1;
        ready_tma_stage = 0;
        ready_tma_phase = 0;
        mma_stage = 0;
        mma_ready_phase = 0;

        // FC2 mode. Roles advance independently across H tiles. Two TMEM
        // accumulator stages let MMA start tile n+1 while the epilogue drains
        // tile n; SMEM stage barriers continue to provide TMA/MMA backpressure.
        if (warp == kTmaWarp && elect_sync()) {
            for (int h_tile = 0; h_tile < kH / kTileN; ++h_tile) {
                const int h = h_tile * kTileN;
                for (int i_stage = 0; i_stage < kBufferI / kStageK; ++i_stage) {
                    const int global_i = chunk * kBufferI + i_stage * kStageK;
                    const int stage_base =
                        workspace_addr + tma_stage * kFc2WorkspaceBytes;
                    const int stage_b = stage_base;
                    const int stage_sfb = stage_b + kWeightSmemBytes;
                    const int local_tma_done =
                        tma_done_addr + tma_stage * sizeof(uint64_t);
                    mbarrier_wait(
                        stage_free_addr + tma_stage * sizeof(uint64_t),
                        tma_free_phase);
                    mbarrier_expect_tx(local_tma_done,
                        kWeightSmemBytes + kScaleStageBytes);
                    tma_load_4d(stage_b, &w2_map,
                        global_i, h + pair_rank * (kTileN / kClusterM),
                        0, 0, local_tma_done);
                    tma_load_4d(stage_sfb, &sfw2_map,
                        0, 0, global_i / kMmaK, h / kTileN,
                        local_tma_done);
                    tma_stage = (tma_stage + 1) % kFc2StageCount;
                    if (tma_stage == 0) {
                        tma_free_phase ^= 1;
                    }
                }
            }
        }
        if (warp == kTmaReadyWarp && elect_sync()) {
            for (int h_tile = 0; h_tile < kH / kTileN; ++h_tile) {
                for (int i_stage = 0; i_stage < kBufferI / kStageK; ++i_stage) {
                    mbarrier_wait(
                        tma_done_addr + ready_tma_stage * sizeof(uint64_t),
                        ready_tma_phase);
                    mbarrier_arrive_on_cluster_cta(
                        pair_ready_addr + ready_tma_stage * sizeof(uint64_t),
                        pair_id * kClusterM);
                    ready_tma_stage =
                        (ready_tma_stage + 1) % kFc2StageCount;
                    if (ready_tma_stage == 0) ready_tma_phase ^= 1;
                }
            }
        }
        if (pair_rank == 0 && warp == kMmaWarp && elect_sync()) {
            int accumulator_stage = 0;
            int accumulator_free_phase = 1;
            for (int h_tile = 0; h_tile < kH / kTileN; ++h_tile) {
                mbarrier_wait(
                    epilogue_free_addr + accumulator_stage * sizeof(uint64_t),
                    accumulator_free_phase);
                const int accumulator_addr =
                    tmem_addr + accumulator_stage * kTileN;
                for (int i_stage = 0; i_stage < kBufferI / kStageK; ++i_stage) {
                    const int stage_base =
                        workspace_addr + mma_stage * kFc2WorkspaceBytes;
                    mbarrier_wait(
                        pair_ready_addr + mma_stage * sizeof(uint64_t),
                        mma_ready_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
                    for (int mma_k = 0; mma_k < kMmaSteps; ++mma_k) {
                        issue_mma_step(accumulator_addr,
                            tmem_addr + kScaleTmemCol,
                            intermediate_values_addr +
                                i_stage * kMatrixSmemBytes,
                            stage_base,
                            intermediate_scales_addr +
                                i_stage * kMmaSteps * 512,
                            stage_base + kWeightSmemBytes,
                            mma_k,
                            i_stage != 0 || mma_k != 0);
                    }
                    commit_pair_mma(
                        stage_free_addr + mma_stage * sizeof(uint64_t), pair_id);
                    mma_stage = (mma_stage + 1) % kFc2StageCount;
                    if (mma_stage == 0) mma_ready_phase ^= 1;
                }
                commit_pair_mma(
                    output_done_addr + accumulator_stage * sizeof(uint64_t),
                    pair_id);
                accumulator_stage =
                    (accumulator_stage + 1) % kAccumulatorStages;
                if (accumulator_stage == 0) accumulator_free_phase ^= 1;
            }
        }

        if (warp < kEpilogueWarps) {
            int accumulator_stage = 0;
            for (int h_tile = 0; h_tile < kH / kTileN; ++h_tile) {
                const int h = h_tile * kTileN;
                const int accumulator_addr =
                    tmem_addr + accumulator_stage * kTileN;
                mbarrier_wait(
                    output_done_addr + accumulator_stage * sizeof(uint64_t),
                    output_phase);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const int local_row = warp * kWarpSize + (tid & 31);
                const int global_row = cta_row + local_row;
                const int tmem_row = pair_rank * kTileM + warp * kWarpSize;
#pragma unroll
                for (int slab = 0; slab < kTileN; slab += kReduceN) {
#pragma unroll
                    for (int group = 0; group < kReduceN / 8; ++group) {
                        float values[8];
                        const int load_addr = accumulator_addr +
                            (tmem_row << 16) +
                            slab + group * 8;
                        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                                     "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                            : "=f"(values[0]), "=f"(values[1]),
                              "=f"(values[2]), "=f"(values[3]),
                              "=f"(values[4]), "=f"(values[5]),
                              "=f"(values[6]), "=f"(values[7])
                            : "r"(load_addr));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
                        for (int col = 0; col < 8; ++col) {
                            finalize_smem[local_row * kReduceN + group * 8 + col] =
                                __float2bfloat16_rn(values[col]);
                        }
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    auto* destination = output +
                        static_cast<size_t>(global_row) * kH + h + slab;
                    if (chunk == 0) {
                        bulk_store_bf16(destination,
                            finalize_smem + local_row * kReduceN,
                            kReduceN * sizeof(__nv_bfloat16));
                    } else {
                        bulk_reduce_add_bf16(destination,
                            finalize_smem + local_row * kReduceN,
                            kReduceN * sizeof(__nv_bfloat16));
                    }
                    bulk_commit_and_wait_read();
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                }
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                asm volatile("bar.sync 2, 128;" ::: "memory");
                if (warp == 0 && elect_sync()) {
                    mbarrier_arrive_on_cluster_cta(
                        epilogue_free_addr +
                            accumulator_stage * sizeof(uint64_t),
                        pair_id * kClusterM);
                }
                accumulator_stage =
                    (accumulator_stage + 1) % kAccumulatorStages;
                if (accumulator_stage == 0) output_phase ^= 1;
            }
        }
        pair_barrier(pair_gather_addr, pair_release_addr,
                     pair_id * kClusterM, pair_sync_phase, tid);
        pair_sync_phase ^= 1;

        // Every chunk may leave the variable-depth TMA ring at a different
        // slot/parity. All users have drained here, so reset the mode-local
        // barriers before the next FC1 chunk reuses the overlaid workspace.
        if (chunk + 1 < kI / kBufferI) {
            if (tid == 0) {
                for (int stage = 0; stage < kBarrierStageCount; ++stage) {
                    asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;"
                        :: "r"(pair_ready_addr +
                            stage * static_cast<int>(sizeof(uint64_t))));
                    mbarrier_init(tma_done_addr + stage * sizeof(uint64_t));
                    mbarrier_init(stage_free_addr + stage * sizeof(uint64_t));
                }
                asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            }
            __syncthreads();
            cluster_barrier();
            tma_stage = 0;
            tma_free_phase = 1;
            ready_tma_stage = 0;
            ready_tma_phase = 0;
            mma_stage = 0;
            mma_ready_phase = 0;
        }
    }

    }

    cluster_barrier();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_addr), "r"(kTmemColumns));
    }
}

template <int kBoxRows>
bool init_matrix_map(
    CUtensorMap* map, const uint8_t* pointer, uint64_t rows, uint64_t cols) {
    constexpr uint32_t rank = 4;
    uint64_t dimensions[rank] = {cols, rows, 1, 1};
    uint64_t strides[rank - 1] = {
        cols / 2, rows * cols / 2, rows * cols / 2};
    uint32_t box[rank] = {kStageK, kBoxRows, 1, 1};
    uint32_t elements[rank] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, rank,
        const_cast<uint8_t*>(pointer), dimensions, strides, box, elements,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

bool init_scale_map(
    CUtensorMap* map, const uint8_t* pointer,
    uint64_t row_tiles, uint64_t k_tiles) {
    constexpr uint32_t rank = 4;
    uint64_t dimensions[rank] = {256, 2, k_tiles, row_tiles};
    uint64_t strides[rank - 1] = {256, 512, 512 * k_tiles};
    uint32_t box[rank] = {256, 2, kMmaSteps, 1};
    uint32_t elements[rank] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank,
        const_cast<uint8_t*>(pointer), dimensions, strides, box, elements,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

}  // namespace

void launch_nvfp4_fc1fc2(
    const uint8_t* a, const uint8_t* w1, const uint8_t* sfa,
    const uint8_t* sfw1, const uint8_t* w2, const uint8_t* sfw2,
    __nv_bfloat16* output, int m, cudaStream_t stream) {
    if (!a || !w1 || !sfa || !sfw1 || !w2 || !sfw2 || !output ||
        m <= 0 || m % kClusterRows) {
        return;
    }
    CUtensorMap a_map;
    CUtensorMap w1_map;
    CUtensorMap sfa_map;
    CUtensorMap sfw1_map;
    CUtensorMap w2_map;
    CUtensorMap sfw2_map;
    const bool maps_ok =
        init_matrix_map<kTileM>(&a_map, a, m, kH) &&
        init_matrix_map<kTileN / kClusterM>(
            &w1_map, w1, kPhysicalFc1N, kH) &&
        init_scale_map(&sfa_map, sfa, m / kTileM, kH / kMmaK) &&
        init_scale_map(&sfw1_map, sfw1,
            kPhysicalFc1N / kTileN, kH / kMmaK) &&
        init_matrix_map<kTileN / kClusterM>(&w2_map, w2, kH, kI) &&
        init_scale_map(&sfw2_map, sfw2, kH / kTileN, kI / kMmaK);
    if (!maps_ok) {
        return;
    }
    auto kernel = nvfp4_fc1fc2_kernel;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kDynamicSmemBytes);
    const int clusters = m / kClusterRows;
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = dim3(2 * clusters, 1, 1);
    launch_config.blockDim = dim3(kWarpCount * kWarpSize, 1, 1);
    launch_config.dynamicSmemBytes = kDynamicSmemBytes;
    launch_config.stream = stream;
    int resident_clusters = 0;
    const cudaError_t occupancy_status = cudaOccupancyMaxActiveClusters(
        &resident_clusters, kernel, &launch_config);
    if (occupancy_status != cudaSuccess || resident_clusters <= 0) {
        int device = 0;
        int sm_count = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        resident_clusters = sm_count / kClusterM;
        if (resident_clusters <= 0) resident_clusters = 1;
    }
    const int workers = clusters < resident_clusters
        ? clusters : resident_clusters;
    kernel<<<dim3(2 * workers, 1, 1), kWarpCount * kWarpSize,
        kDynamicSmemBytes, stream>>>(
            a_map, w1_map, sfa_map, sfw1_map, w2_map, sfw2_map,
            output, clusters);
}

extern "C" void sm110a_nvfp4_fc1fc2_launch(
    const uint8_t* a, const uint8_t* w1, const uint8_t* sfa,
    const uint8_t* sfw1, const uint8_t* w2, const uint8_t* sfw2,
    __nv_bfloat16* output, int m, cudaStream_t stream) {
    launch_nvfp4_fc1fc2(a, w1, sfa, sfw1, w2, sfw2, output, m, stream);
}
