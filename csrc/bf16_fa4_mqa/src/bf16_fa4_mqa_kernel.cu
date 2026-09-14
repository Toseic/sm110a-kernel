#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Packed-MQA folds adjacent query heads into the logical query-row
// dimension. Multiple packed groups in the same batch all reuse its one K/V
// head. The kernel has one schedule: the 16-warp FA4 tactic.
#define SM110A_FA4_MQA_PACK_SIZE 32

namespace {

constexpr int kMqaPackSize = SM110A_FA4_MQA_PACK_SIZE;
constexpr bool kPackMqa = true;
static_assert(kMqaPackSize == 32, "MQA pack size is fixed at 32");

__device__ inline void mbarrier_init(
    int mbar_addr,
    int arrival_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr), "r"(arrival_count));
}

template <int RegisterCount>
__device__ inline void setmaxnreg_decrease() {
    static_assert(RegisterCount >= 24 && RegisterCount <= 256 && RegisterCount % 8 == 0);
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(RegisterCount));
}

template <int RegisterCount>
__device__ inline void setmaxnreg_increase() {
    static_assert(RegisterCount >= 24 && RegisterCount <= 256 && RegisterCount % 8 == 0);
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(RegisterCount));
}

// Wait for the producer to toggle this barrier to `phase`.  The explicit
// parity lets a two-stage ring reuse the same mbarrier on later KV tiles.
__device__ inline void mbarrier_wait(
    int mbar_addr,
    int phase) {
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "FA_MQA_MBAR_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni FA_MQA_MBAR_DONE;\n\t"
                 "bra.uni FA_MQA_MBAR_WAIT;\n\t"
                 "FA_MQA_MBAR_DONE:\n\t"
                 "}" ::"r"(mbar_addr),
        "r"(phase),
        "r"(ticks)
        : "memory");
}

__device__ inline void prefetch_tensormap(
    const void *desc) {
    const uint64_t ptr = reinterpret_cast<uint64_t>(desc);
    asm volatile("prefetch.tensormap [%0];" ::"l"(ptr) : "memory");
}

// Issue one TMA transfer.  `mbar_addr` is completed only after all bytes have
// arrived in shared memory, so consumers may wait on it without a CTA-wide sync.
__device__ inline void tma_4d_gmem2smem(
    int dst,
    const void *tmap_ptr,
    int x,
    int y,
    int z,
    int w,
    int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
        "[%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
        "l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(z),
        "r"(w),
        "r"(mbar_addr)
        : "memory");
}

__device__ inline void tma_5d_gmem2smem(
    int dst,
    const void *tmap_ptr,
    int x,
    int y,
    int z,
    int w,
    int v,
    int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
        "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];" : :
            "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
            "r"(mbar_addr)
        : "memory");
}

__device__ inline void mbarrier_expect_tx(
    int mbar_addr,
    int transaction_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" : :
            "r"(mbar_addr), "r"(transaction_bytes)
        : "memory");
}

__device__ inline void tma_4d_smem2gmem(
    int src,
    const void *tmap_ptr,
    int x,
    int y,
    int z,
    int w) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
                 "[%0, {%1, %2, %3, %4}], [%5];" : :
                     "l"(tmap_ptr),
                     "r"(x),
                     "r"(y),
                     "r"(z),
                     "r"(w),
                     "r"(src)
                 : "memory");
}

__device__ inline void tma_5d_smem2gmem(
    int src,
    const void *tmap_ptr,
    int x,
    int y,
    int z,
    int w,
    int v) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "
                 "[%0, {%1, %2, %3, %4, %5}], [%6];" : :
                     "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
                     "r"(src)
                 : "memory");
}

__device__ inline void tma_store_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ inline uint32_t elect_sync() {
    uint32_t leader = 0;
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "elect.sync _|p, %1;\n\t"
                 "@p mov.s32 %0, 1;\n\t"
                 "}"
        : "+r"(leader)
        : "r"(0xFFFFFFFF));
    return leader;
}

__device__ inline constexpr uint64_t desc_encode(
    uint64_t value) {
    return (value & 0x3FFFFULL) >> 4ULL;
}

__device__ inline uint64_t make_128b_swizzle_desc(
    int smem_addr) {
    constexpr int kStrideByteOffset = 8 * 128;
    return desc_encode(smem_addr) | (desc_encode(kStrideByteOffset) << 32ULL) | (1ULL << 46ULL) |
           (2ULL << 61ULL);
}

// Canonical BF16 MN-major SW128 descriptor for a TMA box ordered as
// [64 elements, K rows, N/64 chunks]. Each N chunk contains the full K tile.
__device__ inline uint64_t make_128b_nmajor_desc(
    int smem_addr,
    int leading_byte_offset) {
    constexpr int kStrideByteOffset = 8 * 128;
    return desc_encode(smem_addr) | (desc_encode(leading_byte_offset) << 16ULL) |
           (desc_encode(kStrideByteOffset) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

__device__ inline void tcgen05_mma_bf16(
    int tmem_addr,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t instr_desc,
    int enable_input_d) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
                 "}" ::"r"(tmem_addr),
        "l"(a_desc),
        "l"(b_desc),
        "r"(instr_desc),
        "r"(enable_input_d));
}

__device__ inline void tcgen05_mma_tmem_a_bf16(
    int tmem_out,
    int tmem_a,
    uint64_t b_desc,
    uint32_t instr_desc,
    int enable_input_d) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t"
                 "}" ::"r"(tmem_out),
        "r"(tmem_a),
        "l"(b_desc),
        "r"(instr_desc),
        "r"(enable_input_d));
}

__device__ inline void tcgen05_ld_f32_x8(
    float *values,
    int tmem_addr) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(values[0]),
        "=f"(values[1]),
        "=f"(values[2]),
        "=f"(values[3]),
        "=f"(values[4]),
        "=f"(values[5]),
        "=f"(values[6]),
        "=f"(values[7])
        : "r"(tmem_addr));
}

__device__ inline void tcgen05_st_f32_x8(
    const float *values,
    int tmem_addr) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(
            tmem_addr),
        "f"(values[0]),
        "f"(values[1]),
        "f"(values[2]),
        "f"(values[3]),
        "f"(values[4]),
        "f"(values[5]),
        "f"(values[6]),
        "f"(values[7]));
}

__device__ inline void tcgen05_ld_f32_x32(
    float *values,
    int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "
        "%31}, [%32];"
        : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]),
        "=f"(values[4]), "=f"(values[5]), "=f"(values[6]), "=f"(values[7]),
        "=f"(values[8]), "=f"(values[9]), "=f"(values[10]), "=f"(values[11]),
        "=f"(values[12]), "=f"(values[13]), "=f"(values[14]), "=f"(values[15]),
        "=f"(values[16]), "=f"(values[17]), "=f"(values[18]), "=f"(values[19]),
        "=f"(values[20]), "=f"(values[21]), "=f"(values[22]), "=f"(values[23]),
        "=f"(values[24]), "=f"(values[25]), "=f"(values[26]), "=f"(values[27]),
        "=f"(values[28]), "=f"(values[29]), "=f"(values[30]), "=f"(values[31])
        : "r"(tmem_addr));
}

__device__ inline void tcgen05_st_f32_x32(
    const float *values,
    int tmem_addr) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
        "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
        "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32};" ::"r"(tmem_addr),
        "f"(values[0]), "f"(values[1]), "f"(values[2]), "f"(values[3]),
        "f"(values[4]), "f"(values[5]), "f"(values[6]), "f"(values[7]),
        "f"(values[8]), "f"(values[9]), "f"(values[10]), "f"(values[11]),
        "f"(values[12]), "f"(values[13]), "f"(values[14]), "f"(values[15]),
        "f"(values[16]), "f"(values[17]), "f"(values[18]), "f"(values[19]),
        "f"(values[20]), "f"(values[21]), "f"(values[22]), "f"(values[23]),
        "f"(values[24]), "f"(values[25]), "f"(values[26]), "f"(values[27]),
        "f"(values[28]), "f"(values[29]), "f"(values[30]), "f"(values[31]));
}

__device__ inline void tcgen05_ld_f32_x64(
    float *values,
    int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "
        "%31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, "
        "%46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, "
        "%61, %62, %63}, [%64];"
        : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]),
        "=f"(values[4]), "=f"(values[5]), "=f"(values[6]), "=f"(values[7]),
        "=f"(values[8]), "=f"(values[9]), "=f"(values[10]), "=f"(values[11]),
        "=f"(values[12]), "=f"(values[13]), "=f"(values[14]), "=f"(values[15]),
        "=f"(values[16]), "=f"(values[17]), "=f"(values[18]), "=f"(values[19]),
        "=f"(values[20]), "=f"(values[21]), "=f"(values[22]), "=f"(values[23]),
        "=f"(values[24]), "=f"(values[25]), "=f"(values[26]), "=f"(values[27]),
        "=f"(values[28]), "=f"(values[29]), "=f"(values[30]), "=f"(values[31]),
        "=f"(values[32]), "=f"(values[33]), "=f"(values[34]), "=f"(values[35]),
        "=f"(values[36]), "=f"(values[37]), "=f"(values[38]), "=f"(values[39]),
        "=f"(values[40]), "=f"(values[41]), "=f"(values[42]), "=f"(values[43]),
        "=f"(values[44]), "=f"(values[45]), "=f"(values[46]), "=f"(values[47]),
        "=f"(values[48]), "=f"(values[49]), "=f"(values[50]), "=f"(values[51]),
        "=f"(values[52]), "=f"(values[53]), "=f"(values[54]), "=f"(values[55]),
        "=f"(values[56]), "=f"(values[57]), "=f"(values[58]), "=f"(values[59]),
        "=f"(values[60]), "=f"(values[61]), "=f"(values[62]), "=f"(values[63])
        : "r"(tmem_addr));
}

__device__ inline float tcgen05_ld_max_f32_x64(float *values, int tmem_addr) {
    float maximum;
    asm volatile(
        "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
        : "=f"(values[0]),
        "=f"(values[1]),
        "=f"(values[2]),
        "=f"(values[3]),
        "=f"(values[4]),
        "=f"(values[5]),
        "=f"(values[6]),
        "=f"(values[7]),
        "=f"(values[8]),
        "=f"(values[9]),
        "=f"(values[10]),
        "=f"(values[11]),
        "=f"(values[12]),
        "=f"(values[13]),
        "=f"(values[14]),
        "=f"(values[15]),
        "=f"(values[16]),
        "=f"(values[17]),
        "=f"(values[18]),
        "=f"(values[19]),
        "=f"(values[20]),
        "=f"(values[21]),
        "=f"(values[22]),
        "=f"(values[23]),
        "=f"(values[24]),
        "=f"(values[25]),
        "=f"(values[26]),
        "=f"(values[27]),
        "=f"(values[28]),
        "=f"(values[29]),
        "=f"(values[30]),
        "=f"(values[31]),
        "=f"(values[32]),
        "=f"(values[33]),
        "=f"(values[34]),
        "=f"(values[35]),
        "=f"(values[36]),
        "=f"(values[37]),
        "=f"(values[38]),
        "=f"(values[39]),
        "=f"(values[40]),
        "=f"(values[41]),
        "=f"(values[42]),
        "=f"(values[43]),
        "=f"(values[44]),
        "=f"(values[45]),
        "=f"(values[46]),
        "=f"(values[47]),
        "=f"(values[48]),
        "=f"(values[49]),
        "=f"(values[50]),
        "=f"(values[51]),
        "=f"(values[52]),
        "=f"(values[53]),
        "=f"(values[54]),
        "=f"(values[55]),
        "=f"(values[56]),
        "=f"(values[57]),
        "=f"(values[58]),
        "=f"(values[59]),
        "=f"(values[60]),
        "=f"(values[61]),
        "=f"(values[62]),
        "=f"(values[63]), "=f"(maximum)
        : "r"(tmem_addr));
    return maximum;
}

__device__ inline uint32_t pack_bf16x2(
    float lo,
    float hi) {
    uint32_t packed;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed) : "f"(lo), "f"(hi));
    return packed;
}

__device__ inline float exp2_approx(
    float value) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ inline void tcgen05_st_b32_x4(
    const uint32_t *values,
    int tmem_addr) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" : :
                     "r"(tmem_addr),
                     "r"(values[0]),
                     "r"(values[1]),
                     "r"(values[2]),
                     "r"(values[3]));
}

struct SoftmaxUpdate {
    // Running maximum in the unscaled QK domain.
    float m;
    float l;
    float alpha;
};

__device__ inline void mbarrier_arrive(int mbar_addr);

// An x128 TMEM load needs at least 147 simultaneously allocatable
// registers.  A 512-thread CTA starts with 128 registers/thread, before
// setmaxnreg redistribution.  Split the load into x64 passes so ptxas can
// compile the entry function at that initial limit; the softmax WG still
// raises its runtime allocation before entering this routine.
template <bool AllScoresValid>
__device__ __attribute__((noinline)) SoftmaxUpdate load_s_and_store_p_chunked(
    int s_tmem_row_addr,
    int tile_key_begin,
    int key_len,
    int mask_q_row,
    int valid_q_row,
    float scale,
    float tilem,
    float tilel,
    float conditional_rescale_threshold,
    float scale_log2e,
    int p_partial_mbar_addr,
    int lane_idx) {
    constexpr int kScoresPerChunk = 64;
    constexpr int kChunkCount = 2;
    float rmax = -__int_as_float(0x7F800000);
    // One bound per tile replaces the per-score (key < key_len) &&
    // (key <= mask_q_row) pair; -1 disables dead rows outright.
    const int key_hi = valid_q_row ? min(key_len - 1, mask_q_row) : -1;

#pragma unroll
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
        float scores[kScoresPerChunk];
        if constexpr (AllScoresValid) {
            const float chunk_max = tcgen05_ld_max_f32_x64(
                scores, s_tmem_row_addr + chunk * kScoresPerChunk);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            rmax = fmaxf(rmax, chunk_max);
        } else {
            tcgen05_ld_f32_x64(scores, s_tmem_row_addr + chunk * kScoresPerChunk);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            const int chunk_key_hi = key_hi - tile_key_begin - chunk * kScoresPerChunk;
            if (chunk_key_hi >= kScoresPerChunk - 1) {
                // Whole chunk inside the attention prefix: no per-score mask.
#pragma unroll
                for (int i = 0; i < kScoresPerChunk; ++i) {
                    rmax = fmaxf(rmax, scores[i]);
                }
            } else if (chunk_key_hi >= 0) {
#pragma unroll
                for (int i = 0; i < kScoresPerChunk; ++i) {
                    if (i <= chunk_key_hi) {
                        rmax = fmaxf(rmax, scores[i]);
                    }
                }
            }
            // chunk_key_hi < 0: every score is masked out, rmax is unchanged.
        }
    }

    float new_m = valid_q_row ? fmaxf(tilem, rmax) : tilem;
    const bool needs_rescale =
        valid_q_row && (new_m - tilem) * scale > conditional_rescale_threshold;
    const bool warp_rescale = __any_sync(0xFFFFFFFFU, needs_rescale);
    if (!warp_rescale) {
        new_m = tilem;
    }
    float alpha =
        valid_q_row && tilel != 0.0f ? exp2_approx((tilem - new_m) * scale_log2e) : 0.0f;
    // Fold the (s - m) * scale_log2e exp argument into one FFMA: the rowmax
    // is pre-scaled once so the per-element stream is fmaf(s, scale, -m*scale),
    // matching what ptxas fails to fuse from the two-op form.
    const float neg_m_scale = -new_m * scale_log2e;
    float partial_sum[4] = {};

#pragma unroll
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
        float scores[kScoresPerChunk];
        tcgen05_ld_f32_x64(scores, s_tmem_row_addr + chunk * kScoresPerChunk);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if constexpr (AllScoresValid) {
#pragma unroll
            for (int i = 0; i < kScoresPerChunk; ++i) {
                scores[i] = exp2_approx(fmaf(scores[i], scale_log2e, neg_m_scale));
                partial_sum[i % 4] += scores[i];
            }
        } else {
            // Per-chunk three-way split of the masked stream: fully-dead
            // chunks skip the exp math entirely, fully-live chunks drop the
            // per-score bound; only the diagonal chunk keeps it.
            const int chunk_key_hi = key_hi - tile_key_begin - chunk * kScoresPerChunk;
            if (chunk_key_hi >= kScoresPerChunk - 1) {
#pragma unroll
                for (int i = 0; i < kScoresPerChunk; ++i) {
                    scores[i] = exp2_approx(fmaf(scores[i], scale_log2e, neg_m_scale));
                    partial_sum[i % 4] += scores[i];
                }
            } else if (chunk_key_hi >= 0) {
#pragma unroll
                for (int i = 0; i < kScoresPerChunk; ++i) {
                    scores[i] = i <= chunk_key_hi
                        ? exp2_approx(fmaf(scores[i], scale_log2e, neg_m_scale))
                        : 0.0f;
                    partial_sum[i % 4] += scores[i];
                }
            } else {
#pragma unroll
                for (int i = 0; i < kScoresPerChunk; ++i) {
                    scores[i] = 0.0f;
                }
            }
        }
#pragma unroll
        for (int group = 0; group < kScoresPerChunk / 8; ++group) {
            uint32_t packed[4];
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int i = group * 8 + pair * 2;
                packed[pair] = pack_bf16x2(scores[i], scores[i + 1]);
            }
            // Two BF16 probabilities share one 32-bit TMEM column.
            tcgen05_st_b32_x4(
                packed, s_tmem_row_addr + chunk * (kScoresPerChunk / 2) + group * 4);
        }
        if (chunk == 0) {
            // Partial-P release: drain only the first 64 P columns and hand
            // them to the MMA warp, whose PV half over columns 0-63 can then
            // overlap the exp2/store tail of chunk 1.
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            if (lane_idx == 0) {
                mbarrier_arrive(p_partial_mbar_addr);
            }
        }
    }

    SoftmaxUpdate update;
    update.m = new_m;
    const float sum = (partial_sum[0] + partial_sum[1]) + (partial_sum[2] + partial_sum[3]);
    update.l = valid_q_row ? alpha * tilel + sum : tilel;
    update.alpha = warp_rescale ? alpha : -alpha;
    return update;
}

__device__ inline void tcgen05_commit_mbarrier(
    int mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
                 "[%0];" ::"r"(mbar_addr)
        : "memory");
}

// Signal both the per-slot MMA completion and the stage-reuse barrier that
// lets the producer overwrite this ring stage's K/V buffer.
__device__ inline void tcgen05_commit_stage_reuse(
    int mma_mbar_addr,
    int stage_reuse_mbar_addr) {
    tcgen05_commit_mbarrier(mma_mbar_addr);
    tcgen05_commit_mbarrier(stage_reuse_mbar_addr);
}

__device__ inline void mbarrier_arrive(
    int mbar_addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(mbar_addr)
        : "memory");
}

// Canonical tcgen05 producer/consumer handoff: tcgen05.ld/st run on the
// generic proxy while tcgen05.mma runs on the asynchronous proxy, so an
// mbarrier release/acquire alone does not order them.
__device__ inline void tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ inline void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__host__ __device__ constexpr size_t align_up(
    size_t value,
    size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Dynamic shared-memory layout for one CTA.  Two Q slots remain resident for
// the whole KV loop.  Each KV ring stage is a single physical buffer: K is
// consumed by QK, then V overwrites precisely that buffer for PV.
template <int TileQ, int TileKV, int QSlots, int KvStages> struct FaMqaSmemLayout {
    static constexpr size_t kAlignment = 128;
    static constexpr size_t kMaxSmemBytes = 227 * 1024;
    // Counts the __shared__ mbarrier arrays declared inside
    // fa_mqa_kernel_wg16: q_prologue/out_store/epilogue/tmem_release plus,
    // per stage, tma1/tma2/k_ready/v_ready and the QSlots-sized
    // mma1/mma2/softmax/rescale/p_partial/p_full rings.
    static constexpr size_t kStaticMbarrierBytes =
        (2 * QSlots + 2 + (4 + 6 * QSlots) * KvStages) * sizeof(uint64_t);
    static constexpr size_t kSoftmaxHandoffBytes =
        2 * QSlots * TileQ * sizeof(float);

    size_t q_offset;
    size_t kv_offset;
    size_t out_offset;
    size_t dynamic_bytes;

    __host__ __device__ static FaMqaSmemLayout make(
        int head_dim) {
        const size_t q_bytes =
            QSlots * TileQ * static_cast<size_t>(head_dim) * sizeof(__nv_bfloat16);
        const size_t kv_bytes =
            KvStages * TileKV * static_cast<size_t>(head_dim) * sizeof(__nv_bfloat16);
        FaMqaSmemLayout layout{};
        layout.q_offset = 0;
        layout.kv_offset = align_up(layout.q_offset + q_bytes, kAlignment);
        layout.out_offset = align_up(layout.kv_offset + kv_bytes, kAlignment);
        layout.dynamic_bytes =
            align_up(layout.out_offset + q_bytes, kAlignment);
        return layout;
    }
};

// Q, K, and V share the contiguous [B, H, S, D] physical layout.  Only the
// sequence length and the number of rows loaded per TMA transaction differ.
inline cudaError_t init_qkv_tmap(
    CUtensorMap *tmap,
    const __nv_bfloat16 *ptr,
    int flattened_head_count,
    int sequence_len,
    int head_dim,
    int tile_rows) {
    constexpr uint32_t kRank = 4;
    // Q has contiguous [B, H, S, D] storage. Split D into 64 BF16 elements
    // (128 B) so the TensorMap can use the maximum 128 B shared-memory swizzle.
    uint64_t global_dim[kRank] = {
        64,
        static_cast<uint64_t>(sequence_len),
        static_cast<uint64_t>(head_dim / 64),
        static_cast<uint64_t>(flattened_head_count),
    };
    uint64_t global_strides[kRank - 1] = {
        static_cast<uint64_t>(head_dim) * sizeof(__nv_bfloat16),
        128,
        static_cast<uint64_t>(sequence_len) * head_dim * sizeof(__nv_bfloat16),
    };
    uint32_t box_dim[kRank] = {64,
        static_cast<uint32_t>(tile_rows),
        static_cast<uint32_t>(head_dim / 64),
        1};
    uint32_t element_strides[kRank] = {1, 1, 1, 1};

    const CUresult result = cuTensorMapEncodeTiled(tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        kRank,
        const_cast<__nv_bfloat16 *>(ptr),
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

// Present BHSD Q/O storage as [64, G, S, D/64, B*(Hq/G)]. A packed 128-row
// tile gathers TileQ/PackSize sequence positions from each query head into
// the same shared-memory layout consumed by the existing 128x128 MMAs.
template <int PackSize>
inline cudaError_t init_packed_qo_tmap(
    CUtensorMap *tmap,
    const __nv_bfloat16 *ptr,
    int batch_query_groups,
    int sequence_len,
    int head_dim,
    int tile_rows) {
    static_assert(PackSize == 4 || PackSize == 8 || PackSize == 16 || PackSize == 32,
        "packed MQA supports 4, 8, 16, or 32 Q heads per CTA group");
    constexpr uint32_t kRank = 5;
    if (tile_rows % PackSize != 0) {
        return cudaErrorInvalidValue;
    }
    uint64_t global_dim[kRank] = {
        64,
        PackSize,
        static_cast<uint64_t>(sequence_len),
        static_cast<uint64_t>(head_dim / 64),
        static_cast<uint64_t>(batch_query_groups),
    };
    uint64_t global_strides[kRank - 1] = {
        static_cast<uint64_t>(sequence_len) * head_dim * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(head_dim) * sizeof(__nv_bfloat16),
        128,
        static_cast<uint64_t>(PackSize) * sequence_len * head_dim *
            sizeof(__nv_bfloat16),
    };
    uint32_t box_dim[kRank] = {
        64,
        PackSize,
        static_cast<uint32_t>(tile_rows / PackSize),
        static_cast<uint32_t>(head_dim / 64),
        1,
    };
    uint32_t element_strides[kRank] = {1, 1, 1, 1, 1};
    const CUresult result = cuTensorMapEncodeTiled(tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        kRank,
        const_cast<__nv_bfloat16 *>(ptr),
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

inline cudaError_t init_qkv_tmaps(
    CUtensorMap *q_tmap,
    CUtensorMap *k_tmap,
    CUtensorMap *v_tmap,
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    int q_batch_heads,
    int kv_batch_heads,
    int query_len,
    int key_len,
    int head_dim,
    int tile_q,
    int tile_kv) {
    cudaError_t status = init_qkv_tmap(q_tmap, q, q_batch_heads, query_len, head_dim, tile_q);
    if (status != cudaSuccess) {
        return status;
    }
    status = init_qkv_tmap(k_tmap, k, kv_batch_heads, key_len, head_dim, tile_kv);
    if (status != cudaSuccess) {
        return status;
    }
    return init_qkv_tmap(v_tmap, v, kv_batch_heads, key_len, head_dim, tile_kv);
}

// A CTA owns two independent logical Q tiles. In packed mode each tile folds
// PackSize query heads into its row dimension; otherwise one CTA owns one query
// head. Every task maps to the batch's single physical K/V head.
template <int TileQ, int TileKV, int QSlots, int KvStages, int HeadDim, bool PackMqa>
__global__ __launch_bounds__(512, 1) void fa_mqa_kernel_wg16(
    const __grid_constant__ CUtensorMap q_tmap,
    const __grid_constant__ CUtensorMap k_tmap,
    const __grid_constant__ CUtensorMap v_tmap,
    const __grid_constant__ CUtensorMap out_tmap,
    int batch,
    int heads,
    int query_len,
    int key_len,
    float scale,
    bool causal) {
    static_assert(TileQ == 128, "the FA MQA kernel currently supports TileQ=128 only");
    static_assert(TileKV > 0, "attention tiles must be positive");
    static_assert(QSlots == 1 || QSlots == 2,
        "the FA MQA mainloop supports one or two Q slots");
    static_assert(KvStages == 2, "the K/V mainloop currently uses two stages");
    static_assert(HeadDim == 128, "the FA MQA kernel currently supports HeadDim=128 only");

    // P and alpha use exp2((raw_score - raw_m) * scale * log2(e)).
    const float scale_log2e = scale * 1.4426950408889634f;

    using SmemLayout = FaMqaSmemLayout<TileQ, TileKV, QSlots, KvStages>;
    extern __shared__ __align__(1024) char smem[];
    const SmemLayout smem_layout = SmemLayout::make(HeadDim);
    auto *q_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.q_offset);
    // K and V intentionally alias.  The K producer may reuse a stage only
    // after both PV consumers commit mma2; the V producer overwrites K only
    // after both QK consumers commit mma1.
    auto *kv_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.kv_offset);
    auto *out_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.out_offset);

    // Producer/consumer graph for each (Q slot, KV stage):
    //   K TMA -> QK MMA -> softmax -> O rescale -> PV MMA -> recycle K/V stage.
    // Each resident Q tile has an independent prologue mbarrier; all remaining
    // edges need one independent barrier per two-stage-ring entry.
    __shared__ __align__(8) uint64_t q_prologue_mbars[QSlots];
    __shared__ __align__(8) uint64_t tma1_mbars[KvStages];
    __shared__ __align__(8) uint64_t tma2_mbars[KvStages];
    // Every QK/PV completion contributes directly to this CTA's stage-reuse
    // barrier. K/V may be overwritten only after both resident Q slots have
    // completed their corresponding MMA.
    __shared__ __align__(8) uint64_t k_ready_mbars[KvStages];
    __shared__ __align__(8) uint64_t v_ready_mbars[KvStages];
    __shared__ __align__(8) uint64_t mma1_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t mma2_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t softmax_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t rescale_mbars[QSlots][KvStages];
    // P-store completion tokens for one (slot, stage).  The stats barrier
    // above is released before the P stores so the rescale WG overlaps them,
    // so the MMA warp needs these separate P tokens before issuing PV: the
    // partial token covers P columns 0-63 (enough for the first PV half),
    // the full token all 128 columns.
    __shared__ __align__(8) uint64_t p_partial_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t p_full_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t epilogue_mbar[1];
    __shared__ __align__(8) uint64_t out_store_ready_mbars[QSlots];
    __shared__ __align__(8) uint64_t tmem_release_mbar[1];
    __shared__ __align__(4) int mma_tmem_addr[1];
    // The softmax WGs retain m/l in registers and publish only the values
    // needed by the separate rescale/epilogue WG.  A slot cannot be
    // overwritten until its preceding PV has committed mma2.
    __shared__ __align__(16) float softmax_alpha[QSlots][TileQ];
    __shared__ __align__(16) float softmax_l[QSlots][TileQ];

    constexpr int kWarpSize = 32;
    // CTA role assignment (16 warps = 512 threads). WG0 has one load-TMA
    // warp, one store-TMA warp, one MMA issuer, and one deliberately idle
    // warp. The two softmax WGs own one Q tile each; WG3 performs O rescale
    // and stages the final epilogue for the store warp.
    constexpr int kTmaLoadWarp = 0;  // Q/K/V shared-memory producer
    constexpr int kTmaStoreWarp = 1; // O global-memory producer
    constexpr int kMmaWarp = 2;      // QK and PV tcgen05 issuer
    constexpr int kIdleWarp = 3;
    constexpr int kWarpsPerGroup = TileQ / kWarpSize;
    constexpr int kSoftmaxWarpBegin = 4;
    constexpr int kSoftmaxWarpEnd = kSoftmaxWarpBegin + QSlots * kWarpsPerGroup;
    constexpr int kEpilogueWarpBegin = kSoftmaxWarpEnd;
    constexpr int kEpilogueWarpEnd = kEpilogueWarpBegin + kWarpsPerGroup;
    constexpr int kTmaMbarArrivalCount = 1;
    constexpr int kMmaMbarArrivalCount = 1;
    // A 128-row TMEM tile is partitioned into four 32-row regions.  Both
    // hand-offs therefore receive one arrival from each warp in their WG.
    constexpr int kSoftmaxMbarArrivalCount = kWarpsPerGroup;
    constexpr int kRescaleMbarArrivalCount = kWarpsPerGroup;
    constexpr int kTmemReleaseMbarArrivalCount = kWarpsPerGroup;
    constexpr int kQPrologueMbarArrivalCount = 1;
    constexpr int kOutStoreReadyMbarArrivalCount = kWarpsPerGroup;

    const int tid = threadIdx.x;
    const int warp_idx = threadIdx.x / kWarpSize;
    const int lane_idx = threadIdx.x % kWarpSize;

    const int q_prologue_mbar_base = static_cast<int>(__cvta_generic_to_shared(q_prologue_mbars));
    const int tma1_mbar_base = static_cast<int>(__cvta_generic_to_shared(tma1_mbars));
    const int tma2_mbar_base = static_cast<int>(__cvta_generic_to_shared(tma2_mbars));
    const int k_ready_mbar_base =
        static_cast<int>(__cvta_generic_to_shared(k_ready_mbars));
    const int v_ready_mbar_base =
        static_cast<int>(__cvta_generic_to_shared(v_ready_mbars));
    const int mma1_mbar_base = static_cast<int>(__cvta_generic_to_shared(mma1_mbars));
    const int mma2_mbar_base = static_cast<int>(__cvta_generic_to_shared(mma2_mbars));
    const int softmax_mbar_base = static_cast<int>(__cvta_generic_to_shared(softmax_mbars));
    const int rescale_mbar_base = static_cast<int>(__cvta_generic_to_shared(rescale_mbars));
    const int p_partial_mbar_base = static_cast<int>(__cvta_generic_to_shared(p_partial_mbars));
    const int p_full_mbar_base = static_cast<int>(__cvta_generic_to_shared(p_full_mbars));
    const int epilogue_mbar_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_mbar));
    const int out_store_ready_mbar_base =
        static_cast<int>(__cvta_generic_to_shared(out_store_ready_mbars));
    const int tmem_release_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(tmem_release_mbar));

    // Follow the BF16 GEMM setup pattern: one thread initializes every
    // barrier slot, publishes the initialization, then the CTA synchronizes.
    if (tid == 0) {
        for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
            mbarrier_init(q_prologue_mbar_base + q_slot * sizeof(uint64_t),
                kQPrologueMbarArrivalCount);
        }
        mbarrier_init(epilogue_mbar_addr, kMmaMbarArrivalCount);
        mbarrier_init(tmem_release_mbar_addr, kTmemReleaseMbarArrivalCount);
        for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
            mbarrier_init(out_store_ready_mbar_base + q_slot * sizeof(uint64_t),
                kOutStoreReadyMbarArrivalCount);
        }
        for (int stage = 0; stage < KvStages; ++stage) {
            const int offset = stage * sizeof(uint64_t);
            mbarrier_init(tma1_mbar_base + offset, kTmaMbarArrivalCount);
            mbarrier_init(tma2_mbar_base + offset, kTmaMbarArrivalCount);
            mbarrier_init(k_ready_mbar_base + offset, QSlots);
            mbarrier_init(v_ready_mbar_base + offset, QSlots);
            for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                const int q_offset = (q_slot * KvStages + stage) * sizeof(uint64_t);
                mbarrier_init(mma1_mbar_base + q_offset, kMmaMbarArrivalCount);
                mbarrier_init(mma2_mbar_base + q_offset, kMmaMbarArrivalCount);
                mbarrier_init(softmax_mbar_base + q_offset, kSoftmaxMbarArrivalCount);
                mbarrier_init(rescale_mbar_base + q_offset, kRescaleMbarArrivalCount);
                mbarrier_init(p_partial_mbar_base + q_offset, kSoftmaxMbarArrivalCount);
                mbarrier_init(p_full_mbar_base + q_offset, kSoftmaxMbarArrivalCount);
            }
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp_idx == kTmaLoadWarp && lane_idx == 0) {
        prefetch_tensormap(&q_tmap);
        prefetch_tensormap(&k_tmap);
        prefetch_tensormap(&v_tmap);
    }
    if (warp_idx == kTmaStoreWarp && lane_idx == 0) {
        prefetch_tensormap(&out_tmap);
    }
    __syncthreads();

    // S0/S1 are temporary score/probability tiles. O0/O1 are the two long-lived
    // unnormalized accumulators, one per resident Q slot.
    // tcgen05 allocation units are power-of-two column counts in [32, 512].
    constexpr int kTmemColumnsPerBank = 128;
    constexpr int kSBankBase = 0;
    constexpr int kOBankBase = QSlots * kTmemColumnsPerBank;
    // One S/O bank pair per resident Q slot.
    constexpr int mma_tmem_columns_per_stage = kTmemColumnsPerBank;
    constexpr int mma_tmem_columns = 2 * QSlots * kTmemColumnsPerBank;
    if (warp_idx == kMmaWarp) {
        const int mma_tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(mma_tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
                         mma_tmem_smem_addr),
            "r"(mma_tmem_columns));
        // This is the CTA's only TMEM allocation. Release the allocation
        // permit immediately so CTAs in later scheduling waves can allocate;
        // the allocation itself remains live until tcgen05.dealloc below.
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();

    // K/V share one stage ring; Q slots have fixed shared-memory and TMEM banks.
    const int q_smem_addr = static_cast<int>(__cvta_generic_to_shared(q_smem));
    const int kv_smem_addr = static_cast<int>(__cvta_generic_to_shared(kv_smem));
    const int out_smem_addr = static_cast<int>(__cvta_generic_to_shared(out_smem));
    const int tmem_addr = mma_tmem_addr[0];

    // Launch assigns exactly one adjacent logical Q-tile pair to each CTA.
    // Unpacked tasks index global query heads; packed tasks index global
    // packed query-head groups. All tasks in one batch share its K/V head.
    const int logical_query_len = query_len * (PackMqa ? kMqaPackSize : 1);
    const int q_tile_count = (logical_query_len + TileQ - 1) / TileQ;
    const int q_pair_count = (q_tile_count + QSlots - 1) / QSlots;
    const int task_id = blockIdx.x;
    const int kv_tile_count = (key_len + TileKV - 1) / TileKV;
    const int task_bh = task_id / q_pair_count;
    const int q_pair_idx = task_id - task_bh * q_pair_count;
    const int query_groups = heads / kMqaPackSize;
    const int tasks_per_batch = PackMqa ? query_groups : heads;
    const int batch_idx = task_bh / tasks_per_batch;
    const int bh = PackMqa ? 0 : task_bh;
    const int qo_bh = PackMqa ? task_bh : bh;
    const int kv_bh = batch_idx;
    int q_tile_start[QSlots];
#pragma unroll
    for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
        q_tile_start[q_slot] = (q_pair_idx * QSlots + q_slot) * TileQ;
    }
    const int logical_q_pair_end = q_tile_start[QSlots - 1] + TileQ < logical_query_len
        ? q_tile_start[QSlots - 1] + TileQ
        : logical_query_len;
    const int q_pair_end = PackMqa
        ? (logical_q_pair_end + kMqaPackSize - 1) / kMqaPackSize
        : logical_q_pair_end;
    // Bottom-right causal alignment (the FA convention): query row r
    // attends keys <= r + (key_len - query_len).
    const int causal_offset = key_len - query_len;
    const int causal_kv_tile_end =
        causal ? (q_pair_end + causal_offset + TileKV - 1) / TileKV : kv_tile_count;
    const int task_kv_tile_count =
        causal_kv_tile_end < kv_tile_count ? causal_kv_tile_end : kv_tile_count;
    const int softmax_warp = warp_idx >= kSoftmaxWarpBegin && warp_idx < kSoftmaxWarpEnd
                                 ? warp_idx - kSoftmaxWarpBegin
                                 : -1;
    const int epilogue_warp = warp_idx >= kEpilogueWarpBegin && warp_idx < kEpilogueWarpEnd
                                  ? warp_idx - kEpilogueWarpBegin
                                  : -1;

    if (softmax_warp >= 0) {
        // Register budget across the four WGs is 32 + 192 + 192 + 96 = 512
        // registers/thread-equivalents, exactly one 64K-register CTA budget.
        setmaxnreg_increase<192>();
        // Each softmax WG owns one Q tile and only touches its S/P TMEM bank.
        const int q_slot = softmax_warp / kWarpsPerGroup;
        const int row_base = (softmax_warp % kWarpsPerGroup) * kWarpSize;
        const int row = row_base + lane_idx;
        const int q_row = q_tile_start[q_slot] + row;
        const int q_sequence_row = PackMqa ? q_row / kMqaPackSize : q_row;
        float tilel = 0.0f;
        float tilem = -__int_as_float(0x7F800000);
        int softmax_mma1_phase[KvStages] = {};
        // The paper's threshold is log2(256) = 8.  m is stored in the
        // unscaled QK domain; its delta is multiplied by scale before
        // this ln(256) = 8 * ln(2) comparison.
        constexpr float kConditionalRescaleThresholdLn = 5.5451774444795623f;
        for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
            const int stage = kv_tile % KvStages;
            if (kv_tile > 0) {
                const int previous_tile = kv_tile - 1;
                const int previous_stage = previous_tile % KvStages;
                const int previous_phase = (previous_tile / KvStages) & 1;
                const int previous_mma2_addr =
                    mma2_mbar_base + (q_slot * KvStages + previous_stage) * sizeof(uint64_t);
                mbarrier_wait(previous_mma2_addr, previous_phase);
            }
            const int mma1_addr = mma1_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
            mbarrier_wait(mma1_addr, softmax_mma1_phase[stage]);
            softmax_mma1_phase[stage] ^= 1;
            tcgen05_fence_after_thread_sync();

            const bool valid_q_row = q_row < logical_query_len;
            const int s_tmem_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
            const int mask_q_row = causal ? q_sequence_row + causal_offset : key_len - 1;
            // Select uniformly across the warp: the earliest row must see
            // the entire KV tile and all 32 query rows must be in bounds.
            const bool all_scores_valid =
                q_tile_start[q_slot] + row_base + kWarpSize <= logical_query_len &&
                (kv_tile + 1) * TileKV <= key_len &&
                (!causal || (kv_tile + 1) * TileKV - 1 <=
                    (PackMqa ? (q_tile_start[q_slot] + row_base) / kMqaPackSize
                             : q_tile_start[q_slot] + row_base) + causal_offset);
            const SoftmaxUpdate softmax = all_scores_valid
                ? load_s_and_store_p_chunked<true>(
                    s_tmem_addr + (row_base << 16), kv_tile * TileKV, key_len,
                    mask_q_row, valid_q_row, scale, tilem, tilel,
                    kConditionalRescaleThresholdLn, scale_log2e,
                    p_partial_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t),
                    lane_idx)
                : load_s_and_store_p_chunked<false>(
                s_tmem_addr + (row_base << 16),
                kv_tile * TileKV,
                key_len,
                mask_q_row,
                valid_q_row,
                scale,
                tilem,
                tilel,
                kConditionalRescaleThresholdLn,
                scale_log2e,
                p_partial_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t),
                lane_idx);
            tilem = softmax.m;
            tilel = softmax.l;
            // Publish the stats immediately: the epilogue WG's O rescale
            // depends only on alpha/l, so it overlaps this WG's remaining
            // exp2/P-store work instead of serializing behind it.
            softmax_alpha[q_slot][row] = softmax.alpha;
            softmax_l[q_slot][row] = tilel;
            __syncwarp();
            if (lane_idx == 0) {
                mbarrier_arrive(
                    softmax_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t));
            }
            // Complete all P stores before releasing the tile's P to the
            // MMA warp on the separate p-full barrier.
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            tcgen05_fence_before_thread_sync();
            if (lane_idx == 0) {
                mbarrier_arrive(
                    p_full_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t));
            }
        }
        return;
    }

    if (epilogue_warp >= 0) {
        // This WG needs only the 32-value O fragment plus scalar loop state.
        // It donates registers before either softmax WG's increase can finish.
        setmaxnreg_decrease<96>();
        int softmax_phase[QSlots][KvStages] = {};
        const int row_base = epilogue_warp * kWarpSize;
        const int row = row_base + lane_idx;
        float final_l[QSlots] = {};

        for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
            const int stage = kv_tile % KvStages;
            for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                const int softmax_addr =
                    softmax_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                mbarrier_wait(softmax_addr, softmax_phase[q_slot][stage]);
                softmax_phase[q_slot][stage] ^= 1;
                tcgen05_fence_after_thread_sync();

                const float alpha = softmax_alpha[q_slot][row];
                final_l[q_slot] = softmax_l[q_slot][row];
                // The sign bit (including -0) denotes the conditional-skip path.
                // Omit the whole TMEM RMW when no row in this warp needs it.
                // Invalid rows (and underflowed alpha) can be zero even
                // when another lane requests correction. TMEM operations
                // must still be executed uniformly by the entire warp.
                const bool warp_needs_rescale = __any_sync(
                    0xFFFFFFFFU, (__float_as_uint(alpha) & 0x80000000U) == 0);
                if (kv_tile != 0 && warp_needs_rescale) {
                    const int o_tmem_addr =
                        tmem_addr + kOBankBase + q_slot * mma_tmem_columns_per_stage;
                    // Four x32 round trips instead of sixteen x8 ones; the
                    // trailing wait::st below is the only store fence.
                    for (int d32 = 0; d32 < HeadDim / 32; ++d32) {
                        float o_fragment[32];
                        const int addr = o_tmem_addr + (row_base << 16) + d32 * 32;
                        tcgen05_ld_f32_x32(o_fragment, addr);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
                        for (int e = 0; e < 32; ++e) {
                            o_fragment[e] *= alpha;
                        }
                        tcgen05_st_f32_x32(o_fragment, addr);
                    }
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                tcgen05_fence_before_thread_sync();
                if (lane_idx == 0) {
                    mbarrier_arrive(
                        rescale_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t));
                }
            }
        }

        mbarrier_wait(epilogue_mbar_addr, 0);
        tcgen05_fence_after_thread_sync();
        for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
            const int q_row = q_tile_start[q_slot] + row;
            const bool valid_q_row = q_row < logical_query_len;
            const float inv_l = valid_q_row ? 1.0f / final_l[q_slot] : 0.0f;
            const int o_tmem_addr =
                tmem_addr + kOBankBase + q_slot * mma_tmem_columns_per_stage;
            for (int d8 = 0; d8 < HeadDim / 8; ++d8) {
                float o_fragment[8];
                tcgen05_ld_f32_x8(o_fragment, o_tmem_addr + (row_base << 16) + d8 * 8);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (valid_q_row) {
                    const int d = d8 * 8;
                    // Match the SW128 tensor-map layout: [D/64, Q, 64].
                    // TMA XORs row bits [9:7] into byte-address bits
                    // [6:4] inside each 1 KiB swizzle atom.
                    const int logical_byte = ((d / 64) * TileQ * 64 + row * 64 + d % 64) *
                                             sizeof(__nv_bfloat16);
                    const int swizzled_byte = logical_byte ^ ((logical_byte >> 3) & 0x70);
                    auto *out_tile_bytes = reinterpret_cast<char *>(out_smem) +
                                           q_slot * TileQ * HeadDim * sizeof(__nv_bfloat16);
                    // Eight adjacent BF16 values fill one aligned SW128
                    // 16-byte atom. Store it together instead of eight
                    // conflicting scalar halfword instructions.
                    const uint4 packed = make_uint4(
                        pack_bf16x2(o_fragment[0] * inv_l, o_fragment[1] * inv_l),
                        pack_bf16x2(o_fragment[2] * inv_l, o_fragment[3] * inv_l),
                        pack_bf16x2(o_fragment[4] * inv_l, o_fragment[5] * inv_l),
                        pack_bf16x2(o_fragment[6] * inv_l, o_fragment[7] * inv_l));
                    *reinterpret_cast<uint4 *>(out_tile_bytes + swizzled_byte) = packed;
                }
            }
            // Let the store warp drain Q0 while this WG stages Q1.
            if (lane_idx == 0) {
                mbarrier_arrive(out_store_ready_mbar_base + q_slot * sizeof(uint64_t));
            }
        }
        // Protect TMEM deallocation until all epilogue reads finish.
        tcgen05_fence_before_thread_sync();
        if (lane_idx == 0) {
            mbarrier_arrive(tmem_release_mbar_addr);
        }
        return;
    }

    // WG0 releases registers before its three driver roles diverge. Warp 3
    // intentionally stays idle, but participates in this warpgroup-uniform
    // redistribution instruction.
    setmaxnreg_decrease<32>();

    if (warp_idx == kIdleWarp) {
        return;
    }

    // Every CTA loads its private Q and K/V tiles.
    if (warp_idx == kTmaLoadWarp) {
        if (lane_idx == 0) {
            int k_ready_phase[KvStages] = {};
            int v_ready_phase[KvStages] = {};
            const int q_transaction_bytes = TileQ * HeadDim * sizeof(__nv_bfloat16);
            for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                const int q_mbar_addr = q_prologue_mbar_base + q_slot * sizeof(uint64_t);
                mbarrier_expect_tx(q_mbar_addr, q_transaction_bytes);
                if constexpr (PackMqa) {
                    tma_5d_gmem2smem(
                        q_smem_addr + q_slot * TileQ * HeadDim * sizeof(__nv_bfloat16),
                        &q_tmap, 0, 0, q_tile_start[q_slot] / kMqaPackSize, 0, qo_bh,
                        q_mbar_addr);
                } else {
                    tma_4d_gmem2smem(
                        q_smem_addr + q_slot * TileQ * HeadDim * sizeof(__nv_bfloat16),
                        &q_tmap, 0, q_tile_start[q_slot], 0, bh, q_mbar_addr);
                }
            }
            // Fill both K stages before waiting for QK. This lets K(i + 1)
            // transfer while the tensor core and softmax pipeline consume i.
            const int k_prologue_tiles =
                task_kv_tile_count < KvStages ? task_kv_tile_count : KvStages;
            const int transaction_bytes = TileKV * HeadDim * sizeof(__nv_bfloat16);
            for (int kv_tile = 0; kv_tile < k_prologue_tiles; ++kv_tile) {
                const int stage = kv_tile;
                const int k_tx_addr = tma1_mbar_base + stage * sizeof(uint64_t);
                mbarrier_expect_tx(k_tx_addr, transaction_bytes);
                tma_4d_gmem2smem(
                    kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16),
                    &k_tmap, 0, kv_tile * TileKV, 0, kv_bh, k_tx_addr);
            }

            for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                const int stage = kv_tile % KvStages;
                const int v_tx_addr = tma2_mbar_base + stage * sizeof(uint64_t);
                mbarrier_wait(k_ready_mbar_base + stage * sizeof(uint64_t),
                    k_ready_phase[stage]);
                k_ready_phase[stage] ^= 1;
                mbarrier_expect_tx(v_tx_addr, transaction_bytes);
                tma_4d_gmem2smem(
                    kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16),
                    &v_tmap, 0, kv_tile * TileKV, 0, kv_bh, v_tx_addr);

                // Once both PV consumers release V(i), immediately refill
                // that stage with K(i + 2). The other K stage was prefetched
                // already, so the MMA issuer need not wait on this recycle.
                const int next_k_tile = kv_tile + KvStages;
                if (next_k_tile < task_kv_tile_count) {
                    const int next_k_tx_addr = tma1_mbar_base + stage * sizeof(uint64_t);
                    mbarrier_wait(v_ready_mbar_base + stage * sizeof(uint64_t),
                        v_ready_phase[stage]);
                    v_ready_phase[stage] ^= 1;
                    mbarrier_expect_tx(next_k_tx_addr, transaction_bytes);
                    tma_4d_gmem2smem(
                        kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16),
                        &k_tmap, 0, next_k_tile * TileKV, 0, kv_bh, next_k_tx_addr);
                }
            }
        }
        return;
    }

    if (warp_idx == kTmaStoreWarp) {
        if (lane_idx == 0) {
            for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                mbarrier_wait(out_store_ready_mbar_base + q_slot * sizeof(uint64_t), 0);
                if constexpr (PackMqa) {
                    tma_5d_smem2gmem(
                        out_smem_addr + q_slot * TileQ * HeadDim * sizeof(__nv_bfloat16),
                        &out_tmap, 0, 0, q_tile_start[q_slot] / kMqaPackSize, 0, qo_bh);
                } else {
                    tma_4d_smem2gmem(
                        out_smem_addr + q_slot * TileQ * HeadDim * sizeof(__nv_bfloat16),
                        &out_tmap, 0, q_tile_start[q_slot], 0, bh);
                }
                tma_store_commit_group();
            }
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
        return;
    }

    // A single elected thread issues tcgen05 operations.  The asynchronous
    // engines perform the actual matrix work while the auxiliary WGs run.
    if (warp_idx == kMmaWarp) {
        int qk_tma1_phase[KvStages] = {};
        int pv_tma2_phase[KvStages] = {};
        int pv_rescale_phase[QSlots][KvStages] = {};
        int pv_ppartial_phase[QSlots][KvStages] = {};
        int pv_pfull_phase[QSlots][KvStages] = {};
        // Q TMA completion replaces the former CTA-wide post-prologue barrier.
        for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
            mbarrier_wait(q_prologue_mbar_base + q_slot * sizeof(uint64_t), 0);
        }
        if (elect_sync()) {
            constexpr uint32_t kQkInstrDesc = (1U << 4U) | (1U << 7U) | (1U << 10U) |
                                              ((uint32_t)TileKV >> 3U << 17U) |
                                              ((uint32_t)TileQ >> 4U << 24U);
            const uint32_t pv_instr_desc = (1U << 4U) | (1U << 7U) | (1U << 10U) | (1U << 16U) |
                                           ((uint32_t)HeadDim >> 3U << 17U) |
                                           ((uint32_t)TileQ >> 4U << 24U);
            if constexpr (QSlots == 1) {
                // Decode specialization: one packed Q tile is resident, so
                // each KV iteration is simply Q0(i) followed by P0(i).
                for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                    constexpr int q_slot = 0;
                    const int stage = kv_tile % KvStages;
                    mbarrier_wait(
                        tma1_mbar_base + stage * sizeof(uint64_t), qk_tma1_phase[stage]);
                    qk_tma1_phase[stage] ^= 1;
                    tcgen05_fence_after_thread_sync();

                    const int q_addr = q_smem_addr;
                    const int k_addr =
                        kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16);
                    const int s_addr = tmem_addr + kSBankBase;
                    for (int d64 = 0; d64 < HeadDim / 64; ++d64)
                        for (int d16 = 0; d16 < 4; ++d16) {
                            tcgen05_mma_bf16(s_addr,
                                make_128b_swizzle_desc(
                                    q_addr + d64 * TileQ * 128 + d16 * 32),
                                make_128b_swizzle_desc(
                                    k_addr + d64 * TileKV * 128 + d16 * 32),
                                kQkInstrDesc,
                                d64 != 0 || d16 != 0);
                        }
                    tcgen05_commit_stage_reuse(
                        mma1_mbar_base + stage * sizeof(uint64_t),
                        k_ready_mbar_base + stage * sizeof(uint64_t));

                    mbarrier_wait(
                        tma2_mbar_base + stage * sizeof(uint64_t), pv_tma2_phase[stage]);
                    pv_tma2_phase[stage] ^= 1;
                    const int rescale_addr =
                        rescale_mbar_base + stage * sizeof(uint64_t);
                    mbarrier_wait(rescale_addr, pv_rescale_phase[q_slot][stage]);
                    pv_rescale_phase[q_slot][stage] ^= 1;
                    const int ppartial_addr =
                        p_partial_mbar_base + stage * sizeof(uint64_t);
                    mbarrier_wait(ppartial_addr, pv_ppartial_phase[q_slot][stage]);
                    pv_ppartial_phase[q_slot][stage] ^= 1;
                    tcgen05_fence_after_thread_sync();

                    const int o_addr = tmem_addr + kOBankBase;
                    const int v_addr =
                        kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16);
                    constexpr int kPvHalfChunks = TileKV / 32;
                    for (int k16 = 0; k16 < kPvHalfChunks; ++k16) {
                        tcgen05_mma_tmem_a_bf16(o_addr,
                            s_addr + k16 * 8,
                            make_128b_nmajor_desc(
                                v_addr + k16 * 16 * 128, TileKV * 128),
                            pv_instr_desc,
                            kv_tile != 0 || k16 != 0);
                    }
                    const int pfull_addr = p_full_mbar_base + stage * sizeof(uint64_t);
                    mbarrier_wait(pfull_addr, pv_pfull_phase[q_slot][stage]);
                    pv_pfull_phase[q_slot][stage] ^= 1;
                    tcgen05_fence_after_thread_sync();
                    for (int k16 = kPvHalfChunks; k16 < TileKV / 16; ++k16) {
                        tcgen05_mma_tmem_a_bf16(o_addr,
                            s_addr + k16 * 8,
                            make_128b_nmajor_desc(
                                v_addr + k16 * 16 * 128, TileKV * 128),
                            pv_instr_desc,
                            kv_tile != 0 || k16 != 0);
                    }
                    tcgen05_commit_stage_reuse(
                        mma2_mbar_base + stage * sizeof(uint64_t),
                        v_ready_mbar_base + stage * sizeof(uint64_t));
                }
                tcgen05_commit_mbarrier(epilogue_mbar_addr);
            } else {
            // Keep the two Q tiles staggered around the PV work.  At iteration
            // i the tensor-core issue order is Q0(i), P1(i - 1), Q1(i), P0(i).
            // P1(0) has no predecessor and is deliberately omitted.  This
            // gives the Q0 rescale work the intervening P1 issue window
            // before its PV is consumed, while P1(i) is flushed at i + 1.
            for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                const int stage = kv_tile % KvStages;
                mbarrier_wait(tma1_mbar_base + stage * sizeof(uint64_t), qk_tma1_phase[stage]);
                qk_tma1_phase[stage] ^= 1;
                tcgen05_fence_after_thread_sync();
                const int k_addr = kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16);

                // Q0(i).
                {
                    constexpr int q_slot = 0;
                    const int s_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int q_addr =
                        q_smem_addr + q_slot * TileQ * HeadDim * sizeof(__nv_bfloat16);
                    for (int d64 = 0; d64 < HeadDim / 64; ++d64)
                        for (int d16 = 0; d16 < 4; ++d16) {
                            tcgen05_mma_bf16(s_addr,
                                make_128b_swizzle_desc(q_addr + d64 * TileQ * 128 + d16 * 32),
                                make_128b_swizzle_desc(k_addr + d64 * TileKV * 128 + d16 * 32),
                                kQkInstrDesc,
                                d64 != 0 || d16 != 0);
                        }
                    tcgen05_commit_stage_reuse(
                        mma1_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t),
                        k_ready_mbar_base + stage * sizeof(uint64_t));
                }

                // P1(i - 1).  There is intentionally no P1 in the prologue.
                if (kv_tile > 0) {
                    const int pv_tile = kv_tile - 1;
                    const int pv_stage = pv_tile % KvStages;
                    constexpr int q_slot = 1;
                    // P0(pv_tile) already waited for this tile's V TMA.  Do
                    // not consume its phase again here: V cannot be recycled
                    // until this P1 commits mma2 below.
                    const int rescale_addr =
                        rescale_mbar_base + (q_slot * KvStages + pv_stage) * sizeof(uint64_t);
                    mbarrier_wait(rescale_addr, pv_rescale_phase[q_slot][pv_stage]);
                    pv_rescale_phase[q_slot][pv_stage] ^= 1;
                    const int ppartial_addr =
                        p_partial_mbar_base + (q_slot * KvStages + pv_stage) * sizeof(uint64_t);
                    mbarrier_wait(ppartial_addr, pv_ppartial_phase[q_slot][pv_stage]);
                    pv_ppartial_phase[q_slot][pv_stage] ^= 1;
                    tcgen05_fence_after_thread_sync();
                    const int o_addr = tmem_addr + kOBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int p_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int v_addr =
                        kv_smem_addr + pv_stage * TileKV * HeadDim * sizeof(__nv_bfloat16);
                    // The P half over columns 0-63 releases before the tail
                    // (FA4's partial-P scheme), so its PV k-steps issue while
                    // the softmax WG finishes the upper half.
                    constexpr int kPvHalfChunks = TileKV / 32;
                    for (int k16 = 0; k16 < kPvHalfChunks; ++k16) {
                        tcgen05_mma_tmem_a_bf16(o_addr,
                            p_addr + k16 * 8,
                            make_128b_nmajor_desc(v_addr + k16 * 16 * 128, TileKV * 128),
                            pv_instr_desc,
                            pv_tile != 0 || k16 != 0);
                    }
                    const int pfull_addr =
                        p_full_mbar_base + (q_slot * KvStages + pv_stage) * sizeof(uint64_t);
                    mbarrier_wait(pfull_addr, pv_pfull_phase[q_slot][pv_stage]);
                    pv_pfull_phase[q_slot][pv_stage] ^= 1;
                    tcgen05_fence_after_thread_sync();
                    for (int k16 = kPvHalfChunks; k16 < TileKV / 16; ++k16) {
                        tcgen05_mma_tmem_a_bf16(o_addr,
                            p_addr + k16 * 8,
                            make_128b_nmajor_desc(v_addr + k16 * 16 * 128, TileKV * 128),
                            pv_instr_desc,
                            pv_tile != 0 || k16 != 0);
                    }
                    tcgen05_commit_stage_reuse(
                        mma2_mbar_base + (q_slot * KvStages + pv_stage) * sizeof(uint64_t),
                        v_ready_mbar_base + pv_stage * sizeof(uint64_t));
                }

                // Q1(i).
                {
                    constexpr int q_slot = 1;
                    const int s_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int q_addr =
                        q_smem_addr + q_slot * TileQ * HeadDim * sizeof(__nv_bfloat16);
                    for (int d64 = 0; d64 < HeadDim / 64; ++d64)
                        for (int d16 = 0; d16 < 4; ++d16) {
                            tcgen05_mma_bf16(s_addr,
                                make_128b_swizzle_desc(q_addr + d64 * TileQ * 128 + d16 * 32),
                                make_128b_swizzle_desc(k_addr + d64 * TileKV * 128 + d16 * 32),
                            kQkInstrDesc,
                            d64 != 0 || d16 != 0);
                        }
                    tcgen05_commit_stage_reuse(
                        mma1_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t),
                        k_ready_mbar_base + stage * sizeof(uint64_t));
                }

                // P0(i).
                {
                    constexpr int q_slot = 0;
                    mbarrier_wait(tma2_mbar_base + stage * sizeof(uint64_t), pv_tma2_phase[stage]);
                    pv_tma2_phase[stage] ^= 1;
                    const int rescale_addr =
                        rescale_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                    mbarrier_wait(rescale_addr, pv_rescale_phase[q_slot][stage]);
                    pv_rescale_phase[q_slot][stage] ^= 1;
                    const int ppartial_addr =
                        p_partial_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                    mbarrier_wait(ppartial_addr, pv_ppartial_phase[q_slot][stage]);
                    pv_ppartial_phase[q_slot][stage] ^= 1;
                    tcgen05_fence_after_thread_sync();
                    const int o_addr = tmem_addr + kOBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int p_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int v_addr =
                        kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16);
                    constexpr int kPvHalfChunks = TileKV / 32;
                    for (int k16 = 0; k16 < kPvHalfChunks; ++k16) {
                        tcgen05_mma_tmem_a_bf16(o_addr,
                            p_addr + k16 * 8,
                            make_128b_nmajor_desc(v_addr + k16 * 16 * 128, TileKV * 128),
                            pv_instr_desc,
                            kv_tile != 0 || k16 != 0);
                    }
                    const int pfull_addr =
                        p_full_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                    mbarrier_wait(pfull_addr, pv_pfull_phase[q_slot][stage]);
                    pv_pfull_phase[q_slot][stage] ^= 1;
                    tcgen05_fence_after_thread_sync();
                    for (int k16 = kPvHalfChunks; k16 < TileKV / 16; ++k16) {
                        tcgen05_mma_tmem_a_bf16(o_addr,
                            p_addr + k16 * 8,
                            make_128b_nmajor_desc(v_addr + k16 * 16 * 128, TileKV * 128),
                            pv_instr_desc,
                            kv_tile != 0 || k16 != 0);
                    }
                    tcgen05_commit_stage_reuse(
                        mma2_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t),
                        v_ready_mbar_base + stage * sizeof(uint64_t));
                }
            }

            // Drain P1 from the final tile; all middle iterations above retain
            // the requested Q0, P1, Q1, P0 tensor-core issue order.
            if (task_kv_tile_count > 0) {
                const int pv_tile = task_kv_tile_count - 1;
                const int pv_stage = pv_tile % KvStages;
                constexpr int q_slot = 1;
                // As above, P0(pv_tile) owns the one TMA-phase wait.
                const int rescale_addr =
                    rescale_mbar_base + (q_slot * KvStages + pv_stage) * sizeof(uint64_t);
                mbarrier_wait(rescale_addr, pv_rescale_phase[q_slot][pv_stage]);
                pv_rescale_phase[q_slot][pv_stage] ^= 1;
                const int ppartial_addr =
                    p_partial_mbar_base + (q_slot * KvStages + pv_stage) * sizeof(uint64_t);
                mbarrier_wait(ppartial_addr, pv_ppartial_phase[q_slot][pv_stage]);
                pv_ppartial_phase[q_slot][pv_stage] ^= 1;
                tcgen05_fence_after_thread_sync();
                const int o_addr = tmem_addr + kOBankBase + q_slot * mma_tmem_columns_per_stage;
                const int p_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
                const int v_addr =
                    kv_smem_addr + pv_stage * TileKV * HeadDim * sizeof(__nv_bfloat16);
                constexpr int kPvHalfChunks = TileKV / 32;
                for (int k16 = 0; k16 < kPvHalfChunks; ++k16) {
                    tcgen05_mma_tmem_a_bf16(o_addr,
                        p_addr + k16 * 8,
                        make_128b_nmajor_desc(v_addr + k16 * 16 * 128, TileKV * 128),
                        pv_instr_desc,
                        pv_tile != 0 || k16 != 0);
                }
                const int pfull_addr =
                    p_full_mbar_base + (q_slot * KvStages + pv_stage) * sizeof(uint64_t);
                mbarrier_wait(pfull_addr, pv_pfull_phase[q_slot][pv_stage]);
                pv_pfull_phase[q_slot][pv_stage] ^= 1;
                tcgen05_fence_after_thread_sync();
                for (int k16 = kPvHalfChunks; k16 < TileKV / 16; ++k16) {
                    tcgen05_mma_tmem_a_bf16(o_addr,
                        p_addr + k16 * 8,
                        make_128b_nmajor_desc(v_addr + k16 * 16 * 128, TileKV * 128),
                        pv_instr_desc,
                        pv_tile != 0 || k16 != 0);
                }
                tcgen05_commit_stage_reuse(
                    mma2_mbar_base + (q_slot * KvStages + pv_stage) * sizeof(uint64_t),
                    v_ready_mbar_base + pv_stage * sizeof(uint64_t));
            }
            // A fresh, non-ring completion point protects the final O from
            // parity wraparound on the recycled mma2 barriers.
            tcgen05_commit_mbarrier(epilogue_mbar_addr);
            }
        }
        mbarrier_wait(tmem_release_mbar_addr, 0);
        tcgen05_fence_after_thread_sync();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_addr),
            "r"(mma_tmem_columns));
        return;
    }
}

// Q=1 MQA decode launcher. It instantiates the shared FA mainloop with one
// resident Q slot: WG0 drives TMA/MMA, WG1 performs softmax, and WG2 handles
// rescale and the epilogue.
cudaError_t bf16_fa4_mqa_decode_cuda(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    __nv_bfloat16 *out,
    int batch,
    int heads,
    int query_len,
    int key_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream) {
    constexpr int kTileQ = 128;
    constexpr int kTileKV = 128;
    constexpr int kQSlots = 1;
    constexpr int kKvStages = 2;
    constexpr int kHeadDim = 128;
    constexpr int kDecodeWarps = 12;

    if (query_len != 1 || head_dim != kHeadDim || heads <= 0 ||
        heads % kMqaPackSize != 0 || key_len <= 0 || batch <= 0 ||
        !(scale > 0.0f) || (causal && key_len < query_len)) {
        return cudaErrorInvalidValue;
    }

    const int query_groups = heads / kMqaPackSize;
    CUtensorMap q_tmap{};
    CUtensorMap k_tmap{};
    CUtensorMap v_tmap{};
    CUtensorMap out_tmap{};
    cudaError_t status = init_packed_qo_tmap<kMqaPackSize>(
        &q_tmap, q, batch * query_groups, query_len, head_dim, kTileQ);
    if (status == cudaSuccess) {
        status = init_qkv_tmap(
            &k_tmap, k, batch, key_len, head_dim, kTileKV);
    }
    if (status == cudaSuccess) {
        status = init_qkv_tmap(
            &v_tmap, v, batch, key_len, head_dim, kTileKV);
    }
    if (status == cudaSuccess) {
        status = init_packed_qo_tmap<kMqaPackSize>(
            &out_tmap, out, batch * query_groups, query_len, head_dim, kTileQ);
    }
    if (status != cudaSuccess) {
        return status;
    }

    using SmemLayout =
        FaMqaSmemLayout<kTileQ, kTileKV, kQSlots, kKvStages>;
    const SmemLayout smem_layout = SmemLayout::make(head_dim);
    if (smem_layout.dynamic_bytes + SmemLayout::kStaticMbarrierBytes +
            SmemLayout::kSoftmaxHandoffBytes >
        SmemLayout::kMaxSmemBytes) {
        return cudaErrorInvalidConfiguration;
    }

    status = cudaFuncSetAttribute(
        fa_mqa_kernel_wg16<
            kTileQ, kTileKV, kQSlots, kKvStages, kHeadDim, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_layout.dynamic_bytes));
    if (status != cudaSuccess) {
        return status;
    }

    const dim3 block(kDecodeWarps * 32);
    const dim3 grid(batch * query_groups);
    fa_mqa_kernel_wg16<
        kTileQ, kTileKV, kQSlots, kKvStages, kHeadDim, true>
        <<<grid, block, smem_layout.dynamic_bytes, stream>>>(
            q_tmap, k_tmap, v_tmap, out_tmap, batch, heads, query_len,
            key_len, scale, causal);
    return cudaGetLastError();
}


} // namespace

// Dispatch between the 12-warp decode and 16-warp prefill schedules.
cudaError_t bf16_fa4_mqa_cuda(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    __nv_bfloat16 *out,
    int batch,
    int heads,
    int query_len,
    int key_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream) {
    if (query_len == 1) {
        return bf16_fa4_mqa_decode_cuda(
            q, k, v, out, batch, heads, query_len, key_len, head_dim,
            scale, causal, stream);
    }
    constexpr int kTileQ = 128;
    constexpr int kTileKV = 128;
    constexpr int kQSlots = 2;
    constexpr int kKvStages = 2;
    constexpr int kHeadDim = 128;

    if (head_dim != kHeadDim || heads <= 0 || heads % kMqaPackSize != 0 ||
        query_len <= 0 || batch <= 0 ||
        key_len <= 0 || !(scale > 0.0f) ||
        (causal && key_len < query_len)) {
        return cudaErrorInvalidValue;
    }

    const int query_groups = heads / kMqaPackSize;
    CUtensorMap q_tmap{};
    CUtensorMap k_tmap{};
    CUtensorMap v_tmap{};
    CUtensorMap out_tmap{};
    cudaError_t status;
    if constexpr (kPackMqa) {
        status = init_packed_qo_tmap<kMqaPackSize>(
            &q_tmap, q, batch * query_groups, query_len, head_dim, kTileQ);
    } else {
        status = init_qkv_tmap(
            &q_tmap, q, batch * heads, query_len, head_dim, kTileQ);
    }
    if (status == cudaSuccess) {
        status = init_qkv_tmap(
            &k_tmap, k, batch, key_len, head_dim, kTileKV);
    }
    if (status == cudaSuccess) {
        status = init_qkv_tmap(
            &v_tmap, v, batch, key_len, head_dim, kTileKV);
    }
    if (status == cudaSuccess) {
        if constexpr (kPackMqa) {
            status = init_packed_qo_tmap<kMqaPackSize>(
                &out_tmap, out, batch * query_groups, query_len, head_dim, kTileQ);
        } else {
            status = init_qkv_tmap(
                &out_tmap, out, batch * heads, query_len, head_dim, kTileQ);
        }
    }
    if (status != cudaSuccess) {
        return status;
    }

    const int logical_query_len =
        query_len * (kPackMqa ? kMqaPackSize : 1);
    const int q_tile_count = (logical_query_len + kTileQ - 1) / kTileQ;
    const int q_pair_count = (q_tile_count + kQSlots - 1) / kQSlots;
    const int tasks_per_batch = kPackMqa ? query_groups : heads;
    const int total_task_count = q_pair_count * batch * tasks_per_batch;

    using SmemLayout =
        FaMqaSmemLayout<kTileQ, kTileKV, kQSlots, kKvStages>;
    const SmemLayout smem_layout = SmemLayout::make(head_dim);
    if (smem_layout.dynamic_bytes + SmemLayout::kStaticMbarrierBytes +
            SmemLayout::kSoftmaxHandoffBytes >
        SmemLayout::kMaxSmemBytes) {
        return cudaErrorInvalidConfiguration;
    }

    status = cudaFuncSetAttribute(
        fa_mqa_kernel_wg16<
            kTileQ, kTileKV, kQSlots, kKvStages, kHeadDim, kPackMqa>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_layout.dynamic_bytes));
    if (status != cudaSuccess) {
        return status;
    }

    const dim3 block(16 * 32);
    const dim3 grid(total_task_count);
    fa_mqa_kernel_wg16<
        kTileQ, kTileKV, kQSlots, kKvStages, kHeadDim, kPackMqa>
        <<<grid, block, smem_layout.dynamic_bytes, stream>>>(
            q_tmap, k_tmap, v_tmap, out_tmap, batch, heads, query_len,
            key_len, scale, causal);
    return cudaGetLastError();
}

extern "C" cudaError_t sm110a_bf16_fa4_mqa_launch(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    __nv_bfloat16 *out,
    int batch,
    int heads,
    int query_len,
    int key_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream) {
    return bf16_fa4_mqa_cuda(
        q, k, v, out, batch, heads, query_len, key_len, head_dim, scale,
        causal, stream);
}
