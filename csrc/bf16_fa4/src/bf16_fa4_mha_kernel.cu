#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

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
                 "FA_MHA_MBAR_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni FA_MHA_MBAR_DONE;\n\t"
                 "bra.uni FA_MHA_MBAR_WAIT;\n\t"
                 "FA_MHA_MBAR_DONE:\n\t"
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

// Keep the full S row in a leaf register frame from the single TMEM load
// through masked softmax and the packed BF16 P stores.
struct SoftmaxUpdate {
    // Running maximum in the unscaled QK domain.
    float m;
    float l;
    float alpha;
};

__device__ __attribute__((noinline)) SoftmaxUpdate load_s_and_store_p_ptx(
    int s_tmem_row_addr,
    int tile_key_begin,
    int key_len,
    int q_row,
    int valid_q_row,
    float scale,
    float tilem,
    float tilel,
    float conditional_rescale_threshold,
    int all_scores_valid,
    float scale_log2e) {
    SoftmaxUpdate update;
    asm volatile(
        "{\n\t"
        ".reg .pred p, pvalid, prescale, pfast;\n\t"
        ".reg .s32 key, addr;\n\t"
        ".reg .f32 score<128>, rmax, v, sum, neg_m_scaled;\n\t"
        ".reg .b32 packed<4>;\n\t"
        "tcgen05.ld.red.sync.aligned.32x32b.x128.max.f32 {score0, score1, score2, score3, score4, "
        "score5, score6, score7, score8, score9, score10, score11, score12, score13, score14, "
        "score15, score16, score17, score18, score19, score20, score21, score22, score23, score24, "
        "score25, score26, score27, score28, score29, score30, score31, score32, score33, score34, "
        "score35, score36, score37, score38, score39, score40, score41, score42, score43, score44, "
        "score45, score46, score47, score48, score49, score50, score51, score52, score53, score54, "
        "score55, score56, score57, score58, score59, score60, score61, score62, score63, score64, "
        "score65, score66, score67, score68, score69, score70, score71, score72, score73, score74, "
        "score75, score76, score77, score78, score79, score80, score81, score82, score83, score84, "
        "score85, score86, score87, score88, score89, score90, score91, score92, score93, score94, "
        "score95, score96, score97, score98, score99, score100, score101, score102, score103, "
        "score104, score105, score106, score107, score108, score109, score110, score111, score112, "
        "score113, score114, score115, score116, score117, score118, score119, score120, score121, "
        "score122, score123, score124, score125, score126, score127}, rmax, [%3];\n\t"
        "tcgen05.wait::ld.sync.aligned;\n\t"
        "setp.ne.s32 pvalid, %7, 0;\n\t"
        // `redVal` is the max of the 128 raw S values for this lane.  It is
        // usable directly only when no causal/tail mask is needed; masked
        // rows must retain the scalar reduction below.
        "setp.ne.s32 pfast, %12, 0;\n\t"
        "@pfast bra L_skip_masked_max;\n\t"
        "mov.b32 rmax, 0xff800000;\n\t"
        "add.s32 key, %4, 0;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score0;\n\t"
        "add.s32 key, %4, 1;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score1;\n\t"
        "add.s32 key, %4, 2;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score2;\n\t"
        "add.s32 key, %4, 3;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score3;\n\t"
        "add.s32 key, %4, 4;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score4;\n\t"
        "add.s32 key, %4, 5;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score5;\n\t"
        "add.s32 key, %4, 6;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score6;\n\t"
        "add.s32 key, %4, 7;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score7;\n\t"
        "add.s32 key, %4, 8;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score8;\n\t"
        "add.s32 key, %4, 9;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score9;\n\t"
        "add.s32 key, %4, 10;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score10;\n\t"
        "add.s32 key, %4, 11;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score11;\n\t"
        "add.s32 key, %4, 12;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score12;\n\t"
        "add.s32 key, %4, 13;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score13;\n\t"
        "add.s32 key, %4, 14;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score14;\n\t"
        "add.s32 key, %4, 15;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score15;\n\t"
        "add.s32 key, %4, 16;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score16;\n\t"
        "add.s32 key, %4, 17;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score17;\n\t"
        "add.s32 key, %4, 18;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score18;\n\t"
        "add.s32 key, %4, 19;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score19;\n\t"
        "add.s32 key, %4, 20;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score20;\n\t"
        "add.s32 key, %4, 21;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score21;\n\t"
        "add.s32 key, %4, 22;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score22;\n\t"
        "add.s32 key, %4, 23;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score23;\n\t"
        "add.s32 key, %4, 24;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score24;\n\t"
        "add.s32 key, %4, 25;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score25;\n\t"
        "add.s32 key, %4, 26;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score26;\n\t"
        "add.s32 key, %4, 27;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score27;\n\t"
        "add.s32 key, %4, 28;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score28;\n\t"
        "add.s32 key, %4, 29;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score29;\n\t"
        "add.s32 key, %4, 30;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score30;\n\t"
        "add.s32 key, %4, 31;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score31;\n\t"
        "add.s32 key, %4, 32;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score32;\n\t"
        "add.s32 key, %4, 33;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score33;\n\t"
        "add.s32 key, %4, 34;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score34;\n\t"
        "add.s32 key, %4, 35;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score35;\n\t"
        "add.s32 key, %4, 36;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score36;\n\t"
        "add.s32 key, %4, 37;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score37;\n\t"
        "add.s32 key, %4, 38;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score38;\n\t"
        "add.s32 key, %4, 39;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score39;\n\t"
        "add.s32 key, %4, 40;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score40;\n\t"
        "add.s32 key, %4, 41;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score41;\n\t"
        "add.s32 key, %4, 42;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score42;\n\t"
        "add.s32 key, %4, 43;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score43;\n\t"
        "add.s32 key, %4, 44;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score44;\n\t"
        "add.s32 key, %4, 45;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score45;\n\t"
        "add.s32 key, %4, 46;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score46;\n\t"
        "add.s32 key, %4, 47;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score47;\n\t"
        "add.s32 key, %4, 48;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score48;\n\t"
        "add.s32 key, %4, 49;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score49;\n\t"
        "add.s32 key, %4, 50;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score50;\n\t"
        "add.s32 key, %4, 51;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score51;\n\t"
        "add.s32 key, %4, 52;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score52;\n\t"
        "add.s32 key, %4, 53;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score53;\n\t"
        "add.s32 key, %4, 54;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score54;\n\t"
        "add.s32 key, %4, 55;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score55;\n\t"
        "add.s32 key, %4, 56;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score56;\n\t"
        "add.s32 key, %4, 57;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score57;\n\t"
        "add.s32 key, %4, 58;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score58;\n\t"
        "add.s32 key, %4, 59;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score59;\n\t"
        "add.s32 key, %4, 60;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score60;\n\t"
        "add.s32 key, %4, 61;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score61;\n\t"
        "add.s32 key, %4, 62;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score62;\n\t"
        "add.s32 key, %4, 63;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score63;\n\t"
        "add.s32 key, %4, 64;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score64;\n\t"
        "add.s32 key, %4, 65;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score65;\n\t"
        "add.s32 key, %4, 66;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score66;\n\t"
        "add.s32 key, %4, 67;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score67;\n\t"
        "add.s32 key, %4, 68;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score68;\n\t"
        "add.s32 key, %4, 69;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score69;\n\t"
        "add.s32 key, %4, 70;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score70;\n\t"
        "add.s32 key, %4, 71;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score71;\n\t"
        "add.s32 key, %4, 72;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score72;\n\t"
        "add.s32 key, %4, 73;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score73;\n\t"
        "add.s32 key, %4, 74;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score74;\n\t"
        "add.s32 key, %4, 75;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score75;\n\t"
        "add.s32 key, %4, 76;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score76;\n\t"
        "add.s32 key, %4, 77;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score77;\n\t"
        "add.s32 key, %4, 78;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score78;\n\t"
        "add.s32 key, %4, 79;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score79;\n\t"
        "add.s32 key, %4, 80;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score80;\n\t"
        "add.s32 key, %4, 81;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score81;\n\t"
        "add.s32 key, %4, 82;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score82;\n\t"
        "add.s32 key, %4, 83;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score83;\n\t"
        "add.s32 key, %4, 84;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score84;\n\t"
        "add.s32 key, %4, 85;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score85;\n\t"
        "add.s32 key, %4, 86;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score86;\n\t"
        "add.s32 key, %4, 87;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score87;\n\t"
        "add.s32 key, %4, 88;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score88;\n\t"
        "add.s32 key, %4, 89;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score89;\n\t"
        "add.s32 key, %4, 90;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score90;\n\t"
        "add.s32 key, %4, 91;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score91;\n\t"
        "add.s32 key, %4, 92;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score92;\n\t"
        "add.s32 key, %4, 93;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score93;\n\t"
        "add.s32 key, %4, 94;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score94;\n\t"
        "add.s32 key, %4, 95;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score95;\n\t"
        "add.s32 key, %4, 96;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score96;\n\t"
        "add.s32 key, %4, 97;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score97;\n\t"
        "add.s32 key, %4, 98;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score98;\n\t"
        "add.s32 key, %4, 99;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score99;\n\t"
        "add.s32 key, %4, 100;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score100;\n\t"
        "add.s32 key, %4, 101;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score101;\n\t"
        "add.s32 key, %4, 102;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score102;\n\t"
        "add.s32 key, %4, 103;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score103;\n\t"
        "add.s32 key, %4, 104;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score104;\n\t"
        "add.s32 key, %4, 105;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score105;\n\t"
        "add.s32 key, %4, 106;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score106;\n\t"
        "add.s32 key, %4, 107;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score107;\n\t"
        "add.s32 key, %4, 108;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score108;\n\t"
        "add.s32 key, %4, 109;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score109;\n\t"
        "add.s32 key, %4, 110;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score110;\n\t"
        "add.s32 key, %4, 111;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score111;\n\t"
        "add.s32 key, %4, 112;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score112;\n\t"
        "add.s32 key, %4, 113;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score113;\n\t"
        "add.s32 key, %4, 114;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score114;\n\t"
        "add.s32 key, %4, 115;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score115;\n\t"
        "add.s32 key, %4, 116;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score116;\n\t"
        "add.s32 key, %4, 117;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score117;\n\t"
        "add.s32 key, %4, 118;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score118;\n\t"
        "add.s32 key, %4, 119;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score119;\n\t"
        "add.s32 key, %4, 120;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score120;\n\t"
        "add.s32 key, %4, 121;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score121;\n\t"
        "add.s32 key, %4, 122;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score122;\n\t"
        "add.s32 key, %4, 123;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score123;\n\t"
        "add.s32 key, %4, 124;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score124;\n\t"
        "add.s32 key, %4, 125;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score125;\n\t"
        "add.s32 key, %4, 126;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score126;\n\t"
        "add.s32 key, %4, 127;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "@p max.f32 rmax, rmax, score127;\n\t"
        "max.f32 %0, %9, rmax;\n\t"
        "@!pvalid mov.f32 %0, %9;\n\t"
        "bra L_masked_max_done;\n\t"
        "L_skip_masked_max:\n\t"
        // The following warp-wide rescale vote may be true because of a
        // different row, so preserve this row's historical running max.
        "max.f32 %0, %9, rmax;\n\t"
        "@!pvalid mov.f32 %0, %9;\n\t"
        "L_masked_max_done:\n\t"
        // m is kept in the unscaled QK domain.  Convert its delta to the
        // natural-log score domain before comparing against ln(256); 8 is
        // log2(256), used only after conversion for ex2.
        "sub.f32 v, %0, %9;\n\t"
        "mul.f32 v, v, %8;\n\t"
        "setp.gt.and.f32 prescale, v, %11, pvalid;\n\t"
        // Match FA4's warp-uniform conditional rescale: if no row in this
        // warp needs it, retain the old normalization base for every row.
        "vote.sync.any.pred prescale, prescale, 0xffffffff;\n\t"
        "@!prescale mov.f32 %0, %9;\n\t"
        "mul.f32 neg_m_scaled, %0, %13;\n\t"
        "neg.f32 neg_m_scaled, neg_m_scaled;\n\t"
        "setp.ne.and.f32 p, %10, 0f00000000, pvalid;\n\t"
        "fma.rn.f32 v, %9, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 %2, v;\n\t"
        "@!p mov.b32 %2, 0;\n\t"
        "mov.b32 sum, 0;\n\t"
        // The attention mask is a contiguous prefix.  Fully valid tiles take
        // this branch, avoiding the 128 per-score bounds checks below.
        "setp.ne.s32 pfast, %12, 0;\n\t"
        // TMEM stores are warp instructions, so the control flow must be
        // uniform even though all_scores_valid is per Q row.
        "vote.sync.all.pred pfast, pfast, 0xffffffff;\n\t"
        "@pfast bra L_p_full;\n\t"
        "add.s32 key, %4, 0;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score0, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score0, v;\n\t"
        "@!p mov.b32 score0, 0;\n\t"
        "add.f32 sum, sum, score0;\n\t"
        "add.s32 key, %4, 1;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score1, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score1, v;\n\t"
        "@!p mov.b32 score1, 0;\n\t"
        "add.f32 sum, sum, score1;\n\t"
        "add.s32 key, %4, 2;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score2, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score2, v;\n\t"
        "@!p mov.b32 score2, 0;\n\t"
        "add.f32 sum, sum, score2;\n\t"
        "add.s32 key, %4, 3;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score3, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score3, v;\n\t"
        "@!p mov.b32 score3, 0;\n\t"
        "add.f32 sum, sum, score3;\n\t"
        "add.s32 key, %4, 4;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score4, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score4, v;\n\t"
        "@!p mov.b32 score4, 0;\n\t"
        "add.f32 sum, sum, score4;\n\t"
        "add.s32 key, %4, 5;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score5, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score5, v;\n\t"
        "@!p mov.b32 score5, 0;\n\t"
        "add.f32 sum, sum, score5;\n\t"
        "add.s32 key, %4, 6;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score6, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score6, v;\n\t"
        "@!p mov.b32 score6, 0;\n\t"
        "add.f32 sum, sum, score6;\n\t"
        "add.s32 key, %4, 7;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score7, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score7, v;\n\t"
        "@!p mov.b32 score7, 0;\n\t"
        "add.f32 sum, sum, score7;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score1, score0;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score3, score2;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score5, score4;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score7, score6;\n\t"
        "mov.b32 addr, %3;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 8;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score8, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score8, v;\n\t"
        "@!p mov.b32 score8, 0;\n\t"
        "add.f32 sum, sum, score8;\n\t"
        "add.s32 key, %4, 9;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score9, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score9, v;\n\t"
        "@!p mov.b32 score9, 0;\n\t"
        "add.f32 sum, sum, score9;\n\t"
        "add.s32 key, %4, 10;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score10, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score10, v;\n\t"
        "@!p mov.b32 score10, 0;\n\t"
        "add.f32 sum, sum, score10;\n\t"
        "add.s32 key, %4, 11;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score11, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score11, v;\n\t"
        "@!p mov.b32 score11, 0;\n\t"
        "add.f32 sum, sum, score11;\n\t"
        "add.s32 key, %4, 12;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score12, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score12, v;\n\t"
        "@!p mov.b32 score12, 0;\n\t"
        "add.f32 sum, sum, score12;\n\t"
        "add.s32 key, %4, 13;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score13, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score13, v;\n\t"
        "@!p mov.b32 score13, 0;\n\t"
        "add.f32 sum, sum, score13;\n\t"
        "add.s32 key, %4, 14;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score14, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score14, v;\n\t"
        "@!p mov.b32 score14, 0;\n\t"
        "add.f32 sum, sum, score14;\n\t"
        "add.s32 key, %4, 15;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score15, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score15, v;\n\t"
        "@!p mov.b32 score15, 0;\n\t"
        "add.f32 sum, sum, score15;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score9, score8;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score11, score10;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score13, score12;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score15, score14;\n\t"
        "add.s32 addr, %3, 4;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 16;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score16, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score16, v;\n\t"
        "@!p mov.b32 score16, 0;\n\t"
        "add.f32 sum, sum, score16;\n\t"
        "add.s32 key, %4, 17;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score17, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score17, v;\n\t"
        "@!p mov.b32 score17, 0;\n\t"
        "add.f32 sum, sum, score17;\n\t"
        "add.s32 key, %4, 18;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score18, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score18, v;\n\t"
        "@!p mov.b32 score18, 0;\n\t"
        "add.f32 sum, sum, score18;\n\t"
        "add.s32 key, %4, 19;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score19, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score19, v;\n\t"
        "@!p mov.b32 score19, 0;\n\t"
        "add.f32 sum, sum, score19;\n\t"
        "add.s32 key, %4, 20;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score20, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score20, v;\n\t"
        "@!p mov.b32 score20, 0;\n\t"
        "add.f32 sum, sum, score20;\n\t"
        "add.s32 key, %4, 21;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score21, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score21, v;\n\t"
        "@!p mov.b32 score21, 0;\n\t"
        "add.f32 sum, sum, score21;\n\t"
        "add.s32 key, %4, 22;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score22, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score22, v;\n\t"
        "@!p mov.b32 score22, 0;\n\t"
        "add.f32 sum, sum, score22;\n\t"
        "add.s32 key, %4, 23;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score23, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score23, v;\n\t"
        "@!p mov.b32 score23, 0;\n\t"
        "add.f32 sum, sum, score23;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score17, score16;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score19, score18;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score21, score20;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score23, score22;\n\t"
        "add.s32 addr, %3, 8;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 24;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score24, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score24, v;\n\t"
        "@!p mov.b32 score24, 0;\n\t"
        "add.f32 sum, sum, score24;\n\t"
        "add.s32 key, %4, 25;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score25, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score25, v;\n\t"
        "@!p mov.b32 score25, 0;\n\t"
        "add.f32 sum, sum, score25;\n\t"
        "add.s32 key, %4, 26;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score26, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score26, v;\n\t"
        "@!p mov.b32 score26, 0;\n\t"
        "add.f32 sum, sum, score26;\n\t"
        "add.s32 key, %4, 27;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score27, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score27, v;\n\t"
        "@!p mov.b32 score27, 0;\n\t"
        "add.f32 sum, sum, score27;\n\t"
        "add.s32 key, %4, 28;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score28, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score28, v;\n\t"
        "@!p mov.b32 score28, 0;\n\t"
        "add.f32 sum, sum, score28;\n\t"
        "add.s32 key, %4, 29;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score29, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score29, v;\n\t"
        "@!p mov.b32 score29, 0;\n\t"
        "add.f32 sum, sum, score29;\n\t"
        "add.s32 key, %4, 30;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score30, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score30, v;\n\t"
        "@!p mov.b32 score30, 0;\n\t"
        "add.f32 sum, sum, score30;\n\t"
        "add.s32 key, %4, 31;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score31, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score31, v;\n\t"
        "@!p mov.b32 score31, 0;\n\t"
        "add.f32 sum, sum, score31;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score25, score24;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score27, score26;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score29, score28;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score31, score30;\n\t"
        "add.s32 addr, %3, 12;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 32;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score32, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score32, v;\n\t"
        "@!p mov.b32 score32, 0;\n\t"
        "add.f32 sum, sum, score32;\n\t"
        "add.s32 key, %4, 33;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score33, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score33, v;\n\t"
        "@!p mov.b32 score33, 0;\n\t"
        "add.f32 sum, sum, score33;\n\t"
        "add.s32 key, %4, 34;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score34, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score34, v;\n\t"
        "@!p mov.b32 score34, 0;\n\t"
        "add.f32 sum, sum, score34;\n\t"
        "add.s32 key, %4, 35;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score35, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score35, v;\n\t"
        "@!p mov.b32 score35, 0;\n\t"
        "add.f32 sum, sum, score35;\n\t"
        "add.s32 key, %4, 36;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score36, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score36, v;\n\t"
        "@!p mov.b32 score36, 0;\n\t"
        "add.f32 sum, sum, score36;\n\t"
        "add.s32 key, %4, 37;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score37, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score37, v;\n\t"
        "@!p mov.b32 score37, 0;\n\t"
        "add.f32 sum, sum, score37;\n\t"
        "add.s32 key, %4, 38;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score38, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score38, v;\n\t"
        "@!p mov.b32 score38, 0;\n\t"
        "add.f32 sum, sum, score38;\n\t"
        "add.s32 key, %4, 39;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score39, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score39, v;\n\t"
        "@!p mov.b32 score39, 0;\n\t"
        "add.f32 sum, sum, score39;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score33, score32;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score35, score34;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score37, score36;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score39, score38;\n\t"
        "add.s32 addr, %3, 16;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 40;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score40, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score40, v;\n\t"
        "@!p mov.b32 score40, 0;\n\t"
        "add.f32 sum, sum, score40;\n\t"
        "add.s32 key, %4, 41;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score41, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score41, v;\n\t"
        "@!p mov.b32 score41, 0;\n\t"
        "add.f32 sum, sum, score41;\n\t"
        "add.s32 key, %4, 42;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score42, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score42, v;\n\t"
        "@!p mov.b32 score42, 0;\n\t"
        "add.f32 sum, sum, score42;\n\t"
        "add.s32 key, %4, 43;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score43, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score43, v;\n\t"
        "@!p mov.b32 score43, 0;\n\t"
        "add.f32 sum, sum, score43;\n\t"
        "add.s32 key, %4, 44;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score44, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score44, v;\n\t"
        "@!p mov.b32 score44, 0;\n\t"
        "add.f32 sum, sum, score44;\n\t"
        "add.s32 key, %4, 45;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score45, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score45, v;\n\t"
        "@!p mov.b32 score45, 0;\n\t"
        "add.f32 sum, sum, score45;\n\t"
        "add.s32 key, %4, 46;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score46, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score46, v;\n\t"
        "@!p mov.b32 score46, 0;\n\t"
        "add.f32 sum, sum, score46;\n\t"
        "add.s32 key, %4, 47;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score47, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score47, v;\n\t"
        "@!p mov.b32 score47, 0;\n\t"
        "add.f32 sum, sum, score47;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score41, score40;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score43, score42;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score45, score44;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score47, score46;\n\t"
        "add.s32 addr, %3, 20;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 48;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score48, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score48, v;\n\t"
        "@!p mov.b32 score48, 0;\n\t"
        "add.f32 sum, sum, score48;\n\t"
        "add.s32 key, %4, 49;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score49, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score49, v;\n\t"
        "@!p mov.b32 score49, 0;\n\t"
        "add.f32 sum, sum, score49;\n\t"
        "add.s32 key, %4, 50;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score50, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score50, v;\n\t"
        "@!p mov.b32 score50, 0;\n\t"
        "add.f32 sum, sum, score50;\n\t"
        "add.s32 key, %4, 51;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score51, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score51, v;\n\t"
        "@!p mov.b32 score51, 0;\n\t"
        "add.f32 sum, sum, score51;\n\t"
        "add.s32 key, %4, 52;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score52, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score52, v;\n\t"
        "@!p mov.b32 score52, 0;\n\t"
        "add.f32 sum, sum, score52;\n\t"
        "add.s32 key, %4, 53;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score53, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score53, v;\n\t"
        "@!p mov.b32 score53, 0;\n\t"
        "add.f32 sum, sum, score53;\n\t"
        "add.s32 key, %4, 54;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score54, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score54, v;\n\t"
        "@!p mov.b32 score54, 0;\n\t"
        "add.f32 sum, sum, score54;\n\t"
        "add.s32 key, %4, 55;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score55, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score55, v;\n\t"
        "@!p mov.b32 score55, 0;\n\t"
        "add.f32 sum, sum, score55;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score49, score48;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score51, score50;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score53, score52;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score55, score54;\n\t"
        "add.s32 addr, %3, 24;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 56;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score56, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score56, v;\n\t"
        "@!p mov.b32 score56, 0;\n\t"
        "add.f32 sum, sum, score56;\n\t"
        "add.s32 key, %4, 57;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score57, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score57, v;\n\t"
        "@!p mov.b32 score57, 0;\n\t"
        "add.f32 sum, sum, score57;\n\t"
        "add.s32 key, %4, 58;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score58, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score58, v;\n\t"
        "@!p mov.b32 score58, 0;\n\t"
        "add.f32 sum, sum, score58;\n\t"
        "add.s32 key, %4, 59;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score59, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score59, v;\n\t"
        "@!p mov.b32 score59, 0;\n\t"
        "add.f32 sum, sum, score59;\n\t"
        "add.s32 key, %4, 60;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score60, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score60, v;\n\t"
        "@!p mov.b32 score60, 0;\n\t"
        "add.f32 sum, sum, score60;\n\t"
        "add.s32 key, %4, 61;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score61, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score61, v;\n\t"
        "@!p mov.b32 score61, 0;\n\t"
        "add.f32 sum, sum, score61;\n\t"
        "add.s32 key, %4, 62;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score62, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score62, v;\n\t"
        "@!p mov.b32 score62, 0;\n\t"
        "add.f32 sum, sum, score62;\n\t"
        "add.s32 key, %4, 63;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score63, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score63, v;\n\t"
        "@!p mov.b32 score63, 0;\n\t"
        "add.f32 sum, sum, score63;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score57, score56;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score59, score58;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score61, score60;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score63, score62;\n\t"
        "add.s32 addr, %3, 28;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 64;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score64, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score64, v;\n\t"
        "@!p mov.b32 score64, 0;\n\t"
        "add.f32 sum, sum, score64;\n\t"
        "add.s32 key, %4, 65;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score65, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score65, v;\n\t"
        "@!p mov.b32 score65, 0;\n\t"
        "add.f32 sum, sum, score65;\n\t"
        "add.s32 key, %4, 66;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score66, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score66, v;\n\t"
        "@!p mov.b32 score66, 0;\n\t"
        "add.f32 sum, sum, score66;\n\t"
        "add.s32 key, %4, 67;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score67, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score67, v;\n\t"
        "@!p mov.b32 score67, 0;\n\t"
        "add.f32 sum, sum, score67;\n\t"
        "add.s32 key, %4, 68;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score68, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score68, v;\n\t"
        "@!p mov.b32 score68, 0;\n\t"
        "add.f32 sum, sum, score68;\n\t"
        "add.s32 key, %4, 69;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score69, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score69, v;\n\t"
        "@!p mov.b32 score69, 0;\n\t"
        "add.f32 sum, sum, score69;\n\t"
        "add.s32 key, %4, 70;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score70, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score70, v;\n\t"
        "@!p mov.b32 score70, 0;\n\t"
        "add.f32 sum, sum, score70;\n\t"
        "add.s32 key, %4, 71;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score71, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score71, v;\n\t"
        "@!p mov.b32 score71, 0;\n\t"
        "add.f32 sum, sum, score71;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score65, score64;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score67, score66;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score69, score68;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score71, score70;\n\t"
        "add.s32 addr, %3, 32;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 72;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score72, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score72, v;\n\t"
        "@!p mov.b32 score72, 0;\n\t"
        "add.f32 sum, sum, score72;\n\t"
        "add.s32 key, %4, 73;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score73, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score73, v;\n\t"
        "@!p mov.b32 score73, 0;\n\t"
        "add.f32 sum, sum, score73;\n\t"
        "add.s32 key, %4, 74;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score74, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score74, v;\n\t"
        "@!p mov.b32 score74, 0;\n\t"
        "add.f32 sum, sum, score74;\n\t"
        "add.s32 key, %4, 75;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score75, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score75, v;\n\t"
        "@!p mov.b32 score75, 0;\n\t"
        "add.f32 sum, sum, score75;\n\t"
        "add.s32 key, %4, 76;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score76, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score76, v;\n\t"
        "@!p mov.b32 score76, 0;\n\t"
        "add.f32 sum, sum, score76;\n\t"
        "add.s32 key, %4, 77;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score77, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score77, v;\n\t"
        "@!p mov.b32 score77, 0;\n\t"
        "add.f32 sum, sum, score77;\n\t"
        "add.s32 key, %4, 78;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score78, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score78, v;\n\t"
        "@!p mov.b32 score78, 0;\n\t"
        "add.f32 sum, sum, score78;\n\t"
        "add.s32 key, %4, 79;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score79, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score79, v;\n\t"
        "@!p mov.b32 score79, 0;\n\t"
        "add.f32 sum, sum, score79;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score73, score72;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score75, score74;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score77, score76;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score79, score78;\n\t"
        "add.s32 addr, %3, 36;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 80;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score80, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score80, v;\n\t"
        "@!p mov.b32 score80, 0;\n\t"
        "add.f32 sum, sum, score80;\n\t"
        "add.s32 key, %4, 81;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score81, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score81, v;\n\t"
        "@!p mov.b32 score81, 0;\n\t"
        "add.f32 sum, sum, score81;\n\t"
        "add.s32 key, %4, 82;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score82, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score82, v;\n\t"
        "@!p mov.b32 score82, 0;\n\t"
        "add.f32 sum, sum, score82;\n\t"
        "add.s32 key, %4, 83;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score83, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score83, v;\n\t"
        "@!p mov.b32 score83, 0;\n\t"
        "add.f32 sum, sum, score83;\n\t"
        "add.s32 key, %4, 84;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score84, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score84, v;\n\t"
        "@!p mov.b32 score84, 0;\n\t"
        "add.f32 sum, sum, score84;\n\t"
        "add.s32 key, %4, 85;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score85, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score85, v;\n\t"
        "@!p mov.b32 score85, 0;\n\t"
        "add.f32 sum, sum, score85;\n\t"
        "add.s32 key, %4, 86;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score86, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score86, v;\n\t"
        "@!p mov.b32 score86, 0;\n\t"
        "add.f32 sum, sum, score86;\n\t"
        "add.s32 key, %4, 87;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score87, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score87, v;\n\t"
        "@!p mov.b32 score87, 0;\n\t"
        "add.f32 sum, sum, score87;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score81, score80;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score83, score82;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score85, score84;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score87, score86;\n\t"
        "add.s32 addr, %3, 40;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 88;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score88, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score88, v;\n\t"
        "@!p mov.b32 score88, 0;\n\t"
        "add.f32 sum, sum, score88;\n\t"
        "add.s32 key, %4, 89;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score89, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score89, v;\n\t"
        "@!p mov.b32 score89, 0;\n\t"
        "add.f32 sum, sum, score89;\n\t"
        "add.s32 key, %4, 90;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score90, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score90, v;\n\t"
        "@!p mov.b32 score90, 0;\n\t"
        "add.f32 sum, sum, score90;\n\t"
        "add.s32 key, %4, 91;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score91, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score91, v;\n\t"
        "@!p mov.b32 score91, 0;\n\t"
        "add.f32 sum, sum, score91;\n\t"
        "add.s32 key, %4, 92;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score92, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score92, v;\n\t"
        "@!p mov.b32 score92, 0;\n\t"
        "add.f32 sum, sum, score92;\n\t"
        "add.s32 key, %4, 93;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score93, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score93, v;\n\t"
        "@!p mov.b32 score93, 0;\n\t"
        "add.f32 sum, sum, score93;\n\t"
        "add.s32 key, %4, 94;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score94, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score94, v;\n\t"
        "@!p mov.b32 score94, 0;\n\t"
        "add.f32 sum, sum, score94;\n\t"
        "add.s32 key, %4, 95;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score95, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score95, v;\n\t"
        "@!p mov.b32 score95, 0;\n\t"
        "add.f32 sum, sum, score95;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score89, score88;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score91, score90;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score93, score92;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score95, score94;\n\t"
        "add.s32 addr, %3, 44;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 96;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score96, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score96, v;\n\t"
        "@!p mov.b32 score96, 0;\n\t"
        "add.f32 sum, sum, score96;\n\t"
        "add.s32 key, %4, 97;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score97, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score97, v;\n\t"
        "@!p mov.b32 score97, 0;\n\t"
        "add.f32 sum, sum, score97;\n\t"
        "add.s32 key, %4, 98;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score98, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score98, v;\n\t"
        "@!p mov.b32 score98, 0;\n\t"
        "add.f32 sum, sum, score98;\n\t"
        "add.s32 key, %4, 99;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score99, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score99, v;\n\t"
        "@!p mov.b32 score99, 0;\n\t"
        "add.f32 sum, sum, score99;\n\t"
        "add.s32 key, %4, 100;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score100, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score100, v;\n\t"
        "@!p mov.b32 score100, 0;\n\t"
        "add.f32 sum, sum, score100;\n\t"
        "add.s32 key, %4, 101;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score101, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score101, v;\n\t"
        "@!p mov.b32 score101, 0;\n\t"
        "add.f32 sum, sum, score101;\n\t"
        "add.s32 key, %4, 102;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score102, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score102, v;\n\t"
        "@!p mov.b32 score102, 0;\n\t"
        "add.f32 sum, sum, score102;\n\t"
        "add.s32 key, %4, 103;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score103, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score103, v;\n\t"
        "@!p mov.b32 score103, 0;\n\t"
        "add.f32 sum, sum, score103;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score97, score96;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score99, score98;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score101, score100;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score103, score102;\n\t"
        "add.s32 addr, %3, 48;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 104;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score104, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score104, v;\n\t"
        "@!p mov.b32 score104, 0;\n\t"
        "add.f32 sum, sum, score104;\n\t"
        "add.s32 key, %4, 105;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score105, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score105, v;\n\t"
        "@!p mov.b32 score105, 0;\n\t"
        "add.f32 sum, sum, score105;\n\t"
        "add.s32 key, %4, 106;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score106, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score106, v;\n\t"
        "@!p mov.b32 score106, 0;\n\t"
        "add.f32 sum, sum, score106;\n\t"
        "add.s32 key, %4, 107;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score107, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score107, v;\n\t"
        "@!p mov.b32 score107, 0;\n\t"
        "add.f32 sum, sum, score107;\n\t"
        "add.s32 key, %4, 108;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score108, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score108, v;\n\t"
        "@!p mov.b32 score108, 0;\n\t"
        "add.f32 sum, sum, score108;\n\t"
        "add.s32 key, %4, 109;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score109, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score109, v;\n\t"
        "@!p mov.b32 score109, 0;\n\t"
        "add.f32 sum, sum, score109;\n\t"
        "add.s32 key, %4, 110;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score110, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score110, v;\n\t"
        "@!p mov.b32 score110, 0;\n\t"
        "add.f32 sum, sum, score110;\n\t"
        "add.s32 key, %4, 111;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score111, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score111, v;\n\t"
        "@!p mov.b32 score111, 0;\n\t"
        "add.f32 sum, sum, score111;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score105, score104;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score107, score106;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score109, score108;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score111, score110;\n\t"
        "add.s32 addr, %3, 52;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 112;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score112, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score112, v;\n\t"
        "@!p mov.b32 score112, 0;\n\t"
        "add.f32 sum, sum, score112;\n\t"
        "add.s32 key, %4, 113;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score113, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score113, v;\n\t"
        "@!p mov.b32 score113, 0;\n\t"
        "add.f32 sum, sum, score113;\n\t"
        "add.s32 key, %4, 114;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score114, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score114, v;\n\t"
        "@!p mov.b32 score114, 0;\n\t"
        "add.f32 sum, sum, score114;\n\t"
        "add.s32 key, %4, 115;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score115, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score115, v;\n\t"
        "@!p mov.b32 score115, 0;\n\t"
        "add.f32 sum, sum, score115;\n\t"
        "add.s32 key, %4, 116;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score116, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score116, v;\n\t"
        "@!p mov.b32 score116, 0;\n\t"
        "add.f32 sum, sum, score116;\n\t"
        "add.s32 key, %4, 117;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score117, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score117, v;\n\t"
        "@!p mov.b32 score117, 0;\n\t"
        "add.f32 sum, sum, score117;\n\t"
        "add.s32 key, %4, 118;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score118, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score118, v;\n\t"
        "@!p mov.b32 score118, 0;\n\t"
        "add.f32 sum, sum, score118;\n\t"
        "add.s32 key, %4, 119;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score119, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score119, v;\n\t"
        "@!p mov.b32 score119, 0;\n\t"
        "add.f32 sum, sum, score119;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score113, score112;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score115, score114;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score117, score116;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score119, score118;\n\t"
        "add.s32 addr, %3, 56;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "add.s32 key, %4, 120;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score120, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score120, v;\n\t"
        "@!p mov.b32 score120, 0;\n\t"
        "add.f32 sum, sum, score120;\n\t"
        "add.s32 key, %4, 121;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score121, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score121, v;\n\t"
        "@!p mov.b32 score121, 0;\n\t"
        "add.f32 sum, sum, score121;\n\t"
        "add.s32 key, %4, 122;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score122, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score122, v;\n\t"
        "@!p mov.b32 score122, 0;\n\t"
        "add.f32 sum, sum, score122;\n\t"
        "add.s32 key, %4, 123;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score123, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score123, v;\n\t"
        "@!p mov.b32 score123, 0;\n\t"
        "add.f32 sum, sum, score123;\n\t"
        "add.s32 key, %4, 124;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score124, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score124, v;\n\t"
        "@!p mov.b32 score124, 0;\n\t"
        "add.f32 sum, sum, score124;\n\t"
        "add.s32 key, %4, 125;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score125, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score125, v;\n\t"
        "@!p mov.b32 score125, 0;\n\t"
        "add.f32 sum, sum, score125;\n\t"
        "add.s32 key, %4, 126;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score126, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score126, v;\n\t"
        "@!p mov.b32 score126, 0;\n\t"
        "add.f32 sum, sum, score126;\n\t"
        "add.s32 key, %4, 127;\n\t"
        "setp.lt.and.s32 p, key, %5, pvalid;\n\t"
        "setp.le.and.s32 p, key, %6, p;\n\t"
        "fma.rn.f32 v, score127, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score127, v;\n\t"
        "@!p mov.b32 score127, 0;\n\t"
        "add.f32 sum, sum, score127;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score121, score120;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score123, score122;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score125, score124;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score127, score126;\n\t"
        "add.s32 addr, %3, 60;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "bra L_p_done;\n\t"
        "L_p_full:\n\t"
        "mov.b32 sum, 0;\n\t"
        "fma.rn.f32 v, score0, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score0, v;\n\t"
        "add.f32 sum, sum, score0;\n\t"
        "fma.rn.f32 v, score1, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score1, v;\n\t"
        "add.f32 sum, sum, score1;\n\t"
        "fma.rn.f32 v, score2, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score2, v;\n\t"
        "add.f32 sum, sum, score2;\n\t"
        "fma.rn.f32 v, score3, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score3, v;\n\t"
        "add.f32 sum, sum, score3;\n\t"
        "fma.rn.f32 v, score4, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score4, v;\n\t"
        "add.f32 sum, sum, score4;\n\t"
        "fma.rn.f32 v, score5, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score5, v;\n\t"
        "add.f32 sum, sum, score5;\n\t"
        "fma.rn.f32 v, score6, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score6, v;\n\t"
        "add.f32 sum, sum, score6;\n\t"
        "fma.rn.f32 v, score7, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score7, v;\n\t"
        "add.f32 sum, sum, score7;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score1, score0;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score3, score2;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score5, score4;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score7, score6;\n\t"
        "mov.b32 addr, %3;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score8, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score8, v;\n\t"
        "add.f32 sum, sum, score8;\n\t"
        "fma.rn.f32 v, score9, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score9, v;\n\t"
        "add.f32 sum, sum, score9;\n\t"
        "fma.rn.f32 v, score10, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score10, v;\n\t"
        "add.f32 sum, sum, score10;\n\t"
        "fma.rn.f32 v, score11, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score11, v;\n\t"
        "add.f32 sum, sum, score11;\n\t"
        "fma.rn.f32 v, score12, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score12, v;\n\t"
        "add.f32 sum, sum, score12;\n\t"
        "fma.rn.f32 v, score13, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score13, v;\n\t"
        "add.f32 sum, sum, score13;\n\t"
        "fma.rn.f32 v, score14, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score14, v;\n\t"
        "add.f32 sum, sum, score14;\n\t"
        "fma.rn.f32 v, score15, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score15, v;\n\t"
        "add.f32 sum, sum, score15;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score9, score8;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score11, score10;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score13, score12;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score15, score14;\n\t"
        "add.s32 addr, %3, 4;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score16, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score16, v;\n\t"
        "add.f32 sum, sum, score16;\n\t"
        "fma.rn.f32 v, score17, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score17, v;\n\t"
        "add.f32 sum, sum, score17;\n\t"
        "fma.rn.f32 v, score18, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score18, v;\n\t"
        "add.f32 sum, sum, score18;\n\t"
        "fma.rn.f32 v, score19, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score19, v;\n\t"
        "add.f32 sum, sum, score19;\n\t"
        "fma.rn.f32 v, score20, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score20, v;\n\t"
        "add.f32 sum, sum, score20;\n\t"
        "fma.rn.f32 v, score21, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score21, v;\n\t"
        "add.f32 sum, sum, score21;\n\t"
        "fma.rn.f32 v, score22, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score22, v;\n\t"
        "add.f32 sum, sum, score22;\n\t"
        "fma.rn.f32 v, score23, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score23, v;\n\t"
        "add.f32 sum, sum, score23;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score17, score16;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score19, score18;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score21, score20;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score23, score22;\n\t"
        "add.s32 addr, %3, 8;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score24, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score24, v;\n\t"
        "add.f32 sum, sum, score24;\n\t"
        "fma.rn.f32 v, score25, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score25, v;\n\t"
        "add.f32 sum, sum, score25;\n\t"
        "fma.rn.f32 v, score26, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score26, v;\n\t"
        "add.f32 sum, sum, score26;\n\t"
        "fma.rn.f32 v, score27, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score27, v;\n\t"
        "add.f32 sum, sum, score27;\n\t"
        "fma.rn.f32 v, score28, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score28, v;\n\t"
        "add.f32 sum, sum, score28;\n\t"
        "fma.rn.f32 v, score29, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score29, v;\n\t"
        "add.f32 sum, sum, score29;\n\t"
        "fma.rn.f32 v, score30, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score30, v;\n\t"
        "add.f32 sum, sum, score30;\n\t"
        "fma.rn.f32 v, score31, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score31, v;\n\t"
        "add.f32 sum, sum, score31;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score25, score24;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score27, score26;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score29, score28;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score31, score30;\n\t"
        "add.s32 addr, %3, 12;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score32, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score32, v;\n\t"
        "add.f32 sum, sum, score32;\n\t"
        "fma.rn.f32 v, score33, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score33, v;\n\t"
        "add.f32 sum, sum, score33;\n\t"
        "fma.rn.f32 v, score34, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score34, v;\n\t"
        "add.f32 sum, sum, score34;\n\t"
        "fma.rn.f32 v, score35, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score35, v;\n\t"
        "add.f32 sum, sum, score35;\n\t"
        "fma.rn.f32 v, score36, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score36, v;\n\t"
        "add.f32 sum, sum, score36;\n\t"
        "fma.rn.f32 v, score37, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score37, v;\n\t"
        "add.f32 sum, sum, score37;\n\t"
        "fma.rn.f32 v, score38, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score38, v;\n\t"
        "add.f32 sum, sum, score38;\n\t"
        "fma.rn.f32 v, score39, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score39, v;\n\t"
        "add.f32 sum, sum, score39;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score33, score32;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score35, score34;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score37, score36;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score39, score38;\n\t"
        "add.s32 addr, %3, 16;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score40, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score40, v;\n\t"
        "add.f32 sum, sum, score40;\n\t"
        "fma.rn.f32 v, score41, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score41, v;\n\t"
        "add.f32 sum, sum, score41;\n\t"
        "fma.rn.f32 v, score42, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score42, v;\n\t"
        "add.f32 sum, sum, score42;\n\t"
        "fma.rn.f32 v, score43, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score43, v;\n\t"
        "add.f32 sum, sum, score43;\n\t"
        "fma.rn.f32 v, score44, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score44, v;\n\t"
        "add.f32 sum, sum, score44;\n\t"
        "fma.rn.f32 v, score45, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score45, v;\n\t"
        "add.f32 sum, sum, score45;\n\t"
        "fma.rn.f32 v, score46, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score46, v;\n\t"
        "add.f32 sum, sum, score46;\n\t"
        "fma.rn.f32 v, score47, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score47, v;\n\t"
        "add.f32 sum, sum, score47;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score41, score40;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score43, score42;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score45, score44;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score47, score46;\n\t"
        "add.s32 addr, %3, 20;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score48, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score48, v;\n\t"
        "add.f32 sum, sum, score48;\n\t"
        "fma.rn.f32 v, score49, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score49, v;\n\t"
        "add.f32 sum, sum, score49;\n\t"
        "fma.rn.f32 v, score50, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score50, v;\n\t"
        "add.f32 sum, sum, score50;\n\t"
        "fma.rn.f32 v, score51, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score51, v;\n\t"
        "add.f32 sum, sum, score51;\n\t"
        "fma.rn.f32 v, score52, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score52, v;\n\t"
        "add.f32 sum, sum, score52;\n\t"
        "fma.rn.f32 v, score53, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score53, v;\n\t"
        "add.f32 sum, sum, score53;\n\t"
        "fma.rn.f32 v, score54, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score54, v;\n\t"
        "add.f32 sum, sum, score54;\n\t"
        "fma.rn.f32 v, score55, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score55, v;\n\t"
        "add.f32 sum, sum, score55;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score49, score48;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score51, score50;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score53, score52;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score55, score54;\n\t"
        "add.s32 addr, %3, 24;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score56, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score56, v;\n\t"
        "add.f32 sum, sum, score56;\n\t"
        "fma.rn.f32 v, score57, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score57, v;\n\t"
        "add.f32 sum, sum, score57;\n\t"
        "fma.rn.f32 v, score58, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score58, v;\n\t"
        "add.f32 sum, sum, score58;\n\t"
        "fma.rn.f32 v, score59, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score59, v;\n\t"
        "add.f32 sum, sum, score59;\n\t"
        "fma.rn.f32 v, score60, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score60, v;\n\t"
        "add.f32 sum, sum, score60;\n\t"
        "fma.rn.f32 v, score61, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score61, v;\n\t"
        "add.f32 sum, sum, score61;\n\t"
        "fma.rn.f32 v, score62, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score62, v;\n\t"
        "add.f32 sum, sum, score62;\n\t"
        "fma.rn.f32 v, score63, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score63, v;\n\t"
        "add.f32 sum, sum, score63;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score57, score56;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score59, score58;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score61, score60;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score63, score62;\n\t"
        "add.s32 addr, %3, 28;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score64, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score64, v;\n\t"
        "add.f32 sum, sum, score64;\n\t"
        "fma.rn.f32 v, score65, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score65, v;\n\t"
        "add.f32 sum, sum, score65;\n\t"
        "fma.rn.f32 v, score66, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score66, v;\n\t"
        "add.f32 sum, sum, score66;\n\t"
        "fma.rn.f32 v, score67, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score67, v;\n\t"
        "add.f32 sum, sum, score67;\n\t"
        "fma.rn.f32 v, score68, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score68, v;\n\t"
        "add.f32 sum, sum, score68;\n\t"
        "fma.rn.f32 v, score69, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score69, v;\n\t"
        "add.f32 sum, sum, score69;\n\t"
        "fma.rn.f32 v, score70, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score70, v;\n\t"
        "add.f32 sum, sum, score70;\n\t"
        "fma.rn.f32 v, score71, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score71, v;\n\t"
        "add.f32 sum, sum, score71;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score65, score64;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score67, score66;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score69, score68;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score71, score70;\n\t"
        "add.s32 addr, %3, 32;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score72, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score72, v;\n\t"
        "add.f32 sum, sum, score72;\n\t"
        "fma.rn.f32 v, score73, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score73, v;\n\t"
        "add.f32 sum, sum, score73;\n\t"
        "fma.rn.f32 v, score74, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score74, v;\n\t"
        "add.f32 sum, sum, score74;\n\t"
        "fma.rn.f32 v, score75, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score75, v;\n\t"
        "add.f32 sum, sum, score75;\n\t"
        "fma.rn.f32 v, score76, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score76, v;\n\t"
        "add.f32 sum, sum, score76;\n\t"
        "fma.rn.f32 v, score77, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score77, v;\n\t"
        "add.f32 sum, sum, score77;\n\t"
        "fma.rn.f32 v, score78, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score78, v;\n\t"
        "add.f32 sum, sum, score78;\n\t"
        "fma.rn.f32 v, score79, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score79, v;\n\t"
        "add.f32 sum, sum, score79;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score73, score72;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score75, score74;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score77, score76;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score79, score78;\n\t"
        "add.s32 addr, %3, 36;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score80, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score80, v;\n\t"
        "add.f32 sum, sum, score80;\n\t"
        "fma.rn.f32 v, score81, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score81, v;\n\t"
        "add.f32 sum, sum, score81;\n\t"
        "fma.rn.f32 v, score82, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score82, v;\n\t"
        "add.f32 sum, sum, score82;\n\t"
        "fma.rn.f32 v, score83, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score83, v;\n\t"
        "add.f32 sum, sum, score83;\n\t"
        "fma.rn.f32 v, score84, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score84, v;\n\t"
        "add.f32 sum, sum, score84;\n\t"
        "fma.rn.f32 v, score85, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score85, v;\n\t"
        "add.f32 sum, sum, score85;\n\t"
        "fma.rn.f32 v, score86, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score86, v;\n\t"
        "add.f32 sum, sum, score86;\n\t"
        "fma.rn.f32 v, score87, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score87, v;\n\t"
        "add.f32 sum, sum, score87;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score81, score80;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score83, score82;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score85, score84;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score87, score86;\n\t"
        "add.s32 addr, %3, 40;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score88, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score88, v;\n\t"
        "add.f32 sum, sum, score88;\n\t"
        "fma.rn.f32 v, score89, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score89, v;\n\t"
        "add.f32 sum, sum, score89;\n\t"
        "fma.rn.f32 v, score90, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score90, v;\n\t"
        "add.f32 sum, sum, score90;\n\t"
        "fma.rn.f32 v, score91, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score91, v;\n\t"
        "add.f32 sum, sum, score91;\n\t"
        "fma.rn.f32 v, score92, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score92, v;\n\t"
        "add.f32 sum, sum, score92;\n\t"
        "fma.rn.f32 v, score93, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score93, v;\n\t"
        "add.f32 sum, sum, score93;\n\t"
        "fma.rn.f32 v, score94, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score94, v;\n\t"
        "add.f32 sum, sum, score94;\n\t"
        "fma.rn.f32 v, score95, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score95, v;\n\t"
        "add.f32 sum, sum, score95;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score89, score88;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score91, score90;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score93, score92;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score95, score94;\n\t"
        "add.s32 addr, %3, 44;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score96, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score96, v;\n\t"
        "add.f32 sum, sum, score96;\n\t"
        "fma.rn.f32 v, score97, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score97, v;\n\t"
        "add.f32 sum, sum, score97;\n\t"
        "fma.rn.f32 v, score98, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score98, v;\n\t"
        "add.f32 sum, sum, score98;\n\t"
        "fma.rn.f32 v, score99, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score99, v;\n\t"
        "add.f32 sum, sum, score99;\n\t"
        "fma.rn.f32 v, score100, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score100, v;\n\t"
        "add.f32 sum, sum, score100;\n\t"
        "fma.rn.f32 v, score101, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score101, v;\n\t"
        "add.f32 sum, sum, score101;\n\t"
        "fma.rn.f32 v, score102, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score102, v;\n\t"
        "add.f32 sum, sum, score102;\n\t"
        "fma.rn.f32 v, score103, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score103, v;\n\t"
        "add.f32 sum, sum, score103;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score97, score96;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score99, score98;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score101, score100;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score103, score102;\n\t"
        "add.s32 addr, %3, 48;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score104, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score104, v;\n\t"
        "add.f32 sum, sum, score104;\n\t"
        "fma.rn.f32 v, score105, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score105, v;\n\t"
        "add.f32 sum, sum, score105;\n\t"
        "fma.rn.f32 v, score106, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score106, v;\n\t"
        "add.f32 sum, sum, score106;\n\t"
        "fma.rn.f32 v, score107, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score107, v;\n\t"
        "add.f32 sum, sum, score107;\n\t"
        "fma.rn.f32 v, score108, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score108, v;\n\t"
        "add.f32 sum, sum, score108;\n\t"
        "fma.rn.f32 v, score109, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score109, v;\n\t"
        "add.f32 sum, sum, score109;\n\t"
        "fma.rn.f32 v, score110, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score110, v;\n\t"
        "add.f32 sum, sum, score110;\n\t"
        "fma.rn.f32 v, score111, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score111, v;\n\t"
        "add.f32 sum, sum, score111;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score105, score104;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score107, score106;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score109, score108;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score111, score110;\n\t"
        "add.s32 addr, %3, 52;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score112, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score112, v;\n\t"
        "add.f32 sum, sum, score112;\n\t"
        "fma.rn.f32 v, score113, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score113, v;\n\t"
        "add.f32 sum, sum, score113;\n\t"
        "fma.rn.f32 v, score114, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score114, v;\n\t"
        "add.f32 sum, sum, score114;\n\t"
        "fma.rn.f32 v, score115, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score115, v;\n\t"
        "add.f32 sum, sum, score115;\n\t"
        "fma.rn.f32 v, score116, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score116, v;\n\t"
        "add.f32 sum, sum, score116;\n\t"
        "fma.rn.f32 v, score117, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score117, v;\n\t"
        "add.f32 sum, sum, score117;\n\t"
        "fma.rn.f32 v, score118, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score118, v;\n\t"
        "add.f32 sum, sum, score118;\n\t"
        "fma.rn.f32 v, score119, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score119, v;\n\t"
        "add.f32 sum, sum, score119;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score113, score112;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score115, score114;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score117, score116;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score119, score118;\n\t"
        "add.s32 addr, %3, 56;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "fma.rn.f32 v, score120, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score120, v;\n\t"
        "add.f32 sum, sum, score120;\n\t"
        "fma.rn.f32 v, score121, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score121, v;\n\t"
        "add.f32 sum, sum, score121;\n\t"
        "fma.rn.f32 v, score122, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score122, v;\n\t"
        "add.f32 sum, sum, score122;\n\t"
        "fma.rn.f32 v, score123, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score123, v;\n\t"
        "add.f32 sum, sum, score123;\n\t"
        "fma.rn.f32 v, score124, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score124, v;\n\t"
        "add.f32 sum, sum, score124;\n\t"
        "fma.rn.f32 v, score125, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score125, v;\n\t"
        "add.f32 sum, sum, score125;\n\t"
        "fma.rn.f32 v, score126, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score126, v;\n\t"
        "add.f32 sum, sum, score126;\n\t"
        "fma.rn.f32 v, score127, %13, neg_m_scaled;\n\t"
        "ex2.approx.ftz.f32 score127, v;\n\t"
        "add.f32 sum, sum, score127;\n\t"
        "cvt.rn.bf16x2.f32 packed0, score121, score120;\n\t"
        "cvt.rn.bf16x2.f32 packed1, score123, score122;\n\t"
        "cvt.rn.bf16x2.f32 packed2, score125, score124;\n\t"
        "cvt.rn.bf16x2.f32 packed3, score127, score126;\n\t"
        "add.s32 addr, %3, 60;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [addr], {packed0, packed1, packed2, packed3};\n\t"
        "L_p_done:\n\t"
        "fma.rn.f32 %1, %2, %10, sum;\n\t"
        "@!pvalid mov.f32 %1, %10;\n\t"
        // `prescale` is warp-uniform after vote.sync.any.  l has already
        // consumed the positive alpha, so use its sign only as the return
        // flag that lets the correction warp skip its O read-modify-write.
        "@!prescale neg.f32 %2, %2;\n\t"
        "}\n"
        : "=f"(update.m), "=f"(update.l), "=f"(update.alpha)
        : "r"(s_tmem_row_addr),
        "r"(tile_key_begin),
        "r"(key_len),
        "r"(q_row),
        "r"(valid_q_row),
        "f"(scale),
        "f"(tilem),
        "f"(tilel),
        "f"(conditional_rescale_threshold),
        "r"(all_scores_valid),
        "f"(scale_log2e)
        : "memory");
    return update;
}

__device__ inline void tcgen05_commit_mbarrier(
    int mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
                 "[%0];" ::"r"(mbar_addr)
        : "memory");
}

__device__ inline void mbarrier_arrive(
    int mbar_addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(mbar_addr)
        : "memory");
}

__host__ __device__ constexpr size_t align_up(
    size_t value,
    size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Dynamic shared-memory layout for one CTA.  Two Q slots remain resident for
// the whole KV loop.  Each KV ring stage is a single physical buffer: K is
// consumed by QK, then V overwrites precisely that buffer for PV.
template <int TileQ, int TileKV, int QSlots, int KvStages> struct FaMhaSmemLayout {
    static constexpr size_t kAlignment = 128;
    static constexpr size_t kMaxSmemBytes = 227 * 1024;
    static constexpr size_t kStaticMbarrierBytes =
        (QSlots + 2 + (2 + 3 * QSlots) * KvStages) * sizeof(uint64_t);

    size_t q_offset;
    size_t kv_offset;
    size_t dynamic_bytes;

    __host__ __device__ static FaMhaSmemLayout make(
        int head_dim) {
        const size_t q_bytes =
            QSlots * TileQ * static_cast<size_t>(head_dim) * sizeof(__nv_bfloat16);
        const size_t kv_bytes =
            KvStages * TileKV * static_cast<size_t>(head_dim) * sizeof(__nv_bfloat16);
        FaMhaSmemLayout layout{};
        layout.q_offset = 0;
        layout.kv_offset = align_up(layout.q_offset + q_bytes, kAlignment);
        layout.dynamic_bytes = align_up(layout.kv_offset + kv_bytes, kAlignment);
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

// A CTA owns two independent Q tiles for one head.  K/V are shared between
// those slots; S is transient, while each slot's unnormalized O stays in TMEM.
template <int TileQ, int TileKV, int QSlots, int KvStages, int HeadDim>
__global__ __launch_bounds__(384, 1) void fa_mha_kernel(const __grid_constant__ CUtensorMap q_tmap,
    const __grid_constant__ CUtensorMap k_tmap,
    const __grid_constant__ CUtensorMap v_tmap,
    __nv_bfloat16 *__restrict__ out,
    int batch,
    int heads,
    int query_len,
    int key_len,
    float scale,
    bool causal) {
    static_assert(TileQ == 128, "the FA MHA kernel currently supports TileQ=128 only");
    static_assert(TileKV > 0, "attention tiles must be positive");
    static_assert(QSlots == 2, "the FA MHA mainloop currently supports two Q slots");
    static_assert(KvStages == 2, "the K/V mainloop currently uses two stages");
    static_assert(HeadDim == 128, "the FA MHA kernel currently supports HeadDim=128 only");

    // P and alpha use exp2((raw_score - raw_m) * scale * log2(e)).
    const float scale_log2e = scale * 1.4426950408889634f;

    using SmemLayout = FaMhaSmemLayout<TileQ, TileKV, QSlots, KvStages>;
    extern __shared__ __align__(1024) char smem[];
    const SmemLayout smem_layout = SmemLayout::make(HeadDim);
    auto *q_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.q_offset);
    // K and V intentionally alias.  The K producer may reuse a stage only
    // after both PV consumers commit mma2; the V producer overwrites K only
    // after both QK consumers commit mma1.
    auto *kv_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.kv_offset);

    // Producer/consumer graph for each (Q slot, KV stage):
    //   K TMA -> QK MMA -> rescale -> PV MMA -> recycle K/V stage.
    // Each resident Q tile has an independent prologue mbarrier; all remaining
    // edges need one independent barrier per two-stage-ring entry.
    __shared__ __align__(8) uint64_t q_prologue_mbars[QSlots];
    __shared__ __align__(8) uint64_t tma1_mbars[KvStages];
    __shared__ __align__(8) uint64_t tma2_mbars[KvStages];
    __shared__ __align__(8) uint64_t mma1_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t mma2_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t rescale_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t epilogue_mbar[1];
    __shared__ __align__(8) uint64_t tmem_release_mbar[1];
    __shared__ __align__(4) int mma_tmem_addr[1];

    constexpr int kWarpSize = 32;
    // CTA role assignment (12 warps = 384 threads).  Each rescale warpgroup
    // owns one Q tile, so its S/P/O dependency chain is independent of the
    // other resident Q tile.
    constexpr int kKTmaWarp = 0;      // K shared-memory producer
    constexpr int kVTmaWarp = 1;      // V shared-memory producer
    constexpr int kMmaWarp = 2;       // QK and PV tcgen05 issuer
    constexpr int kQPrologueWarp = 3; // loads the two resident Q tiles
    constexpr int kRescaleWarpsPerGroup = TileQ / kWarpSize;
    constexpr int kRescaleWarpBegin = 4; // Q0: warps 4-7; Q1: warps 8-11
    constexpr int kRescaleWarpEnd = kRescaleWarpBegin + QSlots * kRescaleWarpsPerGroup;
    constexpr int kTmaMbarArrivalCount = 1;
    constexpr int kMmaMbarArrivalCount = 1;
    // A 128-row TMEM tile is partitioned into four 32-row regions.  Each
    // rescale mbarrier belongs to exactly one Q slot, so only its own four
    // warps arrive; Q0 never waits for Q1 (or vice versa) before PV can issue.
    constexpr int kRescaleMbarArrivalCount = kRescaleWarpsPerGroup;
    constexpr int kTmemReleaseMbarArrivalCount = QSlots * kRescaleWarpsPerGroup;
    constexpr int kQPrologueMbarArrivalCount = 1;

    const int tid = threadIdx.x;
    const int warp_idx = threadIdx.x / kWarpSize;
    const int lane_idx = threadIdx.x % kWarpSize;

    const int q_prologue_mbar_base = static_cast<int>(__cvta_generic_to_shared(q_prologue_mbars));
    const int tma1_mbar_base = static_cast<int>(__cvta_generic_to_shared(tma1_mbars));
    const int tma2_mbar_base = static_cast<int>(__cvta_generic_to_shared(tma2_mbars));
    const int mma1_mbar_base = static_cast<int>(__cvta_generic_to_shared(mma1_mbars));
    const int mma2_mbar_base = static_cast<int>(__cvta_generic_to_shared(mma2_mbars));
    const int rescale_mbar_base = static_cast<int>(__cvta_generic_to_shared(rescale_mbars));
    const int epilogue_mbar_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_mbar));
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
        for (int stage = 0; stage < KvStages; ++stage) {
            const int offset = stage * sizeof(uint64_t);
            mbarrier_init(tma1_mbar_base + offset, kTmaMbarArrivalCount);
            mbarrier_init(tma2_mbar_base + offset, kTmaMbarArrivalCount);
            for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                const int q_offset = (q_slot * KvStages + stage) * sizeof(uint64_t);
                mbarrier_init(mma1_mbar_base + q_offset, kMmaMbarArrivalCount);
                mbarrier_init(mma2_mbar_base + q_offset, kMmaMbarArrivalCount);
                mbarrier_init(rescale_mbar_base + q_offset, kRescaleMbarArrivalCount);
            }
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp_idx == kQPrologueWarp && lane_idx == 0) {
        prefetch_tensormap(&q_tmap);
    }
    if (warp_idx == kKTmaWarp && lane_idx == 0) {
        prefetch_tensormap(&k_tmap);
    }
    if (warp_idx == kVTmaWarp && lane_idx == 0) {
        prefetch_tensormap(&v_tmap);
    }
    __syncthreads();

    // S0/S1 are temporary score/probability tiles. O0/O1 are the two long-lived
    // unnormalized accumulators, one per resident Q slot.
    // tcgen05 allocation units are power-of-two column counts in [32, 512].
    constexpr int kTmemColumnsPerBank = 128;
    constexpr int kSBankBase = 0;
    constexpr int kOBankBase = QSlots * kTmemColumnsPerBank;
    // S0/S1/O0/O1: 4 * 128 = 512 TMEM columns.
    constexpr int mma_tmem_columns_per_stage = kTmemColumnsPerBank;
    constexpr int mma_tmem_columns = 4 * kTmemColumnsPerBank;
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
    const int tmem_addr = mma_tmem_addr[0];

    // Launch assigns exactly one adjacent Q-tile pair to each CTA.
    const int q_tile_count = (query_len + TileQ - 1) / TileQ;
    const int q_pair_count = (q_tile_count + QSlots - 1) / QSlots;
    const int total_task_count = q_pair_count * batch * heads;
    const int task_id = blockIdx.x;
    if (task_id >= total_task_count) {
        return;
    }
    const int kv_tile_count = (key_len + TileKV - 1) / TileKV;
    const int bh = task_id / q_pair_count;
    const int q_pair_idx = task_id - bh * q_pair_count;
    const int q_tile_start[QSlots] = {
        (q_pair_idx * QSlots + 0) * TileQ,
        (q_pair_idx * QSlots + 1) * TileQ,
    };
    const int q_pair_end =
        q_tile_start[1] + TileQ < query_len ? q_tile_start[1] + TileQ : query_len;
    const int task_kv_tile_count = causal ? ((q_pair_end + TileKV - 1) / TileKV < kv_tile_count
                                                    ? (q_pair_end + TileKV - 1) / TileKV
                                                    : kv_tile_count)
                                          : kv_tile_count;
    const int rescale_warp = warp_idx >= kRescaleWarpBegin && warp_idx < kRescaleWarpEnd
                                 ? warp_idx - kRescaleWarpBegin
                                 : -1;

    if (rescale_warp >= 0) {
        setmaxnreg_increase<208>();
        int epilogue_phase = 0;
        // The two four-warp groups have separate Q-tile ownership and
        // separate softmax state.  Consequently, q_slot's mma1, rescale,
        // and PV chain can advance without any rescale-side dependency on
        // the other tile.
        const int q_slot = rescale_warp / kRescaleWarpsPerGroup;
        const int row_base = (rescale_warp % kRescaleWarpsPerGroup) * kWarpSize;
        const int row = row_base + lane_idx;
        const int q_row = q_tile_start[q_slot] + row;
        float tilel = 0.0f;
        float tilem = -__int_as_float(0x7F800000);
        int rescale_mma1_phase[KvStages] = {};
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
            mbarrier_wait(mma1_addr, rescale_mma1_phase[stage]);
            rescale_mma1_phase[stage] ^= 1;

            const bool valid_q_row = q_row < query_len;
            const int s_tmem_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
            const int mask_q_row = causal ? q_row : key_len - 1;
            // The hardware reduction is correct only if all 128 S values
            // belong to this row's attention domain.  The diagonal causal
            // tile and the final KV tail remain on the masked scalar-max path.
            const bool all_scores_valid = valid_q_row && (kv_tile + 1) * TileKV <= key_len &&
                                          (kv_tile + 1) * TileKV <= mask_q_row + 1;
            const SoftmaxUpdate softmax = load_s_and_store_p_ptx(s_tmem_addr + (row_base << 16),
                kv_tile * TileKV,
                key_len,
                mask_q_row,
                valid_q_row,
                scale,
                tilem,
                tilel,
                kConditionalRescaleThresholdLn,
                all_scores_valid,
                scale_log2e);
            tilem = softmax.m;
            tilel = softmax.l;

            // `alpha < 0` denotes the warp-uniform conditional-skip path.
            // Do not merely multiply O by one: omit the whole TMEM
            // read/modify/write, which is the intended FA4 win.
            if (kv_tile != 0 && softmax.alpha > 0.0f) {
                const int o_tmem_addr =
                    tmem_addr + kOBankBase + q_slot * mma_tmem_columns_per_stage;
                for (int d8 = 0; d8 < HeadDim / 8; ++d8) {
                    float o_fragment[8];
                    const int addr = o_tmem_addr + (row_base << 16) + d8 * 8;
                    tcgen05_ld_f32_x8(o_fragment, addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        o_fragment[e] *= softmax.alpha;
                    }
                    tcgen05_st_f32_x8(o_fragment, addr);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            if (lane_idx == 0) {
                mbarrier_arrive(rescale_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t));
            }
        }

        mbarrier_wait(epilogue_mbar_addr, epilogue_phase);
        epilogue_phase ^= 1;
        const bool epilogue_valid_q_row = q_row < query_len;
        const float inv_l = epilogue_valid_q_row ? 1.0f / tilel : 0.0f;
        const int out_base = epilogue_valid_q_row ? (bh * query_len + q_row) * HeadDim : 0;
        const int o_tmem_addr = tmem_addr + kOBankBase + q_slot * mma_tmem_columns_per_stage;
        for (int d8 = 0; d8 < HeadDim / 8; ++d8) {
            float o_fragment[8];
            tcgen05_ld_f32_x8(o_fragment, o_tmem_addr + (row_base << 16) + d8 * 8);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            if (epilogue_valid_q_row) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    out[out_base + d8 * 8 + e] = __float2bfloat16_rn(o_fragment[e] * inv_l);
                }
            }
        }
        // Protect TMEM deallocation until all epilogue reads finish.
        if (lane_idx == 0) {
            mbarrier_arrive(tmem_release_mbar_addr);
        }
        return;
    }

    // WG0 releases registers before its four permanent driver roles diverge.
    setmaxnreg_decrease<80>();

    // Load both resident Q tiles through independent prologue barriers.
    if (warp_idx == kQPrologueWarp) {
        int q_prologue_phase[QSlots] = {};
        if (lane_idx == 0) {
            const int q_transaction_bytes = TileQ * HeadDim * sizeof(__nv_bfloat16);
            for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                const int q_mbar_addr = q_prologue_mbar_base + q_slot * sizeof(uint64_t);
                const int q_transaction_mbar_addr = q_mbar_addr & 0xFEFFFFFF;
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::"r"(
                        q_transaction_mbar_addr),
                    "r"(q_transaction_bytes)
                    : "memory");
                tma_4d_gmem2smem(q_smem_addr + q_slot * TileQ * HeadDim * sizeof(__nv_bfloat16),
                    &q_tmap,
                    0,
                    q_tile_start[q_slot],
                    0,
                    bh,
                    q_transaction_mbar_addr);
                mbarrier_wait(q_mbar_addr, q_prologue_phase[q_slot]);
                q_prologue_phase[q_slot] ^= 1;
            }
        }
        return;
    }

    // K producer: stage may be overwritten only after PV has consumed it
    // for *both* Q slots, hence the two mma2 waits on ring reuse.
    if (warp_idx == kKTmaWarp) {
        int k_recycle_phase[QSlots][KvStages] = {};
        bool k_slot_initialized[KvStages] = {};
        if (lane_idx == 0) {
            for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                const int stage = kv_tile % KvStages;
                if (k_slot_initialized[stage]) {
                    for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                        const int addr =
                            mma2_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                        mbarrier_wait(addr, k_recycle_phase[q_slot][stage]);
                        k_recycle_phase[q_slot][stage] ^= 1;
                    }
                }
                const int addr = tma1_mbar_base + stage * sizeof(uint64_t);
                const int tx_addr = addr & 0xFEFFFFFF;
                const int transaction_bytes = TileKV * HeadDim * sizeof(__nv_bfloat16);
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::"r"(
                        tx_addr),
                    "r"(transaction_bytes)
                    : "memory");
                tma_4d_gmem2smem(kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16),
                    &k_tmap,
                    0,
                    kv_tile * TileKV,
                    0,
                    bh,
                    tx_addr);
                k_slot_initialized[stage] = true;
            }
        }
        return;
    }

    // V producer: QK must finish before V overwrites the stage (mma1), and
    // the prior PV must finish before a reused V stage is overwritten.
    if (warp_idx == kVTmaWarp) {
        int v_mma1_phase[QSlots][KvStages] = {};
        int v_recycle_phase[QSlots][KvStages] = {};
        bool v_slot_initialized[KvStages] = {};
        if (lane_idx == 0) {
            for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                const int stage = kv_tile % KvStages;
                for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                    const int addr =
                        mma1_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                    mbarrier_wait(addr, v_mma1_phase[q_slot][stage]);
                    v_mma1_phase[q_slot][stage] ^= 1;
                }
                if (v_slot_initialized[stage]) {
                    for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                        const int addr =
                            mma2_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                        mbarrier_wait(addr, v_recycle_phase[q_slot][stage]);
                        v_recycle_phase[q_slot][stage] ^= 1;
                    }
                }
                const int addr = tma2_mbar_base + stage * sizeof(uint64_t);
                const int tx_addr = addr & 0xFEFFFFFF;
                const int transaction_bytes = TileKV * HeadDim * sizeof(__nv_bfloat16);
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::"r"(
                        tx_addr),
                    "r"(transaction_bytes)
                    : "memory");
                tma_4d_gmem2smem(kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16),
                    &v_tmap,
                    0,
                    kv_tile * TileKV,
                    0,
                    bh,
                    tx_addr);
                v_slot_initialized[stage] = true;
            }
        }
        return;
    }

    // A single elected thread issues tcgen05 operations.  The asynchronous
    // engines perform the actual matrix work while producer/rescale warps run.
    if (warp_idx == kMmaWarp) {
        int qk_tma1_phase[KvStages] = {};
        int pv_tma2_phase[KvStages] = {};
        int pv_rescale_phase[QSlots][KvStages] = {};
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
            // Keep the two Q tiles staggered around the PV work.  At iteration
            // i the tensor-core issue order is Q0(i), P1(i - 1), Q1(i), P0(i).
            // P1(0) has no predecessor and is deliberately omitted.  This
            // gives the Q0 rescale work the intervening P1 issue window
            // before its PV is consumed, while P1(i) is flushed at i + 1.
            for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                const int stage = kv_tile % KvStages;
                mbarrier_wait(tma1_mbar_base + stage * sizeof(uint64_t), qk_tma1_phase[stage]);
                qk_tma1_phase[stage] ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
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
                    tcgen05_commit_mbarrier(
                        mma1_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t));
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
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    const int o_addr = tmem_addr + kOBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int p_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int v_addr =
                        kv_smem_addr + pv_stage * TileKV * HeadDim * sizeof(__nv_bfloat16);
                    for (int k16 = 0; k16 < TileKV / 16; ++k16) {
                        tcgen05_mma_tmem_a_bf16(o_addr,
                            p_addr + k16 * 8,
                            make_128b_nmajor_desc(v_addr + k16 * 16 * 128, TileKV * 128),
                            pv_instr_desc,
                            pv_tile != 0 || k16 != 0);
                    }
                    tcgen05_commit_mbarrier(
                        mma2_mbar_base + (q_slot * KvStages + pv_stage) * sizeof(uint64_t));
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
                    tcgen05_commit_mbarrier(
                        mma1_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t));
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
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    const int o_addr = tmem_addr + kOBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int p_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int v_addr =
                        kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16);
                    for (int k16 = 0; k16 < TileKV / 16; ++k16) {
                        tcgen05_mma_tmem_a_bf16(o_addr,
                            p_addr + k16 * 8,
                            make_128b_nmajor_desc(v_addr + k16 * 16 * 128, TileKV * 128),
                            pv_instr_desc,
                            kv_tile != 0 || k16 != 0);
                    }
                    tcgen05_commit_mbarrier(
                        mma2_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t));
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
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const int o_addr = tmem_addr + kOBankBase + q_slot * mma_tmem_columns_per_stage;
                const int p_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
                const int v_addr =
                    kv_smem_addr + pv_stage * TileKV * HeadDim * sizeof(__nv_bfloat16);
                for (int k16 = 0; k16 < TileKV / 16; ++k16) {
                    tcgen05_mma_tmem_a_bf16(o_addr,
                        p_addr + k16 * 8,
                        make_128b_nmajor_desc(v_addr + k16 * 16 * 128, TileKV * 128),
                        pv_instr_desc,
                        pv_tile != 0 || k16 != 0);
                }
                tcgen05_commit_mbarrier(
                    mma2_mbar_base + (q_slot * KvStages + pv_stage) * sizeof(uint64_t));
            }
            // A fresh, non-ring completion point protects the final O from
            // parity wraparound on the recycled mma2 barriers.
            tcgen05_commit_mbarrier(epilogue_mbar_addr);
        }
        mbarrier_wait(tmem_release_mbar_addr, 0);
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_addr),
            "r"(mma_tmem_columns));
        return;
    }
}

} // namespace

cudaError_t bf16_fa4_mha_cuda(
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
    constexpr int kQSlots = 2;
    constexpr int kKvStages = 2;
    constexpr int kHeadDim = 128;
    constexpr int kThreads = 12 * 32;
    // S0/S1 plus long-lived O0/O1 consume all 512 available TMEM columns.
    if (head_dim != kHeadDim) {
        return cudaErrorInvalidValue;
    }

    CUtensorMap q_tmap{};
    CUtensorMap k_tmap{};
    CUtensorMap v_tmap{};
    cudaError_t status = init_qkv_tmap(&q_tmap, q, batch * heads, query_len, head_dim, kTileQ);
    if (status != cudaSuccess) {
        return status;
    }
    status = init_qkv_tmap(&k_tmap, k, batch * heads, key_len, head_dim, kTileKV);
    if (status != cudaSuccess) {
        return status;
    }
    status = init_qkv_tmap(&v_tmap, v, batch * heads, key_len, head_dim, kTileKV);
    if (status != cudaSuccess) {
        return status;
    }

    using SmemLayout = FaMhaSmemLayout<kTileQ, kTileKV, kQSlots, kKvStages>;
    const SmemLayout smem_layout = SmemLayout::make(head_dim);
    if (smem_layout.dynamic_bytes + SmemLayout::kStaticMbarrierBytes > SmemLayout::kMaxSmemBytes) {
        return cudaErrorInvalidConfiguration;
    }

    status = cudaFuncSetAttribute(fa_mha_kernel<kTileQ, kTileKV, kQSlots, kKvStages, kHeadDim>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_layout.dynamic_bytes));
    if (status != cudaSuccess) {
        return status;
    }

    const int q_tile_count = (query_len + kTileQ - 1) / kTileQ;
    const int q_pair_count = (q_tile_count + kQSlots - 1) / kQSlots;
    const int total_task_count = q_pair_count * batch * heads;
    // Barrier parity is task-local. Give every (head, Q-pair) task a fresh
    // CTA rather than carrying the two-stage ring state across tasks.

    const dim3 block(kThreads);
    const dim3 grid(total_task_count);
    fa_mha_kernel<kTileQ, kTileKV, kQSlots, kKvStages, kHeadDim>
        <<<grid, block, smem_layout.dynamic_bytes, stream>>>(q_tmap,
            k_tmap,
            v_tmap,
            out,
            batch,
            heads,
            query_len,
            key_len,
            scale,
            causal);
    return cudaGetLastError();
}
