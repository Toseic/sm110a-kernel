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
    float scale_log2e) {
    constexpr int kScoresPerChunk = 64;
    constexpr int kChunkCount = 2;
    float rmax = -__int_as_float(0x7F800000);

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
#pragma unroll
            for (int i = 0; i < kScoresPerChunk; ++i) {
                const int key = tile_key_begin + chunk * kScoresPerChunk + i;
                if (valid_q_row && key < key_len && key <= mask_q_row) {
                    rmax = fmaxf(rmax, scores[i]);
                }
            }
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
    float partial_sum[4] = {};

#pragma unroll
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
        float scores[kScoresPerChunk];
        tcgen05_ld_f32_x64(scores, s_tmem_row_addr + chunk * kScoresPerChunk);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
        for (int i = 0; i < kScoresPerChunk; ++i) {
            const int key = tile_key_begin + chunk * kScoresPerChunk + i;
            const bool score_valid = AllScoresValid ||
                (valid_q_row && key < key_len && key <= mask_q_row);
            scores[i] = score_valid ? exp2_approx((scores[i] - new_m) * scale_log2e) : 0.0f;
            partial_sum[i % 4] += scores[i];
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
        (2 * QSlots + 2 + (2 + 4 * QSlots) * KvStages) * sizeof(uint64_t);
    static constexpr size_t kSoftmaxHandoffBytes =
        2 * QSlots * TileQ * sizeof(float);

    size_t q_offset;
    size_t kv_offset;
    size_t out_offset;
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

// A CTA owns two independent Q tiles for one head.  K/V are shared between
// those slots; S is transient, while each slot's unnormalized O stays in TMEM.
template <int TileQ, int TileKV, int QSlots, int KvStages, int HeadDim>
__global__ __launch_bounds__(512, 1) void fa_mha_kernel(const __grid_constant__ CUtensorMap q_tmap,
    const __grid_constant__ CUtensorMap k_tmap,
    const __grid_constant__ CUtensorMap v_tmap,
    const __grid_constant__ CUtensorMap out_tmap,
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
    auto *out_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.out_offset);

    // Producer/consumer graph for each (Q slot, KV stage):
    //   K TMA -> QK MMA -> softmax -> O rescale -> PV MMA -> recycle K/V stage.
    // Each resident Q tile has an independent prologue mbarrier; all remaining
    // edges need one independent barrier per two-stage-ring entry.
    __shared__ __align__(8) uint64_t q_prologue_mbars[QSlots];
    __shared__ __align__(8) uint64_t tma1_mbars[KvStages];
    __shared__ __align__(8) uint64_t tma2_mbars[KvStages];
    __shared__ __align__(8) uint64_t mma1_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t mma2_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t softmax_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t rescale_mbars[QSlots][KvStages];
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
    constexpr int kSoftmaxWarpBegin = 4; // Q0: warps 4-7; Q1: warps 8-11
    constexpr int kSoftmaxWarpEnd = kSoftmaxWarpBegin + QSlots * kWarpsPerGroup;
    constexpr int kEpilogueWarpBegin = kSoftmaxWarpEnd; // warps 12-15
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
    const int mma1_mbar_base = static_cast<int>(__cvta_generic_to_shared(mma1_mbars));
    const int mma2_mbar_base = static_cast<int>(__cvta_generic_to_shared(mma2_mbars));
    const int softmax_mbar_base = static_cast<int>(__cvta_generic_to_shared(softmax_mbars));
    const int rescale_mbar_base = static_cast<int>(__cvta_generic_to_shared(rescale_mbars));
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
            for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                const int q_offset = (q_slot * KvStages + stage) * sizeof(uint64_t);
                mbarrier_init(mma1_mbar_base + q_offset, kMmaMbarArrivalCount);
                mbarrier_init(mma2_mbar_base + q_offset, kMmaMbarArrivalCount);
                mbarrier_init(softmax_mbar_base + q_offset, kSoftmaxMbarArrivalCount);
                mbarrier_init(rescale_mbar_base + q_offset, kRescaleMbarArrivalCount);
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
    const int out_smem_addr = static_cast<int>(__cvta_generic_to_shared(out_smem));
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

            const bool valid_q_row = q_row < query_len;
            const int s_tmem_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
            const int mask_q_row = causal ? q_row : key_len - 1;
            // Select uniformly across the warp: the earliest row must see
            // the entire KV tile and all 32 query rows must be in bounds.
            const bool all_scores_valid =
                q_tile_start[q_slot] + row_base + kWarpSize <= query_len &&
                (kv_tile + 1) * TileKV <= key_len &&
                (!causal || (kv_tile + 1) * TileKV - 1 <= q_tile_start[q_slot] + row_base);
            const SoftmaxUpdate softmax = all_scores_valid
                ? load_s_and_store_p_chunked<true>(
                    s_tmem_addr + (row_base << 16), kv_tile * TileKV, key_len,
                    mask_q_row, valid_q_row, scale, tilem, tilel,
                    kConditionalRescaleThresholdLn, scale_log2e)
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
                scale_log2e);
            tilem = softmax.m;
            tilel = softmax.l;
            // Complete all P stores before publishing the scalar handoff.
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            softmax_alpha[q_slot][row] = softmax.alpha;
            softmax_l[q_slot][row] = tilel;
            __syncwarp();
            if (lane_idx == 0) {
                mbarrier_arrive(
                    softmax_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t));
            }
        }
        return;
    }

    if (epilogue_warp >= 0) {
        // This WG needs only the 8-value O fragment plus scalar loop state.
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
                    for (int d8 = 0; d8 < HeadDim / 8; ++d8) {
                        float o_fragment[8];
                        const int addr = o_tmem_addr + (row_base << 16) + d8 * 8;
                        tcgen05_ld_f32_x8(o_fragment, addr);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            o_fragment[e] *= alpha;
                        }
                        tcgen05_st_f32_x8(o_fragment, addr);
                    }
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (lane_idx == 0) {
                    mbarrier_arrive(
                        rescale_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t));
                }
            }
        }

        mbarrier_wait(epilogue_mbar_addr, 0);
        for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
            const int q_row = q_tile_start[q_slot] + row;
            const bool valid_q_row = q_row < query_len;
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

    // The sole load warp owns Q, K, and V TMA traffic. For each KV tile it
    // loads K, waits until both QK operations release the aliased stage, then
    // overwrites that stage with V. Before ring reuse, both prior PV users
    // must have completed.
    if (warp_idx == kTmaLoadWarp) {
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
            }
            const int transaction_bytes = TileKV * HeadDim * sizeof(__nv_bfloat16);
            // Fill both K stages before waiting for QK. This lets K(i + 1)
            // transfer while the tensor core and softmax pipeline consume i.
            const int k_prologue_tiles =
                task_kv_tile_count < KvStages ? task_kv_tile_count : KvStages;
            for (int kv_tile = 0; kv_tile < k_prologue_tiles; ++kv_tile) {
                const int stage = kv_tile;
                const int k_addr = tma1_mbar_base + stage * sizeof(uint64_t);
                const int k_tx_addr = k_addr & 0xFEFFFFFF;
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::"r"(
                        k_tx_addr),
                    "r"(transaction_bytes)
                    : "memory");
                tma_4d_gmem2smem(kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16),
                    &k_tmap,
                    0,
                    kv_tile * TileKV,
                    0,
                    bh,
                    k_tx_addr);
            }

            for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                const int stage = kv_tile % KvStages;
                const int mma1_phase = (kv_tile / KvStages) & 1;
                for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                    const int addr =
                        mma1_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                    mbarrier_wait(addr, mma1_phase);
                }
                const int v_addr = tma2_mbar_base + stage * sizeof(uint64_t);
                const int v_tx_addr = v_addr & 0xFEFFFFFF;
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::"r"(
                        v_tx_addr),
                    "r"(transaction_bytes)
                    : "memory");
                tma_4d_gmem2smem(kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16),
                    &v_tmap,
                    0,
                    kv_tile * TileKV,
                    0,
                    bh,
                    v_tx_addr);

                // Once both PV consumers release V(i), immediately refill
                // that stage with K(i + 2). The other K stage was prefetched
                // already, so the MMA issuer need not wait on this recycle.
                const int next_k_tile = kv_tile + KvStages;
                if (next_k_tile < task_kv_tile_count) {
                    const int recycle_phase = (kv_tile / KvStages) & 1;
                    for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                        const int addr =
                            mma2_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                        mbarrier_wait(addr, recycle_phase);
                    }
                    const int next_k_addr = tma1_mbar_base + stage * sizeof(uint64_t);
                    const int next_k_tx_addr = next_k_addr & 0xFEFFFFFF;
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::
                            "r"(next_k_tx_addr),
                            "r"(transaction_bytes)
                        : "memory");
                    tma_4d_gmem2smem(
                        kv_smem_addr + stage * TileKV * HeadDim * sizeof(__nv_bfloat16),
                        &k_tmap,
                        0,
                        next_k_tile * TileKV,
                        0,
                        bh,
                        next_k_tx_addr);
                }
            }
        }
        return;
    }

    if (warp_idx == kTmaStoreWarp) {
        if (lane_idx == 0) {
            for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                mbarrier_wait(out_store_ready_mbar_base + q_slot * sizeof(uint64_t), 0);
                tma_4d_smem2gmem(
                    out_smem_addr + q_slot * TileQ * HeadDim * sizeof(__nv_bfloat16),
                    &out_tmap,
                    0,
                    q_tile_start[q_slot],
                    0,
                    bh);
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
    constexpr int kThreads = 16 * 32;
    // S0/S1 plus long-lived O0/O1 consume all 512 available TMEM columns.
    if (head_dim != kHeadDim) {
        return cudaErrorInvalidValue;
    }

    CUtensorMap q_tmap{};
    CUtensorMap k_tmap{};
    CUtensorMap v_tmap{};
    CUtensorMap out_tmap{};
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
    status = init_qkv_tmap(&out_tmap, out, batch * heads, query_len, head_dim, kTileQ);
    if (status != cudaSuccess) {
        return status;
    }

    using SmemLayout = FaMhaSmemLayout<kTileQ, kTileKV, kQSlots, kKvStages>;
    const SmemLayout smem_layout = SmemLayout::make(head_dim);
    if (smem_layout.dynamic_bytes + SmemLayout::kStaticMbarrierBytes +
            SmemLayout::kSoftmaxHandoffBytes >
        SmemLayout::kMaxSmemBytes) {
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
            out_tmap,
            batch,
            heads,
            query_len,
            key_len,
            scale,
            causal);
    return cudaGetLastError();
}
