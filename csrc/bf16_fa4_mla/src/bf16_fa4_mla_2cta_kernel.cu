#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// MLA decoding: H=128, QK=576, V=512.  A two-CTA cluster splits H over its
// ranks.  The CTAs issue QK and PV cooperatively and each rank computes 64
// heads and all 512 output columns.
#define SM110A_FA4_MLA_PACK_SIZE 64

namespace sm110a::bf16_fa4_mla_2cta {

constexpr int kMlaPackSize = SM110A_FA4_MLA_PACK_SIZE;
static_assert(kMlaPackSize == 64, "MLA decode packs 64 heads per CTA");

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
// Parity lets a barrier be reused for successive K chunks or KV tiles.
__device__ inline void mbarrier_wait(
    int mbar_addr,
    int phase) {
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "FA_MLA_MBAR_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni FA_MLA_MBAR_DONE;\n\t"
                 "bra.uni FA_MLA_MBAR_WAIT;\n\t"
                 "FA_MLA_MBAR_DONE:\n\t"
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
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
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
        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
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
                 "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t"
                 "}" ::"r"(tmem_addr),
        "l"(a_desc),
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


struct SoftmaxUpdate {
    // Running maximum in the unscaled QK domain.
    float m;
    float l;
    float alpha;
};

__device__ inline void mbarrier_arrive(int mbar_addr);

// M=128 cta_group::2 uses layout B. Each CTA owns 64 logical query rows;
// N=64 is routed as four 32-datapath quadrants. The four softmax warps each
// consume one quadrant, and paired warps exchange max/sum through shared
// memory while retaining the existing four-warp organization.
template <bool AllScoresValid>
__device__ __attribute__((noinline)) SoftmaxUpdate load_s_and_store_p_m64(
    int s_tmem_warp_addr,
    char *p_smem_bytes,
    float *exchange,
    int warp_in_group,
    int tile_key_begin,
    int key_len,
    int mask_q_row,
    float scale,
    float tilem,
    float tilel,
    float conditional_rescale_threshold,
    float scale_log2e,
    int lane_idx) {
    constexpr int kScoresPerLane = 32;
    const int half = warp_in_group >> 1;
    const int logical_row = (warp_in_group & 1) * 32 + lane_idx;
    const int key_begin = tile_key_begin + half * kScoresPerLane;
    const int key_hi = min(key_len - 1, mask_q_row);
    float scores[kScoresPerLane];
    tcgen05_ld_f32_x32(scores, s_tmem_warp_addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    float rmax = -__int_as_float(0x7F800000);
#pragma unroll
    for (int i = 0; i < kScoresPerLane; ++i) {
        if constexpr (AllScoresValid) {
            rmax = fmaxf(rmax, scores[i]);
        } else if (key_begin + i <= key_hi) {
            rmax = fmaxf(rmax, scores[i]);
        }
    }
    exchange[half * 64 + logical_row] = rmax;
    asm volatile("bar.sync 1, 128;" ::: "memory");
    rmax = fmaxf(rmax, exchange[(half ^ 1) * 64 + logical_row]);
    // Keep either half from reusing exchange for the partial sum until its
    // peer has consumed the max written above.
    asm volatile("bar.sync 1, 128;" ::: "memory");
    float new_m = fmaxf(tilem, rmax);
    const bool needs_rescale = (new_m - tilem) * scale > conditional_rescale_threshold;
    const bool warp_rescale = __any_sync(0xFFFFFFFFU, needs_rescale);
    if (!warp_rescale) {
        new_m = tilem;
    }
    const float alpha = tilel != 0.0f
        ? exp2_approx((tilem - new_m) * scale_log2e)
        : 0.0f;
    const float neg_m_scale = -new_m * scale_log2e;
    float partial_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kScoresPerLane; ++i) {
        if constexpr (AllScoresValid) {
            scores[i] = exp2_approx(fmaf(scores[i], scale_log2e, neg_m_scale));
        } else {
            scores[i] = key_begin + i <= key_hi
                ? exp2_approx(fmaf(scores[i], scale_log2e, neg_m_scale))
                : 0.0f;
        }
        partial_sum += scores[i];
    }

    // Paired warps store disjoint halves of one SW128 P row.
#pragma unroll
    for (int group = 0; group < kScoresPerLane / 8; ++group) {
        const int i = group * 8;
        const uint4 packed = make_uint4(
            pack_bf16x2(scores[i + 0], scores[i + 1]),
            pack_bf16x2(scores[i + 2], scores[i + 3]),
            pack_bf16x2(scores[i + 4], scores[i + 5]),
            pack_bf16x2(scores[i + 6], scores[i + 7]));
        const int logical_byte =
            (logical_row * 64 + half * kScoresPerLane + i) * sizeof(__nv_bfloat16);
        const int swizzled_byte = logical_byte ^ ((logical_byte >> 3) & 0x70);
        *reinterpret_cast<uint4 *>(p_smem_bytes + swizzled_byte) = packed;
    }
    exchange[half * 64 + logical_row] = partial_sum;
    asm volatile("bar.sync 1, 128;" ::: "memory");
    partial_sum += exchange[(half ^ 1) * 64 + logical_row];
    // The next KV tile reuses the same slots for rmax.
    asm volatile("bar.sync 1, 128;" ::: "memory");

    SoftmaxUpdate update;
    update.m = new_m;
    update.l = alpha * tilel + partial_sum;
    update.alpha = warp_rescale ? alpha : -alpha;
    return update;
}

__device__ inline void tcgen05_commit_mbarrier(
    int mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
                 "multicast::cluster.b64 [%0], %1;" ::"r"(mbar_addr), "h"(int16_t{3})
        : "memory");
}


__device__ inline void mbarrier_arrive(
    int mbar_addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(mbar_addr)
        : "memory");
}

__device__ inline void mbarrier_arrive_on_peer(
    int mbar_addr,
    int peer_cta_rank) {
    int peer_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(peer_addr)
        : "r"(mbar_addr), "r"(peer_cta_rank));
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
        :: "r"(peer_addr)
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

// Q [64,576] is resident. Each stage holds the rank-local halves needed by
// cta_group::2: QK K=[32,576] and PV V=[64,256]. P0 is explicitly allocated;
// P1 steals the QK storage of stage 1 after that stage's QK has completed.
// The final O [64,512] epilogue reuses Q after the last QK.
template <int TileQ, int TileKV, int QSlots, int KvStages> struct FaMlaSmemLayout {
    static constexpr size_t kAlignment = 128;
    static constexpr size_t kMaxSmemBytes = 227 * 1024;
    static constexpr int kKChunkCount = 9;
    static constexpr int kValueChunkCount = 8;
    static constexpr size_t kStaticMbarrierBytes =
        (kKChunkCount +
         KvStages * (2 * kKChunkCount + kValueChunkCount + 5) + 3) *
        sizeof(uint64_t);
    static constexpr size_t kSoftmaxHandoffBytes =
        4 * QSlots * TileQ * sizeof(float);

    size_t q_offset;
    size_t kv_offset;
    size_t p_offset;
    size_t out_offset;
    size_t dynamic_bytes;

    __host__ __device__ static FaMlaSmemLayout make(
        int head_dim) {
        const size_t q_bytes = QSlots * TileQ * static_cast<size_t>(head_dim) *
            sizeof(__nv_bfloat16);
        const size_t qk_k_bytes =
            KvStages * (TileKV / 2) * static_cast<size_t>(head_dim) * sizeof(__nv_bfloat16);
        const size_t pv_v_bytes =
            KvStages * TileKV * size_t{256} * sizeof(__nv_bfloat16);
        const size_t kv_bytes = qk_k_bytes + pv_v_bytes;
        const size_t p_bytes = TileQ * TileKV * sizeof(__nv_bfloat16);
        FaMlaSmemLayout layout{};
        layout.q_offset = 0;
        layout.kv_offset = align_up(layout.q_offset + q_bytes, kAlignment);
        layout.p_offset = align_up(layout.kv_offset + kv_bytes, kAlignment);
        layout.out_offset = layout.q_offset;
        layout.dynamic_bytes = align_up(layout.p_offset + p_bytes, kAlignment);
        return layout;
    }
};

// K/V use contiguous [B,1,S,D] storage. tile_dim selects a contiguous
// feature slice without changing the full tensor's physical strides.
inline cudaError_t init_qkv_tmap(
    CUtensorMap *tmap,
    const __nv_bfloat16 *ptr,
    int flattened_head_count,
    int sequence_len,
    int head_dim,
    int tile_rows, int tile_dim = 0) {
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
        static_cast<uint32_t>((tile_dim ? tile_dim : head_dim) / 64),
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

// Present V=[B,1,S,512] as 64-feature chunks. For PV cta_group::2, each rank
// loads four chunks (256 features); the two M128N256 atoms select two local
// chunks from each rank at a time.
inline cudaError_t init_v_tmap(
    CUtensorMap *tmap,
    const __nv_bfloat16 *ptr,
    int flattened_head_count,
    int sequence_len,
    int head_dim,
    int tile_rows) {
    constexpr uint32_t kRank = 4;
    uint64_t global_dim[kRank] = {
        64,
        static_cast<uint64_t>(sequence_len),
        static_cast<uint64_t>(head_dim / 64),
        static_cast<uint64_t>(flattened_head_count),
    };
    uint64_t global_strides[kRank - 1] = {
        static_cast<uint64_t>(head_dim) * sizeof(__nv_bfloat16),
        64 * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(sequence_len) * head_dim * sizeof(__nv_bfloat16),
    };
    uint32_t box_dim[kRank] = {64, static_cast<uint32_t>(tile_rows), 1, 1};
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
    int tile_rows, int tile_dim = 0) {
    static_assert(PackSize == 64,
        "MLA decoding packs 64 Q heads per CTA");
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
        static_cast<uint32_t>((tile_dim ? tile_dim : head_dim) / 64),
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

// A two-CTA cluster owns one batch item.  Each rank owns 64 query heads;
// QK splits its N=64 key tile and PV splits each N=256 atom across ranks.
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1)
void fa_mla_decode_kernel_m64(
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
    constexpr int TileQ = 64;
    constexpr int TileKV = 64;
    constexpr int HeadDim = 576;
    constexpr int KvStages = 2;
    constexpr int KChunkCount = 9;
    constexpr int ValueChunkCount = 8;
    constexpr int LocalValueChunkCount = ValueChunkCount / 2;
    constexpr int ChunkDim = 64;
    constexpr int QChunkBytes = TileQ * ChunkDim * sizeof(__nv_bfloat16);
    constexpr int HalfChunkBytes = (TileKV / 2) * ChunkDim * sizeof(__nv_bfloat16);
    constexpr int QkStageBytes = KChunkCount * HalfChunkBytes;
    constexpr int PvChunkBytes = TileKV * ChunkDim * sizeof(__nv_bfloat16);
    constexpr int PvStageBytes = LocalValueChunkCount * PvChunkBytes;
    constexpr int StageBytes = QkStageBytes + PvStageBytes;
    constexpr int PStageBytes = TileQ * TileKV * sizeof(__nv_bfloat16);
    constexpr int kWarpSize = 32;
    constexpr int kTmaLoadWarp = 0;
    constexpr int kTmaStoreWarp = 1;
    constexpr int kMmaWarp = 2;
    constexpr int kIdleWarp = 3;
    constexpr int kSoftmaxWarpBegin = 4;
    constexpr int kEpilogueWarpBegin = 8;
    constexpr int kWgArrivalCount = 4;
    using SmemLayout = FaMlaSmemLayout<TileQ, TileKV, 1, KvStages>;

    extern __shared__ __align__(1024) char smem[];
    const SmemLayout smem_layout = SmemLayout::make(HeadDim);
    auto *q_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.q_offset);
    auto *kv_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.kv_offset);
    auto *p0_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.p_offset);
    auto *out_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.out_offset);

    __shared__ __align__(8) uint64_t q_tma_mbars[KChunkCount];
    __shared__ __align__(8) uint64_t k_tma_mbars[KvStages][KChunkCount];
    __shared__ __align__(8) uint64_t v_tma_mbars[KvStages][LocalValueChunkCount];
    __shared__ __align__(8) uint64_t qk_done_mbars[KvStages];
    __shared__ __align__(8) uint64_t s_free_mbars[KvStages];
    __shared__ __align__(8) uint64_t p_ready_mbars[KvStages];
    __shared__ __align__(8) uint64_t softmax_done_mbars[KvStages];
    __shared__ __align__(8) uint64_t rescale_done_mbars[KvStages];
    __shared__ __align__(8) uint64_t chunk_free_mbars[KvStages][KChunkCount];
    __shared__ __align__(8) uint64_t v_free_mbars[KvStages][LocalValueChunkCount];
    __shared__ __align__(8) uint64_t epilogue_mbar[1];
    __shared__ __align__(8) uint64_t out_store_ready_mbar[1];
    __shared__ __align__(8) uint64_t tmem_release_mbar[1];
    __shared__ __align__(4) int mma_tmem_addr[1];
    __shared__ __align__(16) float softmax_alpha[TileQ];
    __shared__ __align__(16) float softmax_l[TileQ];
    __shared__ __align__(16) float softmax_exchange[2 * TileQ];

    const int tid = threadIdx.x;
    const int warp_idx = tid / kWarpSize;
    const int lane_idx = tid % kWarpSize;
    const int q_tma_base = static_cast<int>(__cvta_generic_to_shared(q_tma_mbars));
    const int k_tma_base = static_cast<int>(__cvta_generic_to_shared(k_tma_mbars));
    const int v_tma_base = static_cast<int>(__cvta_generic_to_shared(v_tma_mbars));
    const int qk_done_base = static_cast<int>(__cvta_generic_to_shared(qk_done_mbars));
    const int s_free_base = static_cast<int>(__cvta_generic_to_shared(s_free_mbars));
    const int p_ready_base = static_cast<int>(__cvta_generic_to_shared(p_ready_mbars));
    const int softmax_done_base =
        static_cast<int>(__cvta_generic_to_shared(softmax_done_mbars));
    const int rescale_done_base =
        static_cast<int>(__cvta_generic_to_shared(rescale_done_mbars));
    const int chunk_free_base = static_cast<int>(__cvta_generic_to_shared(chunk_free_mbars));
    const int v_free_base = static_cast<int>(__cvta_generic_to_shared(v_free_mbars));
    const int epilogue_mbar_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_mbar));
    const int out_store_ready_addr =
        static_cast<int>(__cvta_generic_to_shared(out_store_ready_mbar));
    const int tmem_release_addr =
        static_cast<int>(__cvta_generic_to_shared(tmem_release_mbar));

    if (tid == 0) {
        for (int d64 = 0; d64 < KChunkCount; ++d64) {
            mbarrier_init(q_tma_base + d64 * sizeof(uint64_t), 2);
        }
        for (int stage = 0; stage < KvStages; ++stage) {
            for (int d64 = 0; d64 < KChunkCount; ++d64) {
                mbarrier_init(
                    k_tma_base + (stage * KChunkCount + d64) * sizeof(uint64_t), 2);
            }
            for (int d64 = 0; d64 < LocalValueChunkCount; ++d64) {
                mbarrier_init(
                    v_tma_base + (stage * LocalValueChunkCount + d64) * sizeof(uint64_t), 2);
            }
            mbarrier_init(qk_done_base + stage * sizeof(uint64_t), 1);
            mbarrier_init(s_free_base + stage * sizeof(uint64_t), 2 * kWgArrivalCount);
            mbarrier_init(p_ready_base + stage * sizeof(uint64_t), 2 * kWgArrivalCount);
            mbarrier_init(softmax_done_base + stage * sizeof(uint64_t), kWgArrivalCount);
            mbarrier_init(rescale_done_base + stage * sizeof(uint64_t), 2 * kWgArrivalCount);
            for (int chunk = 0; chunk < KChunkCount; ++chunk) {
                mbarrier_init(
                    chunk_free_base + (stage * KChunkCount + chunk) * sizeof(uint64_t), 1);
            }
            for (int chunk = 0; chunk < LocalValueChunkCount; ++chunk) {
                mbarrier_init(
                    v_free_base + (stage * LocalValueChunkCount + chunk) * sizeof(uint64_t), 1);
            }
        }
        mbarrier_init(epilogue_mbar_addr, 1);
        mbarrier_init(out_store_ready_addr, kWgArrivalCount);
        mbarrier_init(tmem_release_addr, kWgArrivalCount);
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
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    // Layout B routes the second half of logical N into datapaths 64..127.
    // Thus S[64,64] occupies 32 columns and O[64,512] occupies 256 columns.
    constexpr int kSCols = 32;
    constexpr int kOBase = 64;
    constexpr int kTmemCols = 512;
    if (warp_idx == kMmaWarp) {
        const int alloc_addr = static_cast<int>(__cvta_generic_to_shared(mma_tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(alloc_addr), "r"(kTmemCols));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"
                     ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    const int q_smem_addr = static_cast<int>(__cvta_generic_to_shared(q_smem));
    const int kv_smem_addr = static_cast<int>(__cvta_generic_to_shared(kv_smem));
    const int p0_smem_addr = static_cast<int>(__cvta_generic_to_shared(p0_smem));
    const int out_smem_addr = static_cast<int>(__cvta_generic_to_shared(out_smem));
    const int query_groups = heads / kMlaPackSize;
    const int qo_bh = blockIdx.x;
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int kv_tile_count = (key_len + TileKV - 1) / TileKV;
    const int causal_offset = key_len - query_len;
    const int task_kv_tile_count = causal
        ? min(kv_tile_count, (1 + causal_offset + TileKV - 1) / TileKV)
        : kv_tile_count;
    const float scale_log2e = scale * 1.4426950408889634f;

    if (warp_idx >= kSoftmaxWarpBegin && warp_idx < kEpilogueWarpBegin) {
        setmaxnreg_increase<192>();
        const int tmem_addr = mma_tmem_addr[0];
        const int warp_in_group = warp_idx - kSoftmaxWarpBegin;
        const int logical_row = (warp_in_group & 1) * 32 + lane_idx;
        float tilem = -__int_as_float(0x7F800000);
        float tilel = 0.0f;
        int qk_phase[KvStages] = {};
        constexpr float kConditionalRescaleThresholdLn = 5.5451774444795623f;
        for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
            const int stage = kv_tile & 1;
            mbarrier_wait(qk_done_base + stage * sizeof(uint64_t), qk_phase[stage]);
            qk_phase[stage] ^= 1;
            tcgen05_fence_after_thread_sync();

            char *p_bytes = stage == 0
                ? reinterpret_cast<char *>(p0_smem)
                : reinterpret_cast<char *>(kv_smem) + stage * StageBytes;
            const int mask_q_row = causal ? causal_offset : key_len - 1;
            const bool all_scores_valid =
                (kv_tile + 1) * TileKV <= key_len &&
                (!causal || (kv_tile + 1) * TileKV - 1 <= mask_q_row);
            const int s_warp_addr = tmem_addr + stage * kSCols +
                (warp_in_group * 32 << 16);
            const SoftmaxUpdate update = all_scores_valid
                ? load_s_and_store_p_m64<true>(s_warp_addr, p_bytes, softmax_exchange,
                    warp_in_group,
                    kv_tile * TileKV, key_len, mask_q_row, scale, tilem, tilel,
                    kConditionalRescaleThresholdLn, scale_log2e, lane_idx)
                : load_s_and_store_p_m64<false>(s_warp_addr, p_bytes, softmax_exchange,
                    warp_in_group,
                    kv_tile * TileKV, key_len, mask_q_row, scale, tilem, tilel,
                    kConditionalRescaleThresholdLn, scale_log2e, lane_idx);
            tilem = update.m;
            tilel = update.l;
            if (warp_in_group < 2) {
                softmax_alpha[logical_row] = update.alpha;
                softmax_l[logical_row] = tilel;
            }
            // Make generic-proxy P stores visible to tcgen05.mma's async proxy.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane_idx == 0) {
                if (cta_rank == 0) {
                    mbarrier_arrive(s_free_base + stage * sizeof(uint64_t));
                    mbarrier_arrive(p_ready_base + stage * sizeof(uint64_t));
                } else {
                    // Only rank 0 issues the collective MMA and waits on
                    // these barriers.  Funnel rank 1's arrivals to rank 0;
                    // symmetric remote arrivals can target rank 1 after its
                    // consumer warps have already reached the kernel tail.
                    mbarrier_arrive_on_peer(
                        s_free_base + stage * sizeof(uint64_t), 0);
                    mbarrier_arrive_on_peer(
                        p_ready_base + stage * sizeof(uint64_t), 0);
                }
                mbarrier_arrive(softmax_done_base + stage * sizeof(uint64_t));
            }
        }
        return;
    }

    if (warp_idx >= kEpilogueWarpBegin && warp_idx < 12) {
        setmaxnreg_decrease<96>();
        const int tmem_addr = mma_tmem_addr[0];
        const int warp_in_group = warp_idx - kEpilogueWarpBegin;
        const int logical_row = (warp_in_group & 1) * 32 + lane_idx;
        const int output_base = (warp_in_group >> 1) * 256;
        int softmax_phase[KvStages] = {};
        float final_l = 0.0f;
        for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
            const int stage = kv_tile & 1;
            mbarrier_wait(
                softmax_done_base + stage * sizeof(uint64_t), softmax_phase[stage]);
            softmax_phase[stage] ^= 1;
            const float alpha = softmax_alpha[logical_row];
            final_l = softmax_l[logical_row];
            const bool warp_needs_rescale = __any_sync(
                0xFFFFFFFFU, (__float_as_uint(alpha) & 0x80000000U) == 0);
            if (kv_tile != 0 && warp_needs_rescale) {
                const int o_warp_addr = tmem_addr + kOBase +
                    (warp_in_group * 32 << 16);
                for (int d32 = 0; d32 < 256 / 32; ++d32) {
                    float values[32];
                    tcgen05_ld_f32_x32(values, o_warp_addr + d32 * 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        values[i] *= alpha;
                    }
                    tcgen05_st_f32_x32(values, o_warp_addr + d32 * 32);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            tcgen05_fence_before_thread_sync();
            if (lane_idx == 0) {
                if (cta_rank == 0) {
                    mbarrier_arrive(rescale_done_base + stage * sizeof(uint64_t));
                } else {
                    mbarrier_arrive_on_peer(
                        rescale_done_base + stage * sizeof(uint64_t), 0);
                }
            }
        }

        mbarrier_wait(epilogue_mbar_addr, 0);
        tcgen05_fence_after_thread_sync();
        const float inv_l = 1.0f / final_l;
        const int o_warp_addr = tmem_addr + kOBase + (warp_in_group * 32 << 16);
        for (int d8 = 0; d8 < 256 / 8; ++d8) {
            float values[8];
            tcgen05_ld_f32_x8(values, o_warp_addr + d8 * 8);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            const int d = output_base + d8 * 8;
            const int logical_byte =
                ((d / 64) * TileQ * 64 + logical_row * 64 + d % 64) *
                sizeof(__nv_bfloat16);
            const int swizzled_byte = logical_byte ^ ((logical_byte >> 3) & 0x70);
            const uint4 packed = make_uint4(
                pack_bf16x2(values[0] * inv_l, values[1] * inv_l),
                pack_bf16x2(values[2] * inv_l, values[3] * inv_l),
                pack_bf16x2(values[4] * inv_l, values[5] * inv_l),
                pack_bf16x2(values[6] * inv_l, values[7] * inv_l));
            *reinterpret_cast<uint4 *>(reinterpret_cast<char *>(out_smem) + swizzled_byte) =
                packed;
        }
        if (lane_idx == 0) {
            mbarrier_arrive(out_store_ready_addr);
        }
        tcgen05_fence_before_thread_sync();
        if (lane_idx == 0) {
            mbarrier_arrive(tmem_release_addr);
        }
        return;
    }

    setmaxnreg_decrease<32>();
    if (warp_idx == kIdleWarp) {
        return;
    }

    if (warp_idx == kTmaLoadWarp) {
        if (lane_idx == 0) {
            const int kv_bh = qo_bh / query_groups;
            for (int d64 = 0; d64 < KChunkCount; ++d64) {
                const int mbar = q_tma_base + d64 * sizeof(uint64_t);
                const int tx_mbar = mbar & 0xFEFFFFFF;
                mbarrier_expect_tx(tx_mbar, QChunkBytes);
                tma_5d_gmem2smem(q_smem_addr + d64 * QChunkBytes, &q_tmap,
                    0, 0, 0, d64, qo_bh, tx_mbar);
            }
            int chunk_free_phase[KvStages][KChunkCount] = {};
            int v_free_phase[KvStages][LocalValueChunkCount] = {};
            for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                const int stage = kv_tile & 1;
                const int stage_addr = kv_smem_addr + stage * StageBytes;
                for (int d64 = 0; d64 < KChunkCount; ++d64) {
                    if (kv_tile >= KvStages) {
                        const int free_addr = chunk_free_base +
                            (stage * KChunkCount + d64) * sizeof(uint64_t);
                        mbarrier_wait(free_addr, chunk_free_phase[stage][d64]);
                        chunk_free_phase[stage][d64] ^= 1;
                    }
                    const int mbar = k_tma_base +
                        (stage * KChunkCount + d64) * sizeof(uint64_t);
                    const int tx_mbar = mbar & 0xFEFFFFFF;
                    mbarrier_expect_tx(tx_mbar, HalfChunkBytes);
                    tma_4d_gmem2smem(stage_addr + d64 * HalfChunkBytes, &k_tmap,
                        0, kv_tile * TileKV + cta_rank * (TileKV / 2), d64, kv_bh, tx_mbar);
                }
                const int v_stage_addr = stage_addr + QkStageBytes;
                for (int d64 = 0; d64 < LocalValueChunkCount; ++d64) {
                    if (kv_tile >= KvStages) {
                        const int free_addr = v_free_base +
                            (stage * LocalValueChunkCount + d64) * sizeof(uint64_t);
                        mbarrier_wait(free_addr, v_free_phase[stage][d64]);
                        v_free_phase[stage][d64] ^= 1;
                    }
                    const int mbar = v_tma_base +
                        (stage * LocalValueChunkCount + d64) * sizeof(uint64_t);
                    const int tx_mbar = mbar & 0xFEFFFFFF;
                    mbarrier_expect_tx(tx_mbar, PvChunkBytes);
                    tma_4d_gmem2smem(v_stage_addr + d64 * PvChunkBytes, &v_tmap,
                        0, kv_tile * TileKV, d64 + cta_rank * LocalValueChunkCount,
                        kv_bh, tx_mbar);
                }
            }
        }
        return;
    }

    if (warp_idx == kTmaStoreWarp) {
        if (lane_idx == 0) {
            mbarrier_wait(out_store_ready_addr, 0);
            tma_5d_smem2gmem(out_smem_addr, &out_tmap, 0, 0, 0, 0, qo_bh);
            tma_store_commit_group();
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
        return;
    }

    if (warp_idx == kMmaWarp) {
        const int tmem_addr = mma_tmem_addr[0];
        if (cta_rank == 0 && elect_sync()) {
            constexpr uint32_t kQkInstrDesc =
                (1U << 4U) | (1U << 7U) | (1U << 10U) |
                (8U << 17U) | (8U << 24U); // collective M128 N64
            constexpr uint32_t kPvInstrDesc =
                (1U << 4U) | (1U << 7U) | (1U << 10U) |
                (1U << 16U) | (32U << 17U) | (8U << 24U); // collective M128 N256
            int k_phase[KvStages][KChunkCount] = {};
            int v_phase[KvStages][LocalValueChunkCount] = {};
            int s_free_phase[KvStages] = {};
            int p_ready_phase[KvStages] = {};
            int rescale_phase[KvStages] = {};

            auto issue_qk = [&](int kv_tile) {
                const int stage = kv_tile & 1;
                if (kv_tile >= KvStages) {
                    mbarrier_wait(
                        s_free_base + stage * sizeof(uint64_t), s_free_phase[stage]);
                    s_free_phase[stage] ^= 1;
                }
                const int s_addr = tmem_addr + stage * kSCols;
                const int stage_addr = kv_smem_addr + stage * StageBytes;
                for (int d64 = 0; d64 < KChunkCount; ++d64) {
                    mbarrier_wait(
                        q_tma_base + d64 * sizeof(uint64_t), 0);
                    mbarrier_wait(
                        k_tma_base + (stage * KChunkCount + d64) * sizeof(uint64_t),
                        k_phase[stage][d64]);
                    k_phase[stage][d64] ^= 1;
                    tcgen05_fence_after_thread_sync();
                    for (int d16 = 0; d16 < 4; ++d16) {
                        tcgen05_mma_bf16(s_addr,
                            make_128b_swizzle_desc(
                                q_smem_addr + d64 * QChunkBytes + d16 * 32),
                            make_128b_swizzle_desc(
                                stage_addr + d64 * HalfChunkBytes + d16 * 32),
                            kQkInstrDesc, d64 != 0 || d16 != 0);
                    }
                    // Stage 1 becomes P1 after QK. Do not let its TMA
                    // producer overwrite that storage until PV has consumed P1.
                    if (stage == 0) {
                        tcgen05_commit_mbarrier(
                            chunk_free_base +
                            (stage * KChunkCount + d64) * sizeof(uint64_t));
                    }
                }
                tcgen05_commit_mbarrier(qk_done_base + stage * sizeof(uint64_t));
            };

            auto issue_pv = [&](int kv_tile) {
                const int stage = kv_tile & 1;
                mbarrier_wait(
                    p_ready_base + stage * sizeof(uint64_t), p_ready_phase[stage]);
                p_ready_phase[stage] ^= 1;
                mbarrier_wait(
                    rescale_done_base + stage * sizeof(uint64_t), rescale_phase[stage]);
                rescale_phase[stage] ^= 1;
                const int stage_addr = kv_smem_addr + stage * StageBytes;
                const int v_addr = stage_addr + QkStageBytes;
                const int p_addr = stage == 0
                    ? p0_smem_addr
                    : stage_addr;
                for (int atom = 0; atom < 2; ++atom) {
                    for (int local_chunk = atom * 2; local_chunk < atom * 2 + 2;
                        ++local_chunk) {
                        mbarrier_wait(
                            v_tma_base +
                                (stage * LocalValueChunkCount + local_chunk) * sizeof(uint64_t),
                            v_phase[stage][local_chunk]);
                        v_phase[stage][local_chunk] ^= 1;
                    }
                    tcgen05_fence_after_thread_sync();
                    const int d_addr = tmem_addr + kOBase + atom * 128;
                    for (int k16 = 0; k16 < 4; ++k16) {
                        tcgen05_mma_bf16(d_addr,
                            make_128b_swizzle_desc(p_addr + k16 * 32),
                            make_128b_nmajor_desc(
                                v_addr + atom * 2 * PvChunkBytes + k16 * 16 * 128,
                                PvChunkBytes),
                            kPvInstrDesc, kv_tile != 0 || k16 != 0);
                    }
                    for (int local_chunk = atom * 2; local_chunk < atom * 2 + 2;
                        ++local_chunk) {
                        tcgen05_commit_mbarrier(
                            v_free_base +
                            (stage * LocalValueChunkCount + local_chunk) * sizeof(uint64_t));
                    }
                }
                if (stage == 1) {
                    for (int chunk = 0; chunk < KChunkCount; ++chunk) {
                        tcgen05_commit_mbarrier(
                            chunk_free_base +
                            (stage * KChunkCount + chunk) * sizeof(uint64_t));
                    }
                }
            };

            for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                issue_qk(kv_tile);
                if (kv_tile > 0) {
                    issue_pv(kv_tile - 1);
                }
            }
            issue_pv(task_kv_tile_count - 1);
            tcgen05_commit_mbarrier(epilogue_mbar_addr);
        }
        mbarrier_wait(tmem_release_addr, 0);
        tcgen05_fence_after_thread_sync();
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                     :: "r"(tmem_addr), "r"(kTmemCols));
        return;
    }
}

// Q=1 MLA decode launcher. It instantiates the shared FA mainloop with one
// resident Q slot: WG0 drives TMA/MMA, WG1 performs softmax, and WG2 handles
// rescale and the epilogue.
cudaError_t bf16_fa4_mla_decode_cuda(
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
    constexpr int kTileQ = 64;
    constexpr int kTileKV = 64;
    constexpr int kQSlots = 1;
    constexpr int kKvStages = 2;
    constexpr int kHeadDim = 576;
    constexpr int kDecodeWarps = 12;

    if (query_len != 1 || head_dim != kHeadDim || heads != 128 ||
        heads % kMlaPackSize != 0 || key_len <= 0 || batch <= 0 ||
        !(scale > 0.0f) || (causal && key_len < query_len)) {
        return cudaErrorInvalidValue;
    }

    const int query_groups = heads / kMlaPackSize;
    CUtensorMap q_tmap{};
    CUtensorMap k_tmap{};
    CUtensorMap v_tmap{};
    CUtensorMap out_tmap{};
    cudaError_t status = init_packed_qo_tmap<kMlaPackSize>(
        &q_tmap, q, batch * query_groups, query_len, head_dim, kTileQ, 64);
    if (status == cudaSuccess) {
        status = init_qkv_tmap(
            &k_tmap, k, batch, key_len, head_dim, kTileKV / 2, 64);
    }
    if (status == cudaSuccess) {
        status = init_v_tmap(
            &v_tmap, v, batch, key_len, 512, kTileKV);
    }
    if (status == cudaSuccess) {
        status = init_packed_qo_tmap<kMlaPackSize>(
            &out_tmap, out, batch * query_groups, query_len, 512, kTileQ, 512);
    }
    if (status != cudaSuccess) {
        return status;
    }

    using SmemLayout =
        FaMlaSmemLayout<kTileQ, kTileKV, kQSlots, kKvStages>;
    const SmemLayout smem_layout = SmemLayout::make(head_dim);
    if (smem_layout.dynamic_bytes + SmemLayout::kStaticMbarrierBytes +
            SmemLayout::kSoftmaxHandoffBytes >
        SmemLayout::kMaxSmemBytes) {
        return cudaErrorInvalidConfiguration;
    }

    status = cudaFuncSetAttribute(
        fa_mla_decode_kernel_m64,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_layout.dynamic_bytes));
    if (status != cudaSuccess) {
        return status;
    }

    const dim3 block(kDecodeWarps * 32);
    const dim3 grid(batch * query_groups);
    fa_mla_decode_kernel_m64
        <<<grid, block, smem_layout.dynamic_bytes, stream>>>(
            q_tmap, k_tmap, v_tmap, out_tmap, batch, heads, query_len,
            key_len, scale, causal);
    return cudaGetLastError();
}


} // namespace sm110a::bf16_fa4_mla_2cta

using namespace sm110a::bf16_fa4_mla_2cta;


extern "C" cudaError_t sm110a_bf16_fa4_mla_2cta_launch(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    __nv_bfloat16 *out, int batch, int heads, int query_len, int key_len,
    int head_dim, float scale, bool causal, cudaStream_t stream) {
    return bf16_fa4_mla_decode_cuda(q, k, v, out, batch, heads,
        query_len, key_len, head_dim, scale, causal, stream);
}
