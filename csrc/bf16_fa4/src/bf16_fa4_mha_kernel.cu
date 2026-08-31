#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

__device__ inline void mbarrier_init(int mbar_addr, int arrival_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr),
                 "r"(arrival_count));
}

// Wait for the producer to toggle this barrier to `phase`.  The explicit
// parity lets a two-stage ring reuse the same mbarrier on later KV tiles.
__device__ inline void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "FA_MHA_MBAR_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni FA_MHA_MBAR_DONE;\n\t"
                 "bra.uni FA_MHA_MBAR_WAIT;\n\t"
                 "FA_MHA_MBAR_DONE:\n\t"
                 "}" :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ inline void prefetch_tensormap(const void *desc) {
    const uint64_t ptr = reinterpret_cast<uint64_t>(desc);
    asm volatile("prefetch.tensormap [%0];" :: "l"(ptr) : "memory");
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
        "[%0], [%1, {%2, %3, %4, %5}], [%6];" :: "r"(dst), "l"(tmap_ptr),
        "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar_addr) : "memory");
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

__device__ inline constexpr uint64_t desc_encode(uint64_t value) {
    return (value & 0x3FFFFULL) >> 4ULL;
}

__device__ inline uint64_t make_128b_swizzle_desc(int smem_addr) {
    constexpr int kStrideByteOffset = 8 * 128;
    return desc_encode(smem_addr) | (desc_encode(kStrideByteOffset) << 32ULL) |
        (1ULL << 46ULL) | (2ULL << 61ULL);
}

// Canonical BF16 MN-major SW128 descriptor for a TMA box ordered as
// [64 elements, K rows, N/64 chunks]. Each N chunk contains the full K tile.
__device__ inline uint64_t make_128b_nmajor_desc(
    int smem_addr, int leading_byte_offset) {
    constexpr int kStrideByteOffset = 8 * 128;
    return desc_encode(smem_addr) | (desc_encode(leading_byte_offset) << 16ULL) |
        (desc_encode(kStrideByteOffset) << 32ULL) | (1ULL << 46ULL) |
        (2ULL << 61ULL);
}

__device__ inline void tcgen05_mma_bf16(
    int tmem_addr, uint64_t a_desc, uint64_t b_desc, uint32_t instr_desc, int enable_input_d) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
                 "}"
                 :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(instr_desc),
                 "r"(enable_input_d));
}

__device__ inline void tcgen05_mma_tmem_a_bf16(
    int tmem_out, int tmem_a, uint64_t b_desc, uint32_t instr_desc, int enable_input_d) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t"
                 "}"
                 :: "r"(tmem_out), "r"(tmem_a), "l"(b_desc), "r"(instr_desc),
                 "r"(enable_input_d));
}

__device__ inline void tcgen05_ld_f32_x8(float *values, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]),
          "=f"(values[4]), "=f"(values[5]), "=f"(values[6]), "=f"(values[7])
        : "r"(tmem_addr));
}

__device__ inline void tcgen05_st_f32_x8(const float *values, int tmem_addr) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
        :: "r"(tmem_addr), "f"(values[0]), "f"(values[1]), "f"(values[2]),
           "f"(values[3]), "f"(values[4]), "f"(values[5]), "f"(values[6]),
           "f"(values[7]));
}

// The f16-kind TMEM-A operand packs two adjacent BF16 values into each 32-bit
// TMEM word. Keep the register pair packed and store without `.unpack::16b`.
__device__ inline void tcgen05_st_bf16_x8(const float *values, int tmem_addr) {
    uint32_t packed[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t lo = static_cast<uint32_t>(
            __bfloat16_as_ushort(__float2bfloat16_rn(values[2 * i])));
        const uint32_t hi = static_cast<uint32_t>(
            __bfloat16_as_ushort(__float2bfloat16_rn(values[2 * i + 1])));
        packed[i] = lo | (hi << 16);
    }
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]));
}

__device__ inline void tcgen05_commit_mbarrier(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
        "[%0];" :: "r"(mbar_addr) : "memory");
}

__device__ inline void mbarrier_arrive(int mbar_addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::
                     "r"(mbar_addr)
                 : "memory");
}

__host__ __device__ constexpr size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Dynamic shared-memory layout for one CTA.  Two Q slots remain resident for
// the whole KV loop, while K/V use a separate two-entry producer/consumer ring.
template <int TileQ, int TileKV, int QSlots, int KvStages>
struct FaMhaSmemLayout {
    static constexpr size_t kAlignment = 128;
    static constexpr size_t kMaxSmemBytes = 227 * 1024;
    static constexpr size_t kStaticMbarrierBytes =
        (QSlots + 1 + (2 + 3 * QSlots) * KvStages) * sizeof(uint64_t);

    size_t q_offset;
    size_t k_offset;
    size_t v_offset;
    size_t tilel_offset;
    size_t tilem_offset;
    size_t dynamic_bytes;

    __host__ __device__ static FaMhaSmemLayout make(int head_dim) {
        const size_t q_bytes = QSlots * TileQ * static_cast<size_t>(head_dim) *
            sizeof(__nv_bfloat16);
        const size_t kv_bytes = KvStages * TileKV * static_cast<size_t>(head_dim) *
            sizeof(__nv_bfloat16);
        const size_t tilelm_bytes = QSlots * TileQ * sizeof(float);

        FaMhaSmemLayout layout{};
        layout.q_offset = 0;
        layout.k_offset = align_up(layout.q_offset + q_bytes, kAlignment);
        layout.v_offset = align_up(layout.k_offset + kv_bytes, kAlignment);
        layout.tilel_offset = align_up(layout.v_offset + kv_bytes, kAlignment);
        layout.tilem_offset = align_up(layout.tilel_offset + tilelm_bytes, kAlignment);
        layout.dynamic_bytes = align_up(layout.tilem_offset + tilelm_bytes, kAlignment);
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
    uint32_t box_dim[kRank] = {64, static_cast<uint32_t>(tile_rows),
        static_cast<uint32_t>(head_dim / 64), 1};
    uint32_t element_strides[kRank] = {1, 1, 1, 1};

    const CUresult result = cuTensorMapEncodeTiled(
        tmap,
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
template <int TileQ, int TileKV, int QSlots, int KvStages>
__global__ void fa_mha_kernel(
    const __grid_constant__ CUtensorMap q_tmap,
    const __grid_constant__ CUtensorMap k_tmap,
    const __grid_constant__ CUtensorMap v_tmap,
    __nv_bfloat16 *__restrict__ out,
    int batch,
    int heads,
    int query_len,
    int key_len,
    int head_dim,
    float scale,
    bool causal) {
    static_assert(TileQ == 128, "the FA MHA kernel currently supports TileQ=128 only");
    static_assert(TileKV > 0, "attention tiles must be positive");
    static_assert(QSlots == 2, "the FA MHA mainloop currently supports two Q slots");
    static_assert(KvStages == 2, "the K/V mainloop currently uses two stages");

    using SmemLayout = FaMhaSmemLayout<TileQ, TileKV, QSlots, KvStages>;
    extern __shared__ __align__(1024) char smem[];
    const SmemLayout smem_layout = SmemLayout::make(head_dim);
    auto *q_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.q_offset);
    auto *k_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.k_offset);
    auto *v_smem = reinterpret_cast<__nv_bfloat16 *>(smem + smem_layout.v_offset);
    auto *tilel = reinterpret_cast<float *>(smem + smem_layout.tilel_offset);
    auto *tilem = reinterpret_cast<float *>(smem + smem_layout.tilem_offset);

    // Producer/consumer graph for each (Q slot, KV stage):
    //   K TMA -> QK MMA -> MUFU -> PV MMA -> recycle K/V stage.
    // Each resident Q tile has an independent prologue mbarrier; all remaining
    // edges need one independent barrier per two-stage-ring entry.
    __shared__ __align__(8) uint64_t q_prologue_mbars[QSlots];
    __shared__ __align__(8) uint64_t tma1_mbars[KvStages];
    __shared__ __align__(8) uint64_t tma2_mbars[KvStages];
    __shared__ __align__(8) uint64_t mma1_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t mma2_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t mufu_mbars[QSlots][KvStages];
    __shared__ __align__(8) uint64_t epilogue_mbar[1];
    __shared__ __align__(4) int mma_tmem_addr[1];

    constexpr int kWarpSize = 32;
    // CTA role assignment (16 warps = 512 threads).  The two MUFU workgroups
    // are independent because they transform S0 and S1 for separate Q tiles.
    constexpr int kKTmaWarp = 0;            // K shared-memory producer
    constexpr int kVTmaWarp = 1;            // V shared-memory producer
    constexpr int kMmaWarp = 2;             // QK and PV tcgen05 issuer
    constexpr int kQPrologueWarp = 3;       // loads the two resident Q tiles
    constexpr int kMufu0WarpBegin = 4;     // Q0: warp 4-7
    constexpr int kMufu0WarpEnd = 8;
    constexpr int kMufu1WarpBegin = 8;     // Q1: warp 8-11
    constexpr int kMufu1WarpEnd = 12;
    constexpr int kEpilogueWarpBegin = 12; // warp 12-15
    constexpr int kEpilogueWarpEnd = 16;
    constexpr int kTmaMbarArrivalCount = 1;
    constexpr int kMmaMbarArrivalCount = 1;
    // A 128-row TMEM tile is partitioned into four 32-row regions. MUFU
    // warps 4-7 collectively own one stage and each arrives once.
    constexpr int kMufuMbarArrivalCount = kMufu0WarpEnd - kMufu0WarpBegin;
    constexpr int kQPrologueMbarArrivalCount = 1;

    const int tid = threadIdx.x;
    const int warp_idx = threadIdx.x / kWarpSize;
    const int lane_idx = threadIdx.x % kWarpSize;

    const int q_prologue_mbar_base =
        static_cast<int>(__cvta_generic_to_shared(q_prologue_mbars));
    const int tma1_mbar_base = static_cast<int>(__cvta_generic_to_shared(tma1_mbars));
    const int tma2_mbar_base = static_cast<int>(__cvta_generic_to_shared(tma2_mbars));
    const int mma1_mbar_base = static_cast<int>(__cvta_generic_to_shared(mma1_mbars));
    const int mma2_mbar_base = static_cast<int>(__cvta_generic_to_shared(mma2_mbars));
    const int mufu_mbar_base = static_cast<int>(__cvta_generic_to_shared(mufu_mbars));
    const int epilogue_mbar_addr =
        static_cast<int>(__cvta_generic_to_shared(epilogue_mbar));

    // Follow the BF16 GEMM setup pattern: one thread initializes every
    // barrier slot, publishes the initialization, then the CTA synchronizes.
    if (tid == 0) {
        for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
            mbarrier_init(q_prologue_mbar_base + q_slot * sizeof(uint64_t),
                kQPrologueMbarArrivalCount);
        }
        mbarrier_init(epilogue_mbar_addr, kMmaMbarArrivalCount);
        for (int stage = 0; stage < KvStages; ++stage) {
            const int offset = stage * sizeof(uint64_t);
            mbarrier_init(tma1_mbar_base + offset, kTmaMbarArrivalCount);
            mbarrier_init(tma2_mbar_base + offset, kTmaMbarArrivalCount);
            for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                const int q_offset = (q_slot * KvStages + stage) * sizeof(uint64_t);
                mbarrier_init(mma1_mbar_base + q_offset, kMmaMbarArrivalCount);
                mbarrier_init(mma2_mbar_base + q_offset, kMmaMbarArrivalCount);
                mbarrier_init(mufu_mbar_base + q_offset, kMufuMbarArrivalCount);
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
        const int mma_tmem_smem_addr =
            static_cast<int>(__cvta_generic_to_shared(mma_tmem_addr));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
                "r"(mma_tmem_smem_addr), "r"(mma_tmem_columns));
    }
    __syncthreads();

    // K/V use the stage ring; Q slots have fixed shared-memory and TMEM banks.
    const int q_smem_addr = static_cast<int>(__cvta_generic_to_shared(q_smem));
    const int k_smem_addr = static_cast<int>(__cvta_generic_to_shared(k_smem));
    const int v_smem_addr = static_cast<int>(__cvta_generic_to_shared(v_smem));
    const int tmem_addr = mma_tmem_addr[0];

    // Persistent schedule: one task is (one flattened B*H head, two adjacent
    // 128-row Q tiles).  Each resident CTA strides over tasks, avoiding a huge
    // launch grid for large batch/head counts.
    const int q_tile_count = (query_len + TileQ - 1) / TileQ;
    const int q_pair_count = (q_tile_count + QSlots - 1) / QSlots;
    const int flattened_head_count = batch * heads;
    const int total_task_count = q_pair_count * flattened_head_count;
    const int kv_tile_count = (key_len + TileKV - 1) / TileKV;
    int q_prologue_phase[QSlots] = {};
    int k_recycle_phase[QSlots][KvStages] = {};
    int v_mma1_phase[QSlots][KvStages] = {};
    int v_recycle_phase[QSlots][KvStages] = {};
    int mufu_mma1_phase[QSlots][KvStages] = {};
    int qk_tma1_phase[KvStages] = {};
    int pv_tma2_phase[KvStages] = {};
    int pv_mufu_phase[QSlots][KvStages] = {};
    int epilogue_phase = 0;
    bool k_slot_initialized[KvStages] = {};
    bool v_slot_initialized[KvStages] = {};

    for (int task_id = blockIdx.x; task_id < total_task_count; task_id += gridDim.x) {
        const int bh = task_id / q_pair_count;
        const int q_pair_idx = task_id - bh * q_pair_count;
        const int q_tile_start[QSlots] = {
            (q_pair_idx * QSlots + 0) * TileQ,
            (q_pair_idx * QSlots + 1) * TileQ,
        };
        const int q_pair_end = q_tile_start[1] + TileQ < query_len
            ? q_tile_start[1] + TileQ : query_len;
        // For causal attention, no row in this Q-pair can observe keys beyond
        // q_pair_end. Non-causal attention consumes the whole KV sequence.
        const int task_kv_tile_count = causal
            ? ((q_pair_end + TileKV - 1) / TileKV < kv_tile_count
                   ? (q_pair_end + TileKV - 1) / TileKV : kv_tile_count)
            : kv_tile_count;

        // Online-softmax state starts empty for every row in both Q tiles:
        // m=-inf, l=0, and O is initialized below to the zero numerator.
        for (int element = tid; element < QSlots * TileQ; element += blockDim.x) {
            tilel[element] = 0.0f;
            tilem[element] = -__int_as_float(0x7F800000);
        }
        __syncthreads();

        const int mufu_q_slot = warp_idx >= kMufu0WarpBegin && warp_idx < kMufu0WarpEnd ? 0 :
            (warp_idx >= kMufu1WarpBegin && warp_idx < kMufu1WarpEnd ? 1 : -1);
        if (mufu_q_slot >= 0) {
            // Each MUFU WG clears its own long-lived unnormalized accumulator:
            // O_tilde = sum(exp(S - m) * V). It is normalized only in epilogue.
            const int mufu_warp = warp_idx - (mufu_q_slot == 0 ? kMufu0WarpBegin : kMufu1WarpBegin);
            const int row_base = mufu_warp * kWarpSize;
            float zero[8] = {};
            const int o_tmem_addr = tmem_addr + kOBankBase +
                mufu_q_slot * mma_tmem_columns_per_stage;
            for (int d8 = 0; d8 < head_dim / 8; ++d8) {
                tcgen05_st_f32_x8(zero, o_tmem_addr + (row_base << 16) + d8 * 8);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }

        // Load both resident Q tiles through independent prologue barriers.
        if (warp_idx == kQPrologueWarp && lane_idx == 0) {
            const int q_transaction_bytes = TileQ * head_dim * sizeof(__nv_bfloat16);
            for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                const int q_mbar_addr =
                    q_prologue_mbar_base + q_slot * sizeof(uint64_t);
                const int q_transaction_mbar_addr = q_mbar_addr & 0xFEFFFFFF;
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::
                                 "r"(q_transaction_mbar_addr), "r"(q_transaction_bytes)
                             : "memory");
                tma_4d_gmem2smem(q_smem_addr + q_slot * TileQ * head_dim * sizeof(__nv_bfloat16),
                    &q_tmap, 0, q_tile_start[q_slot], 0, bh, q_transaction_mbar_addr);
                mbarrier_wait(q_mbar_addr, q_prologue_phase[q_slot]);
                q_prologue_phase[q_slot] ^= 1;
            }
        }
        __syncthreads();

        // K producer: stage may be overwritten only after PV has consumed it
        // for *both* Q slots, hence the two mma2 waits on ring reuse.
        if (warp_idx == kKTmaWarp && lane_idx == 0) {
            for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                const int stage = kv_tile % KvStages;
                if (k_slot_initialized[stage]) {
                    for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                        const int addr = mma2_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                        mbarrier_wait(addr, k_recycle_phase[q_slot][stage]);
                        k_recycle_phase[q_slot][stage] ^= 1;
                    }
                }
                const int addr = tma1_mbar_base + stage * sizeof(uint64_t);
                const int tx_addr = addr & 0xFEFFFFFF;
                const int transaction_bytes = TileKV * head_dim * sizeof(__nv_bfloat16);
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::
                                 "r"(tx_addr), "r"(transaction_bytes) : "memory");
                tma_4d_gmem2smem(k_smem_addr + stage * TileKV * head_dim * sizeof(__nv_bfloat16),
                    &k_tmap, 0, kv_tile * TileKV, 0, bh, tx_addr);
                k_slot_initialized[stage] = true;
            }
        }

        // V producer: QK must finish before V overwrites the stage (mma1), and
        // the prior PV must finish before a reused V stage is overwritten.
        if (warp_idx == kVTmaWarp && lane_idx == 0) {
            for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                const int stage = kv_tile % KvStages;
                for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                    const int addr = mma1_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                    mbarrier_wait(addr, v_mma1_phase[q_slot][stage]);
                    v_mma1_phase[q_slot][stage] ^= 1;
                }
                if (v_slot_initialized[stage]) {
                    for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                        const int addr = mma2_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                        mbarrier_wait(addr, v_recycle_phase[q_slot][stage]);
                        v_recycle_phase[q_slot][stage] ^= 1;
                    }
                }
                const int addr = tma2_mbar_base + stage * sizeof(uint64_t);
                const int tx_addr = addr & 0xFEFFFFFF;
                const int transaction_bytes = TileKV * head_dim * sizeof(__nv_bfloat16);
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::
                                 "r"(tx_addr), "r"(transaction_bytes) : "memory");
                tma_4d_gmem2smem(v_smem_addr + stage * TileKV * head_dim * sizeof(__nv_bfloat16),
                    &v_tmap, 0, kv_tile * TileKV, 0, bh, tx_addr);
                v_slot_initialized[stage] = true;
            }
        }

        if (mufu_q_slot >= 0) {
            // Four warps cover 4 x 32 rows of one Q tile.  They consume FP32 S
            // written by QK, produce packed BF16 P in the same TMEM bank, and
            // update the online-softmax (m, l) state for their own rows.
            const int mufu_warp = warp_idx - (mufu_q_slot == 0 ? kMufu0WarpBegin : kMufu1WarpBegin);
            const int row_base = mufu_warp * kWarpSize;
            const int row = row_base + lane_idx;
            const int q_row = q_tile_start[mufu_q_slot] + row;
            const bool valid_q_row = q_row < query_len;
            const int o_tmem_addr = tmem_addr + kOBankBase + mufu_q_slot * mma_tmem_columns_per_stage;
            for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                const int stage = kv_tile % KvStages;
                if (kv_tile > 0) {
                    const int previous_tile = kv_tile - 1;
                    const int previous_stage = previous_tile % KvStages;
                    const int previous_phase = (previous_tile / KvStages) & 1;
                    const int previous_mma2_addr = mma2_mbar_base +
                        (mufu_q_slot * KvStages + previous_stage) * sizeof(uint64_t);
                    // O is a single long-lived accumulator per Q slot. Its
                    // alpha rescale must not race the preceding asynchronous PV.
                    mbarrier_wait(previous_mma2_addr, previous_phase);
                }
                const int mma1_addr = mma1_mbar_base + (mufu_q_slot * KvStages + stage) * sizeof(uint64_t);
                mbarrier_wait(mma1_addr, mufu_mma1_phase[mufu_q_slot][stage]);
                mufu_mma1_phase[mufu_q_slot][stage] ^= 1;
                const int s_tmem_addr = tmem_addr + kSBankBase + mufu_q_slot * mma_tmem_columns_per_stage;
                float score[8];
                float row_max = -__int_as_float(0x7F800000);
                for (int col8 = 0; col8 < TileKV / 8; ++col8) {
                    tcgen05_ld_f32_x8(score, s_tmem_addr + (row_base << 16) + col8 * 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int key_col = kv_tile * TileKV + col8 * 8 + e;
                        if (!valid_q_row || key_col >= key_len || (causal && key_col > q_row)) {
                            score[e] = -__int_as_float(0x7F800000);
                        }
                        row_max = fmaxf(row_max, score[e] * scale);
                    }
                }
                const int ml_idx = mufu_q_slot * TileQ + row;
                // Online softmax recurrence for this KV tile.  `alpha` rescales
                // both the previous denominator l and the unnormalized O_tilde.
                const float old_m = tilem[ml_idx];
                const float old_l = tilel[ml_idx];
                const float new_m = valid_q_row ? fmaxf(old_m, row_max) : old_m;
                const float alpha = valid_q_row && old_l != 0.0f ? expf(old_m - new_m) : 0.0f;
                float tile_sum = 0.0f;
                for (int col8 = 0; col8 < TileKV / 8; ++col8) {
                    const int score_addr = s_tmem_addr + (row_base << 16) + col8 * 8;
                    tcgen05_ld_f32_x8(score, score_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int key_col = kv_tile * TileKV + col8 * 8 + e;
                        const bool valid_key = valid_q_row && key_col < key_len &&
                            (!causal || key_col <= q_row);
                        const float p = valid_key ? expf(score[e] * scale - new_m) : 0.0f;
                        score[e] = p;
                        tile_sum += p;
                    }
                    // Compact FP32 S (one value per word) into packed BF16 P
                    // (two values per word). Forward compaction cannot clobber
                    // a score fragment that has not been read yet.
                    const int p_addr = s_tmem_addr + (row_base << 16) + col8 * 4;
                    tcgen05_st_bf16_x8(score, p_addr);
                }
                tilem[ml_idx] = new_m;
                tilel[ml_idx] = valid_q_row ? alpha * old_l + tile_sum : old_l;
                // Apply alpha before issuing this tile's PV MMA:
                // O_tilde_new = alpha * O_tilde_old + P_tile @ V_tile.
                for (int d8 = 0; d8 < head_dim / 8; ++d8) {
                    float o_fragment[8];
                    const int addr = o_tmem_addr + (row_base << 16) + d8 * 8;
                    tcgen05_ld_f32_x8(o_fragment, addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) o_fragment[e] *= alpha;
                    tcgen05_st_f32_x8(o_fragment, addr);
                }
                // P remains unnormalized and was written in the 16-bit TMEM-A
                // format expected by the subsequent kind::f16 P@V MMA.
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (lane_idx == 0) {
                    mbarrier_arrive(mufu_mbar_base + (mufu_q_slot * KvStages + stage) * sizeof(uint64_t));
                }
            }
        }

        // A single elected thread issues tcgen05 operations.  The asynchronous
        // engines perform the actual matrix work while producer/MUFU warps run.
        if (warp_idx == kMmaWarp && elect_sync()) {
            constexpr uint32_t kQkInstrDesc = (1U << 4U) | (1U << 7U) | (1U << 10U) |
                ((uint32_t)TileKV >> 3U << 17U) | ((uint32_t)TileQ >> 4U << 24U);
            const uint32_t pv_instr_desc = (1U << 4U) | (1U << 7U) | (1U << 10U) | (1U << 16U) |
                ((uint32_t)head_dim >> 3U << 17U) | ((uint32_t)TileQ >> 4U << 24U);
            for (int kv_tile = 0; kv_tile < task_kv_tile_count; ++kv_tile) {
                const int stage = kv_tile % KvStages;
                mbarrier_wait(tma1_mbar_base + stage * sizeof(uint64_t), qk_tma1_phase[stage]);
                qk_tma1_phase[stage] ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const int k_addr = k_smem_addr + stage * TileKV * head_dim * sizeof(__nv_bfloat16);
                // QK: write FP32 scores for each resident Q tile into S0/S1.
                for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                    const int s_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int q_addr = q_smem_addr + q_slot * TileQ * head_dim * sizeof(__nv_bfloat16);
                    for (int d64 = 0; d64 < head_dim / 64; ++d64) for (int d16 = 0; d16 < 4; ++d16) {
                        tcgen05_mma_bf16(s_addr,
                            make_128b_swizzle_desc(q_addr + d64 * TileQ * 128 + d16 * 32),
                            make_128b_swizzle_desc(k_addr + d64 * TileKV * 128 + d16 * 32),
                            kQkInstrDesc, d64 != 0 || d16 != 0);
                    }
                    tcgen05_commit_mbarrier(mma1_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t));
                }
                mbarrier_wait(tma2_mbar_base + stage * sizeof(uint64_t), pv_tma2_phase[stage]);
                pv_tma2_phase[stage] ^= 1;
                // PV: wait until MUFU has converted S to packed BF16 P, then
                // accumulate P @ V directly into the long-lived O bank.
                for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                    const int mufu_addr = mufu_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t);
                    mbarrier_wait(mufu_addr, pv_mufu_phase[q_slot][stage]);
                    pv_mufu_phase[q_slot][stage] ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    const int o_addr = tmem_addr + kOBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int p_addr = tmem_addr + kSBankBase + q_slot * mma_tmem_columns_per_stage;
                    const int v_addr = v_smem_addr + stage * TileKV * head_dim * sizeof(__nv_bfloat16);
                    for (int k16 = 0; k16 < TileKV / 16; ++k16) {
                        tcgen05_mma_tmem_a_bf16(o_addr, p_addr + k16 * 8,
                            make_128b_nmajor_desc(v_addr + k16 * 16 * 128,
                                TileKV * 128),
                            pv_instr_desc, 1);
                    }
                    tcgen05_commit_mbarrier(mma2_mbar_base + (q_slot * KvStages + stage) * sizeof(uint64_t));
                }
            }
            // A fresh, non-ring completion point protects the final O from
            // parity wraparound on the recycled mma2 barriers.
            tcgen05_commit_mbarrier(epilogue_mbar_addr);
        }

        // Epilogue has its own four warps: once the final PV commit is visible,
        // read O_tilde and perform the only division in the whole mainloop,
        // O = O_tilde / l, before storing BF16 output.
        if (warp_idx >= kEpilogueWarpBegin && warp_idx < kEpilogueWarpEnd) {
            const int row_base = (warp_idx - kEpilogueWarpBegin) * kWarpSize;
            const int row = row_base + lane_idx;
            mbarrier_wait(epilogue_mbar_addr, epilogue_phase);
            epilogue_phase ^= 1;
            for (int q_slot = 0; q_slot < QSlots; ++q_slot) {
                const int q_row = q_tile_start[q_slot] + row;
                const bool valid_q_row = q_row < query_len;
                const float inv_l = valid_q_row
                    ? 1.0f / tilel[q_slot * TileQ + row] : 0.0f;
                const int o_addr =
                    tmem_addr + kOBankBase + q_slot * mma_tmem_columns_per_stage;
                const int out_base = valid_q_row
                    ? (bh * query_len + q_row) * head_dim : 0;
                for (int d8 = 0; d8 < head_dim / 8; ++d8) {
                    float o_fragment[8];
                    // tcgen05.ld is warp-collective, so invalid tail lanes must
                    // still participate. Only the ordinary global stores diverge.
                    tcgen05_ld_f32_x8(
                        o_fragment, o_addr + (row_base << 16) + d8 * 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (valid_q_row) {
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) out[out_base + d8 * 8 + e] =
                            __float2bfloat16_rn(o_fragment[e] * inv_l);
                    }
                }
            }
        }

        __syncthreads();
    }

    __syncthreads();
    if (warp_idx == kMmaWarp) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::
                         "r"(tmem_addr), "r"(mma_tmem_columns));
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
    constexpr int kThreads = 16 * 32;
    // S0/S1 plus long-lived O0/O1 consume all 512 available TMEM columns.
    if (head_dim != 128) {
        return cudaErrorInvalidValue;
    }

    CUtensorMap q_tmap{};
    CUtensorMap k_tmap{};
    CUtensorMap v_tmap{};
    cudaError_t status = init_qkv_tmap(
        &q_tmap, q, batch * heads, query_len, head_dim, kTileQ);
    if (status != cudaSuccess) {
        return status;
    }
    status = init_qkv_tmap(
        &k_tmap, k, batch * heads, key_len, head_dim, kTileKV);
    if (status != cudaSuccess) {
        return status;
    }
    status = init_qkv_tmap(
        &v_tmap, v, batch * heads, key_len, head_dim, kTileKV);
    if (status != cudaSuccess) {
        return status;
    }

    using SmemLayout = FaMhaSmemLayout<kTileQ, kTileKV, kQSlots, kKvStages>;
    const SmemLayout smem_layout = SmemLayout::make(head_dim);
    if (smem_layout.dynamic_bytes + SmemLayout::kStaticMbarrierBytes >
        SmemLayout::kMaxSmemBytes) {
        return cudaErrorInvalidConfiguration;
    }

    status = cudaFuncSetAttribute(
        fa_mha_kernel<kTileQ, kTileKV, kQSlots, kKvStages>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_layout.dynamic_bytes));
    if (status != cudaSuccess) {
        return status;
    }

    int device = 0;
    status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    int sm_count = 0;
    status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) {
        return status;
    }
    const int q_tile_count = (query_len + kTileQ - 1) / kTileQ;
    const int q_pair_count = (q_tile_count + kQSlots - 1) / kQSlots;
    const int total_task_count = q_pair_count * batch * heads;
    // Launch no more CTAs than SMs; the device-side task-stride loop above
    // distributes all remaining Q-pairs to these persistent CTAs.
    const int persistent_cta_count =
        total_task_count < sm_count ? total_task_count : sm_count;

    const dim3 block(kThreads);
    const dim3 grid(persistent_cta_count);
    fa_mha_kernel<kTileQ, kTileKV, kQSlots, kKvStages><<<grid, block, smem_layout.dynamic_bytes, stream>>>(q_tmap,
        k_tmap,
        v_tmap,
        out,
        batch,
        heads,
        query_len,
        key_len,
        head_dim,
        scale,
        causal);
    return cudaGetLastError();
}
