#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#ifndef SM110A_TOPK_STAGES
#define SM110A_TOPK_STAGES 2
#endif
#ifndef SM110A_TOPK_TMA
#define SM110A_TOPK_TMA 0
#endif

namespace {
constexpr int Stages = SM110A_TOPK_STAGES;
constexpr bool UseTma = SM110A_TOPK_TMA;
static_assert(Stages >= 2 && Stages <= 8, "TopK supports 2 through 8 stages");
// TODO(sm110a): support short candidate lists (KVlen < 512). Both the Python
// facade and C++ binding currently require k == 512 and KVlen >= 512, matching
// this compile-time constant. The DSV4 ratio-4 CSA indexer therefore rejects
// contexts shorter than 2048 tokens, even though Qlen == 1 decode is supported.
// When selecting min(512, KVlen), KVlen <= 512 needs no selection: return all
// candidate indices. Add that fast path and update the API checks/output shape;
// keep prefill causal masking and cache-index offsets in the model adapter.
constexpr int TopK = 512;
constexpr int Threads = 256;
constexpr int Block = 1024;
constexpr int B2 = 1024;

// NaNs rank above +inf; signed zeros compare equal.
__device__ __forceinline__ uint32_t key(
    float x) {
    uint32_t result;
    // Classify zero/NaN independently of the sign transform. The LUT is
    // a ^ (b | c): 0xf0 ^ (0xcc | 0xaa) = 0x1e. No .ftz: keep subnormals.
    asm("{ .reg .b32 bits, sign, ordered; .reg .pred zero, nan;\n"
        "mov.b32 bits, %1;\n"
        "shr.s32 sign, bits, 31;\n"
        "lop3.b32 ordered, bits, sign, 0x80000000, 0x1e;\n"
        "setp.eq.f32 zero, %1, 0f00000000;\n"
        "setp.nan.f32 nan, %1, %1;\n"
        "selp.b32 ordered, 0x80000000, ordered, zero;\n"
        "selp.b32 %0, 0xffffffff, ordered, nan; }"
        : "=r"(result) : "f"(x));
    return result;
}

struct RadixStorage {
    int histogram[256];
    int warp_sums[Threads / 32];
    int digit;
    int above;
};
struct ScanResult {
    int rank;
    int total;
};

// Each warp reduces the eight warp totals in registers. ProtectReuse=false is
// valid only when the caller supplies a barrier before the next scratch write.
template <bool ProtectReuse = true, bool CountFour = false>
__device__ __forceinline__ ScanResult block_scan(
    int value,
    RadixStorage &s) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int sum;
    // Filter counts are in [0, 4]; bit planes shorten their warp prefix.
    if constexpr (CountFour) {
        unsigned inclusive_lanes;
        asm("mov.u32 %0, %%lanemask_le;" : "=r"(inclusive_lanes));
        sum = 0;
#pragma unroll
        for (int bit = 0; bit < 3; ++bit) {
            unsigned plane;
            asm("{ .reg .b32 masked; .reg .pred p;\n"
                "and.b32 masked, %1, %2;\n"
                "setp.ne.b32 p, masked, 0;\n"
                "vote.sync.ballot.b32 %0, p, 0xffffffff; }"
                : "=r"(plane) : "r"(value), "r"(1 << bit));
            sum += __popc(plane & inclusive_lanes) << bit;
        }
    } else {
        sum = value;
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            asm("{ .reg .pred p; .reg .b32 other;\n"
                "shfl.sync.up.b32 other|p, %0, %1, 0, 0xffffffff;\n"
                "@p add.s32 %0, %0, other; }"
                : "+r"(sum) : "r"(offset));
        }
    }
    if (lane == 31)
        s.warp_sums[warp] = sum;
    __syncthreads();
    const int warp_sum = lane < Threads / 32 ? s.warp_sums[lane] : 0;
    int preceding, total;
    // Every warp uses the same full member mask; zero unwanted inputs
    // instead of using a per-lane reduction mask (which would be invalid).
    const int preceding_value = lane < warp ? warp_sum : 0;
    if constexpr (CountFour) {
        // Both block counts are <= 1024, so the fields cannot carry into
        // each other. One reduction supplies the total and preceding count.
        static_assert(Threads * 4 < 2048);
        const int packed = warp_sum | (preceding_value << 11);
        int both;
        asm("redux.sync.add.s32 %0, %1, 0xffffffff;"
            : "=r"(both) : "r"(packed));
        total = both & 2047;
        preceding = both >> 11;
    } else {
        asm("redux.sync.add.s32 %0, %1, 0xffffffff;"
            : "=r"(total) : "r"(warp_sum));
        asm("redux.sync.add.s32 %0, %1, 0xffffffff;"
            : "=r"(preceding) : "r"(preceding_value));
    }
    // preceding is already zero for warp 0; no extra select is necessary.
    const ScanResult result = {sum - value + preceding, total};
    if constexpr (ProtectReuse)
        __syncthreads();
    return result;
}

struct Selection {
    uint32_t threshold_key;
    int equal_needed;
};

// Find the threshold among shared-memory monotonic keys. Normalize promotes
// the raw FP32 suffix [keyed_count, count) once, during the first histogram.
// That round's existing completion barrier publishes the normalized keys.
// All 256 consumer threads participate; count >= k > 0.
template <typename T, bool Normalize>
__device__ __forceinline__ Selection radix_threshold(
    uint32_t *values,
    int count,
    int k,
    RadixStorage &s,
    int keyed_count) {
    uint32_t prefix = 0;
    uint32_t mask = 0;
    int remaining = k;
#pragma unroll
    for (int shift = 24; shift >= (sizeof(T) == 2 ? 16 : 0); shift -= 8) {
        s.histogram[threadIdx.x] = 0;
        __syncthreads();
        // Avoid automatic eight-way unrolling and its remainder paths.
#pragma unroll 4
        for (int i = threadIdx.x; i < count; i += Threads) {
            uint32_t v = values[i];
            if constexpr (Normalize) {
                if (shift == 24 && i >= keyed_count) {
                    v = key(__uint_as_float(v));
                    values[i] = v;
                }
            }
            if ((v & mask) == prefix)
                atomicAdd(&s.histogram[(v >> shift) & 255u], 1);
        }
        __syncthreads();
        const int digit = 255 - threadIdx.x;
        const int bucket_count = s.histogram[digit];
        const ScanResult scan = block_scan<false>(bucket_count, s);
        if (scan.rank < remaining && scan.rank + bucket_count >= remaining) {
            s.digit = digit;
            s.above = scan.rank;
        }
        // Also protects scan scratch reads before the next scan invocation.
        __syncthreads();
        prefix |= uint32_t(s.digit) << shift;
        mask |= 255u << shift;
        remaining -= s.above;
        // The next round's histogram barriers precede the next digit write.
    }
    if constexpr (sizeof(T) == 2) {
        // Keys came from exact BF16-to-FP32 expansions. Once the high half is
        // known, the low half is zero for nonnegative values, all ones for
        // negative values and for the canonical NaN key.
        if (!(prefix & 0x80000000u) || prefix == 0xffff0000u)
            prefix |= 0xffffu;
    }
    return {prefix, remaining};
}

// Device primitive: compact count shared keys to K in place,
// carrying their GMEM indices. Input order is stable, so ties keep lower indices.
// Each tile is loaded into registers before any overlapping output write.
// Returns the selected threshold key in every thread's register.
template <typename T, bool Normalize = false>
__device__ __forceinline__ uint32_t radixselectTopk(
    uint32_t *values,
    int32_t *indices,
    int count,
    int k,
    RadixStorage &s,
    int keyed_count = 0) {
    // The first histogram initialization barrier publishes candidate writes.
    const Selection selection = radix_threshold<T, Normalize>(values, count, k, s, keyed_count);
    int equal_seen = 0;
    int emitted = 0;
    // Each thread owns four adjacent candidates. Register loads precede the
    // scan barriers, so stable in-place writes cannot clobber unread input.
    // One scan covers 1024 candidates instead of 256.
    constexpr int Items = 4;
    for (int base = 0; base < count; base += Threads * Items) {
        uint32_t value[Items];
        int32_t index[Items];
        unsigned equals = 0, greater = 0;
        const int first = base + threadIdx.x * Items;
        if (first + Items <= count) {
            // Four adjacent words per lane: wide shared loads avoid the
            // four-way bank conflicts of four separate strided word loads.
            const uint4 v = *reinterpret_cast<const uint4 *>(values + first);
            const int4 ix = *reinterpret_cast<const int4 *>(indices + first);
            value[0] = v.x;
            value[1] = v.y;
            value[2] = v.z;
            value[3] = v.w;
            index[0] = ix.x;
            index[1] = ix.y;
            index[2] = ix.z;
            index[3] = ix.w;
        } else {
#pragma unroll
            for (int item = 0; item < Items; ++item) {
                const int i = first + item;
                value[item] = i < count ? values[i] : 0u;
                index[item] = i < count ? indices[i] : 0;
            }
        }
#pragma unroll
        for (int item = 0; item < Items; ++item) {
            const int i = first + item;
            const uint32_t v = value[item];
            equals |= unsigned(i < count && v == selection.threshold_key) << item;
            greater |= unsigned(i < count && v > selection.threshold_key) << item;
        }
        // At most 1024 equals: reserve 11 bits to prevent a carry.
        const ScanResult both = block_scan(__popc(equals) | (__popc(greater) << 11), s);
        const int equal_rank = both.rank & 2047;
        const int equal_total = both.total & 2047;
        const int equal_budget = max(0, selection.equal_needed - equal_seen);
        int destination = emitted + (both.rank >> 11) + min(equal_rank, equal_budget);
        int local_equal = 0;
#pragma unroll
        for (int item = 0; item < Items; ++item) {
            const bool equal = equals & (1u << item);
            const int equal_index = equal_seen + equal_rank + local_equal;
            if ((greater & (1u << item)) ||
                (equal && equal_index < selection.equal_needed)) {
                values[destination] = value[item];
                indices[destination++] = index[item];
            }
            local_equal += equal;
        }
        equal_seen += equal_total;
        emitted += (both.total >> 11) + min(equal_total, equal_budget);
        // Every candidate in this tile was loaded before scan barriers.
        // Stable writes never overlap the next tile's input.
    }
    __syncthreads();
    return selection.threshold_key;
}

__device__ __forceinline__ uint4 load_four(
    const float *input) {
    return *reinterpret_cast<const uint4 *>(input);
}
__device__ __forceinline__ uint4 load_four(
    const __nv_bfloat16 *input) {
    const uint2 packed = *reinterpret_cast<const uint2 *>(input);
    return make_uint4(packed.x, packed.y, 0, 0);
}
__device__ __forceinline__ float packed_value(
    uint4 data,
    int bit,
    const float *) {
    const uint32_t bits = bit == 0 ? data.x : bit == 1 ? data.y : bit == 2 ? data.z : data.w;
    return __uint_as_float(bits);
}
__device__ __forceinline__ float packed_value(
    uint4 data,
    int bit,
    const __nv_bfloat16 *) {
    const uint32_t pair = bit < 2 ? data.x : data.y;
    return __bfloat162float(__ushort_as_bfloat16(uint16_t(pair >> ((bit & 1) * 16))));
}
__device__ __forceinline__ uint32_t packed_mask(
    uint4 data,
    uint32_t threshold,
    const float *type) {
    if (threshold == 0xffffffffu)
        return 0;
    const uint32_t bits = threshold ^
        ((threshold & 0x80000000u) ? 0x80000000u : 0xffffffffu);
    const float value = __uint_as_float(bits);
    uint32_t mask = 0;
#pragma unroll
    for (int bit = 0; bit < 4; ++bit) {
        uint32_t comparison;
        // Unordered greater preserves NaN ordering and treats signed zeros
        // equally. No .ftz: FP32 subnormal scores must retain their ordering.
        asm("set.gtu.u32.f32 %0, %1, %2;" : "=r"(comparison)
            : "f"(packed_value(data, bit, type)), "f"(value));
        mask |= comparison & (1u << bit);
    }
    return mask;
}
__device__ __forceinline__ uint32_t packed_mask(
    uint4 data,
    uint32_t threshold,
    const __nv_bfloat16 *) {
    // All NaNs share the maximum monotonic key. No candidate can beat it.
    if (threshold == 0xffffffffu)
        return 0;
    const uint32_t float_bits = threshold ^ ((threshold & 0x80000000u) ? 0x80000000u : 0xffffffffu);
    // The selected threshold is an exact BF16 expansion; its upper half is
    // already the BF16 encoding. Avoid a redundant FP32-to-BF16 conversion.
    uint32_t pair_threshold;
    asm("prmt.b32 %0, %1, %1, 0x3232;" : "=r"(pair_threshold) : "r"(float_bits));
    uint32_t cmp01, cmp23, bytes, mask;
    // Unordered greater includes candidate NaNs, matching the public ordering.
    asm("set.gtu.s32.bf16x2 %0, %1, %2;" : "=r"(cmp01) : "r"(data.x), "r"(pair_threshold));
    asm("set.gtu.s32.bf16x2 %0, %1, %2;" : "=r"(cmp23) : "r"(data.y), "r"(pair_threshold));
    // Pack four -1/0 comparison results into signed bytes, then weight them
    // by {-1,-2,-4,-8} to produce the four-bit keep mask in one dot product.
    asm("prmt.b32 %0, %1, %2, 0x6420;" : "=r"(bytes) : "r"(cmp01), "r"(cmp23));
    asm("dp4a.s32.s32 %0, %1, %2, 0;" : "=r"(mask) : "r"(bytes), "r"(0xf8fcfeffu));
    return mask;
}

template <typename T>
__device__ __forceinline__ int filter_mask(
    const T *input,
    int64_t base,
    int n,
    uint32_t threshold,
    bool have_threshold,
    uint32_t *values,
    int32_t *indices,
    int count,
    RadixStorage &s) {
    // Contiguous ownership preserves input order after the per-thread count scan.
    const int offset = threadIdx.x * 4;
    const uint4 data = load_four(input + offset);
    const int64_t remaining = int64_t(n) - base - offset;
    if (!have_threshold) {
        // Before the first prune every input survives, so its destination is
        // known without a scan. Earlier full tiles make count a multiple of 4.
        if (remaining >= 4) {
            const uint4 raw = make_uint4(__float_as_uint(packed_value(data, 0, input)),
                __float_as_uint(packed_value(data, 1, input)), __float_as_uint(packed_value(data, 2, input)),
                __float_as_uint(packed_value(data, 3, input)));
            const int i = int(base) + offset;
            *reinterpret_cast<uint4 *>(values + count + offset) = raw;
            *reinterpret_cast<int4 *>(indices + count + offset) = make_int4(i, i + 1, i + 2, i + 3);
        } else {
#pragma unroll
            for (int bit = 0; bit < 4; ++bit) {
                if (bit < remaining) {
                    values[count + offset + bit] = __float_as_uint(packed_value(data, bit, input));
                    indices[count + offset + bit] = int32_t(base + offset + bit);
                }
            }
        }
        // All stage reads must finish before any thread issues the next copy.
        __syncthreads();
        return count + int(min(int64_t(Block), int64_t(n) - base));
    }
    const int valid = remaining <= 0 ? 0 : remaining >= 4 ? 4 : int(remaining);
    uint32_t mask =
        packed_mask(data, threshold, input) & ((1u << valid) - 1u);
    // The next input wait, radix initialization, or final publication supplies
    // a CTA barrier before warp_sums can be written again. The scan's first
    // barrier already protects this stage's reads against its next prefetch.
    const ScanResult scan = block_scan<false, true>(__popc(mask), s);
    int destination = count + scan.rank;
    // Constant item indices eliminate the bit-scan loop and dynamic packed
    // extraction. Ascending stores preserve stable ordering within a thread.
#pragma unroll
    for (int bit = 0; bit < 4; ++bit) {
        if (mask & (1u << bit)) {
            values[destination] = __float_as_uint(packed_value(data, bit, input));
            indices[destination++] = int32_t(base + offset + bit);
        }
    }
    return count + scan.total;
}

template <int CopyBytes, typename T>
__device__ __forceinline__ void copy_input_async(
    T *stage,
    const T *row,
    int64_t base,
    int n) {
    constexpr int Elements = CopyBytes / sizeof(T);
    for (int offset = threadIdx.x * Elements; offset < Block; offset += Threads * Elements) {
        const int64_t i = base + offset;
        const int available = i < n ? int((n - i) < Elements ? n - i : Elements) * sizeof(T) : 0;
        const T *source = row + (i < n ? i : base);
        const uint32_t destination = __cvta_generic_to_shared(stage + offset);
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(destination),
            "l"(source),
            "n"(CopyBytes),
            "r"(available)
            : "memory");
    }
}

// Commit one input block per thread. Partial copies zero-fill the tail;
// consumers still mask by n so padding cannot enter the candidate buffer.
template <typename T>
__device__ __forceinline__ void prefetch_input(
    T *stage,
    const T *row,
    int64_t base,
    int n) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(row);
    if ((address & 15u) == 0) {
        copy_input_async<16>(stage, row, base, n);
    } else if ((address & 3u) == 0) {
        copy_input_async<4>(stage, row, base, n);
    } else {
        // BF16 permits a 2-byte-aligned row (including contiguous odd widths).
        for (int offset = threadIdx.x; offset < Block; offset += Threads) {
            const int64_t i = base + offset;
            stage[offset] = i < n ? row[i] : T{};
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int Pending>
__device__ __forceinline__ void wait_copy_groups(int pending) {
    if (pending == Pending)
        asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
    else if constexpr (Pending > 0)
        wait_copy_groups<Pending - 1>(pending);
}

__device__ __forceinline__ void wait_stage(
    uint64_t *barrier,
    int phase) {
    const uint32_t address = __cvta_generic_to_shared(barrier);
    asm volatile("{ .reg .pred done;\n"
                 "wait_topk_stage:\n"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1;\n"
                 "@!done bra wait_topk_stage;\n}" ::"r"(address),
        "r"(phase)
        : "memory");
}

template <typename T>
__device__ __forceinline__ bool use_tma_block(
    const T *row,
    int64_t base,
    int n) {
    return UseTma && (reinterpret_cast<uintptr_t>(row) & 15u) == 0 && base + Block <= n;
}

template <typename T>
__device__ __forceinline__ void issue_input(
    T *stage, const T *row, int64_t base, int n, uint64_t *barrier) {
    if (use_tma_block(row, base, n)) {
        if (threadIdx.x == 0) {
            const uint32_t address = __cvta_generic_to_shared(barrier);
            const uint32_t destination = __cvta_generic_to_shared(stage);
            constexpr int Bytes = Block * sizeof(T);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(address), "r"(Bytes) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                         "[%0], [%1], %2, [%3];"
                :: "r"(destination), "l"(row + base), "r"(Bytes), "r"(address) : "memory");
        }
    } else {
        prefetch_input(stage, row, base, n);
    }
}

template <typename T, int Split>
__global__ void topk_buffered(
    const T *scores,
    int32_t *out,
    int n) {
    constexpr int k = TopK;
    __shared__ RadixStorage scratch;
    __shared__ int retained;
    __shared__ __align__(16) T input[Stages][Block];
    __shared__ __align__(8) uint64_t ready[Stages];
    extern __shared__ __align__(16) uint32_t buffer[];
    // Before an append count < k+B2; one block can add at most Block entries.
    constexpr int capacity = k + B2 + Block;
    static_assert(capacity % 4 == 0, "candidate arrays require 16-byte alignment");
    // Retained candidates are keys; appended candidates are raw FP32 bits.
    // The first histogram normalizes the suffix once. Later rounds,
    // compaction, and DSM merges reuse keys rather than reclassifying floats.
    uint32_t *values = buffer;
    int32_t *indices = reinterpret_cast<int32_t *>(buffer + capacity);
    const int rank = blockIdx.x % Split;
    const int row_id = blockIdx.x / Split;
    const int64_t total_blocks = (int64_t(n) + Block - 1) / Block;
    const int64_t start = total_blocks * rank / Split * Block;
    const int64_t end = total_blocks * (rank + 1) / Split * Block;
    const int local_n = int(min(int64_t(n), end) - min(int64_t(n), start));
    // Empty partitions still participate in every cluster barrier.
    const T *row = scores + int64_t(row_id) * n + min(int64_t(n), start);
    int32_t *result = out + int64_t(row_id) * k;
    int count = 0;
    uint32_t threshold = 0;
    bool have_threshold = false;
    if constexpr (UseTma) {
        if (threadIdx.x == 0) {
            for (int slot = 0; slot < Stages; ++slot) {
                const uint32_t address = __cvta_generic_to_shared(ready + slot);
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(address) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();
    }
    const int64_t blocks = (int64_t(local_n) + Block - 1) / Block;
    for (int slot = 0; slot < Stages; ++slot) {
        if (slot < blocks)
            issue_input(input[slot], row, int64_t(slot) * Block, local_n, ready + slot);
    }
    for (int64_t block = 0; block < blocks; ++block) {
        const int stage = block % Stages;
        const int64_t base = block * Block;
        if (use_tma_block(row, base, local_n)) {
            wait_stage(ready + stage, (block / Stages) & 1);
        } else {
            const int pending = UseTma && (reinterpret_cast<uintptr_t>(row) & 15u) == 0
                ? 0 : int(blocks - block - 1 < Stages - 1 ? blocks - block - 1 : Stages - 1);
            wait_copy_groups<Stages - 1>(pending);
        }
        __syncthreads();
        count = filter_mask(input[stage], base, local_n, threshold,
            have_threshold, values, indices, count, scratch);
        if (count >= k + B2) {
            threshold = radixselectTopk<T, true>(values, indices, count, k, scratch,
                have_threshold ? k : 0);
            have_threshold = true;
            count = k;
        }
        if (block + Stages < blocks)
            issue_input(input[stage], row, (block + Stages) * Block, local_n, ready + stage);
    }
    if (count > k) {
        radixselectTopk<T, true>(values, indices, count, k, scratch,
            have_threshold ? k : 0);
    } else {
        __syncthreads();
        if constexpr (Split > 1) {
            if (!have_threshold) {
                // Small/empty partitions did not select, but DSM peers expect keys.
                for (int i = threadIdx.x; i < count; i += Threads)
                    values[i] = key(__uint_as_float(values[i]));
            }
        }
    }
    count = min(count, k);
    for (int i = threadIdx.x; i < count; i += Threads)
        indices[i] += int32_t(start);
    if (threadIdx.x == 0)
        retained = count;

    if constexpr (Split > 1) {
        auto cluster = cooperative_groups::this_cluster();
        // Each level publishes local results before peers read DSM. The next
        // level's barrier also prevents a source CTA from exiting too early.
#pragma unroll
        for (int stride = 1; stride < Split; stride *= 2) {
            cluster.sync();
            if (rank % (2 * stride) == 0) {
                const int peer = rank + stride;
                const uint32_t *peer_values = cluster.map_shared_rank(values, peer);
                const int32_t *peer_indices = cluster.map_shared_rank(indices, peer);
                const int peer_count = *cluster.map_shared_rank(&retained, peer);
                // Both sides retain at most K, so the merge has <= 2*K
                // candidates and needs only one final selection.
                for (int i = threadIdx.x; i < peer_count; i += Threads) {
                    values[count + i] = peer_values[i];
                    indices[count + i] = peer_indices[i];
                }
                count += peer_count;
                if (count > k) {
                    radixselectTopk<T>(values, indices, count, k, scratch);
                    count = k;
                } else {
                    __syncthreads();
                }
                if (threadIdx.x == 0)
                    retained = count;
            }
        }
        cluster.sync();
    }
    if (rank == 0) {
        for (int i = threadIdx.x; i < k; i += Threads)
            result[i] = indices[i];
    }
}

template <typename T, int Split>
void launch_split(
    const T *scores, int32_t *out, int rows, int n, cudaStream_t stream) {
    const size_t smem = size_t(TopK + B2 + Block) * (sizeof(uint32_t) + sizeof(int32_t));
    constexpr size_t StaticBytes =
        sizeof(RadixStorage) + Stages * Block * sizeof(T) + Stages * sizeof(uint64_t) + 48;
    if (smem + StaticBytes > 48 * 1024)
        cudaFuncSetAttribute(topk_buffered<T, Split>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if constexpr (Split == 1) {
        topk_buffered<T, Split><<<rows, Threads, smem, stream>>>(scores, out, n);
    } else {
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(static_cast<unsigned>(rows) * Split);
        config.blockDim = dim3(Threads);
        config.dynamicSmemBytes = smem;
        config.stream = stream;
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {Split, 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;
        cudaLaunchKernelEx(&config, topk_buffered<T, Split>, scores, out, n);
    }
}

template <typename T>
void launch(const T *scores, int32_t *out, int rows, int n, int split,
    cudaStream_t stream) {
    switch (split) {
    case 1: launch_split<T, 1>(scores, out, rows, n, stream); break;
    case 2: launch_split<T, 2>(scores, out, rows, n, stream); break;
    case 4: launch_split<T, 4>(scores, out, rows, n, stream); break;
    case 8: launch_split<T, 8>(scores, out, rows, n, stream); break;
    }
}
} // namespace

extern "C" void sm110a_topk_launch(
    const void *scores, int32_t *out, int rows, int n, int dtype,
    int split, cudaStream_t stream) {
    if (!rows)
        return;
    if (dtype == 0)
        launch(static_cast<const float *>(scores), out, rows, n, split, stream);
    else
        launch(static_cast<const __nv_bfloat16 *>(scores), out, rows, n, split, stream);
}
