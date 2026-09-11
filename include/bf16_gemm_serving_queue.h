#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#include <cuda/atomic>
#endif

namespace sm110a::bf16_gemm_serving {

#if defined(__CUDACC__)
#define SM110A_SERVING_HD __host__ __device__
#else
#define SM110A_SERVING_HD
#endif

inline constexpr uint32_t kServingQueueMagic = 0x31515342u;  // "BSQ1"
inline constexpr uint32_t kServingQueueAbiVersion = 1;

enum class ServingLifecycle : uint32_t {
    kUninitialized = 0,
    kRunning = 1,
    kDrainAndStop = 2,
    kStopped = 3,
};

enum class ServingGroupState : uint64_t {
    kFree = 0,
    kReady = 1,
    kOutputReady = 2,
};

// One device-scope atomic word publishes all producer-owned group metadata.
// A ticket is the monotonic group generation, valid_rows counts rows across
// the complete cluster-M group, and flush distinguishes an explicit partial
// publication from an ordinary full group.
inline constexpr uint32_t kServingStateBits = 2;
inline constexpr uint32_t kServingFlushShift = 2;
inline constexpr uint32_t kServingValidRowsShift = 4;
inline constexpr uint32_t kServingValidRowsBits = 12;
inline constexpr uint32_t kServingTicketShift = 16;
inline constexpr uint64_t kServingTicketValueMask =
    (uint64_t{1} << (64 - kServingTicketShift)) - 1;
inline constexpr uint64_t kServingStateMask = (uint64_t{1} << kServingStateBits) - 1;
inline constexpr uint64_t kServingFlushMask = uint64_t{1} << kServingFlushShift;
inline constexpr uint64_t kServingValidRowsMask =
    ((uint64_t{1} << kServingValidRowsBits) - 1) << kServingValidRowsShift;
inline constexpr uint64_t kServingTicketMask =
    kServingTicketValueMask << kServingTicketShift;

SM110A_SERVING_HD constexpr uint64_t encode_serving_group_state(
    ServingGroupState state,
    uint64_t group_ticket,
    uint32_t valid_rows,
    bool flush) {
    return (static_cast<uint64_t>(state) & kServingStateMask) |
        (flush ? kServingFlushMask : 0) |
        ((static_cast<uint64_t>(valid_rows) << kServingValidRowsShift) &
            kServingValidRowsMask) |
        ((group_ticket << kServingTicketShift) & kServingTicketMask);
}

SM110A_SERVING_HD constexpr ServingGroupState decode_serving_group_state(uint64_t word) {
    return static_cast<ServingGroupState>(word & kServingStateMask);
}

SM110A_SERVING_HD constexpr bool decode_serving_group_flush(uint64_t word) {
    return (word & kServingFlushMask) != 0;
}

SM110A_SERVING_HD constexpr uint32_t decode_serving_group_valid_rows(uint64_t word) {
    return static_cast<uint32_t>(
        (word & kServingValidRowsMask) >> kServingValidRowsShift);
}

SM110A_SERVING_HD constexpr uint64_t decode_serving_group_ticket(uint64_t word) {
    return (word & kServingTicketMask) >> kServingTicketShift;
}

struct alignas(128) ServingQueueConfigLine {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t header_bytes;
    uint32_t group_meta_bytes;
    uint32_t slot_count;
    uint32_t group_capacity;
    uint32_t n;
    uint32_t k;
    uint32_t grid_n;
    uint32_t tile_m;
    uint32_t tile_n;
    uint32_t tile_k;
    uint32_t cluster_m;
    uint32_t epilogue_block_n;
    uint32_t reserved[18];
};

struct alignas(128) ServingQueueCounterLine {
    uint64_t value;
    uint8_t reserved[120];
};

struct alignas(128) ServingQueueLifecycleLine {
    uint32_t state;
    uint32_t stopped_clusters;
    uint32_t launched_clusters;
    uint32_t reserved0;
    uint32_t reserved[28];
};

// Each independently contended counter owns a cache line. Group metadata is
// stored immediately after this fixed-size header in the same allocation.
struct alignas(128) ServingQueueControl {
    ServingQueueConfigLine config;
    ServingQueueCounterLine published_tail;  // group-ticket units
    ServingQueueCounterLine dispatch_head;   // flattened (group,n-tile) units
    ServingQueueCounterLine recycle_head;    // group-ticket units
    ServingQueueLifecycleLine lifecycle;
};

struct alignas(32) ServingGroupMeta {
    uint64_t state_word;
    // N-tile completion tally for GEMM-side OUTPUT_READY publication; reset
    // to zero by the producer before READY.
    uint32_t completion_count;
    // Arrival tally for multi-block test I/O copies (producer publish and
    // consumer recycle). Reset discipline: whoever publishes the next state
    // for the slot zeroes it first, so every phase starts from zero.
    uint32_t io_arrival_count;
    uint64_t reserved1;
    uint64_t reserved2;
};

SM110A_SERVING_HD constexpr size_t serving_queue_control_bytes(uint32_t group_capacity) {
    return sizeof(ServingQueueControl) +
        static_cast<size_t>(group_capacity) * sizeof(ServingGroupMeta);
}

SM110A_SERVING_HD inline ServingGroupMeta* serving_group_meta(ServingQueueControl* control) {
    return reinterpret_cast<ServingGroupMeta*>(
        reinterpret_cast<uint8_t*>(control) + sizeof(ServingQueueControl));
}

SM110A_SERVING_HD inline const ServingGroupMeta* serving_group_meta(
    const ServingQueueControl* control) {
    return reinterpret_cast<const ServingGroupMeta*>(
        reinterpret_cast<const uint8_t*>(control) + sizeof(ServingQueueControl));
}

static_assert(sizeof(ServingQueueConfigLine) == 128);
static_assert(sizeof(ServingQueueCounterLine) == 128);
static_assert(sizeof(ServingQueueLifecycleLine) == 128);
static_assert(sizeof(ServingQueueControl) == 5 * 128);
static_assert(alignof(ServingQueueControl) == 128);
static_assert(sizeof(ServingGroupMeta) == 32);
static_assert(alignof(ServingGroupMeta) == 32);
static_assert(std::is_standard_layout_v<ServingQueueControl>);
static_assert(std::is_trivially_copyable_v<ServingQueueControl>);
static_assert(std::is_standard_layout_v<ServingGroupMeta>);
static_assert(std::is_trivially_copyable_v<ServingGroupMeta>);
static_assert(offsetof(ServingQueueControl, published_tail) == 128);
static_assert(offsetof(ServingQueueControl, dispatch_head) == 256);
static_assert(offsetof(ServingQueueControl, recycle_head) == 384);
static_assert(offsetof(ServingQueueControl, lifecycle) == 512);

#if defined(__CUDACC__)
template <typename T>
__device__ inline T serving_load_acquire(T* ptr) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ref(*ptr);
    return ref.load(cuda::memory_order_acquire);
}

template <typename T>
__device__ inline T serving_load_relaxed(T* ptr) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ref(*ptr);
    return ref.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ inline void serving_store_release(T* ptr, T value) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ref(*ptr);
    ref.store(value, cuda::memory_order_release);
}

template <typename T>
__device__ inline void serving_store_relaxed(T* ptr, T value) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ref(*ptr);
    ref.store(value, cuda::memory_order_relaxed);
}

template <typename T>
__device__ inline bool serving_compare_exchange_weak_relaxed(
    T* ptr,
    T& expected,
    T desired) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ref(*ptr);
    return ref.compare_exchange_weak(
        expected,
        desired,
        cuda::memory_order_relaxed,
        cuda::memory_order_relaxed);
}

template <typename T>
__device__ inline T serving_fetch_add_acq_rel(T* ptr, T value) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ref(*ptr);
    return ref.fetch_add(value, cuda::memory_order_acq_rel);
}
#endif

#undef SM110A_SERVING_HD

}  // namespace sm110a::bf16_gemm_serving
