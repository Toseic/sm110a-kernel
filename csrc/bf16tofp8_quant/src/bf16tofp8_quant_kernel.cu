#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "bf16tofp8_quant.h"
#include "blocked_quant.h"

namespace sm110a::bf16tofp8_quant {

constexpr int kRows = 32;
constexpr int kBlock = 128;
constexpr int kBlocksPerStore = 2;
constexpr int kComputeWarps = 4;
constexpr int kLoadWarp = 4;
constexpr int kStoreWarp = 5;
constexpr int kThreads = 32 * (kStoreWarp + 1);
constexpr int kLanesPerRow = 4;
constexpr int kValuesPerLane = kBlock / kLanesPerRow;
constexpr int kInputBytes = kRows * kBlock * sizeof(__nv_bfloat16);
constexpr int kOutputBytes = kRows * kBlock * kBlocksPerStore * sizeof(uint8_t);
constexpr int kStageBytes = kInputBytes + kOutputBytes;
constexpr int kBarrierCount = 4;
constexpr float kE4M3Max = 448.0f;
static_assert(kRows == kComputeWarps * 32 / kLanesPerRow);

__device__ __forceinline__ void wait_barrier(int address, int phase) {
    constexpr uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred done;\n\t"
                 "WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
                 "done, [%0], %1, %2;\n\t"
                 "@done bra.uni COMPLETE;\n\t"
                 "bra.uni WAIT;\n\t"
                 "COMPLETE:\n\t"
                 "}" :: "r"(address), "r"(phase), "r"(kSuspendTicks) : "memory");
}

// E8M0 scales are powers of two, so scaling in bf16 is exact; the bf16x2 to
// e4m3x2 conversion is then a single hardware instruction per pair.
__device__ __forceinline__ uint32_t fp8x4_bf16(uint32_t w0, uint32_t w1) {
    unsigned short lo, hi;
    asm("{cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;}" : "=h"(lo) : "r"(w0));
    asm("{cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;}" : "=h"(hi) : "r"(w1));
    uint32_t packed;
    asm("mov.b32 %0, {%1, %2};" : "=r"(packed) : "h"(lo), "h"(hi));
    return packed;
}

// Streaming access: each byte is touched once, so loads are marked
// evict_first and stores evict_unchanged to keep the lines from displacing
// useful L2 state; measured ~1-1.5% faster than default policy.
__device__ __forceinline__ uint64_t l2_load_policy() {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
                 : "=l"(policy));
    return policy;
}

__device__ __forceinline__ uint64_t l2_store_policy() {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_unchanged.b64 %0, 1.0;"
                 : "=l"(policy));
    return policy;
}

__global__ __launch_bounds__(kThreads)
void bf16tofp8_quant_kernel(const __grid_constant__ CUtensorMap input_map,
                            const __grid_constant__ CUtensorMap output_map,
                            uint8_t* __restrict__ scales,
                            int tokens, int d, int tiles, int stages) {
    if constexpr (SM110A_QUANT_PDL) {
        if (threadIdx.x == 0) cudaTriggerProgrammaticLaunchCompletion();
    }

    // Stage barriers: input ready/free, output ready/free. Loader and storer
    // each own a TMA warp; four other warps convert 32 rows per tile.
    extern __shared__ __align__(128) uint8_t shared[];
    const int tid = threadIdx.x;
    if (tid < stages * kBarrierCount) {
        const int stage = tid % stages;
        const int group = tid / stages;
        uint64_t* barriers = reinterpret_cast<uint64_t*>(shared + stages * kStageBytes);
        const int address = static_cast<int>(__cvta_generic_to_shared(
            barriers + group * stages + stage));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                     "r"(address), "r"((group == 1 || group == 2) ? kComputeWarps : 1));
    }
    __syncthreads();

    const int k_blocks = d / kBlock;
    const int k_groups = (k_blocks + kBlocksPerStore - 1) / kBlocksPerStore;
    const int iterations = (tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
    if (tid == kLoadWarp * 32) {
        const uint64_t load_policy = l2_load_policy();
        const int data0 = static_cast<int>(__cvta_generic_to_shared(shared));
        const int ready0 = data0 + stages * kStageBytes;
        const int free0 = ready0 + stages * static_cast<int>(sizeof(uint64_t));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const int tile = blockIdx.x + iteration * gridDim.x;
#pragma unroll
            for (int part = 0; part < kBlocksPerStore; ++part) {
                const int sub_iteration = iteration * kBlocksPerStore + part;
                const int stage = sub_iteration % stages;
                if (sub_iteration >= stages) {
                    wait_barrier(free0 + stage * sizeof(uint64_t),
                                 ((sub_iteration - stages) / stages) & 1);
                }
                const int ready = ready0 + stage * sizeof(uint64_t);
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
                             "_, [%0], %1;" :: "r"(ready), "r"(kInputBytes) : "memory");
                asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global."
                             "mbarrier::complete_tx::bytes.L2::cache_hint "
                             "[%0], [%1, {%2, %3}], [%4], %5;" ::
                             "r"(data0 + stage * kStageBytes), "l"(&input_map),
                             "r"(((tile % k_groups) * kBlocksPerStore + part) * kBlock),
                             "r"((tile / k_groups) * kRows),
                             "r"(ready), "l"(load_policy) : "memory");
            }
        }
    } else if (tid == kStoreWarp * 32) {
        const uint64_t store_policy = l2_store_policy();
        const int data0 = static_cast<int>(__cvta_generic_to_shared(shared));
        const int output_ready0 = data0 + stages * kStageBytes +
            2 * stages * static_cast<int>(sizeof(uint64_t));
        const int output_free0 = output_ready0 +
            stages * static_cast<int>(sizeof(uint64_t));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const int stage = iteration % stages;
            wait_barrier(output_ready0 + stage * sizeof(uint64_t),
                         (iteration / stages) & 1);
            const int tile = blockIdx.x + iteration * gridDim.x;
            asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
                         ".L2::cache_hint "
                         "[%0, {%1, %2}], [%3], %4;" ::
                         "l"(&output_map),
                         "r"((tile % k_groups) * kBlocksPerStore * kBlock),
                         "r"((tile / k_groups) * kRows),
                         "r"(data0 + stage * kStageBytes + kInputBytes),
                         "l"(store_policy) : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::
                         "r"(output_free0 + stage * static_cast<int>(sizeof(uint64_t)))
                         : "memory");
        }
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    } else if (tid < kComputeWarps * 32) {
        const int lane = tid % kLanesPerRow;
        const int row_in_tile = tid / kLanesPerRow;
        const int data0 = static_cast<int>(__cvta_generic_to_shared(shared));
        const int ready0 = data0 + stages * kStageBytes;
        const int input_free0 = ready0 + stages * static_cast<int>(sizeof(uint64_t));
        const int output_ready0 = input_free0 + stages * static_cast<int>(sizeof(uint64_t));
        const int output_free0 = output_ready0 + stages * static_cast<int>(sizeof(uint64_t));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const int output_stage = iteration % stages;
            if (iteration >= stages) {
                wait_barrier(output_free0 + output_stage * sizeof(uint64_t),
                             ((iteration - stages) / stages) & 1);
            }
            const int tile = blockIdx.x + iteration * gridDim.x;
            const int row = (tile / k_groups) * kRows + row_in_tile;
#pragma unroll
            for (int part = 0; part < kBlocksPerStore; ++part) {
                const int sub_iteration = iteration * kBlocksPerStore + part;
                const int input_stage = sub_iteration % stages;
                wait_barrier(ready0 + input_stage * sizeof(uint64_t),
                             (sub_iteration / stages) & 1);
                const int k_tile = (tile % k_groups) * kBlocksPerStore + part;
                const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(
                    shared + input_stage * kStageBytes) + row_in_tile * kBlock +
                    lane * kValuesPerLane;
                __nv_bfloat16 values[kValuesPerLane];
#pragma unroll
                for (int i = 0; i < kValuesPerLane / 8; ++i) {
                    reinterpret_cast<uint4*>(values)[i] =
                        reinterpret_cast<const uint4*>(src)[i];
                }
                __syncwarp();
                if ((tid & 31) == 0) {
                    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::
                                 "r"(input_free0 + input_stage * static_cast<int>(sizeof(uint64_t)))
                                 : "memory");
                }
                const __nv_bfloat162* pairs =
                    reinterpret_cast<const __nv_bfloat162*>(values);
                __nv_bfloat162 amax2 = __habs2(pairs[0]);
#pragma unroll
                for (int i = 1; i < kValuesPerLane / 2; ++i) {
                    amax2 = __hmax2(amax2, __habs2(pairs[i]));
                }
#pragma unroll
                for (int offset = kLanesPerRow / 2; offset; offset >>= 1) {
                    amax2 = __hmax2(amax2, __shfl_down_sync(
                        0xffffffff, amax2, offset, kLanesPerRow));
                }
                float scale = 1.0f;
                if (lane == 0) {
                    const float amax = fmaxf(
                        fabsf(__bfloat162float(amax2.x)),
                        fabsf(__bfloat162float(amax2.y)));
                    const __nv_fp8_e8m0 e8m0(amax / kE4M3Max);
                    if (row < tokens && k_tile < k_blocks) {
                        if constexpr (SM110A_QUANT_BLOCKED)
                            store_blocked_quant_scale(scales, row, k_tile, k_blocks, e8m0.__x);
                        else
                            scales[static_cast<size_t>(row) * k_blocks + k_tile] = e8m0.__x;
                    }
                    scale = ldexpf(1.0f, 127 - static_cast<int>(e8m0.__x));
                }
                scale = __shfl_sync(0xffffffff, scale, 0, kLanesPerRow);
                const __nv_bfloat162 rscale2 =
                    __float22bfloat162_rn(make_float2(scale, scale));
                uint4 packed[kValuesPerLane / 16];
                uint32_t* packed_words = reinterpret_cast<uint32_t*>(packed);
                const uint32_t* words = reinterpret_cast<const uint32_t*>(values);
#pragma unroll
                for (int i = 0; i < kValuesPerLane / 4; ++i) {
                    const __nv_bfloat162 scaled0 = __hmul2(pairs[2 * i], rscale2);
                    const __nv_bfloat162 scaled1 = __hmul2(pairs[2 * i + 1], rscale2);
                    packed_words[i] = fp8x4_bf16(
                        reinterpret_cast<const uint32_t&>(scaled0),
                        reinterpret_cast<const uint32_t&>(scaled1));
                }
                uint4* dst = reinterpret_cast<uint4*>(shared +
                    output_stage * kStageBytes + kInputBytes +
                    row_in_tile * kBlock * kBlocksPerStore +
                    part * kBlock + lane * kValuesPerLane);
#pragma unroll
                for (int i = 0; i < kValuesPerLane / 16; ++i) {
                    dst[i] = packed[i];
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if ((tid & 31) == 0) {
                asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::
                             "r"(output_ready0 + output_stage * static_cast<int>(sizeof(uint64_t)))
                             : "memory");
            }
        }
    }
}

} // namespace sm110a::bf16tofp8_quant

using namespace sm110a::bf16tofp8_quant;

void launch_bf16tofp8_quant(const __nv_bfloat16* input, uint8_t* output,
                            uint8_t* scales, int tokens, int d,
                            cudaStream_t stream) {
    CUtensorMap input_map;
    const uint64_t dimensions[2] = {static_cast<uint64_t>(d),
                                     static_cast<uint64_t>(tokens)};
    const uint64_t strides[1] = {static_cast<uint64_t>(d) * sizeof(__nv_bfloat16)};
    const uint32_t box[2] = {kBlock, kRows};
    const uint32_t element_strides[2] = {1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &input_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<__nv_bfloat16*>(input), dimensions, strides, box,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(result == CUDA_SUCCESS);
    CUtensorMap output_map;
    const uint64_t output_strides[1] = {static_cast<uint64_t>(d)};
    const uint32_t output_box[2] = {kBlock * kBlocksPerStore, kRows};
    const CUresult output_result = cuTensorMapEncodeTiled(
        &output_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        output, dimensions, output_strides, output_box,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(output_result == CUDA_SUCCESS);
    int device, sms, max_shared;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&max_shared,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    static bool shared_configured = false;
    if (!shared_configured) {
        cudaFuncSetAttribute(bf16tofp8_quant_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             max_shared);
        shared_configured = true;
    }
    const int k_groups = (d / kBlock + kBlocksPerStore - 1) / kBlocksPerStore;
    const int tiles = ((tokens + kRows - 1) / kRows) * k_groups;
    const int ctas = std::min(tiles, sms);
    const int max_stages = max_shared /
        (kStageBytes + kBarrierCount * sizeof(uint64_t));
    // Beyond ~4 stages the deeper input prefetch does not add bandwidth;
    // read/write turnaround on the memory side is the binding limit.
    const int stages = std::min({max_stages, 4, (tiles + ctas - 1) / ctas});
    const int shared_bytes = stages *
        (kStageBytes + kBarrierCount * sizeof(uint64_t));
    bf16tofp8_quant_kernel<<<ctas, kThreads, shared_bytes, stream>>>(
        input_map, output_map, scales, tokens, d, tiles, stages);
}

extern "C" void sm110a_bf16tofp8_quant_launch(
    const __nv_bfloat16* input, uint8_t* output, uint8_t* scales,
    int tokens, int d, cudaStream_t stream) {
    launch_bf16tofp8_quant(input, output, scales, tokens, d, stream);
}
