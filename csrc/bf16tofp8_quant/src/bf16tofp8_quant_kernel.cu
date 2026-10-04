#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "bf16tofp8_quant.h"

namespace {

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

__device__ __forceinline__ uint32_t fp8x4(
    float x0, float x1, float x2, float x3, float scale) {
    return __nv_fp8x4_e4m3(make_float4(
        x0 / scale, x1 / scale, x2 / scale, x3 / scale)).__x;
}

__global__ __launch_bounds__(kThreads)
void bf16tofp8_quant_kernel(const __grid_constant__ CUtensorMap input_map,
                            const __grid_constant__ CUtensorMap output_map,
                            uint8_t* __restrict__ scales,
                            int tokens, int d, int tiles, int stages) {
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
                             "mbarrier::complete_tx::bytes "
                             "[%0], [%1, {%2, %3}], [%4];" ::
                             "r"(data0 + stage * kStageBytes), "l"(&input_map),
                             "r"(((tile % k_groups) * kBlocksPerStore + part) * kBlock),
                             "r"((tile / k_groups) * kRows),
                             "r"(ready) : "memory");
            }
        }
    } else if (tid == kStoreWarp * 32) {
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
            asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
                         "[%0, {%1, %2}], [%3];" ::
                         "l"(&output_map),
                         "r"((tile % k_groups) * kBlocksPerStore * kBlock),
                         "r"((tile / k_groups) * kRows),
                         "r"(data0 + stage * kStageBytes + kInputBytes) : "memory");
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
                float amax = 0.0f;
#pragma unroll
                for (int i = 0; i < kValuesPerLane; ++i) {
                    amax = fmaxf(amax, fabsf(__bfloat162float(values[i])));
                }
#pragma unroll
                for (int offset = kLanesPerRow / 2; offset; offset >>= 1) {
                    amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax,
                                                        offset, kLanesPerRow));
                }
                float scale = 1.0f;
                if (lane == 0) {
                    const __nv_fp8_e8m0 e8m0(amax / kE4M3Max);
                    if (row < tokens && k_tile < k_blocks) {
                        scales[static_cast<size_t>(row) * k_blocks + k_tile] = e8m0.__x;
                    }
                    scale = ldexpf(1.0f, static_cast<int>(e8m0.__x) - 127);
                }
                scale = __shfl_sync(0xffffffff, scale, 0, kLanesPerRow);
                uint4 packed[kValuesPerLane / 16];
                uint32_t* words = reinterpret_cast<uint32_t*>(packed);
#pragma unroll
                for (int i = 0; i < kValuesPerLane; i += 4) {
                    words[i / 4] = fp8x4(
                        __bfloat162float(values[i]),
                        __bfloat162float(values[i + 1]),
                        __bfloat162float(values[i + 2]),
                        __bfloat162float(values[i + 3]), scale);
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

}  // namespace

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
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(output_result == CUDA_SUCCESS);
    int device, sms, max_shared;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device);
    const int k_groups = (d / kBlock + kBlocksPerStore - 1) / kBlocksPerStore;
    const int tiles = ((tokens + kRows - 1) / kRows) * k_groups;
    const int ctas = std::min(tiles, sms);
    const int max_stages = max_shared /
        (kStageBytes + kBarrierCount * sizeof(uint64_t));
    const int stages = std::min(max_stages, (tiles + ctas - 1) / ctas);
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
