#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "bf16tonvfp4_quant.h"

namespace sm110a::bf16tonvfp4_quant {

constexpr int kRows = 32;
constexpr int kBlock = 128;
constexpr int kFp4Block = 16;
constexpr int kBlocksPerStore = 2;
constexpr int kComputeWarps = 4;
constexpr int kLoadWarp = 4;
constexpr int kStoreWarp = 5;
constexpr int kThreads = 32 * (kStoreWarp + 1);
constexpr int kLanesPerRow = 4;
constexpr int kValuesPerLane = kBlock / kLanesPerRow;
constexpr int kLaneBlocks = kValuesPerLane / kFp4Block;
constexpr int kPackedBytesPerLane = kValuesPerLane / 2;
constexpr int kRowPackedBytes = kBlock * kBlocksPerStore / 2;
constexpr int kInputBytes = kRows * kBlock * sizeof(__nv_bfloat16);
constexpr int kOutputBytes = kRows * kRowPackedBytes;
constexpr int kStageBytes = kInputBytes + kOutputBytes;
constexpr int kBarrierCount = 4;
// Blocked (SWIZZLE_32_4_4) scale layout: a 128-row x 4-block logical tile is
// 512 bytes stored as four 32-row x 16 B core matrices; row r's four bytes of
// one tile live at (r % 32) * 16 + (r / 32) * 4 within it.
constexpr int kScaleCoreBytes = 16;
constexpr int kScaleTileBytes = 512;
constexpr float kE2m1Max = 6.0f;
static_assert(kRows == kComputeWarps * 32 / kLanesPerRow);
static_assert(kBlock % (kLanesPerRow * kFp4Block) == 0);

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

// amax / 6 rounded to the stored UE4M3 scale byte, matching
// (blocks.abs().amax(-1) / 6).to(torch.float8_e4m3fn): round-to-nearest.
__device__ __forceinline__ uint8_t e4m3_scale_byte(float amax) {
    return __nv_cvt_float_to_fp8(__fdiv_rn(amax, kE2m1Max),
                                 __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float e4m3_value(uint8_t bits) {
    return __half2float(__ushort_as_half(
        __nv_cvt_fp8_to_halfraw(bits, __NV_E4M3).x));
}

// Pack two normalized floats into one E2M1 byte.  PyTorch keeps the even
// element in the low nibble; the CUDA header's float2 helper maps x.x to the
// low nibble of cvt.rn.satfinite.e2m1x2.f32 (verified bit-exactly).
__device__ __forceinline__ uint8_t e2m1x2_rn(float even, float odd) {
    return static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(
        make_float2(even, odd), __NV_E2M1, cudaRoundNearest));
}

// The reference packs values with torch.bucketize over the conversion
// midpoints, which maps an exact midpoint to the smaller-magnitude code.
// Hardware cvt.rn rounds ties to even, so exactly 0.75, 1.75, and 3.5 (and
// their negatives) convert one code too high; detect them and step back.
__device__ __forceinline__ bool e2m1_tie_down(float v) {
    const float a = fabsf(v);
    return a == 0.75f || a == 1.75f || a == 3.5f;
}

__global__ __launch_bounds__(kThreads)
void bf16tonvfp4_quant_kernel(const __grid_constant__ CUtensorMap input_map,
                              const __grid_constant__ CUtensorMap output_map,
                              uint8_t* __restrict__ scales,
                              int scale_col_tiles, int k_groups, int tiles,
                              int stages) {
    // Stage barriers: input ready/free, output ready/free. Loader and storer
    // each own a TMA warp; four other warps quantize 32 rows per tile.
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
                         "r"((tile % k_groups) * kRowPackedBytes),
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
        const int lane = tid & (kLanesPerRow - 1);
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
            const int segment = tile % k_groups;
            const int row_tile = tile / k_groups;
            // tokens is a multiple of 128 and tiles cover 32 rows, so every
            // row is in bounds; scale_quarter is the 32-row quarter index
            // inside the 128-row scale group and row_tile >> 2 the group.
            const int scale_quarter = row_tile & 3;
            const size_t scale_row_group =
                static_cast<size_t>(row_tile >> 2) * scale_col_tiles;
#pragma unroll
            for (int part = 0; part < kBlocksPerStore; ++part) {
                const int sub_iteration = iteration * kBlocksPerStore + part;
                const int input_stage = sub_iteration % stages;
                // This part's 128-column segment index, mirroring the loader.
                const int k_tile = segment * kBlocksPerStore + part;
                wait_barrier(ready0 + input_stage * sizeof(uint64_t),
                             (sub_iteration / stages) & 1);
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
                // Each lane owns two 16-value NVFP4 blocks: amax -> UE4M3
                // scale, then normalize in fp32 and pack E2M1 pairs.
                uint8_t scale_bytes[kLaneBlocks];
                float block_scale[kLaneBlocks];
#pragma unroll
                for (int b = 0; b < kLaneBlocks; ++b) {
                    __nv_bfloat162 amax2 = __habs2(pairs[b * (kFp4Block / 2)]);
#pragma unroll
                    for (int i = 1; i < kFp4Block / 2; ++i) {
                        amax2 = __hmax2(amax2, __habs2(pairs[b * (kFp4Block / 2) + i]));
                    }
                    const float amax = fmaxf(
                        fabsf(__bfloat162float(amax2.x)),
                        fabsf(__bfloat162float(amax2.y)));
                    scale_bytes[b] = e4m3_scale_byte(amax);
                    block_scale[b] = e4m3_value(scale_bytes[b]);
                    if (block_scale[b] == 0.0f) {
                        block_scale[b] = 1.0f;
                    }
                }
                // Even lanes of a row group store one 4-byte group of the
                // blocked layout; the odd partner's bytes arrive by shuffle.
                const uint32_t my_bytes =
                    static_cast<uint32_t>(scale_bytes[0]) |
                    (static_cast<uint32_t>(scale_bytes[1]) << 8);
                const uint32_t partner_bytes =
                    __shfl_down_sync(0xffffffffu, my_bytes, 1, kLanesPerRow);
                if ((tid & 1) == 0) {
                    // Group (lane >> 1) covers the four blocks owned by this
                    // lane pair; it is valid iff its 64-column span is inside
                    // D (D is a multiple of 64).
                    const int col_tile = k_tile * (kBlock / 64) + (lane >> 1);
                    if (col_tile < scale_col_tiles) {
                        uint32_t* dst = reinterpret_cast<uint32_t*>(
                            scales +
                            ((scale_row_group + col_tile) * kScaleTileBytes) +
                            row_in_tile * kScaleCoreBytes +
                            scale_quarter * (kScaleCoreBytes / 4));
                        *dst = my_bytes | (partner_bytes << 16);
                    }
                }
                uint2 packed[kLaneBlocks];
#pragma unroll
                for (int b = 0; b < kLaneBlocks; ++b) {
                    uint8_t bytes[kPackedBytesPerLane / kLaneBlocks];
#pragma unroll
                    for (int i = 0; i < kFp4Block / 2; ++i) {
                        const float2 f = __bfloat1622float2(pairs[b * (kFp4Block / 2) + i]);
                        // +0.0 canonicalizes -0 to the positive code the
                        // bucketize-based reference produces.
                        const float lo = __fdiv_rn(f.x, block_scale[b]) + 0.0f;
                        const float hi = __fdiv_rn(f.y, block_scale[b]) + 0.0f;
                        uint8_t byte = e2m1x2_rn(lo, hi);
                        // A tie value always converted to the even code
                        // {2, 4, 6}, so stepping back one code is exact.
                        byte -= static_cast<uint8_t>(
                            (e2m1_tie_down(lo) ? 0x1u : 0u) +
                            (e2m1_tie_down(hi) ? 0x10u : 0u));
                        bytes[i] = byte;
                    }
                    packed[b] = reinterpret_cast<const uint2&>(bytes);
                }
                uint2* dst = reinterpret_cast<uint2*>(shared +
                    output_stage * kStageBytes + kInputBytes +
                    row_in_tile * kRowPackedBytes +
                    part * (kBlock / 2) + lane * kPackedBytesPerLane);
#pragma unroll
                for (int b = 0; b < kLaneBlocks; ++b) {
                    dst[b] = packed[b];
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

} // namespace sm110a::bf16tonvfp4_quant

using namespace sm110a::bf16tonvfp4_quant;

void launch_bf16tonvfp4_quant(const __nv_bfloat16* input, uint8_t* packed,
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
    const uint64_t output_dimensions[2] = {static_cast<uint64_t>(d / 2),
                                            static_cast<uint64_t>(tokens)};
    const uint64_t output_strides[1] = {static_cast<uint64_t>(d / 2)};
    const uint32_t output_box[2] = {kRowPackedBytes, kRows};
    const CUresult output_result = cuTensorMapEncodeTiled(
        &output_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        packed, output_dimensions, output_strides, output_box,
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
        cudaFuncSetAttribute(bf16tonvfp4_quant_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             max_shared);
        shared_configured = true;
    }
    const int k_groups = (d + kBlock * kBlocksPerStore - 1) /
        (kBlock * kBlocksPerStore);
    const int tiles = ((tokens + kRows - 1) / kRows) * k_groups;
    const int ctas = std::min(tiles, sms);
    const int max_stages = max_shared /
        (kStageBytes + kBarrierCount * sizeof(uint64_t));
    // Beyond ~4 stages the deeper input prefetch does not add bandwidth;
    // read/write turnaround on the memory side is the binding limit.
    const int stages = std::min({max_stages, 4, (tiles + ctas - 1) / ctas});
    const int shared_bytes = stages *
        (kStageBytes + kBarrierCount * sizeof(uint64_t));
    bf16tonvfp4_quant_kernel<<<ctas, kThreads, shared_bytes, stream>>>(
        input_map, output_map, scales, d / 64, k_groups, tiles, stages);
}

extern "C" void sm110a_bf16tonvfp4_quant_launch(
    const __nv_bfloat16* input, uint8_t* packed, uint8_t* scales,
    int tokens, int d, cudaStream_t stream) {
    launch_bf16tonvfp4_quant(input, packed, scales, tokens, d, stream);
}
