#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "bf16tofp4_quant.h"

namespace {

constexpr int kRows = 32;
constexpr int kBlock = 128;
constexpr int kQuantBlock = 32;
constexpr int kBlocksPerStore = 2;
constexpr int kComputeWarps = 4;
constexpr int kLoadWarp = 4;
constexpr int kStoreWarp = 5;
constexpr int kThreads = 32 * (kStoreWarp + 1);
constexpr int kLanesPerRow = 4;
constexpr int kValuesPerLane = kBlock / kLanesPerRow;
constexpr int kInputBytes = kRows * kBlock * sizeof(__nv_bfloat16);
constexpr int kQuantOutBytes = kRows * kBlock * kBlocksPerStore / 2;
constexpr int kDequantOutBytes =
    kRows * kBlock * kBlocksPerStore * sizeof(__nv_bfloat16);
constexpr int kBarrierCount = 4;
constexpr float kFp4Max = 6.0f;
// Blocks whose amax falls below 6 * 2^-126 clamp there so the scale exponent
// never underflows the E8M0 range; identical to the TileLang reference.
constexpr float kFp4AmaxFloor = kFp4Max * 1.17549435082228751e-38f;
static_assert(kRows == kComputeWarps * 32 / kLanesPerRow);
static_assert(kValuesPerLane == kQuantBlock,
              "one lane owns exactly one E2M1 quantization block per stage");

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

// E8M0 scales are powers of two, so the bf16 multiply by the exact reciprocal
// is lossless and each bf16x2 pair converts with one hardware instruction.
// The PTX semantics place the lower bf16 (element 2i) in the low nibble and
// the upper bf16 (element 2i + 1) in the high nibble, matching the PyTorch
// float4_e2m1fn_x2 packing.
__device__ __forceinline__ uint32_t fp4x4_bf16(uint32_t w0, uint32_t w1,
                                               uint32_t w2, uint32_t w3) {
    uint32_t packed;
    asm("{.reg .b8 b0, b1, b2, b3;\n\t"
        "cvt.rn.satfinite.e2m1x2.bf16x2 b0, %1;\n\t"
        "cvt.rn.satfinite.e2m1x2.bf16x2 b1, %2;\n\t"
        "cvt.rn.satfinite.e2m1x2.bf16x2 b2, %3;\n\t"
        "cvt.rn.satfinite.e2m1x2.bf16x2 b3, %4;\n\t"
        "mov.b32 %0, {b0, b1, b2, b3};\n\t}"
        : "=r"(packed) : "r"(w0), "r"(w1), "r"(w2), "r"(w3));
    return packed;
}

// Decode four packed E2M1x2 bytes into four exact f16x2 pairs for the fused
// quantize+dequantize path.
__device__ __forceinline__ void fp4x4_to_f16x2(uint32_t packed, uint32_t (&f16x2)[4]) {
    asm("{.reg .b8 b0, b1, b2, b3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
        : "=r"(f16x2[0]), "=r"(f16x2[1]), "=r"(f16x2[2]), "=r"(f16x2[3])
        : "r"(packed));
}

// 2^ceil(log2(amax / 6)) via IEEE 754 bit tricks (fast_round_scale).
__device__ __forceinline__ int round_scale_byte(float amax) {
    const int bits = __float_as_int(amax * (1.0f / kFp4Max));
    const int log2_ceil = ((bits >> 23) & 0xFF) - 127 +
        ((bits & 0x7FFFFF) != 0 ? 1 : 0);
    return min(max(log2_ceil + 127, 0), 255);
}

// Streaming access: each byte is touched once, so loads are marked
// evict_first and stores evict_unchanged to keep the lines from displacing
// useful L2 state; measured ~1-1.5% faster than default policy on the
// sibling FP8 quantizer.
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


// Fallback for the trailing tokens % kRows rows (at most 31): a plain
// elementwise kernel with the same per-32-block math as the TMA path, so the
// bulk TMA launch never sees row-clipped tensor boxes. One warp per row;
// each thread owns the 32-value blocks threadIdx.x, +32, +64, ...
template <bool kDequant>
__global__ __launch_bounds__(128)
void bf16tofp4_tail_kernel(const __nv_bfloat16* __restrict__ input,
                           uint8_t* __restrict__ output,
                           uint8_t* __restrict__ scales,
                           int tokens_base, int tail_rows, int d) {
    const int row = tokens_base + blockIdx.x * (128 / 32) + threadIdx.x / 32;
    if (row >= tokens_base + tail_rows) {
        return;
    }
    const int tid_in_row = threadIdx.x % 32;
    const int blocks = d / kQuantBlock;
    for (int block_idx = tid_in_row; block_idx < blocks; block_idx += 32) {
        const __nv_bfloat16* src = input +
            static_cast<size_t>(row) * d + block_idx * kQuantBlock;
        __nv_bfloat16 values[kQuantBlock];
#pragma unroll
        for (int i = 0; i < kQuantBlock / 8; ++i) {
            reinterpret_cast<uint4*>(values)[i] =
                reinterpret_cast<const uint4*>(src)[i];
        }
        const __nv_bfloat162* pairs =
            reinterpret_cast<const __nv_bfloat162*>(values);
        __nv_bfloat162 amax2 = __habs2(pairs[0]);
#pragma unroll
        for (int i = 1; i < kQuantBlock / 2; ++i) {
            amax2 = __hmax2(amax2, __habs2(pairs[i]));
        }
        float amax = fmaxf(fabsf(__bfloat162float(amax2.x)),
                           fabsf(__bfloat162float(amax2.y)));
        amax = fmaxf(amax, kFp4AmaxFloor);
        const int scale_byte = round_scale_byte(amax);
        scales[static_cast<size_t>(row) * blocks + block_idx] =
            static_cast<uint8_t>(scale_byte);
        const float reciprocal = ldexpf(1.0f, 127 - scale_byte);
        const __nv_bfloat162 rscale2 =
            __float22bfloat162_rn(make_float2(reciprocal, reciprocal));
        __nv_bfloat162 scaled[kQuantBlock / 2];
#pragma unroll
        for (int i = 0; i < kQuantBlock / 2; ++i) {
            scaled[i] = __hmul2(pairs[i], rscale2);
        }
        const uint32_t* words = reinterpret_cast<const uint32_t*>(scaled);
        uint4 packed_vec[kQuantBlock / 16];
        uint32_t* packed_words = reinterpret_cast<uint32_t*>(packed_vec);
#pragma unroll
        for (int i = 0; i < kQuantBlock / 8; ++i) {
            packed_words[i] = fp4x4_bf16(words[4 * i], words[4 * i + 1],
                                         words[4 * i + 2], words[4 * i + 3]);
        }
        if (kDequant) {
            const float scale_f = ldexpf(1.0f, scale_byte - 127);
            __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(output) +
                static_cast<size_t>(row) * d + block_idx * kQuantBlock;
#pragma unroll
            for (int i = 0; i < kQuantBlock / 8; ++i) {
                uint32_t f16x2[4];
                fp4x4_to_f16x2(packed_words[i], f16x2);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 decoded =
                        __half22float2(reinterpret_cast<__half2&>(f16x2[j]));
                    reinterpret_cast<__nv_bfloat162*>(dst)[4 * i + j] =
                        __float22bfloat162_rn(make_float2(
                            decoded.x * scale_f, decoded.y * scale_f));
                }
            }
        } else {
            *reinterpret_cast<uint4*>(
                output + static_cast<size_t>(row) * (d / 2) +
                block_idx * (kQuantBlock / 2)) = packed_vec[0];
        }
    }
}

template <bool kDequant>
__global__ __launch_bounds__(kThreads)
void bf16tofp4_quant_kernel(const __grid_constant__ CUtensorMap input_map,
                            const __grid_constant__ CUtensorMap output_map,
                            uint8_t* __restrict__ scales,
                            int tokens, int d, int tiles, int stages) {
    constexpr int kStageBytes = kDequant ? kInputBytes + kDequantOutBytes
                                         : kInputBytes + kQuantOutBytes;
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

    // Each tile covers kBlocksPerStore kBlock-column load stages, matching
    // the fp8 kernel's k_groups; counting 128-column groups instead would
    // append fully out-of-range tiles whose TMA stores corrupt live output.
    const int col_tiles =
        (d + kBlock * kBlocksPerStore - 1) / (kBlock * kBlocksPerStore);
    const int quant_blocks = d / kQuantBlock;
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
                // The last column tile is shifted back so its 256-column
                // box always ends at d: TMA stores must never straddle the
                // packed row extent (the box is wider than short rows).
                const int tile_col = (tile % col_tiles) * (kBlock * kBlocksPerStore);
                const int tile_x = (tile_col + kBlock * kBlocksPerStore <= d)
                    ? tile_col : d - kBlock * kBlocksPerStore;
                const int part_col = tile_x + part * kBlock;
                const int ready = ready0 + stage * sizeof(uint64_t);
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
                             "_, [%0], %1;" :: "r"(ready), "r"(kInputBytes) : "memory");
                asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global."
                             "mbarrier::complete_tx::bytes.L2::cache_hint "
                             "[%0], [%1, {%2, %3}], [%4], %5;" ::
                             "r"(data0 + stage * kStageBytes), "l"(&input_map),
                             "r"(part_col),
                             "r"((tile / col_tiles) * kRows),
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
            const int tile_col = (tile % col_tiles) * (kBlock * kBlocksPerStore);
            const int tile_x = (tile_col + kBlock * kBlocksPerStore <= d)
                ? tile_col : d - kBlock * kBlocksPerStore;
            asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
                         ".L2::cache_hint "
                         "[%0, {%1, %2}], [%3], %4;" ::
                         "l"(&output_map),
                         "r"(kDequant ? tile_x : tile_x / 2),
                         "r"((tile / col_tiles) * kRows),
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
        // Bytes per row of packed (or dequantized BF16) output per store.
        const int row_out_bytes = kDequant
            ? kBlock * kBlocksPerStore * static_cast<int>(sizeof(__nv_bfloat16))
            : kBlock * kBlocksPerStore / 2;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const int output_stage = iteration % stages;
            if (iteration >= stages) {
                wait_barrier(output_free0 + output_stage * sizeof(uint64_t),
                             ((iteration - stages) / stages) & 1);
            }
            const int tile = blockIdx.x + iteration * gridDim.x;
            const int row = (tile / col_tiles) * kRows + row_in_tile;
#pragma unroll
            for (int part = 0; part < kBlocksPerStore; ++part) {
                const int sub_iteration = iteration * kBlocksPerStore + part;
                const int input_stage = sub_iteration % stages;
                const int tile_col = (tile % col_tiles) * (kBlock * kBlocksPerStore);
                const int tile_x = (tile_col + kBlock * kBlocksPerStore <= d)
                    ? tile_col : d - kBlock * kBlocksPerStore;
                const int part_col = tile_x + part * kBlock;
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
                // Each lane covers exactly one 32-value quantization block,
                // so the amax reduction stays lane-local.
                __nv_bfloat162 amax2 = __habs2(pairs[0]);
#pragma unroll
                for (int i = 1; i < kValuesPerLane / 2; ++i) {
                    amax2 = __hmax2(amax2, __habs2(pairs[i]));
                }
                float amax = fmaxf(
                    fabsf(__bfloat162float(amax2.x)),
                    fabsf(__bfloat162float(amax2.y)));
                amax = fmaxf(amax, kFp4AmaxFloor);
                const int scale_byte = round_scale_byte(amax);
                const int quant_block = tile_x / kQuantBlock +
                    part * (kBlock / kQuantBlock) + lane;
                if (row < tokens && quant_block < quant_blocks) {
                    scales[static_cast<size_t>(row) * quant_blocks + quant_block] =
                        static_cast<uint8_t>(scale_byte);
                }
                const float reciprocal =
                    ldexpf(1.0f, 127 - scale_byte);
                const __nv_bfloat162 rscale2 =
                    __float22bfloat162_rn(make_float2(reciprocal, reciprocal));
                __nv_bfloat162 scaled[kValuesPerLane / 2];
#pragma unroll
                for (int i = 0; i < kValuesPerLane / 2; ++i) {
                    scaled[i] = __hmul2(pairs[i], rscale2);
                }
                const uint32_t* words = reinterpret_cast<const uint32_t*>(scaled);
                uint4 packed_vec[kValuesPerLane / 16];
                uint32_t* packed_words = reinterpret_cast<uint32_t*>(packed_vec);
#pragma unroll
                for (int i = 0; i < kValuesPerLane / 8; ++i) {
                    packed_words[i] = fp4x4_bf16(
                        words[4 * i], words[4 * i + 1],
                        words[4 * i + 2], words[4 * i + 3]);
                }
                if (kDequant) {
                    // Fused quantize+dequantize: decode the E2M1 codes and
                    // rescale, matching TileLang's fp32 dequant multiply.
                    const float scale_f = ldexpf(1.0f, scale_byte - 127);
                    uint4 out[kValuesPerLane / 8];
#pragma unroll
                    for (int i = 0; i < kValuesPerLane / 8; ++i) {
                        uint32_t f16x2[4];
                        fp4x4_to_f16x2(packed_words[i], f16x2);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float2 decoded =
                                __half22float2(reinterpret_cast<__half2&>(f16x2[j]));
                            const __nv_bfloat162 round_tripped =
                                __float22bfloat162_rn(make_float2(
                                    decoded.x * scale_f, decoded.y * scale_f));
                            reinterpret_cast<uint32_t*>(out)[4 * i + j] =
                                reinterpret_cast<const uint32_t&>(round_tripped);
                        }
                    }
                    uint4* dst = reinterpret_cast<uint4*>(shared +
                        output_stage * kStageBytes + kInputBytes +
                        row_in_tile * row_out_bytes + part * kBlock *
                        static_cast<int>(sizeof(__nv_bfloat16)) +
                        lane * kValuesPerLane * static_cast<int>(sizeof(__nv_bfloat16)));
#pragma unroll
                    for (int i = 0; i < kValuesPerLane / 8; ++i) {
                        dst[i] = out[i];
                    }
                } else {
                    uint4* dst = reinterpret_cast<uint4*>(shared +
                        output_stage * kStageBytes + kInputBytes +
                        row_in_tile * row_out_bytes + part * kBlock / 2 +
                        lane * kValuesPerLane / 2);
                    dst[0] = packed_vec[0];
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

template <bool kDequant>
void launch_bf16tofp4_quant_impl(const __nv_bfloat16* input, uint8_t* output,
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
    constexpr int kOutWidth = kDequant ? kBlock * kBlocksPerStore
                                       : kBlock * kBlocksPerStore / 2;
    CUtensorMap output_map;
    const uint64_t output_strides[1] = {
        kDequant ? static_cast<uint64_t>(d) * sizeof(__nv_bfloat16)
                 : static_cast<uint64_t>(d) / 2};
    const uint32_t output_box[2] = {kOutWidth, kRows};
    const CUresult output_result = cuTensorMapEncodeTiled(
        &output_map,
        kDequant ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2, output, dimensions, output_strides, output_box,
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
        cudaFuncSetAttribute(bf16tofp4_quant_kernel<kDequant>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             max_shared);
        shared_configured = true;
    }
    constexpr int kStageBytes = kDequant ? kInputBytes + kDequantOutBytes
                                         : kInputBytes + kQuantOutBytes;
    const int col_tiles =
        (d + kBlock * kBlocksPerStore - 1) / (kBlock * kBlocksPerStore);
    // The bulk TMA launch covers full 32-row groups only, so no tensor box is
    // ever row-clipped, and d >= kBlock * kBlocksPerStore lets the last
    // column tile shift back to a fully in-bounds box (TMA stores must never
    // straddle the packed row extent). Everything else runs on the fallback
    // kernel.
    const int tokens_full = (d >= kBlock * kBlocksPerStore)
        ? (tokens / kRows) * kRows : 0;
    const int tiles = (tokens_full / kRows) * col_tiles;
    if (tiles > 0) {
        const int ctas = std::min(tiles, sms);
        const int max_stages = max_shared /
            (kStageBytes + kBarrierCount * sizeof(uint64_t));
        // Beyond ~4 stages the deeper input prefetch does not add bandwidth;
        // read/write turnaround on the memory side is the binding limit.
        const int stages = std::min({max_stages, 4, (tiles + ctas - 1) / ctas});
        const int shared_bytes = stages *
            (kStageBytes + kBarrierCount * sizeof(uint64_t));
        bf16tofp4_quant_kernel<kDequant><<<ctas, kThreads, shared_bytes, stream>>>(
            input_map, output_map, scales, tokens_full, d, tiles, stages);
    }
    const int tail_rows = tokens - tokens_full;
    if (tail_rows > 0) {
        bf16tofp4_tail_kernel<kDequant><<<(tail_rows + 3) / 4, 128, 0, stream>>>(
            input, output, scales, tokens_full, tail_rows, d);
    }
}

}  // namespace

void launch_bf16tofp4_quant(const __nv_bfloat16* input, uint8_t* output,
                            uint8_t* scales, int tokens, int d,
                            cudaStream_t stream) {
    launch_bf16tofp4_quant_impl<false>(input, output, scales, tokens, d, stream);
}

void launch_bf16tofp4_quant_dequant(const __nv_bfloat16* input,
                                    __nv_bfloat16* output, uint8_t* scales,
                                    int tokens, int d, cudaStream_t stream) {
    launch_bf16tofp4_quant_impl<true>(input, reinterpret_cast<uint8_t*>(output),
                                      scales, tokens, d, stream);
}

extern "C" void sm110a_bf16tofp4_quant_launch(
    const __nv_bfloat16* input, uint8_t* output, uint8_t* scales,
    int tokens, int d, cudaStream_t stream) {
    launch_bf16tofp4_quant(input, output, scales, tokens, d, stream);
}

extern "C" void sm110a_bf16tofp4_quant_dequant_launch(
    const __nv_bfloat16* input, __nv_bfloat16* output, uint8_t* scales,
    int tokens, int d, cudaStream_t stream) {
    launch_bf16tofp4_quant_dequant(input, output, scales, tokens, d, stream);
}
