#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bf16_moe_finalize.h"

namespace sm110a::bf16_moe_finalize {

constexpr int kFinalizeThreads = 256;

__global__ void finalize_kernel(
    const __nv_bfloat16* grouped_fc2,
    const int32_t* assignment_to_grouped,
    const __nv_bfloat16* assignment_weights,
    __nv_bfloat16* out,
    int tokens,
    int topk,
    int h) {
    const int token = blockIdx.x;
    if (token >= tokens)
        return;
    for (int column = threadIdx.x; column < h; column += blockDim.x) {
        float sum = 0.0f;
        for (int rank = 0; rank < topk; ++rank) {
            const int grouped_row = assignment_to_grouped[token * topk + rank];
            const float weight = __bfloat162float(assignment_weights[token * topk + rank]);
            sum += weight * __bfloat162float(
                grouped_fc2[static_cast<size_t>(grouped_row) * h + column]);
        }
        out[static_cast<size_t>(token) * h + column] = __float2bfloat16_rn(sum);
    }
}

} // namespace sm110a::bf16_moe_finalize

using namespace sm110a::bf16_moe_finalize;

void launch_moe_finalize(const __nv_bfloat16* grouped_fc2,
                         const int32_t* assignment_to_grouped,
                         const __nv_bfloat16* assignment_weights,
                         __nv_bfloat16* out,
                         int tokens, int topk, int h,
                         cudaStream_t stream) {
    if (!grouped_fc2 || !assignment_to_grouped || !assignment_weights || !out ||
        tokens <= 0 || topk <= 0 || h <= 0)
        return;
    finalize_kernel<<<tokens, kFinalizeThreads, 0, stream>>>(
        grouped_fc2, assignment_to_grouped, assignment_weights, out, tokens, topk, h);
}
