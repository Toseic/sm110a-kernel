#pragma once

#include "common.h"

// H=6144, I=2048 W4A8 fusion: FP8 A, block-interleaved packed E2M1 W1,
// padded blocked E8M0 scales, FP8/E8M0 intermediate, BF16 output. Grouped
// input is single-expert M%(128*cluster_m)==0. Routed metadata uses FC1
// tile_n=256 and the matching cluster_m. Resident per-row h_split variants
// exchange a reusable global image; additional rows execute in waves.
// Routing weights are BF16 for FC2 weighting, FP32 for FC1 weighting.
// COMPACT_WEIGHT_SCALES=1 variants consume SFW1/SFW2 packed by taking byte
// 0 of each padded four-byte group (one quarter of the original storage).
// SFA and intermediate scale formats stay padded. The JIT selects the ABI's
// weight scale layout at compile time; callers must pass the matching format.
cudaError_t launch_mxfp4_fc1fc2(const uint8_t *a,
    const uint8_t *w1,
    const uint8_t *sfa,
    const uint8_t *sfw1,
    const uint8_t *w2,
    const uint8_t *sfw2,
    const int32_t *source_rows,
    const int32_t *grouped_assignments,
    const int32_t *tasks,
    const int32_t *state,
    const void *routing_weights,
    __nv_bfloat16 *output,
    uint8_t *workspace,
    float swiglu_limit,
    int tokens,
    int m,
    int experts,
    int topk,
    int row_task_count,
    int max_tasks,
    cudaStream_t stream);
