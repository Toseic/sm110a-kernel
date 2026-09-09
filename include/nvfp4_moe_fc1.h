#pragma once

#include "common.h"

// A/B contain two packed E2M1 values per byte. A is the unpermuted token
// matrix [tokens, K/2], while C is the expert-grouped FC1 output [M, N].
// source_rows maps each grouped output row back to its source token row.
void launch_nvfp4_fc1_main(
    const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
    const uint8_t* sfb, const int32_t* source_rows, const int32_t* tasks,
    int32_t* state, __nv_bfloat16* c, int tokens, int m, int experts, int n,
    int k, int max_tasks, int max_active_clusters, cudaStream_t stream);
