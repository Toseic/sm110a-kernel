#pragma once

#include "common.h"

// A/B contain two packed E2M1 values per byte. A is the unpermuted token
// matrix [tokens, K/2], while C is the packed expert-grouped FC1 output
// [M, N/2]. SFC is its blocked UE4M3 scale tensor with one scale per 16
// logical output elements.
// source_rows maps each grouped output row back to its source token row.
void launch_nvfp4_fc1_main(
    const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
    const uint8_t* sfb, const int32_t* source_rows, const int32_t* tasks,
    int32_t* state, uint8_t* c, uint8_t* sfc, int tokens, int m, int experts, int n,
    int k, int max_tasks, int max_active_clusters, cudaStream_t stream);
