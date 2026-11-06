#pragma once
#include <cstdint>
namespace sm110a::compressor_pool {
struct Params {
    const void *kv, *score;
    const float *ape;
    float *kv_state, *score_state;
    void *output;
    int64_t batch, sequence, d, start, input_batch_stride, input_row_stride;
};
} // namespace sm110a::compressor_pool
