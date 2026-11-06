#pragma once
#include <cstdint>
namespace sm110a::compressor_store {
struct Params {
    const void *input, *frequencies;
    const float *weight;
    void *cache_values, *cache_scales, *cache_rope;
    void *packed_values, *packed_scales, *packed_rope;
    int64_t rows, sequence, capacity, position, frequency_stride;
    float eps;
    bool conjugate;
};
} // namespace sm110a::compressor_store
