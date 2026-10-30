#pragma once
#include <cstdint>

namespace sm110a::rope {
struct Layout {
    int64_t pairs, sequence, heads;
    int64_t batch_stride, sequence_stride, head_stride;
    int64_t frequency_stride;
};
} // namespace sm110a::rope
