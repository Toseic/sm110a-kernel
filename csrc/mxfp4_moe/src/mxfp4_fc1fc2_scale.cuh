#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Exact positive power of two used by the E8M0 quantizer. Include the two
// subnormal cases so this matches ldexpf(1, 127-raw) for every uint8 value.
__device__ __forceinline__ float reciprocal_e8m0(
    uint8_t raw) {
    const uint32_t bits =
        raw < 254 ? static_cast<uint32_t>(254 - raw) << 23 : 0x00400000u >> (raw - 254);
    return __uint_as_float(bits);
}

// ceil(log2(amax/448)) without a general FP32 division. At exponent 8,
// the first float above the threshold still divides to 2^-127 after RN.
__device__ __forceinline__ uint8_t encode_activation_scale(
    float amax) {
    const uint32_t bits = __float_as_uint(amax);
    const int exponent = (bits >> 23) & 255;
    const uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 255)
        return fraction ? 255 : 254;
    const int scale = exponent - 8 + (fraction > (0x600000u + (exponent == 8)));
    return static_cast<uint8_t>(scale < 0 ? 0 : scale);
}
