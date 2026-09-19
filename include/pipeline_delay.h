// Device-only cycle busy-wait primitive for software-pipeline experiments.
// Kernel-specific point names and defaults belong in local *_pipeline_delay.cuh
// files; this public header intentionally has no pipeline-topology knowledge.
#pragma once

#include <stdint.h>

namespace sm110a {

// The observed delay is at least Cycles. clock64() and loop bookkeeping add a
// small architecture-dependent overshoot, so compare against a zero baseline.
template <uint64_t Cycles>
__device__ __forceinline__ void pipeline_delay() {
  if constexpr (Cycles != 0) {
    const uint64_t start = clock64();
    while (clock64() - start < Cycles) {
      asm volatile("" ::: "memory");
    }
  }
}

}  // namespace sm110a
