#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

namespace sm110a {
// Keep the launch attribute local: it applies to the dependent kernel only.
template <bool Enabled, typename Kernel, typename... Args>
inline void launch_pdl(cudaLaunchConfig_t* config, Kernel kernel, Args... args) {
    cudaLaunchAttribute attribute{};
    if constexpr (Enabled) {
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        config->attrs = &attribute;
        config->numAttrs = 1;
    }
    const auto status = cudaLaunchKernelEx(config, kernel, args...);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace sm110a
