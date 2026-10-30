#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#ifndef SM110A_PDL
#define SM110A_PDL 0
#endif

namespace sm110a {
// Call from every thread, before any read of a predecessor's output.
// Triggering permits launch; it does not publish data. The consumer must wait.
template <bool Enabled = SM110A_PDL>
__device__ __forceinline__ void pdl_trigger(
    bool issuer = threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    if constexpr (Enabled) {
        if (issuer)
            cudaTriggerProgrammaticLaunchCompletion();
    }
}

template <bool Enabled = SM110A_PDL>
__device__ __forceinline__ void pdl_wait() {
    if constexpr (Enabled) cudaGridDependencySynchronize();
}

// Conservative bidirectional support. Kernels with independent setup can
// separate trigger and wait, moving the wait to the first dependent load.
template <bool Enabled = SM110A_PDL>
__device__ __forceinline__ void pdl_entry() {
    pdl_trigger<Enabled>();
    pdl_wait<Enabled>();
}

// Preserve cluster and other attributes, and never retain pointers to locals
// in the caller's config. The serialization attribute belongs to the consumer.
template <bool Enabled, typename Kernel, typename... Args>
inline void launch_pdl(const cudaLaunchConfig_t* config, Kernel kernel, Args... args) {
    cudaLaunchConfig_t launch = *config;
    std::vector<cudaLaunchAttribute> attributes;
    if constexpr (Enabled) {
        if (config->numAttrs)
            attributes.assign(config->attrs, config->attrs + config->numAttrs);
        bool found = false;
        for (auto& attribute : attributes) {
            if (attribute.id == cudaLaunchAttributeProgrammaticStreamSerialization) {
                attribute.val.programmaticStreamSerializationAllowed = 1;
                found = true;
            }
        }
        if (!found) {
            cudaLaunchAttribute attribute{};
            attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute.val.programmaticStreamSerializationAllowed = 1;
            attributes.push_back(attribute);
        }
        launch.attrs = attributes.data();
        launch.numAttrs = static_cast<unsigned int>(attributes.size());
    }
    const auto status = cudaLaunchKernelEx(&launch, kernel, args...);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// Ordinary launch syntax when disabled; extended launch only for PDL variants.
template <bool Enabled = SM110A_PDL, typename Kernel, typename... Args>
inline void launch_kernel(Kernel kernel, dim3 grid, dim3 block, size_t smem,
                          cudaStream_t stream, Args... args) {
    if constexpr (Enabled) {
        cudaLaunchConfig_t config{};
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = smem;
        config.stream = stream;
        launch_pdl<true>(&config, kernel, args...);
    } else {
        kernel<<<grid, block, smem, stream>>>(args...);
    }
}
}  // namespace sm110a
