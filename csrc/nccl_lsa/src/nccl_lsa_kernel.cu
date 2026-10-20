#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#include "nccl_lsa.h"

namespace sm110a::nccl_lsa {

void check_cuda(
    cudaError_t status,
    const char *expression) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(expression) + " failed: " + cudaGetErrorString(status));
    }
}

void check_nccl(
    ncclResult_t status,
    const char *expression) {
    if (status != ncclSuccess) {
        throw std::runtime_error(
            std::string(expression) + " failed: " + ncclGetErrorString(status));
    }
}

__global__ void nccl_lsa_all_reduce_kernel(
    ncclDevComm dev_comm,
    ncclWindow_t window,
    std::size_t count) {
    ncclLsaBarrierSession<ncclCoopCta> barrier{ncclCoopCta(),
        dev_comm,
        ncclTeamTagLsa(),
        blockIdx.x};
    barrier.sync(ncclCoopCta(), cuda::memory_order_acquire);

    const int rank = dev_comm.lsaRank;
    const int world_size = dev_comm.lsaSize;
    const std::size_t global_thread = threadIdx.x + blockDim.x * (rank + blockIdx.x * world_size);
    const std::size_t global_threads =
        static_cast<std::size_t>(blockDim.x) * gridDim.x * world_size;

    for (std::size_t index = global_thread; index < count; index += global_threads) {
        float sum = 0.0f;
        for (int peer = 0; peer < world_size; ++peer) {
            const auto *peer_data = static_cast<const float *>(ncclGetLsaPointer(window, 0, peer));
            sum += peer_data[index];
        }
        for (int peer = 0; peer < world_size; ++peer) {
            auto *peer_data = static_cast<float *>(ncclGetLsaPointer(window, 0, peer));
            peer_data[index] = sum;
        }
    }

    barrier.sync(ncclCoopCta(), cuda::memory_order_release);
}

} // namespace sm110a::nccl_lsa

using namespace sm110a::nccl_lsa;

struct Sm110aNcclLsaState {
    int device = 0;
    int rank = 0;
    int world_size = 0;
    std::size_t capacity = 0;
    ncclComm_t comm = nullptr;
    float *buffer = nullptr;
    ncclWindow_t window = nullptr;
    ncclDevComm dev_comm{};
    bool dev_comm_created = false;
};

std::size_t sm110a_nccl_lsa_unique_id_size() { return sizeof(ncclUniqueId); }

void sm110a_nccl_lsa_get_unique_id(
    void *bytes,
    std::size_t size) {
    if (bytes == nullptr || size != sizeof(ncclUniqueId)) {
        throw std::invalid_argument("NCCL unique ID buffer has the wrong size");
    }
    ncclUniqueId id{};
    check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    std::memcpy(bytes, &id, sizeof(id));
}

Sm110aNcclLsaState *sm110a_nccl_lsa_create(
    const void *unique_id,
    std::size_t unique_id_size,
    int rank,
    int world_size,
    int device,
    std::size_t capacity) {
    if (unique_id == nullptr || unique_id_size != sizeof(ncclUniqueId)) {
        throw std::invalid_argument("NCCL unique ID has the wrong size");
    }
    if (world_size <= 0 || rank < 0 || rank >= world_size) {
        throw std::invalid_argument("invalid NCCL rank or world size");
    }
    if (capacity == 0) {
        throw std::invalid_argument("capacity must be positive");
    }

    auto *state = new Sm110aNcclLsaState;
    state->device = device;
    state->rank = rank;
    state->world_size = world_size;
    state->capacity = capacity;

    try {
        check_cuda(cudaSetDevice(device), "cudaSetDevice");

        ncclUniqueId id{};
        std::memcpy(&id, unique_id, sizeof(id));
        check_nccl(ncclCommInitRank(&state->comm, world_size, id, rank), "ncclCommInitRank");
        check_nccl(
            ncclMemAlloc(reinterpret_cast<void **>(&state->buffer), capacity * sizeof(float)),
            "ncclMemAlloc");
        check_nccl(ncclCommWindowRegister(state->comm,
                       state->buffer,
                       capacity * sizeof(float),
                       &state->window,
                       NCCL_WIN_COLL_SYMMETRIC),
            "ncclCommWindowRegister");

        ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        requirements.lsaBarrierCount = 1;
        check_nccl(ncclDevCommCreate(state->comm, &requirements, &state->dev_comm),
            "ncclDevCommCreate");
        state->dev_comm_created = true;

        if (state->dev_comm.lsaSize != world_size) {
            throw std::runtime_error("all ranks must be in one NCCL load/store-accessible team");
        }
        return state;
    } catch (...) {
        sm110a_nccl_lsa_destroy(state);
        throw;
    }
}

void sm110a_nccl_lsa_destroy(
    Sm110aNcclLsaState *state) noexcept {
    if (state == nullptr) {
        return;
    }
    cudaSetDevice(state->device);
    cudaDeviceSynchronize();
    if (state->dev_comm_created) {
        ncclDevCommDestroy(state->comm, &state->dev_comm);
    }
    if (state->window != nullptr) {
        ncclCommWindowDeregister(state->comm, state->window);
    }
    if (state->buffer != nullptr) {
        ncclMemFree(state->buffer);
    }
    if (state->comm != nullptr) {
        ncclCommDestroy(state->comm);
    }
    delete state;
}

void sm110a_nccl_lsa_all_reduce(
    Sm110aNcclLsaState *state,
    const float *input,
    float *output,
    std::size_t count,
    cudaStream_t stream) {
    if (state == nullptr) {
        throw std::invalid_argument("NCCL LSA communicator is closed");
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("input and output pointers must be non-null");
    }
    if (count == 0 || count > state->capacity) {
        throw std::invalid_argument("count must be in [1, capacity]");
    }

    check_cuda(cudaSetDevice(state->device), "cudaSetDevice");
    check_cuda(cudaMemcpyAsync(state->buffer,
                   input,
                   count * sizeof(float),
                   cudaMemcpyDeviceToDevice,
                   stream),
        "cudaMemcpyAsync(input)");
    nccl_lsa_all_reduce_kernel<<<1, 256, 0, stream>>>(state->dev_comm, state->window, count);
    check_cuda(cudaGetLastError(), "nccl_lsa_all_reduce_kernel");
    check_cuda(cudaMemcpyAsync(output,
                   state->buffer,
                   count * sizeof(float),
                   cudaMemcpyDeviceToDevice,
                   stream),
        "cudaMemcpyAsync(output)");
}
