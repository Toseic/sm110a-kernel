#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

struct Sm110aNcclLsaState;

std::size_t sm110a_nccl_lsa_unique_id_size();
void sm110a_nccl_lsa_get_unique_id(void* bytes, std::size_t size);

Sm110aNcclLsaState* sm110a_nccl_lsa_create(
    const void* unique_id,
    std::size_t unique_id_size,
    int rank,
    int world_size,
    int device,
    std::size_t capacity);

void sm110a_nccl_lsa_destroy(Sm110aNcclLsaState* state) noexcept;

void sm110a_nccl_lsa_all_reduce(
    Sm110aNcclLsaState* state,
    const float* input,
    float* output,
    std::size_t count,
    cudaStream_t stream);
