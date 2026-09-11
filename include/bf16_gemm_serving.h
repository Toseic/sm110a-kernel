#pragma once

#include "common.h"
#include "bf16_gemm_serving_queue.h"

void launch_bf16_gemm_serving(const __nv_bfloat16* d_a, const __nv_bfloat16* d_b,
                      __nv_bfloat16* d_c, int m, int n, int k,
                      bool swap_ab, cudaStream_t stream,
                      int max_active_clusters = 0);

void launch_bf16_gemm_serving_profile(const __nv_bfloat16* d_a, const __nv_bfloat16* d_b,
                              __nv_bfloat16* d_c, int m, int n, int k,
                              int64_t* profiler, int num_entries,
                              bool swap_ab, cudaStream_t stream,
                              int max_active_clusters = 0);

cudaError_t initialize_bf16_gemm_serving_queue(
    sm110a::bf16_gemm_serving::ServingQueueControl* control,
    size_t control_bytes,
    uint32_t slot_count,
    uint32_t n,
    uint32_t k,
    uint32_t launched_clusters,
    cudaStream_t stream);

cudaError_t launch_bf16_gemm_persistent_serving(
    const __nv_bfloat16* a_ring,
    const __nv_bfloat16* b_nt,
    __nv_bfloat16* c_ring,
    sm110a::bf16_gemm_serving::ServingQueueControl* control,
    uint32_t slot_count,
    uint32_t n,
    uint32_t k,
    uint32_t launched_clusters,
    cudaStream_t stream);

cudaError_t request_bf16_gemm_serving_drain(
    sm110a::bf16_gemm_serving::ServingQueueControl* control,
    cudaStream_t stream);

cudaError_t launch_bf16_gemm_serving_test_producer(
    const __nv_bfloat16* rows,
    __nv_bfloat16* a_ring,
    sm110a::bf16_gemm_serving::ServingQueueControl* control,
    uint64_t group_ticket,
    uint32_t valid_rows,
    bool flush,
    uint64_t delay_cycles,
    bool bypass_payload,
    cudaStream_t stream);

cudaError_t launch_bf16_gemm_serving_test_consumer(
    const __nv_bfloat16* c_ring,
    __nv_bfloat16* rows,
    sm110a::bf16_gemm_serving::ServingQueueControl* control,
    uint64_t group_ticket,
    uint32_t expected_valid_rows,
    bool bypass_payload,
    cudaStream_t stream);
