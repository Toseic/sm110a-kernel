#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t status__ = (expr);                                             \
    if (status__ != cudaSuccess) {                                             \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "    \
                << cudaGetErrorString(status__) << std::endl;                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(expr)                                                     \
  do {                                                                         \
    cublasStatus_t status__ = (expr);                                          \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                   \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__           \
                << ": status=" << static_cast<int>(status__) << std::endl;     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

inline double tflops_from_ms(int m, int n, int k, float elapsed_ms) {
  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                       static_cast<double>(k);
  return flops / (elapsed_ms * 1.0e-3) / 1.0e12;
}

inline float abs_diff(float a, float b) { return std::fabs(a - b); }
