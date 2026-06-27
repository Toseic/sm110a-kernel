#include "common.h"
#include "bf16_gemm.h"

#include <algorithm>
#include <iomanip>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Options {
  int warmup = 20;
  int iters = 20;
  bool run_check = true;
  bool run_accuracy = true;
  bool single_shape = false;
  int m = 0;
  int n = 0;
  int k = 0;
};

struct Result {
  float naive_ms = 0.0f;
  float cublas_lt_ms = 0.0f;
  float max_abs_diff_lt = 0.0f;
};

constexpr std::size_t kCublasLtWorkspaceBytes = 32ull * 1024ull * 1024ull;
constexpr int kAccuracyM = 512;
constexpr int kAccuracyN = 512;
constexpr int kAccuracyK = 512;
constexpr int kAccuracyWarmup = 2;
constexpr int kAccuracyIters = 5;

float benchmark_bf16_gemm(const __nv_bfloat16* d_a, const __nv_bfloat16* d_b,
                          __nv_bfloat16* d_c, int m, int n, int k, int warmup,
                          int iters) {
  for (int i = 0; i < warmup; ++i) {
    launch_bf16_gemm(d_a, d_b, d_c, m, n, k);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch_bf16_gemm(d_a, d_b, d_c, m, n, k);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return elapsed_ms / static_cast<float>(iters);
}

float benchmark_cublas_lt(cublasLtHandle_t handle, const __nv_bfloat16* d_a,
                          const __nv_bfloat16* d_b, __nv_bfloat16* d_c, int m,
                          int n, int k, int warmup, int iters, void* workspace,
                          std::size_t workspace_size) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  CHECK_CUBLAS(cublasLtMatmulDescCreate(&operation_desc,
                                        CUBLAS_COMPUTE_32F_FAST_16BF,
                                        CUDA_R_32F));
  // Inputs are k-major in memory. The kernel does C(M, N) = A(M, K) * B(K, N)
  // where A and B are both stored with K as the fastest-changing dimension.
  // cuBLASLt is column-major, so:
  //   d_a (K-major (M, K)) is seen as a (K, M) col-major view = A^T.
  //   d_b (K-major (K, N)) is seen as a (K, N) col-major view = B.
  // To produce C (N, M) col-major = C^T, we compute
  //   C^T = op(A_cm) * op(B_cm) = B^T * A^T = (A * B)^T,
  // i.e. transA = T on the (K, N) layout that backs d_b.
  const cublasOperation_t trans_a = CUBLAS_OP_T;
  const cublasOperation_t trans_b = CUBLAS_OP_N;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, k, n, k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, k, m, k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, n, m, n));
  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size,
      sizeof(workspace_size)));

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      handle, operation_desc, a_desc, b_desc, c_desc, c_desc, preference, 1,
      &heuristic, &returned_results));
  if (returned_results == 0) {
    std::cerr << "cublasLtMatmulAlgoGetHeuristic returned no algorithm"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  for (int i = 0; i < warmup; ++i) {
    CHECK_CUBLAS(cublasLtMatmul(
        handle, operation_desc, &alpha, d_b, a_desc, d_a, b_desc, &beta, d_c,
        c_desc, d_c, c_desc, &heuristic.algo, workspace, workspace_size, 0));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CHECK_CUBLAS(cublasLtMatmul(
        handle, operation_desc, &alpha, d_b, a_desc, d_a, b_desc, &beta, d_c,
        c_desc, d_c, c_desc, &heuristic.algo, workspace, workspace_size, 0));
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
  CHECK_CUBLAS(cublasLtMatmulDescDestroy(operation_desc));
  return elapsed_ms / static_cast<float>(iters);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* name) {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << std::endl;
        std::exit(EXIT_FAILURE);
      }
      return std::atoi(argv[++i]);
    };

    if (arg == "--m") {
      options.m = require_value("--m");
    } else if (arg == "--n") {
      options.n = require_value("--n");
    } else if (arg == "--k") {
      options.k = require_value("--k");
    } else if (arg == "--warmup") {
      options.warmup = require_value("--warmup");
    } else if (arg == "--iters") {
      options.iters = require_value("--iters");
    } else if (arg == "--no-accuracy") {
      options.run_accuracy = false;
    } else if (arg == "--skip-check") {
      options.run_check = false;
    } else {
      std::cerr << "unknown argument: " << arg << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  options.single_shape = options.m > 0 || options.n > 0 || options.k > 0;
  if (options.single_shape &&
      (options.m <= 0 || options.n <= 0 || options.k <= 0)) {
    std::cerr << "--m, --n, --k must all be provided for single-shape mode"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return options;
}

std::vector<std::tuple<int, int, int>> default_shapes() {
  return {
      {256, 256, 256},
      {512, 512, 512},
      {1024, 1024, 1024},
      {2048, 2048, 2048},
      {2560, 2048, 2048},
      {4096, 4096, 4096},
      {4096, 11008, 4096},
      {8192, 4096, 4096},
  };
}

void initialize_host(std::vector<__nv_bfloat16>& a,
                     std::vector<__nv_bfloat16>& b) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto& x : a) {
    x = __float2bfloat16(dist(rng));
  }
  for (auto& x : b) {
    x = __float2bfloat16(dist(rng));
  }
}

Result run_case(int m, int n, int k, cublasLtHandle_t lt_handle,
                const Options& options) {
  const std::size_t a_elems = static_cast<std::size_t>(m) * k;
  const std::size_t b_elems = static_cast<std::size_t>(k) * n;
  const std::size_t c_elems = static_cast<std::size_t>(m) * n;

  std::vector<__nv_bfloat16> h_a(a_elems);
  std::vector<__nv_bfloat16> h_b(b_elems);
  std::vector<__nv_bfloat16> h_naive(c_elems);
  initialize_host(h_a, h_b);

  __nv_bfloat16* d_a = nullptr;
  __nv_bfloat16* d_b = nullptr;
  __nv_bfloat16* d_naive = nullptr;
  __nv_bfloat16* d_cublas_lt = nullptr;
  void* lt_workspace = nullptr;

  CHECK_CUDA(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_naive, c_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_cublas_lt, c_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&lt_workspace, kCublasLtWorkspaceBytes));

  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), a_elems * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), b_elems * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  Result result;
  result.naive_ms = benchmark_bf16_gemm(
      d_a, d_b, d_naive, m, n, k, options.warmup, options.iters);
  result.cublas_lt_ms =
      benchmark_cublas_lt(lt_handle, d_a, d_b, d_cublas_lt, m, n, k,
                          options.warmup, options.iters, lt_workspace,
                          kCublasLtWorkspaceBytes);

  if (options.run_check) {
    CHECK_CUDA(cudaMemcpy(h_naive.data(), d_naive,
                          c_elems * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    std::vector<__nv_bfloat16> h_cublas_lt(c_elems);
    CHECK_CUDA(cudaMemcpy(h_cublas_lt.data(), d_cublas_lt,
                          c_elems * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < c_elems; ++i) {
      result.max_abs_diff_lt =
          std::max(result.max_abs_diff_lt,
                   abs_diff(__bfloat162float(h_naive[i]),
                            __bfloat162float(h_cublas_lt[i])));
    }
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_naive));
  CHECK_CUDA(cudaFree(d_cublas_lt));
  CHECK_CUDA(cudaFree(lt_workspace));
  return result;
}

void print_header() {
  std::cout << std::left << std::setw(8) << "M" << std::setw(8) << "N"
            << std::setw(8) << "K" << std::setw(14) << "Naive(ms)"
            << std::setw(14) << "Naive TF/s" << std::setw(16)
            << "cuBLASLt(ms)" << std::setw(16) << "cuBLASLt TF/s"
            << std::setw(14) << "LtSpeedup" << std::setw(14) << "DiffLt"
            << std::endl;
}

void print_result(int m, int n, int k, const Result& result) {
  const double naive_tflops = tflops_from_ms(m, n, k, result.naive_ms);
  const double cublas_lt_tflops = tflops_from_ms(m, n, k, result.cublas_lt_ms);
  const double lt_speedup = result.naive_ms / result.cublas_lt_ms;

  std::cout << std::left << std::setw(8) << m << std::setw(8) << n
            << std::setw(8) << k << std::setw(14) << std::fixed
            << std::setprecision(4) << result.naive_ms << std::setw(14)
            << std::setprecision(3) << naive_tflops << std::setw(16)
            << std::setprecision(4) << result.cublas_lt_ms << std::setw(16)
            << std::setprecision(3) << cublas_lt_tflops << std::setw(14)
            << std::setprecision(2) << lt_speedup << std::setw(14)
            << std::setprecision(6) << result.max_abs_diff_lt << std::endl;
}

void run_accuracy_check(cublasLtHandle_t lt_handle) {
  Options accuracy_options;
  accuracy_options.warmup = kAccuracyWarmup;
  accuracy_options.iters = kAccuracyIters;
  accuracy_options.run_check = true;

  std::cout << "[accuracy] small shape with correctness check" << std::endl;
  print_header();
  const Result result =
      run_case(kAccuracyM, kAccuracyN, kAccuracyK, lt_handle, accuracy_options);
  print_result(kAccuracyM, kAccuracyN, kAccuracyK, result);
  std::cout << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);

  cublasLtHandle_t lt_handle;
  CHECK_CUBLAS(cublasLtCreate(&lt_handle));

  if (!options.single_shape && options.run_accuracy) {
    run_accuracy_check(lt_handle);
  }

  if (!options.single_shape && options.run_accuracy && !options.run_check) {
    std::cout << "[performance] default sweep without correctness check"
              << std::endl;
  }

  print_header();
  if (options.single_shape) {
    const Result result =
        run_case(options.m, options.n, options.k, lt_handle, options);
    print_result(options.m, options.n, options.k, result);
  } else {
    for (const auto& [m, n, k] : default_shapes()) {
      const Result result = run_case(m, n, k, lt_handle, options);
      print_result(m, n, k, result);
    }
  }

  CHECK_CUBLAS(cublasLtDestroy(lt_handle));
  return 0;
}
