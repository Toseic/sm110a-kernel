#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <vector>

#include "bf16_gemm.h"

namespace {

constexpr int kBlockM = 128;
constexpr int kBlockN = 256;
constexpr int kBlockK = 64;
constexpr std::size_t kDefaultCublasLtWorkspaceBytes = 32ull * 1024ull * 1024ull;

void check_cublas_status(cublasStatus_t status, const char* expr) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              expr,
              " failed with cuBLAS status ",
              static_cast<int>(status));
}

void check_inputs(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16,
              "A must have dtype torch.bfloat16");
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16,
              "B must have dtype torch.bfloat16");
  TORCH_CHECK(a.dim() == 2, "A must be 2D");
  TORCH_CHECK(b.dim() == 2, "B must be 2D");
  TORCH_CHECK(a.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same device");
  TORCH_CHECK(a.size(1) == b.size(0),
              "Inner dimensions must match: got ",
              a.size(0),
              "x",
              a.size(1),
              " and ",
              b.size(0),
              "x",
              b.size(1));
  TORCH_CHECK(a.size(0) % kBlockM == 0,
              "M must be a multiple of ",
              kBlockM,
              ", got ",
              a.size(0));
  TORCH_CHECK(b.size(1) % kBlockN == 0,
              "N must be a multiple of ",
              kBlockN,
              ", got ",
              b.size(1));
  TORCH_CHECK(a.size(1) % kBlockK == 0,
              "K must be a multiple of ",
              kBlockK,
              ", got ",
              a.size(1));
}

void check_output(const torch::Tensor& out, int64_t m, int64_t n,
                  const c10::Device& device) {
  TORCH_CHECK(out.is_cuda(), "Output must be a CUDA tensor");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16,
              "Output must have dtype torch.bfloat16");
  TORCH_CHECK(out.dim() == 2, "Output must be 2D");
  TORCH_CHECK(out.is_contiguous(), "Output must be contiguous");
  TORCH_CHECK(out.device() == device, "Output must be on the same device");
  TORCH_CHECK(out.size(0) == m && out.size(1) == n,
              "Output must have shape (",
              m,
              ", ",
              n,
              "), got (",
              out.size(0),
              ", ",
              out.size(1),
              ")");
}

void check_workspace(const torch::Tensor& workspace, const c10::Device& device) {
  TORCH_CHECK(workspace.is_cuda(), "Workspace must be a CUDA tensor");
  TORCH_CHECK(workspace.scalar_type() == torch::kUInt8,
              "Workspace must have dtype torch.uint8");
  TORCH_CHECK(workspace.is_contiguous(), "Workspace must be contiguous");
  TORCH_CHECK(workspace.device() == device,
              "Workspace must be on the same device");
  TORCH_CHECK(workspace.numel() > 0, "Workspace must be non-empty");
}

cublasLtHandle_t get_cublaslt_handle() {
  static cublasLtHandle_t handle = [] {
    cublasLtHandle_t h = nullptr;
    check_cublas_status(cublasLtCreate(&h), "cublasLtCreate");
    return h;
  }();
  return handle;
}

void run_cublaslt(torch::Tensor a, torch::Tensor b, torch::Tensor out,
                  torch::Tensor workspace, int iters) {
  check_inputs(a, b);
  check_output(out, a.size(0), b.size(1), a.device());
  check_workspace(workspace, a.device());
  TORCH_CHECK(iters > 0, "iters must be positive");
  c10::cuda::CUDAGuard device_guard(a.device());

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int64_t m = a.size(0);
  const int64_t n = b.size(1);
  const int64_t k = a.size(1);

  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  check_cublas_status(
      cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F_FAST_16BF,
                               CUDA_R_32F),
      "cublasLtMatmulDescCreate");
  // cuBLASLt uses column-major layouts.  The row-major B(K, N) buffer is
  // therefore the column-major view B^T(N, K), while the row-major A(M, K)
  // buffer is the column-major view A^T(K, M).  Passing B as operand A and A
  // as operand B computes C^T = B^T * A^T, which is stored in the row-major
  // C(M, N) buffer through the column-major C(N, M) layout.
  check_cublas_status(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, n, k, n),
                      "cublasLtMatrixLayoutCreate(a_desc)");
  check_cublas_status(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, k, m, k),
                      "cublasLtMatrixLayoutCreate(b_desc)");
  check_cublas_status(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, n, m, n),
                      "cublasLtMatrixLayoutCreate(c_desc)");
  check_cublas_status(cublasLtMatmulPreferenceCreate(&preference),
                      "cublasLtMatmulPreferenceCreate");

  const std::size_t workspace_size = workspace.numel();
  check_cublas_status(cublasLtMatmulPreferenceSetAttribute(
                          preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                          &workspace_size, sizeof(workspace_size)),
                      "cublasLtMatmulPreferenceSetAttribute");

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  check_cublas_status(cublasLtMatmulAlgoGetHeuristic(
                          get_cublaslt_handle(), operation_desc, a_desc, b_desc,
                          c_desc, c_desc, preference, 1, &heuristic,
                          &returned_results),
                      "cublasLtMatmulAlgoGetHeuristic");
  TORCH_CHECK(returned_results > 0,
              "cublasLtMatmulAlgoGetHeuristic returned no algorithm");

  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  for (int i = 0; i < iters; ++i) {
    check_cublas_status(cublasLtMatmul(
                            get_cublaslt_handle(), operation_desc, &alpha,
                            reinterpret_cast<const __nv_bfloat16*>(
                                b.data_ptr<at::BFloat16>()),
                            a_desc,
                            reinterpret_cast<const __nv_bfloat16*>(
                                a.data_ptr<at::BFloat16>()),
                            b_desc, &beta,
                            reinterpret_cast<__nv_bfloat16*>(
                                out.data_ptr<at::BFloat16>()),
                            c_desc,
                            reinterpret_cast<__nv_bfloat16*>(
                                out.data_ptr<at::BFloat16>()),
                            c_desc, &heuristic.algo, workspace.data_ptr(),
                            workspace_size, stream),
                        "cublasLtMatmul");
  }

  check_cublas_status(cublasLtMatmulPreferenceDestroy(preference),
                      "cublasLtMatmulPreferenceDestroy");
  check_cublas_status(cublasLtMatrixLayoutDestroy(c_desc),
                      "cublasLtMatrixLayoutDestroy(c_desc)");
  check_cublas_status(cublasLtMatrixLayoutDestroy(b_desc),
                      "cublasLtMatrixLayoutDestroy(b_desc)");
  check_cublas_status(cublasLtMatrixLayoutDestroy(a_desc),
                      "cublasLtMatrixLayoutDestroy(a_desc)");
  check_cublas_status(cublasLtMatmulDescDestroy(operation_desc),
                      "cublasLtMatmulDescDestroy");
}

torch::Tensor cublaslt_gemm(torch::Tensor a, torch::Tensor b) {
  check_inputs(a, b);
  c10::cuda::CUDAGuard device_guard(a.device());

  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  auto workspace = torch::empty(
      {static_cast<long long>(kDefaultCublasLtWorkspaceBytes)},
      out.options().dtype(torch::kUInt8));
  run_cublaslt(a, b, out, workspace, 1);
  return out;
}

torch::Tensor cublaslt_gemm_out(torch::Tensor a, torch::Tensor b, torch::Tensor out,
                                torch::Tensor workspace) {
  run_cublaslt(a, b, out, workspace, 1);
  return out;
}

torch::Tensor cublaslt_gemm_out_batched(torch::Tensor a, torch::Tensor b,
                                        torch::Tensor out, torch::Tensor workspace,
                                        int iters) {
  run_cublaslt(a, b, out, workspace, iters);
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cublaslt_gemm", &cublaslt_gemm, "cuBLASLt BF16 GEMM");
  m.def("cublaslt_gemm_out", &cublaslt_gemm_out,
        "cuBLASLt BF16 GEMM with explicit output and workspace");
  m.def("cublaslt_gemm_out_batched", &cublaslt_gemm_out_batched,
        "cuBLASLt BF16 GEMM with explicit output, workspace, and batched iterations");
}
