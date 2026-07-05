#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <limits>
#include <vector>

#include "bf16_gemm.h"
#include "bf16_grouped_m_gemm.h"

namespace {

constexpr int kBlockM = 128;
constexpr int kBlockN = 256;
constexpr int kBlockK = 64;
constexpr int kGroupedBlockK = 128;
constexpr int kProfilerBlockCount = 20;
constexpr int kProfilerWarpsPerBlock = 6;
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

void check_profiler(const torch::Tensor& profiler, int num_entries,
                    const c10::Device& device) {
  TORCH_CHECK(num_entries > 0, "num_entries must be positive");
  TORCH_CHECK(profiler.is_cuda(), "Profiler buffer must be a CUDA tensor");
  TORCH_CHECK(profiler.scalar_type() == torch::kInt64,
              "Profiler buffer must have dtype torch.int64");
  TORCH_CHECK(profiler.dim() == 2, "Profiler buffer must be 2D");
  TORCH_CHECK(profiler.is_contiguous(), "Profiler buffer must be contiguous");
  TORCH_CHECK(profiler.device() == device,
              "Profiler buffer must be on the same device");
  TORCH_CHECK(profiler.size(0) >= kProfilerBlockCount * kProfilerWarpsPerBlock,
              "Profiler buffer must provide at least ",
              kProfilerBlockCount * kProfilerWarpsPerBlock,
              " rows");
  TORCH_CHECK(profiler.size(1) == 1 + static_cast<int64_t>(num_entries) * 4,
              "Profiler buffer must have shape (*, ",
              1 + static_cast<int64_t>(num_entries) * 4,
              ")");
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
  const cublasOperation_t trans_a = CUBLAS_OP_T;
  const cublasOperation_t trans_b = CUBLAS_OP_N;
  check_cublas_status(cublasLtMatmulDescSetAttribute(
                          operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a,
                          sizeof(trans_a)),
                      "cublasLtMatmulDescSetAttribute(TRANSA)");
  check_cublas_status(cublasLtMatmulDescSetAttribute(
                          operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b,
                          sizeof(trans_b)),
                      "cublasLtMatmulDescSetAttribute(TRANSB)");

  check_cublas_status(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, k, n, k),
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

torch::Tensor bf16_gemm(torch::Tensor a, torch::Tensor b) {
  check_inputs(a, b);
  c10::cuda::CUDAGuard device_guard(a.device());

  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  launch_bf16_gemm(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor bf16_gemm_out(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  check_inputs(a, b);
  check_output(out, a.size(0), b.size(1), a.device());
  c10::cuda::CUDAGuard device_guard(a.device());

  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  launch_bf16_gemm(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor bf16_gemm_profile(torch::Tensor a, torch::Tensor b,
                                torch::Tensor profiler, int64_t num_entries) {
  check_inputs(a, b);
  TORCH_CHECK(num_entries <= std::numeric_limits<int>::max(),
              "num_entries is too large");
  check_profiler(profiler, static_cast<int>(num_entries), a.device());
  c10::cuda::CUDAGuard device_guard(a.device());

  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  launch_bf16_gemm_profile(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)),
      profiler.data_ptr<int64_t>(),
      static_cast<int>(num_entries),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor bf16_gemm_dynamic_schedule(torch::Tensor a, torch::Tensor b) {
  check_inputs(a, b);
  c10::cuda::CUDAGuard device_guard(a.device());

  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  launch_bf16_gemm_dynamic_schedule(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor bf16_gemm_dynamic_schedule_out(torch::Tensor a, torch::Tensor b,
                                             torch::Tensor out) {
  check_inputs(a, b);
  check_output(out, a.size(0), b.size(1), a.device());
  c10::cuda::CUDAGuard device_guard(a.device());

  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  launch_bf16_gemm_dynamic_schedule(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor bf16_gemm_dynamic_schedule_profile(
    torch::Tensor a, torch::Tensor b, torch::Tensor profiler,
    int64_t num_entries) {
  check_inputs(a, b);
  TORCH_CHECK(num_entries <= std::numeric_limits<int>::max(),
              "num_entries is too large");
  check_profiler(profiler, static_cast<int>(num_entries), a.device());
  c10::cuda::CUDAGuard device_guard(a.device());

  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  launch_bf16_gemm_dynamic_schedule_profile(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)),
      profiler.data_ptr<int64_t>(),
      static_cast<int>(num_entries),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
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

void check_group_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_layout_tensor(const torch::Tensor& layout) {
  TORCH_CHECK(layout.is_cuda(), "grouped_layout must be a CUDA tensor");
  TORCH_CHECK(layout.scalar_type() == torch::kInt32,
              "grouped_layout must have dtype torch.int32");
  TORCH_CHECK(layout.dim() == 1, "grouped_layout must be 1D");
  TORCH_CHECK(layout.is_contiguous(), "grouped_layout must be contiguous");
}

void check_grouped_inputs(const torch::Tensor& a, const torch::Tensor& b,
                          const torch::Tensor& layout) {
  check_group_tensor(a, "A");
  check_group_tensor(b, "B");
  check_layout_tensor(layout);
  TORCH_CHECK(a.dim() == 2, "A must be 2D");
  TORCH_CHECK(b.dim() == 3, "B must be 3D");
  TORCH_CHECK(layout.size(0) == a.size(0),
              "grouped_layout must have length M");
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same device");
  TORCH_CHECK(a.device() == layout.device(),
              "A and grouped_layout must be on the same device");
  TORCH_CHECK(a.size(1) == b.size(2),
              "A and B inner dimensions must match");
  TORCH_CHECK(b.size(0) > 0, "B must contain at least one expert");
  TORCH_CHECK(a.size(0) > 0, "A must contain at least one row");
  TORCH_CHECK(b.size(1) % kBlockN == 0,
              "N must be a multiple of ", kBlockN);
  TORCH_CHECK(a.size(1) % kGroupedBlockK == 0,
              "K must be a multiple of ", kGroupedBlockK);
}

void check_grouped_output(const torch::Tensor& a, const torch::Tensor& b,
                          const torch::Tensor& out) {
  TORCH_CHECK(out.is_cuda(), "Output must be a CUDA tensor");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16,
              "Output must have dtype torch.bfloat16");
  TORCH_CHECK(out.dim() == 2, "Output must be 2D");
  TORCH_CHECK(out.is_contiguous(), "Output must be contiguous");
  TORCH_CHECK(out.device() == a.device(), "Output must be on the same device");
  TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == b.size(1),
              "Output shape must be (", a.size(0), ", ", b.size(1), ")");
}

torch::Tensor bf16_grouped_m_gemm(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor grouped_layout) {
  check_grouped_inputs(a, b, grouped_layout);
  c10::cuda::CUDAGuard device_guard(a.device());

  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  launch_bf16_grouped_m_gemm(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      grouped_layout.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor bf16_grouped_m_gemm_out(torch::Tensor a, torch::Tensor b,
                                      torch::Tensor grouped_layout,
                                      torch::Tensor out) {
  check_grouped_inputs(a, b, grouped_layout);
  check_grouped_output(a, b, out);
  c10::cuda::CUDAGuard device_guard(a.device());

  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  launch_bf16_grouped_m_gemm(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      grouped_layout.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bf16_gemm", &bf16_gemm, "SM110A BF16 GEMM");
  m.def("bf16_gemm_out", &bf16_gemm_out, "SM110A BF16 GEMM with explicit output");
  m.def("bf16_gemm_profile", &bf16_gemm_profile,
        "SM110A BF16 GEMM with profiler output");
  m.def("bf16_gemm_dynamic_schedule", &bf16_gemm_dynamic_schedule,
        "SM110A BF16 GEMM with dynamic schedule");
  m.def("bf16_gemm_dynamic_schedule_out", &bf16_gemm_dynamic_schedule_out,
        "SM110A BF16 GEMM with dynamic schedule and explicit output");
  m.def("bf16_gemm_dynamic_schedule_profile", &bf16_gemm_dynamic_schedule_profile,
        "SM110A BF16 GEMM with dynamic schedule and profiler output");
  m.def("cublaslt_gemm", &cublaslt_gemm, "cuBLASLt BF16 GEMM");
  m.def("cublaslt_gemm_out", &cublaslt_gemm_out,
        "cuBLASLt BF16 GEMM with explicit output and workspace");
  m.def("cublaslt_gemm_out_batched", &cublaslt_gemm_out_batched,
        "cuBLASLt BF16 GEMM with explicit output, workspace, and batched iterations");
  m.def("bf16_grouped_m_gemm", &bf16_grouped_m_gemm,
        "SM110A grouped M BF16 GEMM scaffold");
  m.def("bf16_grouped_m_gemm_out", &bf16_grouped_m_gemm_out,
        "SM110A grouped M BF16 GEMM scaffold with explicit outputs");
}
