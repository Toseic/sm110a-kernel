#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "bf16_grouped_m_gemm.h"

#ifndef SM110A_GROUPED_TILE_N
#error "SM110A_GROUPED_TILE_N must be defined by the JIT compiler"
#endif

#ifndef SM110A_GROUPED_TILE_K
#error "SM110A_GROUPED_TILE_K must be defined by the JIT compiler"
#endif

namespace {

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_inputs(const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& layout) {
  check_tensor(a, "A");
  check_tensor(b, "B");
  TORCH_CHECK(layout.is_cuda(), "grouped_layout must be a CUDA tensor");
  TORCH_CHECK(layout.scalar_type() == torch::kInt32,
              "grouped_layout must have dtype torch.int32");
  TORCH_CHECK(layout.is_contiguous(), "grouped_layout must be contiguous");
  TORCH_CHECK(a.dim() == 2, "A must be 2D");
  TORCH_CHECK(b.dim() == 3, "B must be 3D");
  TORCH_CHECK(layout.dim() == 1, "grouped_layout must be 1D");
  TORCH_CHECK(layout.size(0) == a.size(0),
              "grouped_layout must have length M");
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same device");
  TORCH_CHECK(a.device() == layout.device(),
              "A and grouped_layout must be on the same device");
  TORCH_CHECK(a.size(1) == b.size(2),
              "A and B inner dimensions must match");
  TORCH_CHECK(a.size(0) > 0, "A must contain at least one row");
  TORCH_CHECK(b.size(0) > 0, "B must contain at least one expert");
  TORCH_CHECK(b.size(1) % SM110A_GROUPED_TILE_N == 0,
              "N must be a multiple of ", SM110A_GROUPED_TILE_N);
  TORCH_CHECK(a.size(1) % SM110A_GROUPED_TILE_K == 0,
              "K must be a multiple of ", SM110A_GROUPED_TILE_K);
}

void check_output(const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& out) {
  check_tensor(out, "Output");
  TORCH_CHECK(out.device() == a.device(),
              "Output must be on the same device as A");
  TORCH_CHECK(out.dim() == 2, "Output must be 2D");
  TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == b.size(1),
              "Output shape must be (", a.size(0), ", ", b.size(1), ")");
}

torch::Tensor grouped_gemm_out(torch::Tensor a, torch::Tensor b,
                               torch::Tensor layout, torch::Tensor out) {
  check_inputs(a, b, layout);
  check_output(a, b, out);
  c10::cuda::CUDAGuard device_guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  launch_bf16_grouped_m_gemm(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      layout.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor grouped_gemm(torch::Tensor a, torch::Tensor b,
                           torch::Tensor layout) {
  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  return grouped_gemm_out(a, b, layout, out);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bf16_grouped_m_gemm", &grouped_gemm,
        "JIT-compiled grouped M BF16 GEMM");
  m.def("bf16_grouped_m_gemm_out", &grouped_gemm_out,
        "JIT-compiled grouped M BF16 GEMM with explicit output");
}
