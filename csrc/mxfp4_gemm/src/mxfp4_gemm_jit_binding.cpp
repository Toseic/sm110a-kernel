#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "mxfp4_gemm.h"

#ifndef SM110A_MXFP4_TILE_M
#error "SM110A_MXFP4_TILE_M must be defined by the JIT compiler"
#endif
#ifndef SM110A_MXFP4_TILE_N
#error "SM110A_MXFP4_TILE_N must be defined by the JIT compiler"
#endif
#ifndef SM110A_MXFP4_TILE_K
#error "SM110A_MXFP4_TILE_K must be defined by the JIT compiler"
#endif
#ifndef SM110A_MXFP4_CLUSTER_M
#error "SM110A_MXFP4_CLUSTER_M must be defined by the JIT compiler"
#endif

namespace {

void check_fp4(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat4_e2m1fn_x2,
              name, " must have dtype torch.float4_e2m1fn_x2");
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_scale(const torch::Tensor& tensor, const char* name,
                 int64_t expected_numel, const c10::Device& device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat8_e8m0fnu,
              name, " must have dtype torch.float8_e8m0fnu (UE8M0 bits)");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.device() == device, name, " must be on the same device as A");
  TORCH_CHECK(tensor.numel() == expected_numel, name, " must contain ",
              expected_numel, " blocked scale bytes, got ", tensor.numel());
}

void check_inputs(const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& sfa, const torch::Tensor& sfb) {
  check_fp4(a, "A");
  check_fp4(b, "B");
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same device");
  const int64_t m = a.size(0);
  const int64_t n = b.size(0);
  const int64_t k = a.size(1) * 2;
  TORCH_CHECK(b.size(1) * 2 == k,
              "A and transposed-B packed K dimensions must match");
  TORCH_CHECK(m % (SM110A_MXFP4_TILE_M * SM110A_MXFP4_CLUSTER_M) == 0,
              "M must be a multiple of ",
              SM110A_MXFP4_TILE_M * SM110A_MXFP4_CLUSTER_M);
  TORCH_CHECK(n % SM110A_MXFP4_TILE_N == 0,
              "N must be a multiple of ", SM110A_MXFP4_TILE_N);
  TORCH_CHECK(n % 128 == 0,
              "N must be a multiple of 128 for blocked MXFP4 scales");
  TORCH_CHECK(k % SM110A_MXFP4_TILE_K == 0,
              "K must be a multiple of ", SM110A_MXFP4_TILE_K);
  check_scale(sfa, "SFA", m * (k / 16), a.device());
  check_scale(sfb, "SFB", n * (k / 16), a.device());
}

void check_output(const torch::Tensor& out, int64_t m, int64_t n,
                  const c10::Device& device) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16,
              "Output must be a CUDA torch.bfloat16 tensor");
  TORCH_CHECK(out.dim() == 2 && out.is_contiguous(),
              "Output must be a contiguous 2D tensor");
  TORCH_CHECK(out.device() == device, "Output must be on the same device as A");
  TORCH_CHECK(out.size(0) == m && out.size(1) == n,
              "Output must have shape (", m, ", ", n, ")");
}

torch::Tensor mxfp4_gemm_out(torch::Tensor a, torch::Tensor b,
                             torch::Tensor sfa, torch::Tensor sfb,
                             torch::Tensor out) {
  check_inputs(a, b, sfa, sfb);
  check_output(out, a.size(0), b.size(0), a.device());
  c10::cuda::CUDAGuard device_guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  launch_mxfp4_gemm(
      reinterpret_cast<const uint8_t*>(a.data_ptr()),
      reinterpret_cast<const uint8_t*>(b.data_ptr()),
      reinterpret_cast<const uint8_t*>(sfa.data_ptr()),
      reinterpret_cast<const uint8_t*>(sfb.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)), static_cast<int>(b.size(0)),
      static_cast<int>(a.size(1) * 2), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor mxfp4_gemm(torch::Tensor a, torch::Tensor b,
                         torch::Tensor sfa, torch::Tensor sfb) {
  check_inputs(a, b, sfa, sfb);
  auto out = torch::empty({a.size(0), b.size(0)},
                          a.options().dtype(torch::kBFloat16));
  return mxfp4_gemm_out(a, b, sfa, sfb, out);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mxfp4_gemm", &mxfp4_gemm, py::arg("a"), py::arg("b"),
        py::arg("sfa"), py::arg("sfb"));
  m.def("mxfp4_gemm_out", &mxfp4_gemm_out, py::arg("a"), py::arg("b"),
        py::arg("sfa"), py::arg("sfb"), py::arg("out"));
}
