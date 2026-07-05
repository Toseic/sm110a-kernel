#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "bf16_gemm.h"

#ifndef SM110A_BF16_TILE_M
#error "SM110A_BF16_TILE_M must be defined by the JIT compiler"
#endif

#ifndef SM110A_BF16_TILE_N
#error "SM110A_BF16_TILE_N must be defined by the JIT compiler"
#endif

#ifndef SM110A_BF16_TILE_K
#error "SM110A_BF16_TILE_K must be defined by the JIT compiler"
#endif

#ifndef SM110A_BF16_CLUSTER_M
#error "SM110A_BF16_CLUSTER_M must be defined by the JIT compiler"
#endif

namespace {

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_inputs(const torch::Tensor& a, const torch::Tensor& b) {
  check_tensor(a, "A");
  check_tensor(b, "B");
  TORCH_CHECK(a.dim() == 2, "A must be 2D");
  TORCH_CHECK(b.dim() == 2, "B must be 2D");
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
  TORCH_CHECK(a.size(0) % (SM110A_BF16_TILE_M * SM110A_BF16_CLUSTER_M) == 0,
              "M must be a multiple of ",
              SM110A_BF16_TILE_M * SM110A_BF16_CLUSTER_M,
              ", got ",
              a.size(0));
  TORCH_CHECK(b.size(1) % SM110A_BF16_TILE_N == 0,
              "N must be a multiple of ",
              SM110A_BF16_TILE_N,
              ", got ",
              b.size(1));
  TORCH_CHECK(a.size(1) % SM110A_BF16_TILE_K == 0,
              "K must be a multiple of ",
              SM110A_BF16_TILE_K,
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

torch::Tensor bf16_gemm_out_jit(torch::Tensor a, torch::Tensor b,
                                torch::Tensor out) {
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

torch::Tensor bf16_gemm_jit(torch::Tensor a, torch::Tensor b) {
  check_inputs(a, b);
  c10::cuda::CUDAGuard device_guard(a.device());

  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  return bf16_gemm_out_jit(a, b, out);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bf16_gemm", &bf16_gemm_jit,
        "JIT-compiled BF16 GEMM");
  m.def("bf16_gemm_out", &bf16_gemm_out_jit,
        "JIT-compiled BF16 GEMM with explicit output");
}
