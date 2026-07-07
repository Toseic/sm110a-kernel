#include <cstdint>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

using GroupedGemmLauncher = void (*)(
    const __nv_bfloat16*, const __nv_bfloat16*, const int32_t*,
    __nv_bfloat16*, int, int, int, int, bool, cudaStream_t);

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_inputs(const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& layout, int64_t tile_m,
                  int64_t tile_n, int64_t tile_k, int64_t cluster_m,
                  bool swap_ab) {
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
  const int64_t logical_n_tile = swap_ab ? cluster_m * tile_m : tile_n;
  TORCH_CHECK(logical_n_tile > 0 && b.size(1) % logical_n_tile == 0,
              "N must be a multiple of ", logical_n_tile,
              " when swap_ab=", swap_ab);
  TORCH_CHECK(tile_k > 0 && a.size(1) % tile_k == 0,
              "K must be a multiple of ", tile_k);
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
                               torch::Tensor layout, torch::Tensor out,
                               uint64_t launcher_address, int64_t tile_m,
                               int64_t tile_n, int64_t tile_k,
                               int64_t cluster_m, bool swap_ab) {
  TORCH_CHECK(launcher_address != 0, "grouped GEMM launcher is null");
  check_inputs(
      a, b, layout, tile_m, tile_n, tile_k, cluster_m, swap_ab);
  check_output(a, b, out);
  c10::cuda::CUDAGuard device_guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  const auto launcher = reinterpret_cast<GroupedGemmLauncher>(
      static_cast<uintptr_t>(launcher_address));
  launcher(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      layout.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)),
      swap_ab,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor grouped_gemm(torch::Tensor a, torch::Tensor b,
                           torch::Tensor layout, uint64_t launcher_address,
                           int64_t tile_m, int64_t tile_n, int64_t tile_k,
                           int64_t cluster_m, bool swap_ab) {
  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  return grouped_gemm_out(
      a, b, layout, out, launcher_address, tile_m, tile_n, tile_k,
      cluster_m, swap_ab);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bf16_grouped_m_gemm", &grouped_gemm,
        "JIT-compiled grouped M BF16 GEMM",
        py::arg("a"), py::arg("b"), py::arg("grouped_layout"),
        py::arg("launcher_address"), py::arg("tile_m"),
        py::arg("tile_n"), py::arg("tile_k"), py::arg("cluster_m"),
        py::arg("swap_ab") = false);
  m.def("bf16_grouped_m_gemm_out", &grouped_gemm_out,
        "JIT-compiled grouped M BF16 GEMM with explicit output",
        py::arg("a"), py::arg("b"), py::arg("grouped_layout"),
        py::arg("out"), py::arg("launcher_address"), py::arg("tile_m"),
        py::arg("tile_n"), py::arg("tile_k"), py::arg("cluster_m"),
        py::arg("swap_ab") = false);
}
