#include <cstdint>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

using Fc1Launcher = void (*)(
    const __nv_bfloat16*,
    const __nv_bfloat16*,
    const int32_t*,
    __nv_bfloat16*,
    int,
    int,
    int,
    int,
    cudaStream_t);

void check_bf16_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kBFloat16,
      name,
      " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_inputs(
    const torch::Tensor& a,
    const torch::Tensor& b_interleaved,
    const torch::Tensor& layout,
    const torch::Tensor& out,
    int64_t tile_n,
    int64_t tile_k) {
  check_bf16_cuda_contiguous(a, "A");
  check_bf16_cuda_contiguous(b_interleaved, "B_interleaved");
  check_bf16_cuda_contiguous(out, "Output");
  TORCH_CHECK(layout.is_cuda(), "grouped_layout must be a CUDA tensor");
  TORCH_CHECK(
      layout.scalar_type() == torch::kInt32,
      "grouped_layout must have dtype torch.int32");
  TORCH_CHECK(layout.is_contiguous(), "grouped_layout must be contiguous");
  TORCH_CHECK(a.dim() == 2, "A must be 2D");
  TORCH_CHECK(b_interleaved.dim() == 3, "B_interleaved must be 3D");
  TORCH_CHECK(out.dim() == 2, "Output must be 2D");
  TORCH_CHECK(layout.dim() == 1, "grouped_layout must be 1D");
  TORCH_CHECK(a.device() == b_interleaved.device(), "A and B must be on same device");
  TORCH_CHECK(a.device() == layout.device(), "A and grouped_layout must be on same device");
  TORCH_CHECK(a.device() == out.device(), "A and Output must be on same device");
  TORCH_CHECK(layout.size(0) == a.size(0), "grouped_layout length must equal M");
  TORCH_CHECK(a.size(1) == b_interleaved.size(2), "A and B inner dimensions must match");
  TORCH_CHECK(out.size(0) == a.size(0), "Output rows must equal M");
  TORCH_CHECK(b_interleaved.size(1) == 2 * out.size(1), "B_interleaved width must be 2 * output width");
  TORCH_CHECK(a.size(0) > 0, "A must contain at least one row");
  TORCH_CHECK(b_interleaved.size(0) > 0, "B_interleaved must contain at least one expert");
  TORCH_CHECK(tile_n > 0 && out.size(1) % (tile_n / 2) == 0, "logical N must be divisible by tile_n / 2");
  TORCH_CHECK(tile_k > 0 && a.size(1) % tile_k == 0, "K must be divisible by tile_k");
}

torch::Tensor fc1_block_interleaved_swiglu_out(
    torch::Tensor a,
    torch::Tensor b_interleaved,
    torch::Tensor layout,
    torch::Tensor out,
    uint64_t launcher_address,
    int64_t tile_n,
    int64_t tile_k) {
  TORCH_CHECK(launcher_address != 0, "fc1 block-interleaved swiglu launcher is null");
  TORCH_CHECK(tile_n % 2 == 0, "tile_n must be even");
  check_inputs(a, b_interleaved, layout, out, tile_n, tile_k);

  c10::cuda::CUDAGuard device_guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  const auto launcher =
      reinterpret_cast<Fc1Launcher>(static_cast<uintptr_t>(launcher_address));
  launcher(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b_interleaved.data_ptr<at::BFloat16>()),
      layout.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b_interleaved.size(0)),
      static_cast<int>(out.size(1)),
      static_cast<int>(a.size(1)),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor fc1_block_interleaved_swiglu(
    torch::Tensor a,
    torch::Tensor b_interleaved,
    torch::Tensor layout,
    uint64_t launcher_address,
    int64_t tile_n,
    int64_t tile_k) {
  auto out = torch::empty({a.size(0), b_interleaved.size(1) / 2}, a.options());
  return fc1_block_interleaved_swiglu_out(
      a, b_interleaved, layout, out, launcher_address, tile_n, tile_k);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "fc1_block_interleaved_swiglu",
      &fc1_block_interleaved_swiglu,
      "JIT-compiled block-interleaved FC1 SwiGLU",
      py::arg("a"),
      py::arg("b_interleaved"),
      py::arg("grouped_layout"),
      py::arg("launcher_address"),
      py::arg("tile_n"),
      py::arg("tile_k"));
  m.def(
      "fc1_block_interleaved_swiglu_out",
      &fc1_block_interleaved_swiglu_out,
      "JIT-compiled block-interleaved FC1 SwiGLU with explicit output",
      py::arg("a"),
      py::arg("b_interleaved"),
      py::arg("grouped_layout"),
      py::arg("out"),
      py::arg("launcher_address"),
      py::arg("tile_n"),
      py::arg("tile_k"));
}
