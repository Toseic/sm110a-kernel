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
    const __nv_bfloat16*,
    __nv_bfloat16*,
    int,
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
    const torch::Tensor& router_scores,
    const torch::Tensor& out,
    int64_t topk,
    int64_t tile_n,
    int64_t tile_k) {
  check_bf16_cuda_contiguous(a, "A");
  check_bf16_cuda_contiguous(b_interleaved, "B_interleaved");
  check_bf16_cuda_contiguous(out, "Output");
  TORCH_CHECK(router_scores.is_cuda(), "router_scores must be a CUDA tensor");
  TORCH_CHECK(
      router_scores.scalar_type() == torch::kBFloat16,
      "router_scores must have dtype torch.bfloat16");
  TORCH_CHECK(router_scores.is_contiguous(), "router_scores must be contiguous");
  TORCH_CHECK(a.dim() == 2, "A must be 2D");
  TORCH_CHECK(b_interleaved.dim() == 3, "B_interleaved must be 3D");
  TORCH_CHECK(router_scores.dim() == 2, "router_scores must be 2D");
  TORCH_CHECK(out.dim() == 2, "Output must be 2D");
  TORCH_CHECK(a.device() == b_interleaved.device(), "A and B must be on same device");
  TORCH_CHECK(
      a.device() == router_scores.device(),
      "A and router_scores must be on same device");
  TORCH_CHECK(a.device() == out.device(), "A and Output must be on same device");
  TORCH_CHECK(
      router_scores.size(0) == a.size(0),
      "A and router_scores token dimensions must match");
  TORCH_CHECK(
      router_scores.size(1) == b_interleaved.size(0),
      "router_scores expert dimension must match B");
  TORCH_CHECK(a.size(1) == b_interleaved.size(2), "A and B inner dimensions must match");
  TORCH_CHECK(topk > 0, "topk must be positive");
  TORCH_CHECK(topk <= router_scores.size(1), "topk must not exceed the number of experts");
  TORCH_CHECK(out.size(0) == a.size(0) * topk, "Output rows must equal tokens * topk");
  TORCH_CHECK(
      b_interleaved.size(1) == 2 * out.size(1),
      "B_interleaved width must be 2 * output width");
  TORCH_CHECK(a.size(0) > 0, "A must contain at least one token");
  TORCH_CHECK(b_interleaved.size(0) > 0, "B_interleaved must contain at least one expert");
  TORCH_CHECK(
      tile_n > 0 && out.size(1) % (tile_n / 2) == 0,
      "logical N must be divisible by tile_n / 2");
  TORCH_CHECK(tile_k > 0 && a.size(1) % tile_k == 0, "K must be divisible by tile_k");
}

torch::Tensor fc1_block_interleaved_swiglu_out(
    torch::Tensor a,
    torch::Tensor b_interleaved,
    torch::Tensor router_scores,
    int64_t topk,
    torch::Tensor out,
    uint64_t launcher_address,
    int64_t tile_n,
    int64_t tile_k) {
  TORCH_CHECK(launcher_address != 0, "fc1 block-interleaved swiglu launcher is null");
  TORCH_CHECK(tile_n % 2 == 0, "tile_n must be even");
  check_inputs(a, b_interleaved, router_scores, out, topk, tile_n, tile_k);

  c10::cuda::CUDAGuard device_guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  const auto launcher =
      reinterpret_cast<Fc1Launcher>(static_cast<uintptr_t>(launcher_address));
  launcher(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b_interleaved.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          router_scores.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b_interleaved.size(0)),
      static_cast<int>(topk),
      static_cast<int>(out.size(1)),
      static_cast<int>(a.size(1)),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor fc1_block_interleaved_swiglu(
    torch::Tensor a,
    torch::Tensor b_interleaved,
    torch::Tensor router_scores,
    int64_t topk,
    uint64_t launcher_address,
    int64_t tile_n,
    int64_t tile_k) {
  auto out = torch::empty({a.size(0) * topk, b_interleaved.size(1) / 2}, a.options());
  return fc1_block_interleaved_swiglu_out(
      a, b_interleaved, router_scores, topk, out, launcher_address, tile_n, tile_k);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "fc1_block_interleaved_swiglu",
      &fc1_block_interleaved_swiglu,
      "JIT-compiled routed block-interleaved FC1 SwiGLU",
      py::arg("a"),
      py::arg("b_interleaved"),
      py::arg("router_scores"),
      py::arg("topk"),
      py::arg("launcher_address"),
      py::arg("tile_n"),
      py::arg("tile_k"));
  m.def(
      "fc1_block_interleaved_swiglu_out",
      &fc1_block_interleaved_swiglu_out,
      "JIT-compiled routed block-interleaved FC1 SwiGLU with explicit output",
      py::arg("a"),
      py::arg("b_interleaved"),
      py::arg("router_scores"),
      py::arg("topk"),
      py::arg("out"),
      py::arg("launcher_address"),
      py::arg("tile_n"),
      py::arg("tile_k"));
}
