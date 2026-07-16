#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

using Fc2Launcher = void (*)(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, const int32_t*, const int32_t*, int32_t*, int, cudaStream_t);

void check_bf16_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor fc2_finalize_out(
    torch::Tensor fc1, torch::Tensor b, torch::Tensor router_scores, int64_t topk,
    torch::Tensor out, uint64_t launcher_address, int64_t tile_n, int64_t tile_k) {
  check_bf16_cuda_contiguous(fc1, "FC1 output");
  check_bf16_cuda_contiguous(b, "B");
  check_bf16_cuda_contiguous(router_scores, "router_scores");
  check_bf16_cuda_contiguous(out, "Output");
  TORCH_CHECK(fc1.dim() == 2, "FC1 output must be 2D");
  TORCH_CHECK(b.dim() == 3, "B must have shape [experts, H, I]");
  TORCH_CHECK(router_scores.dim() == 2, "router_scores must be 2D");
  TORCH_CHECK(out.dim() == 2, "Output must be 2D");
  TORCH_CHECK(fc1.device() == b.device() && fc1.device() == router_scores.device() &&
                  fc1.device() == out.device(),
              "all tensors must be on the same device");
  TORCH_CHECK(topk > 0 && topk <= b.size(0), "topk must be in [1, experts]");
  TORCH_CHECK(router_scores.size(0) * topk == fc1.size(0),
              "FC1 rows must equal router_scores.size(0) * topk");
  TORCH_CHECK(router_scores.size(1) == b.size(0), "router_scores expert dimension must match B");
  TORCH_CHECK(fc1.size(1) == b.size(2), "FC1 width must equal B I dimension");
  TORCH_CHECK(out.size(0) == router_scores.size(0) && out.size(1) == b.size(1),
              "Output must have shape [tokens, H]");
  TORCH_CHECK(tile_n > 0 && b.size(1) % tile_n == 0, "H must be divisible by tile_n");
  TORCH_CHECK(tile_k > 0 && fc1.size(1) % tile_k == 0, "I must be divisible by tile_k");
  TORCH_CHECK(launcher_address != 0, "FC2 launcher is null");

  c10::cuda::CUDAGuard device_guard(fc1.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(fc1.device().index());
  const auto launcher = reinterpret_cast<Fc2Launcher>(static_cast<uintptr_t>(launcher_address));
  launcher(
      reinterpret_cast<const __nv_bfloat16*>(fc1.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(router_scores.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(router_scores.size(0)), static_cast<int>(b.size(0)),
      static_cast<int>(topk), static_cast<int>(b.size(1)), static_cast<int>(fc1.size(1)), nullptr,
      nullptr, nullptr, 0, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor fc2_finalize_with_routing_out(
    torch::Tensor fc1, torch::Tensor b, int64_t topk, torch::Tensor routing_weights,
    torch::Tensor routing,
    torch::Tensor tasks, torch::Tensor state, torch::Tensor out, uint64_t launcher_address,
    int64_t tile_n, int64_t tile_k) {
  check_bf16_cuda_contiguous(fc1, "FC1 output");
  check_bf16_cuda_contiguous(b, "B");
  check_bf16_cuda_contiguous(routing_weights, "routing_weights");
  check_bf16_cuda_contiguous(out, "Output");
  TORCH_CHECK(routing.is_cuda() && routing.scalar_type() == torch::kInt32 && routing.is_contiguous(),
              "routing must be a contiguous CUDA torch.int32 tensor");
  TORCH_CHECK(tasks.is_cuda() && tasks.scalar_type() == torch::kInt32 && tasks.is_contiguous() &&
                  tasks.dim() == 2 && tasks.size(1) == 5,
              "tasks must be a contiguous CUDA int32 [N, 5] tensor");
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kInt32 && state.is_contiguous() &&
                  state.numel() == 2,
              "state must be a contiguous CUDA int32 {task_count, next_cluster_tile} tensor");
  TORCH_CHECK(fc1.dim() == 2 && b.dim() == 3 && out.dim() == 2, "invalid FC2 tensor rank");
  TORCH_CHECK(fc1.device() == b.device() && fc1.device() == routing_weights.device() &&
                  fc1.device() == routing.device() &&
                  fc1.device() == tasks.device() && fc1.device() == state.device() && fc1.device() == out.device(),
              "all tensors must be on the same device");
  TORCH_CHECK(topk > 0 && topk <= b.size(0) && fc1.size(0) % topk == 0, "invalid topk");
  const auto tokens = fc1.size(0) / topk;
  TORCH_CHECK(routing_weights.dim() == 2 && routing_weights.size(0) == tokens &&
                  routing_weights.size(1) == topk,
              "routing_weights must have shape [tokens, topk]");
  TORCH_CHECK(routing.numel() == 2 * b.size(0) + 3 * fc1.size(0), "routing has an invalid size");
  TORCH_CHECK(fc1.size(1) == b.size(2) && out.size(0) == tokens && out.size(1) == b.size(1),
              "incompatible FC2 shapes");
  TORCH_CHECK(tile_n > 0 && b.size(1) % tile_n == 0 && tile_k > 0 && fc1.size(1) % tile_k == 0,
              "FC2 shape is incompatible with the selected tile size");
  TORCH_CHECK(launcher_address != 0, "FC2 launcher is null");
  c10::cuda::CUDAGuard device_guard(fc1.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(fc1.device().index());
  const auto launcher = reinterpret_cast<Fc2Launcher>(static_cast<uintptr_t>(launcher_address));
  launcher(reinterpret_cast<const __nv_bfloat16*>(fc1.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(routing_weights.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), static_cast<int>(tokens),
      static_cast<int>(b.size(0)), static_cast<int>(topk), static_cast<int>(b.size(1)),
      static_cast<int>(fc1.size(1)), routing.data_ptr<int32_t>(), tasks.data_ptr<int32_t>(),
      state.data_ptr<int32_t>(), static_cast<int>(tasks.size(0)), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor fc2_finalize(
    torch::Tensor fc1, torch::Tensor b, torch::Tensor router_scores, int64_t topk,
    uint64_t launcher_address, int64_t tile_n, int64_t tile_k) {
  auto out = torch::empty({router_scores.size(0), b.size(1)}, fc1.options());
  return fc2_finalize_out(fc1, b, router_scores, topk, out, launcher_address, tile_n, tile_k);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fc2_finalize", &fc2_finalize, "Routed FC2 grouped GEMM and token finalize");
  m.def("fc2_finalize_out", &fc2_finalize_out, "Routed FC2 with explicit output");
  m.def("fc2_finalize_with_routing_out", &fc2_finalize_with_routing_out,
        "Routed FC2 with weighted token finalize using FC1 routing metadata");
}
