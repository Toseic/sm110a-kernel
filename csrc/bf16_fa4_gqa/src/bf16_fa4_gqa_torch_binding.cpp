#include <cmath>
#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

using FaGqaLauncher = cudaError_t (*)(const __nv_bfloat16*, const __nv_bfloat16*,
                                      const __nv_bfloat16*, __nv_bfloat16*, int,
                                      int, int, int, int, int, float, bool,
                                      cudaStream_t);

void check_qkv(const torch::Tensor& q, const torch::Tensor& k,
               const torch::Tensor& v) {
  for (const auto* tensor : {&q, &k, &v}) {
    TORCH_CHECK(tensor->is_cuda(), "Q, K, and V must be CUDA tensors");
    TORCH_CHECK(tensor->scalar_type() == torch::kBFloat16,
                "Q, K, and V must have dtype torch.bfloat16");
    TORCH_CHECK(tensor->dim() == 4,
                "Q, K, and V must have shape [batch, heads, sequence, head_dim]");
    TORCH_CHECK(tensor->is_contiguous(), "Q, K, and V must be contiguous");
  }
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
              "Q, K, and V must be on the same CUDA device");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
              "Q, K, and V batch dimensions must match");
  TORCH_CHECK(k.size(1) == v.size(1), "K and V head dimensions must match");
  TORCH_CHECK(q.size(1) >= k.size(1) && q.size(1) % k.size(1) == 0,
              "Q head count must be a positive multiple of the K/V head count");
  TORCH_CHECK(q.size(1) == 4 * k.size(1),
              "this GQA kernel is specialized for exactly four Q heads per K/V head");
  TORCH_CHECK(q.size(2) >= 128,
              "this prefill GQA kernel requires query length >= 128");
  TORCH_CHECK(k.size(2) == v.size(2), "K and V sequence lengths must match");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3),
              "Q, K, and V head dimensions must match");
  TORCH_CHECK(q.size(2) > 0 && k.size(2) > 0 && q.size(3) > 0,
              "sequence lengths and head_dim must be positive");
}

void check_out(const torch::Tensor& out, const torch::Tensor& q) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 &&
                  out.is_contiguous(),
              "out must be a contiguous CUDA torch.bfloat16 tensor");
  TORCH_CHECK(out.device() == q.device() && out.sizes() == q.sizes(),
              "out must have the same device and shape as Q");
}

torch::Tensor run_out(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                      torch::Tensor out, uint64_t launcher_address, bool causal,
                      double scale) {
  TORCH_CHECK(launcher_address != 0, "FA MHA launcher is null");
  check_qkv(q, k, v);
  check_out(out, q);
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
              "scale must be finite and positive");
  c10::cuda::CUDAGuard device_guard(q.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(q.device().index());
  const auto launcher =
      reinterpret_cast<FaGqaLauncher>(static_cast<uintptr_t>(launcher_address));
  C10_CUDA_CHECK(launcher(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(q.size(0)), static_cast<int>(q.size(1)), static_cast<int>(k.size(1)),
      static_cast<int>(q.size(2)), static_cast<int>(k.size(2)),
      static_cast<int>(q.size(3)), static_cast<float>(scale), causal, stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor fa_gqa(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                     uint64_t launcher, bool causal, double scale) {
  return run_out(q, k, v, torch::empty_like(q), launcher, causal, scale);
}

torch::Tensor fa_gqa_out(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         torch::Tensor out, uint64_t launcher, bool causal,
                         double scale) {
  return run_out(q, k, v, out, launcher, causal, scale);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa_gqa", &fa_gqa);
  m.def("fa_gqa_out", &fa_gqa_out);
}
