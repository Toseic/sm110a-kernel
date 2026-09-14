#include <cmath>
#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

using FaMlaLauncher = cudaError_t (*)(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, int, int, float, bool, cudaStream_t);

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
  TORCH_CHECK(k.size(1) == 1 && v.size(1) == 1,
              "MLA requires exactly one K/V head");
  TORCH_CHECK(q.size(1) == 128,
              "MLA decoding requires H=128");
  TORCH_CHECK(k.size(2) == v.size(2), "K and V sequence lengths must match");
  TORCH_CHECK(q.size(3) == 576 && k.size(3) == 576 && v.size(3) == 512,
              "MLA requires Q/K dimension 576 and V dimension 512");
  TORCH_CHECK(q.size(0) > 0 && q.size(2) == 1 && k.size(2) > 0,
              "MLA decoding requires positive batch/KV length and Q length 1");
}

void check_out(const torch::Tensor& out, const torch::Tensor& q) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 &&
                  out.is_contiguous(),
              "out must be a contiguous CUDA torch.bfloat16 tensor");
  TORCH_CHECK(out.device() == q.device() && out.dim() == 4 &&
                  out.size(0) == q.size(0) && out.size(1) == 128 &&
                  out.size(2) == 1 && out.size(3) == 512,
              "out must be on the Q device with shape [B, 128, 1, 512]");
}

torch::Tensor run_out(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                      torch::Tensor out, uint64_t launcher_address,
                      bool causal, double scale) {
  TORCH_CHECK(launcher_address != 0, "FA MLA launcher is null");
  check_qkv(q, k, v);
  check_out(out, q);
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
              "scale must be finite and positive");
  c10::cuda::CUDAGuard device_guard(q.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(q.device().index());
  const auto launcher = reinterpret_cast<FaMlaLauncher>(
      static_cast<uintptr_t>(launcher_address));
  C10_CUDA_CHECK(launcher(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
      static_cast<int>(q.size(2)), static_cast<int>(k.size(2)),
      static_cast<int>(q.size(3)), static_cast<float>(scale), causal, stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor fa_mla(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                     uint64_t launcher, bool causal, double scale) {
  check_qkv(q, k, v);
  return run_out(q, k, v, torch::empty({q.size(0), 128, 1, 512}, q.options()), launcher, causal, scale);
}

torch::Tensor fa_mla_out(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         torch::Tensor out, uint64_t launcher, bool causal,
                         double scale) {
  return run_out(q, k, v, out, launcher, causal, scale);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa_mla", &fa_mla);
  m.def("fa_mla_out", &fa_mla_out);
}

