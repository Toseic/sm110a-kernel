#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cmath>

cudaError_t bf16_fa4_mha_cuda(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, int batch, int heads, int query_len, int key_len,
    int head_dim, float scale, bool causal, cudaStream_t stream);

namespace {

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
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
                  q.size(1) == k.size(1) && q.size(1) == v.size(1),
              "Q, K, and V batch and head dimensions must match");
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

torch::Tensor fa_mha_out(torch::Tensor q, torch::Tensor k,
                                  torch::Tensor v, torch::Tensor out,
                                  bool causal, double scale) {
  check_qkv(q, k, v);
  check_out(out, q);
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
              "scale must be finite and positive");
  c10::cuda::CUDAGuard device_guard(q.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(q.device().index());
  C10_CUDA_CHECK(bf16_fa4_mha_cuda(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      q.size(0), q.size(1), q.size(2), k.size(2), q.size(3),
      static_cast<float>(scale), causal, stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor fa_mha(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              bool causal, double scale) {
  auto out = torch::empty_like(q);
  return fa_mha_out(q, k, v, out, causal, scale);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa_mha", &fa_mha, "BF16 FA MHA (CUDA)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false,
        py::arg("scale") = 1.0);
  m.def("fa_mha_out", &fa_mha_out,
        "BF16 FA MHA out (CUDA)", py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("out"), py::arg("causal") = false,
        py::arg("scale") = 1.0);
}
