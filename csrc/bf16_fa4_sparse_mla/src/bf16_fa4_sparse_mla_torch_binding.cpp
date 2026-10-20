#include <cmath>
#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace sm110a::bf16_fa4_sparse_mla_torch_binding {

using SparseMlaLauncher = cudaError_t (*)(
    const __nv_bfloat16*, const __nv_bfloat16*, const int32_t*,
    const float*, __nv_bfloat16*, int, int, int, int, int, float,
    cudaStream_t);

void check_inputs(const torch::Tensor& q, const torch::Tensor& kv,
                  const torch::Tensor& indices, const torch::Tensor& sink,
                  const torch::Tensor& out) {
  for (const auto* tensor : {&q, &kv, &indices, &sink, &out}) {
    TORCH_CHECK(tensor->is_cuda() && tensor->is_contiguous(),
                "all sparse MLA tensors must be contiguous CUDA tensors");
    TORCH_CHECK(tensor->device() == q.device(),
                "all sparse MLA tensors must be on the Q device");
  }
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
                  kv.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16,
              "Q, KV and output must be BF16");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32 &&
                  sink.scalar_type() == torch::kFloat32,
              "indices must be int32 and attention sink must be float32");
  TORCH_CHECK(q.dim() == 4 && q.size(0) > 0 && q.size(1) == 64 &&
                  q.size(2) == 1 && q.size(3) == 512,
              "Q must have shape [B, 64, 1, 512]");
  TORCH_CHECK(kv.dim() == 3 && kv.size(0) == q.size(0) &&
                  kv.size(1) > 0 && kv.size(2) == 512,
              "KV must have shape [B, N, 512]");
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) == q.size(0) &&
                  indices.size(1) > 0,
              "indices must have shape [B, K]");
  TORCH_CHECK(sink.dim() == 1 && sink.size(0) == 64,
              "attention sink must have shape [64]");
  TORCH_CHECK(out.sizes() == q.sizes(),
              "output must have shape [B, 64, 1, 512]");
}

torch::Tensor sparse_mla_out(torch::Tensor q, torch::Tensor kv,
                             torch::Tensor indices, torch::Tensor sink,
                             torch::Tensor out, uint64_t launcher_address,
                             double scale) {
  TORCH_CHECK(launcher_address != 0, "sparse MLA launcher is null");
  check_inputs(q, kv, indices, sink, out);
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
              "scale must be finite and positive");
  c10::cuda::CUDAGuard guard(q.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(q.device().index());
  const auto launcher = reinterpret_cast<SparseMlaLauncher>(
      static_cast<uintptr_t>(launcher_address));
  C10_CUDA_CHECK(launcher(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr<at::BFloat16>()),
      indices.data_ptr<int32_t>(), sink.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(q.size(0)), 64, 1, static_cast<int>(kv.size(1)),
      static_cast<int>(indices.size(1)), static_cast<float>(scale), stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor sparse_mla(torch::Tensor q, torch::Tensor kv,
                         torch::Tensor indices, torch::Tensor sink,
                         uint64_t launcher_address, double scale) {
  TORCH_CHECK(q.dim() == 4 && q.size(0) > 0, "Q must be a four-dimensional tensor");
  return sparse_mla_out(q, kv, indices, sink, torch::empty_like(q),
                        launcher_address, scale);
}

} // namespace sm110a::bf16_fa4_sparse_mla_torch_binding

using namespace sm110a::bf16_fa4_sparse_mla_torch_binding;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_mla", &sparse_mla);
  m.def("sparse_mla_out", &sparse_mla_out);
}
