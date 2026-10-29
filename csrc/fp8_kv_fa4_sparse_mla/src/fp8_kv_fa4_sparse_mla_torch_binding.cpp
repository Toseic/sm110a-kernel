#include <cmath>
#include <cstdint>
#include <limits>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace sm110a::fp8_kv_fa4_sparse_mla_torch_binding {

using SparseMlaLauncher = cudaError_t (*)(
    const __nv_bfloat16*, const uint8_t*, const void*, const __nv_bfloat16*, bool,
    const int32_t*, const float*, __nv_bfloat16*, int, int, int, int, int, float,
    cudaStream_t);

void check_inputs(const torch::Tensor& q, const torch::Tensor& kv,
                  const torch::Tensor& scales, const torch::Tensor& rope,
                  const torch::Tensor& indices, const torch::Tensor& sink,
                  const torch::Tensor& out) {
  for (const auto* tensor : {&q, &kv, &scales, &rope, &indices, &sink, &out}) {
    TORCH_CHECK(tensor->is_cuda() && tensor->is_contiguous(),
                "all FP8 KV MLA tensors must be contiguous CUDA tensors");
    TORCH_CHECK(tensor->device() == q.device(),
                "all FP8 KV MLA tensors must be on the Q device");
  }
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
                  rope.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
              "Q, RoPE KV and output must be BF16");
  TORCH_CHECK(kv.scalar_type() == at::kFloat8_e4m3fn,
              "non-RoPE KV must be FP8 E4M3");
  TORCH_CHECK(scales.scalar_type() == at::kFloat8_e8m0fnu ||
                  scales.scalar_type() == at::kFloat,
              "KV scales must be E8M0 or FP32");
  TORCH_CHECK(indices.scalar_type() == at::kInt && sink.scalar_type() == at::kFloat,
              "indices must be int32 and sink must be FP32");
  TORCH_CHECK(q.dim() == 4 && q.size(0) > 0 && q.size(1) == 64 &&
                  q.size(2) > 0 && q.size(3) == 512,
              "Q must have shape [B,64,Q,512]");
  TORCH_CHECK(kv.dim() == 3 && kv.size(0) == q.size(0) &&
                  kv.size(1) > 0 && kv.size(2) == 448,
              "non-RoPE KV must have shape [B,N,448]");
  TORCH_CHECK(scales.dim() == 3 && scales.size(0) == q.size(0) &&
                  scales.size(1) == kv.size(1) && scales.size(2) == 7,
              "scales must have shape [B,N,7] (block size 64)");
  TORCH_CHECK(rope.dim() == 3 && rope.size(0) == q.size(0) &&
                  rope.size(1) == kv.size(1) && rope.size(2) == 64,
              "RoPE KV must have shape [B,N,64]");
  TORCH_CHECK((indices.dim() == 2 && q.size(2) == 1 &&
                  indices.size(0) == q.size(0) && indices.size(1) > 0) ||
              (indices.dim() == 3 && indices.size(0) == q.size(0) &&
                  indices.size(1) == q.size(2) && indices.size(2) > 0),
              "indices must be [B,K] for decode or [B,Q,K]");
  TORCH_CHECK(sink.dim() == 1 && sink.size(0) == 64, "sink must have shape [64]");
  TORCH_CHECK(out.sizes() == q.sizes(), "output must have the same shape as Q");
  TORCH_CHECK(q.size(0) * q.size(2) <= std::numeric_limits<int>::max() &&
                  kv.size(1) <= std::numeric_limits<int>::max() &&
                  indices.size(-1) <= std::numeric_limits<int>::max(),
              "FP8 KV MLA dimensions exceed the launcher integer range");
}

torch::Tensor sparse_mla_out(torch::Tensor q, torch::Tensor kv,
                             torch::Tensor scales, torch::Tensor rope,
                             torch::Tensor indices, torch::Tensor sink,
                             torch::Tensor out, uint64_t launcher_address,
                             double scale) {
  TORCH_CHECK(launcher_address != 0, "FP8 KV MLA launcher is null");
  check_inputs(q, kv, scales, rope, indices, sink, out);
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "scale must be finite and positive");
  c10::cuda::CUDAGuard guard(q.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(q.device().index());
  const auto launcher = reinterpret_cast<SparseMlaLauncher>(
      static_cast<uintptr_t>(launcher_address));
  C10_CUDA_CHECK(launcher(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint8_t*>(kv.data_ptr()), scales.data_ptr(),
      reinterpret_cast<const __nv_bfloat16*>(rope.data_ptr<at::BFloat16>()),
      scales.scalar_type() == at::kFloat, indices.data_ptr<int32_t>(),
      sink.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(q.size(0)), 64, static_cast<int>(q.size(2)),
      static_cast<int>(kv.size(1)), static_cast<int>(indices.size(-1)),
      static_cast<float>(scale), stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor sparse_mla(torch::Tensor q, torch::Tensor kv, torch::Tensor scales,
                         torch::Tensor rope, torch::Tensor indices, torch::Tensor sink,
                         uint64_t launcher_address, double scale) {
  return sparse_mla_out(q, kv, scales, rope, indices, sink, torch::empty_like(q),
                        launcher_address, scale);
}

} // namespace sm110a::fp8_kv_fa4_sparse_mla_torch_binding
using namespace sm110a::fp8_kv_fa4_sparse_mla_torch_binding;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_mla", &sparse_mla);
  m.def("sparse_mla_out", &sparse_mla_out);
}
