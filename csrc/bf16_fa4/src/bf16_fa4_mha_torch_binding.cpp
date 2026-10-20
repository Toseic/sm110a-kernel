#include <cmath>
#include <cstdint>
#include <limits>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace sm110a::bf16_fa4_mha_torch_binding {

using FaMhaLauncher = cudaError_t (*)(const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, float, bool,
    const int32_t*, int, int, int, const int32_t*, int, const int32_t*, cudaStream_t);

void check_bf16(const torch::Tensor& tensor, const torch::Device& device,
                const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device &&
              tensor.scalar_type() == torch::kBFloat16 && tensor.is_contiguous(),
              name, " must be contiguous CUDA BF16 on the Q device");
}

void check_int32(const torch::Tensor& tensor, const torch::Device& device,
                 const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device &&
              tensor.scalar_type() == torch::kInt32 && tensor.is_contiguous(),
              name, " must be contiguous CUDA int32 on the Q device");
}

torch::Tensor run_out(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                      torch::Tensor out, uint64_t launcher_address, bool causal,
                      double scale, torch::Tensor block_table,
                      c10::optional<torch::Tensor> cu_seqlens_q,
                      torch::Tensor kv_seqlens, int64_t max_q_len,
                      int64_t max_kv_len) {
  TORCH_CHECK(launcher_address != 0, "FA MHA launcher is null");
  check_bf16(q, q.device(), "Q");
  check_bf16(k, q.device(), "K");
  check_bf16(v, q.device(), "V");
  check_bf16(out, q.device(), "out");
  check_int32(block_table, q.device(), "block_table");
  check_int32(kv_seqlens, q.device(), "kv_seqlens");
  TORCH_CHECK(k.dim() == 4 && k.sizes() == v.sizes() && k.size(2) > 0 &&
              k.size(2) <= 128 && 128 % k.size(2) == 0 &&
              k.size(0) > 0 && k.size(1) > 0 && k.size(3) == 128,
              "paged K/V must have matching shape [num_pages, heads, page_size, 128], "
              "where page_size divides 128");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) > 0,
              "block_table must have shape [batch, max_pages]");
  const int64_t batch = block_table.size(0);
  const int64_t heads = k.size(1);
  const bool packed_q = cu_seqlens_q.has_value();
  if (packed_q) {
    check_int32(*cu_seqlens_q, q.device(), "cu_seqlens_q");
    TORCH_CHECK(cu_seqlens_q->dim() == 1 && cu_seqlens_q->numel() == batch + 1,
                "cu_seqlens_q must have batch+1 entries");
    TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(1) == heads &&
                q.size(2) == 128, "packed Q must have shape [sum_q, heads, 128]");
    TORCH_CHECK(max_q_len > 0 && max_q_len <= std::numeric_limits<int>::max(),
                "max_q_len must be positive and fit int32");
  } else {
    TORCH_CHECK(q.dim() == 4 && q.size(0) == batch && q.size(1) == heads &&
                q.size(2) > 0 && q.size(3) == 128,
                "Q must have shape [batch, heads, q_len, 128]");
  }
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have the same shape as Q");
  TORCH_CHECK(kv_seqlens.dim() == 1 && kv_seqlens.numel() == batch,
              "kv_seqlens must have batch entries");
  TORCH_CHECK(max_kv_len > 0 && max_kv_len <= std::numeric_limits<int>::max() &&
              block_table.size(1) >= (max_kv_len + k.size(2) - 1) / k.size(2),
              "max_kv_len must be positive and covered by block_table");
  TORCH_CHECK(batch <= std::numeric_limits<int>::max() &&
              heads <= std::numeric_limits<int>::max() &&
              k.size(0) <= std::numeric_limits<int>::max() &&
              block_table.size(1) <= std::numeric_limits<int>::max(),
              "dimensions must fit int32");
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "scale must be finite and positive");

  c10::cuda::CUDAGuard device_guard(q.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(q.device().index());
  const auto launcher = reinterpret_cast<FaMhaLauncher>(
      static_cast<uintptr_t>(launcher_address));
  C10_CUDA_CHECK(launcher(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(batch), static_cast<int>(heads),
      static_cast<int>(packed_q ? max_q_len : q.size(2)),
      static_cast<int>(max_kv_len), 128, static_cast<float>(scale), causal,
      block_table.data_ptr<int32_t>(), static_cast<int>(block_table.size(1)),
      static_cast<int>(k.size(0)),
      static_cast<int>(k.size(2)),
      packed_q ? cu_seqlens_q->data_ptr<int32_t>() : nullptr,
      packed_q ? static_cast<int>(q.size(0)) : 0,
      kv_seqlens.data_ptr<int32_t>(), stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor fa_mha(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                     uint64_t launcher, bool causal, double scale,
                     torch::Tensor block_table,
                     c10::optional<torch::Tensor> cu_seqlens_q,
                     torch::Tensor kv_seqlens, int64_t max_q_len,
                     int64_t max_kv_len) {
  return run_out(q, k, v, torch::empty_like(q), launcher, causal, scale,
                 block_table, cu_seqlens_q, kv_seqlens, max_q_len, max_kv_len);
}

torch::Tensor fa_mha_out(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         torch::Tensor out, uint64_t launcher, bool causal,
                         double scale, torch::Tensor block_table,
                         c10::optional<torch::Tensor> cu_seqlens_q,
                         torch::Tensor kv_seqlens, int64_t max_q_len,
                         int64_t max_kv_len) {
  return run_out(q, k, v, out, launcher, causal, scale,
                 block_table, cu_seqlens_q, kv_seqlens, max_q_len, max_kv_len);
}

} // namespace sm110a::bf16_fa4_mha_torch_binding

using namespace sm110a::bf16_fa4_mha_torch_binding;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa_mha", &fa_mha);
  m.def("fa_mha_out", &fa_mha_out);
}
