#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

using Launcher = void (*)(
    const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
    const int32_t*, const int32_t*, int32_t*, __nv_bfloat16*,
    const __nv_bfloat16*, int, int, int, int, int, int, int, int, cudaStream_t,
    __nv_bfloat16*);
using MetadataLauncher = void (*)(
    const int32_t*, int32_t*, int32_t*, int32_t*, int32_t*, int, int, int,
    int, int, int, bool, cudaStream_t);

void check_fp4(const torch::Tensor& x, const char* name, int dim) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat4_e2m1fn_x2 &&
                  x.dim() == dim && x.is_contiguous(),
              name, " must be contiguous CUDA float4_e2m1fn_x2");
}

void check_fp8(const torch::Tensor& x, const char* name, int dim) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat8_e4m3fn &&
                  x.dim() == dim && x.is_contiguous(),
              name, " must be contiguous CUDA float8_e4m3fn");
}

void check_scale(const torch::Tensor& x, int64_t size, const char* name,
                 const c10::Device& device) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat8_e8m0fnu &&
                  x.is_contiguous() && x.numel() == size && x.device() == device,
              name, " has invalid blocked scale storage");
}

std::vector<torch::Tensor> prepare_metadata(
    torch::Tensor layout, int64_t experts, int64_t n, uint64_t address,
    int64_t tile_m, int64_t tile_n, int64_t cluster_m) {
  TORCH_CHECK(layout.is_cuda() && layout.scalar_type() == torch::kInt32 &&
                  layout.dim() == 1 && layout.is_contiguous(),
              "grouped_layout must be contiguous CUDA int32");
  TORCH_CHECK(layout.numel() > 0 && layout.numel() % 128 == 0 &&
                  experts > 0 && n > 0 && n % tile_n == 0 && address != 0,
              "invalid MXFP4 FC2 metadata shape/config");
  const int64_t max_tasks =
      ((layout.numel() + cluster_m * tile_m - 1) /
           (cluster_m * tile_m) +
       experts) * (n / tile_n);
  auto offsets = torch::empty({experts}, layout.options());
  auto counts = torch::empty({experts}, layout.options());
  auto tasks = torch::empty({max_tasks, 5}, layout.options());
  auto state = torch::empty({2}, layout.options());
  c10::cuda::CUDAGuard guard(layout.device());
  auto stream = c10::cuda::getCurrentCUDAStream(layout.device().index());
  reinterpret_cast<MetadataLauncher>(address)(
      layout.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
      counts.data_ptr<int32_t>(), tasks.data_ptr<int32_t>(),
      state.data_ptr<int32_t>(), static_cast<int>(layout.numel()),
      static_cast<int>(experts), static_cast<int>(n), static_cast<int>(tile_m),
      static_cast<int>(tile_n), static_cast<int>(cluster_m), false, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {offsets, counts, tasks, state};
}

torch::Tensor run_out(
    torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb,
    torch::Tensor routing, torch::Tensor routing_weights,
    torch::Tensor tasks, torch::Tensor state, int64_t topk,
    torch::Tensor out, uint64_t address, int64_t max_active_clusters,
    bool deterministic, bool compact_weight_scales) {
  check_fp8(a, "A", 2);
  check_fp4(b, "B", 3);
  TORCH_CHECK(routing.is_cuda() && routing.scalar_type() == torch::kInt32 &&
                  routing.dim() == 1 && routing.is_contiguous(),
              "routing must be contiguous CUDA int32");
  TORCH_CHECK(routing_weights.is_cuda() &&
                  routing_weights.scalar_type() == torch::kBFloat16 &&
                  routing_weights.dim() == 2 && routing_weights.is_contiguous(),
              "routing_weights must be contiguous CUDA BF16 [tokens, topk]");
  TORCH_CHECK(tasks.is_cuda() && tasks.scalar_type() == torch::kInt32 &&
                  tasks.dim() == 2 && tasks.size(1) == 5 && tasks.is_contiguous(),
              "tasks must be contiguous CUDA int32 [T,5]");
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kInt32 &&
                  state.numel() == 2 && state.is_contiguous(),
              "state must be contiguous CUDA int32[2]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 &&
                  out.dim() == 2 && out.is_contiguous(),
              "output must be contiguous CUDA BF16 [tokens,N]");

  const int64_t m = a.size(0);
  const int64_t experts = b.size(0);
  const int64_t n = b.size(1);
  const int64_t k = a.size(1);
  TORCH_CHECK(address != 0 && max_active_clusters >= 0 && m % 128 == 0 && n % 128 == 0 && k % 64 == 0,
              "MXFP4 FC2 requires M/N/K divisible by 128/128/64");
  TORCH_CHECK(b.size(2) * 2 == k && a.device() == b.device(),
              "incompatible MXFP4 A/B shapes");
  TORCH_CHECK(topk > 0 && m % topk == 0 &&
                  routing_weights.size(0) == m / topk &&
                  routing_weights.size(1) == topk,
              "invalid FC2 topk/routing_weights shape");
  const int64_t tokens = m / topk;
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({tokens, n}),
              "output must have shape [tokens,N]");
  TORCH_CHECK(routing.numel() == 2 * experts + 3 * m,
              "routing has an invalid packed metadata size");
  // Blocked scales: one 512-byte tile per (128 rows, K=32 step), live byte
  // in column 0.
  check_scale(sfa, m * (k / 32) * 4, "SFA", a.device());
  check_scale(sfb, experts * n * (k / 32) * (compact_weight_scales ? 1 : 4), "SFB", a.device());
  TORCH_CHECK(a.device() == sfa.device() && a.device() == sfb.device() &&
                  a.device() == routing.device() &&
                  a.device() == routing_weights.device() &&
                  a.device() == tasks.device() && a.device() == state.device() &&
                  a.device() == out.device(),
              "all MXFP4 FC2 tensors must share a CUDA device");

  c10::cuda::CUDAGuard guard(a.device());
  auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  const auto launcher = reinterpret_cast<Launcher>(
      static_cast<uintptr_t>(address));
  // The fused epilogue indexes the metadata by grouped row to recover the
  // expanded assignment id, so it consumes grouped_assignments (offset
  // 2*experts + m), not the inverse assignment_to_grouped map.
  const auto* grouped_assignments = routing.data_ptr<int32_t>() +
                                    2 * experts + m;
  // Distinct assignments have distinct scratch rows; a second CUDA kernel
  // sums BF16 expert outputs in FP32 without nondeterministic BF16 atomics.
  auto scratch = deterministic
      ? torch::empty({m, n}, out.options()) : torch::Tensor();
  launcher(reinterpret_cast<const uint8_t*>(a.data_ptr()),
           reinterpret_cast<const uint8_t*>(b.data_ptr()),
           reinterpret_cast<const uint8_t*>(sfa.data_ptr()),
           reinterpret_cast<const uint8_t*>(sfb.data_ptr()),
           grouped_assignments, tasks.data_ptr<int32_t>(),
           state.data_ptr<int32_t>(),
           reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
           reinterpret_cast<const __nv_bfloat16*>(routing_weights.data_ptr<at::BFloat16>()),
           static_cast<int>(tokens), static_cast<int>(m), static_cast<int>(experts),
           static_cast<int>(n), static_cast<int>(k), static_cast<int>(topk),
           static_cast<int>(tasks.size(0)), static_cast<int>(max_active_clusters), stream,
           deterministic ? reinterpret_cast<__nv_bfloat16*>(scratch.data_ptr()) : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor run(
    torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb,
    torch::Tensor routing, torch::Tensor routing_weights,
    torch::Tensor tasks, torch::Tensor state, int64_t topk,
    uint64_t address, int64_t max_active_clusters, bool deterministic, bool compact_weight_scales) {
  TORCH_CHECK(topk > 0, "topk must be positive");
  auto out = torch::empty({a.size(0) / topk, b.size(1)},
                          a.options().dtype(torch::kBFloat16));
  return run_out(a, b, sfa, sfb, routing, routing_weights, tasks, state,
                 topk, out, address, max_active_clusters, deterministic, compact_weight_scales);
}

// Use the same CUDA memset operations as the serial launcher, moved ahead
// of FC1 so no ordinary stream operation breaks the PDL dependency edge.
void prepare_pdl_out(torch::Tensor out, torch::Tensor state, bool clear_output) {
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2 &&
                  out.scalar_type() == at::kBFloat16,
              "output must be a contiguous 2D CUDA BF16 tensor");
  TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.dim() == 1 &&
                  state.scalar_type() == at::kInt && state.numel() >= 2 &&
                  state.device() == out.device(),
              "state must be contiguous CUDA int32 with at least two entries on output device");
  c10::cuda::CUDAGuard guard(out.device());
  auto stream = c10::cuda::getCurrentCUDAStream(out.device().index());
  if (clear_output) {
    C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr(), 0,
                                 out.numel() * out.element_size(), stream));
  }
  C10_CUDA_CHECK(cudaMemsetAsync(state.data_ptr<int32_t>() + 1, 0,
                               sizeof(int32_t), stream));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare_pdl_out", &prepare_pdl_out);
  m.def("prepare_mxfp4_fc2_metadata", &prepare_metadata);
  m.def("mxfp4_fc2", &run);
  m.def("mxfp4_fc2_out", &run_out);
}
