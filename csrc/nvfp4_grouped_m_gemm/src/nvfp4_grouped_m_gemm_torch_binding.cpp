#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace sm110a::nvfp4_grouped_m_gemm_torch_binding {
using Launcher = void (*)(const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
    const int32_t*, int32_t*, __nv_bfloat16*, int, int, int, int, int, cudaStream_t);
using MetadataLauncher = void (*)(const int32_t*, int32_t*, int32_t*, int32_t*, int32_t*,
    int, int, int, int, int, int, bool, cudaStream_t);

void check_fp4(const torch::Tensor& x, const char* name, int dim) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat4_e2m1fn_x2 &&
      x.dim() == dim && x.is_contiguous(), name, " must be contiguous CUDA float4_e2m1fn_x2");
}
void check_scale(const torch::Tensor& x, int64_t size, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat8_e4m3fn &&
      x.is_contiguous() && x.numel() == size, name, " has invalid blocked scale storage");
}

std::vector<torch::Tensor> prepare(torch::Tensor layout, int64_t experts, int64_t n,
    uint64_t address, int64_t tile_m, int64_t tile_n, int64_t cluster_m) {
  TORCH_CHECK(layout.is_cuda() && layout.scalar_type() == torch::kInt32 &&
      layout.dim() == 1 && layout.is_contiguous(), "layout must be contiguous CUDA int32");
  TORCH_CHECK(layout.numel() % 128 == 0 && n % tile_n == 0,
      "M and N must satisfy NVFP4 blocked layout constraints");
  const int64_t max_tasks = ((layout.numel() + cluster_m * tile_m - 1) /
      (cluster_m * tile_m) + experts) * (n / tile_n);
  auto offsets = torch::empty({experts}, layout.options());
  auto counts = torch::empty({experts}, layout.options());
  auto tasks = torch::empty({max_tasks, 5}, layout.options());
  auto state = torch::empty({2}, layout.options());
  c10::cuda::CUDAGuard guard(layout.device());
  auto stream = c10::cuda::getCurrentCUDAStream(layout.device().index());
  reinterpret_cast<MetadataLauncher>(address)(layout.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
      counts.data_ptr<int32_t>(), tasks.data_ptr<int32_t>(), state.data_ptr<int32_t>(),
      layout.numel(), experts, n, tile_m, tile_n, cluster_m, false, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {offsets, counts, tasks, state};
}

torch::Tensor run_out(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb,
    torch::Tensor tasks, torch::Tensor state, torch::Tensor out, uint64_t address) {
  check_fp4(a, "A", 2); check_fp4(b, "B", 3);
  const int64_t m = a.size(0), e = b.size(0), n = b.size(1), k = a.size(1) * 2;
  TORCH_CHECK(b.size(2) * 2 == k && a.device() == b.device(), "incompatible A/B");
  check_scale(sfa, m * (k / 16), "SFA");
  check_scale(sfb, e * n * (k / 16), "SFB");
  TORCH_CHECK(tasks.is_cuda() && tasks.scalar_type() == torch::kInt32 && tasks.size(1) == 5,
      "tasks must be CUDA int32 [T,5]");
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kInt32 && state.numel() == 2,
      "state must be CUDA int32[2]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.is_contiguous() &&
      out.sizes() == torch::IntArrayRef({m, n}), "output must be contiguous BF16 [M,N]");
  c10::cuda::CUDAGuard guard(a.device());
  auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  reinterpret_cast<Launcher>(address)(reinterpret_cast<uint8_t*>(a.data_ptr()),
      reinterpret_cast<uint8_t*>(b.data_ptr()), reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      reinterpret_cast<uint8_t*>(sfb.data_ptr()), tasks.data_ptr<int32_t>(), state.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), m, e, n, k, tasks.size(0), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
torch::Tensor run(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb,
    torch::Tensor tasks, torch::Tensor state, uint64_t address) {
  auto out = torch::empty({a.size(0), b.size(1)}, a.options().dtype(torch::kBFloat16));
  return run_out(a,b,sfa,sfb,tasks,state,out,address);
}
} // namespace sm110a::nvfp4_grouped_m_gemm_torch_binding

using namespace sm110a::nvfp4_grouped_m_gemm_torch_binding;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare_grouped_m_metadata", &prepare);
  m.def("nvfp4_grouped_m_gemm", &run);
  m.def("nvfp4_grouped_m_gemm_out", &run_out);
}
