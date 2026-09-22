#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {
using Launcher = void (*)(const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
    const int32_t*, const int32_t*, int32_t*, uint8_t*, uint8_t*, int, int, int, int,
    int, int, int, cudaStream_t);
using MetadataLauncher = void (*)(const int32_t*, int32_t*, int32_t*, int32_t*, int32_t*,
    int, int, int, int, int, int, bool, cudaStream_t);
using RoutingLauncher = void (*)(const __nv_bfloat16*, int, int, int, int32_t*, int32_t*,
    int32_t*, int32_t*, int32_t*, int, int, int, int, cudaStream_t);

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
  TORCH_CHECK(address != 0 && (cluster_m == 1 || cluster_m == 2),
      "invalid NVFP4 FC1 task launcher/config");
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

std::vector<torch::Tensor> prepare_routing(
    torch::Tensor router, int64_t topk, int64_t n, int64_t tile_m,
    int64_t tile_n, int64_t cluster_m, uint64_t address) {
  TORCH_CHECK(router.is_cuda() && router.scalar_type() == torch::kBFloat16 &&
      router.dim() == 2 && router.is_contiguous(),
      "router_scores must be contiguous CUDA BF16 [tokens,experts]");
  TORCH_CHECK(topk > 0 && topk <= router.size(1) && address != 0, "invalid top-k routing input");
  const int64_t tokens = router.size(0), experts = router.size(1), rows = tokens * topk;
  auto options = router.options().dtype(torch::kInt32);
  auto routing = torch::empty({2 * experts + 3 * rows}, options);
  auto source_rows = torch::empty({rows}, options);
  auto selected = torch::empty({rows}, options);
  const int64_t max_tasks =
      ((rows + cluster_m * tile_m - 1) / (cluster_m * tile_m) + experts) *
      (n / tile_n);
  auto tasks = torch::empty({max_tasks, 5}, options);
  auto state = torch::empty({2}, options);
  c10::cuda::CUDAGuard guard(router.device());
  auto stream = c10::cuda::getCurrentCUDAStream(router.device().index());
  reinterpret_cast<RoutingLauncher>(address)(
      reinterpret_cast<const __nv_bfloat16*>(router.data_ptr<at::BFloat16>()),
      tokens, experts, topk, routing.data_ptr<int32_t>(), source_rows.data_ptr<int32_t>(),
      selected.data_ptr<int32_t>(), tasks.data_ptr<int32_t>(), state.data_ptr<int32_t>(),
      n, tile_m, tile_n, cluster_m, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {routing, source_rows, tasks, state};
}

torch::Tensor run_out(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb,
    torch::Tensor source_rows, torch::Tensor tasks, torch::Tensor state,
    torch::Tensor out, torch::Tensor out_scale, uint64_t address, int64_t max_active_clusters) {
  check_fp4(a, "A", 2); check_fp4(b, "B", 3);
  const int64_t tokens = a.size(0), m = out.size(0);
  const int64_t e = b.size(0), n = b.size(1), k = a.size(1) * 2;
  TORCH_CHECK(address != 0 && max_active_clusters >= 0 && tokens % 128 == 0 && n % 128 == 0 && k % 64 == 0,
      "NVFP4 FC1 requires tokens/physical-N/K divisible by 128/128/64");
  TORCH_CHECK(b.size(2) * 2 == k && a.device() == b.device(), "incompatible A/B");
  check_scale(sfa, tokens * (k / 16), "SFA");
  check_scale(sfb, e * n * (k / 16), "SFB");
  TORCH_CHECK(tasks.is_cuda() && tasks.scalar_type() == torch::kInt32 && tasks.size(1) == 5,
      "tasks must be CUDA int32 [T,5]");
  TORCH_CHECK(source_rows.is_cuda() && source_rows.scalar_type() == torch::kInt32 &&
      source_rows.is_contiguous() && source_rows.numel() == m,
      "source_rows must be contiguous CUDA int32 [M]");
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kInt32 && state.numel() == 2,
      "state must be CUDA int32[2]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat4_e2m1fn_x2 &&
      out.is_contiguous() && out.sizes() == torch::IntArrayRef({m, n / 4}),
      "output must be contiguous float4_e2m1fn_x2 [M, physical-N/4]");
  check_scale(out_scale, m * (n / 32), "output scale");
  TORCH_CHECK(a.device() == sfa.device() && a.device() == sfb.device() &&
      a.device() == source_rows.device() && a.device() == tasks.device() &&
      a.device() == state.device() && a.device() == out.device() &&
      a.device() == out_scale.device(),
      "all NVFP4 FC1 tensors must share a CUDA device");
  c10::cuda::CUDAGuard guard(a.device());
  auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  reinterpret_cast<Launcher>(address)(reinterpret_cast<uint8_t*>(a.data_ptr()),
      reinterpret_cast<uint8_t*>(b.data_ptr()), reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      reinterpret_cast<uint8_t*>(sfb.data_ptr()), source_rows.data_ptr<int32_t>(),
      tasks.data_ptr<int32_t>(), state.data_ptr<int32_t>(),
      reinterpret_cast<uint8_t*>(out.data_ptr()),
      reinterpret_cast<uint8_t*>(out_scale.data_ptr()), tokens, m, e, n, k,
      tasks.size(0), static_cast<int>(max_active_clusters), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
std::vector<torch::Tensor> run(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb,
    torch::Tensor source_rows, torch::Tensor tasks, torch::Tensor state, uint64_t address, int64_t max_active_clusters) {
  const auto m = source_rows.numel();
  const auto logical_n = b.size(1) / 2;
  auto out = torch::empty({m, logical_n / 2},
      a.options().dtype(at::kFloat4_e2m1fn_x2));
  auto out_scale = torch::empty({m * (logical_n / 16)},
      a.options().dtype(at::kFloat8_e4m3fn));
  run_out(a,b,sfa,sfb,source_rows,tasks,state,out,out_scale,address,max_active_clusters);
  return {out, out_scale};
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare_nvfp4_fc1_tasks", &prepare);
  m.def("prepare_nvfp4_fc1_routing", &prepare_routing);
  m.def("nvfp4_fc1", &run);
  m.def("nvfp4_fc1_out", &run_out);
}
