#include <cstdint>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace sm110a::bf16_moe_torch_binding {

using Fc1Launcher = void (*)(
    const __nv_bfloat16*,
    const __nv_bfloat16*,
    const __nv_bfloat16*,
    __nv_bfloat16*,
    int,
    int,
    int,
    int,
    int,
    int*,
    cudaStream_t);
using Fc1MetadataLauncher = void (*)(const __nv_bfloat16*, int, int, int, int32_t*,
    int32_t*, int32_t*, int32_t*, int, int, cudaStream_t);
using Fc1MainLauncher = void (*)(const __nv_bfloat16*, const __nv_bfloat16*, const int32_t*,
    const int32_t*, int32_t*, __nv_bfloat16*, int, int, int, int, int, cudaStream_t);

void check_bf16_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kBFloat16,
      name,
      " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_inputs(
    const torch::Tensor& a,
    const torch::Tensor& b_interleaved,
    const torch::Tensor& router_scores,
    const torch::Tensor& out,
    int64_t topk,
    int64_t tile_n,
    int64_t tile_k) {
  check_bf16_cuda_contiguous(a, "A");
  check_bf16_cuda_contiguous(b_interleaved, "B_interleaved");
  check_bf16_cuda_contiguous(out, "Output");
  TORCH_CHECK(router_scores.is_cuda(), "router_scores must be a CUDA tensor");
  TORCH_CHECK(
      router_scores.scalar_type() == torch::kBFloat16,
      "router_scores must have dtype torch.bfloat16");
  TORCH_CHECK(router_scores.is_contiguous(), "router_scores must be contiguous");
  TORCH_CHECK(a.dim() == 2, "A must be 2D");
  TORCH_CHECK(b_interleaved.dim() == 3, "B_interleaved must be 3D");
  TORCH_CHECK(router_scores.dim() == 2, "router_scores must be 2D");
  TORCH_CHECK(out.dim() == 2, "Output must be 2D");
  TORCH_CHECK(a.device() == b_interleaved.device(), "A and B must be on same device");
  TORCH_CHECK(
      a.device() == router_scores.device(),
      "A and router_scores must be on same device");
  TORCH_CHECK(a.device() == out.device(), "A and Output must be on same device");
  TORCH_CHECK(
      router_scores.size(0) == a.size(0),
      "A and router_scores token dimensions must match");
  TORCH_CHECK(
      router_scores.size(1) == b_interleaved.size(0),
      "router_scores expert dimension must match B");
  TORCH_CHECK(a.size(1) == b_interleaved.size(2), "A and B inner dimensions must match");
  TORCH_CHECK(topk > 0, "topk must be positive");
  TORCH_CHECK(topk <= router_scores.size(1), "topk must not exceed the number of experts");
  TORCH_CHECK(out.size(0) == a.size(0) * topk, "Output rows must equal tokens * topk");
  TORCH_CHECK(
      b_interleaved.size(1) == 2 * out.size(1),
      "B_interleaved width must be 2 * output width");
  TORCH_CHECK(a.size(0) > 0, "A must contain at least one token");
  TORCH_CHECK(b_interleaved.size(0) > 0, "B_interleaved must contain at least one expert");
  TORCH_CHECK(
      tile_n > 0 && out.size(1) % (tile_n / 2) == 0,
      "logical N must be divisible by tile_n / 2");
  TORCH_CHECK(tile_k > 0 && a.size(1) % tile_k == 0, "K must be divisible by tile_k");
}

torch::Tensor fc1_block_interleaved_swiglu_out(
    torch::Tensor a,
    torch::Tensor b_interleaved,
    torch::Tensor router_scores,
    int64_t topk,
    torch::Tensor out,
    uint64_t launcher_address,
    int64_t tile_n,
    int64_t tile_k) {
  TORCH_CHECK(launcher_address != 0, "fc1 block-interleaved swiglu launcher is null");
  TORCH_CHECK(tile_n % 2 == 0, "tile_n must be even");
  check_inputs(a, b_interleaved, router_scores, out, topk, tile_n, tile_k);

  c10::cuda::CUDAGuard device_guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  const auto launcher =
      reinterpret_cast<Fc1Launcher>(static_cast<uintptr_t>(launcher_address));
  launcher(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b_interleaved.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          router_scores.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b_interleaved.size(0)),
      static_cast<int>(topk),
      static_cast<int>(out.size(1)),
      static_cast<int>(a.size(1)),
      nullptr,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

py::tuple fc1_block_interleaved_swiglu_with_routing(
    torch::Tensor a, torch::Tensor b_interleaved, torch::Tensor router_scores,
    int64_t topk, uint64_t launcher_address, int64_t tile_n, int64_t tile_k) {
  TORCH_CHECK(launcher_address != 0, "fc1 block-interleaved swiglu launcher is null");
  TORCH_CHECK(tile_n % 2 == 0, "tile_n must be even");
  auto out = torch::empty({a.size(0) * topk, b_interleaved.size(1) / 2}, a.options());
  check_inputs(a, b_interleaved, router_scores, out, topk, tile_n, tile_k);
  auto routing = torch::empty({2 * b_interleaved.size(0) + 3 * out.size(0)},
                              a.options().dtype(torch::kInt32));
  c10::cuda::CUDAGuard device_guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  const auto launcher = reinterpret_cast<Fc1Launcher>(static_cast<uintptr_t>(launcher_address));
  launcher(reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b_interleaved.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(router_scores.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)), static_cast<int>(b_interleaved.size(0)),
      static_cast<int>(topk), static_cast<int>(out.size(1)), static_cast<int>(a.size(1)),
      routing.data_ptr<int32_t>(), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return py::make_tuple(out, routing);
}

torch::Tensor fc1_block_interleaved_swiglu(
    torch::Tensor a,
    torch::Tensor b_interleaved,
    torch::Tensor router_scores,
    int64_t topk,
    uint64_t launcher_address,
    int64_t tile_n,
    int64_t tile_k) {
  auto out = torch::empty({a.size(0) * topk, b_interleaved.size(1) / 2}, a.options());
  return fc1_block_interleaved_swiglu_out(
      a, b_interleaved, router_scores, topk, out, launcher_address, tile_n, tile_k);
}

py::tuple prepare_fc1_metadata(torch::Tensor router, int64_t topk, int64_t n,
    uint64_t address, int64_t tile_m, int64_t tile_n, int64_t cluster_m, int64_t cluster_n) {
  check_bf16_cuda_contiguous(router, "router_scores");
  TORCH_CHECK(router.dim() == 2 && topk > 0 && topk <= router.size(1), "invalid FC1 router input");
  TORCH_CHECK(n > 0 && n % ((tile_n / 2) * cluster_n) == 0, "N is incompatible with FC1 config");
  const int64_t tokens = router.size(0), experts = router.size(1), rows = tokens * topk;
  const int64_t max_tasks = ((rows + cluster_m * tile_m - 1) / (cluster_m * tile_m) + experts) *
      (n / ((tile_n / 2) * cluster_n));
  auto opt = router.options().dtype(torch::kInt32);
  auto routing = torch::empty({2 * experts + 3 * rows}, opt);
  auto source_rows = torch::empty({rows}, opt);
  auto tasks = torch::empty({max_tasks, 5}, opt);
  auto state = torch::empty({2}, opt);
  c10::cuda::CUDAGuard guard(router.device());
  auto launch = reinterpret_cast<Fc1MetadataLauncher>(static_cast<uintptr_t>(address));
  TORCH_CHECK(launch, "FC1 metadata launcher is null");
  launch(reinterpret_cast<const __nv_bfloat16*>(router.data_ptr<at::BFloat16>()), int(tokens), int(experts),
      int(topk), routing.data_ptr<int32_t>(), source_rows.data_ptr<int32_t>(), tasks.data_ptr<int32_t>(),
      state.data_ptr<int32_t>(), int(max_tasks), int(n), c10::cuda::getCurrentCUDAStream(router.device().index()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return py::make_tuple(routing, source_rows, tasks, state);
}

torch::Tensor fc1_main_out(torch::Tensor a, torch::Tensor b, torch::Tensor source_rows,
    torch::Tensor tasks, torch::Tensor state, torch::Tensor out, uint64_t address) {
  check_bf16_cuda_contiguous(a, "A"); check_bf16_cuda_contiguous(b, "B_interleaved"); check_bf16_cuda_contiguous(out, "Output");
  TORCH_CHECK(source_rows.is_cuda() && source_rows.scalar_type() == torch::kInt32 && source_rows.is_contiguous(), "source_rows must be CUDA int32");
  TORCH_CHECK(tasks.is_cuda() && tasks.scalar_type() == torch::kInt32 && tasks.is_contiguous() && tasks.dim() == 2 && tasks.size(1) == 5, "tasks must be CUDA int32 [N,5]");
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kInt32 && state.is_contiguous() && state.numel() == 2, "state must be CUDA int32 [2]");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 3 && out.dim() == 2 && source_rows.numel() == out.size(0) && a.size(1) == b.size(2) && b.size(1) == 2 * out.size(1), "invalid FC1 main shapes");
  TORCH_CHECK(a.device() == b.device() && a.device() == source_rows.device() && a.device() == tasks.device() && a.device() == state.device() && a.device() == out.device(), "FC1 tensors must share a device");
  c10::cuda::CUDAGuard guard(a.device()); auto launch = reinterpret_cast<Fc1MainLauncher>(static_cast<uintptr_t>(address));
  TORCH_CHECK(launch, "FC1 main launcher is null");
  launch(reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()), source_rows.data_ptr<int32_t>(), tasks.data_ptr<int32_t>(), state.data_ptr<int32_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), int(out.size(0)), int(b.size(0)), int(out.size(1)), int(a.size(1)), int(tasks.size(0)), c10::cuda::getCurrentCUDAStream(a.device().index()));
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

} // namespace sm110a::bf16_moe_torch_binding

using namespace sm110a::bf16_moe_torch_binding;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare_fc1_metadata", &prepare_fc1_metadata);
  m.def("fc1_main_out", &fc1_main_out);
}
