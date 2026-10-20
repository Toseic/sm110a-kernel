#include <cstdint>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace sm110a::bf16_grouped_m_gemm_torch_binding {

using GroupedGemmLauncher = void (*)(
    const __nv_bfloat16*, const __nv_bfloat16*, const int32_t*, int32_t*,
    __nv_bfloat16*, int, int, int, int, int, bool, cudaStream_t);
using MetadataLauncher = void (*)(
    const int32_t*, int32_t*, int32_t*, int32_t*, int32_t*, int, int, int,
    int, int, int, bool, cudaStream_t);

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_inputs(const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& tasks, const torch::Tensor& state, int64_t tile_m,
                  int64_t tile_n, int64_t tile_k, int64_t cluster_m,
                  bool swap_ab) {
  check_tensor(a, "A");
  check_tensor(b, "B");
  TORCH_CHECK(tasks.is_cuda() && tasks.scalar_type() == torch::kInt32 && tasks.is_contiguous(),
              "metadata.tasks must be a contiguous CUDA int32 tensor");
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kInt32 && state.is_contiguous(),
              "metadata.state must be a contiguous CUDA int32 tensor");
  TORCH_CHECK(a.dim() == 2, "A must be 2D");
  TORCH_CHECK(b.dim() == 3, "B must be 3D");
  TORCH_CHECK(tasks.dim() == 2 && tasks.size(1) == 5, "metadata.tasks must have shape [N, 5]");
  TORCH_CHECK(state.numel() == 2, "metadata.state must contain {task_count, next_cluster_tile/reserved}");
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same device");
  TORCH_CHECK(a.device() == tasks.device() && a.device() == state.device(),
              "A and metadata must be on the same device");
  TORCH_CHECK(a.size(1) == b.size(2),
              "A and B inner dimensions must match");
  TORCH_CHECK(a.size(0) > 0, "A must contain at least one row");
  TORCH_CHECK(b.size(0) > 0, "B must contain at least one expert");
  const int64_t logical_n_tile = swap_ab ? cluster_m * tile_m : tile_n;
  TORCH_CHECK(logical_n_tile > 0 && b.size(1) % logical_n_tile == 0,
              "N must be a multiple of ", logical_n_tile,
              " when swap_ab=", swap_ab);
  TORCH_CHECK(tile_k > 0 && a.size(1) % tile_k == 0,
              "K must be a multiple of ", tile_k);
}

void check_output(const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& out) {
  check_tensor(out, "Output");
  TORCH_CHECK(out.device() == a.device(),
              "Output must be on the same device as A");
  TORCH_CHECK(out.dim() == 2, "Output must be 2D");
  TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == b.size(1),
              "Output shape must be (", a.size(0), ", ", b.size(1), ")");
}

std::vector<torch::Tensor> prepare_metadata(torch::Tensor layout, int64_t experts,
                                            int64_t n, uint64_t launcher_address,
                                            int64_t tile_m, int64_t tile_n,
                                            int64_t cluster_m, bool swap_ab) {
  TORCH_CHECK(launcher_address != 0, "grouped metadata launcher is null");
  TORCH_CHECK(layout.is_cuda() && layout.scalar_type() == torch::kInt32 && layout.is_contiguous(),
              "grouped_layout must be a contiguous CUDA int32 tensor");
  TORCH_CHECK(layout.dim() == 1 && layout.numel() > 0 && experts > 0, "invalid grouped metadata shape");
  const int64_t logical_n_tile = swap_ab ? cluster_m * tile_m : tile_n;
  TORCH_CHECK(n > 0 && n % logical_n_tile == 0, "N is incompatible with grouped metadata config");
  const int64_t task_rows = swap_ab ? tile_n : cluster_m * tile_m;
  const int64_t max_tasks = ((layout.numel() + task_rows - 1) / task_rows + experts) *
                            (n / logical_n_tile);
  auto options = layout.options();
  auto offsets = torch::empty({experts}, options);
  auto counts = torch::empty({experts}, options);
  auto tasks = torch::empty({max_tasks, 5}, options);
  auto state = torch::empty({2}, options);
  c10::cuda::CUDAGuard device_guard(layout.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(layout.device().index());
  const auto launcher = reinterpret_cast<MetadataLauncher>(static_cast<uintptr_t>(launcher_address));
  launcher(layout.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(), counts.data_ptr<int32_t>(),
           tasks.data_ptr<int32_t>(), state.data_ptr<int32_t>(), static_cast<int>(layout.numel()),
           static_cast<int>(experts), static_cast<int>(n), static_cast<int>(tile_m),
           static_cast<int>(tile_n), static_cast<int>(cluster_m), swap_ab, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {offsets, counts, tasks, state};
}

torch::Tensor grouped_gemm_out(torch::Tensor a, torch::Tensor b,
                               torch::Tensor tasks, torch::Tensor state, torch::Tensor out,
                               uint64_t launcher_address, int64_t tile_m,
                               int64_t tile_n, int64_t tile_k,
                               int64_t cluster_m, bool swap_ab) {
  TORCH_CHECK(launcher_address != 0, "grouped GEMM launcher is null");
  check_inputs(
      a, b, tasks, state, tile_m, tile_n, tile_k, cluster_m, swap_ab);
  check_output(a, b, out);
  c10::cuda::CUDAGuard device_guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
  const auto launcher = reinterpret_cast<GroupedGemmLauncher>(
      static_cast<uintptr_t>(launcher_address));
  launcher(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      tasks.data_ptr<int32_t>(), state.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)),
      static_cast<int>(tasks.size(0)),
      swap_ab,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor grouped_gemm(torch::Tensor a, torch::Tensor b,
                           torch::Tensor tasks, torch::Tensor state, uint64_t launcher_address,
                           int64_t tile_m, int64_t tile_n, int64_t tile_k,
                           int64_t cluster_m, bool swap_ab) {
  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  return grouped_gemm_out(
      a, b, tasks, state, out, launcher_address, tile_m, tile_n, tile_k,
      cluster_m, swap_ab);
}

} // namespace sm110a::bf16_grouped_m_gemm_torch_binding

using namespace sm110a::bf16_grouped_m_gemm_torch_binding;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare_grouped_m_metadata", &prepare_metadata,
        py::arg("grouped_layout"), py::arg("experts"), py::arg("n"),
        py::arg("launcher_address"), py::arg("tile_m"), py::arg("tile_n"),
        py::arg("cluster_m"), py::arg("swap_ab") = false);
  m.def("bf16_grouped_m_gemm", &grouped_gemm,
        "JIT-compiled grouped M BF16 GEMM",
        py::arg("a"), py::arg("b"), py::arg("tasks"), py::arg("state"),
        py::arg("launcher_address"), py::arg("tile_m"),
        py::arg("tile_n"), py::arg("tile_k"), py::arg("cluster_m"),
        py::arg("swap_ab") = false);
  m.def("bf16_grouped_m_gemm_out", &grouped_gemm_out,
        "JIT-compiled grouped M BF16 GEMM with explicit output",
        py::arg("a"), py::arg("b"), py::arg("tasks"), py::arg("state"),
        py::arg("out"), py::arg("launcher_address"), py::arg("tile_m"),
        py::arg("tile_n"), py::arg("tile_k"), py::arg("cluster_m"),
        py::arg("swap_ab") = false);
}
