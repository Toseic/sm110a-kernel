#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

using Bf16GemmLauncher = void (*)(const __nv_bfloat16*, const __nv_bfloat16*,
                                   __nv_bfloat16*, int, int, int, bool,
                                   cudaStream_t);

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_config(int64_t tile_m, int64_t tile_n, int64_t tile_k,
                  int64_t cluster_m) {
  TORCH_CHECK(tile_m > 0 && tile_n > 0 && tile_k > 0 && cluster_m > 0,
              "invalid BF16 GEMM configuration");
}

void check_inputs(const torch::Tensor& a, const torch::Tensor& b,
                  int64_t tile_m, int64_t tile_n, int64_t tile_k,
                  int64_t cluster_m, bool swap_ab) {
  check_tensor(a, "A");
  check_tensor(b, "B");
  check_config(tile_m, tile_n, tile_k, cluster_m);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "A and B must be 2D");
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same device");
  TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions must match");
  const int64_t physical_m = swap_ab ? b.size(1) : a.size(0);
  const int64_t physical_n = swap_ab ? a.size(0) : b.size(1);
  TORCH_CHECK(physical_m % (tile_m * cluster_m) == 0,
              swap_ab ? "N" : "M", " is incompatible with the kernel tile");
  TORCH_CHECK(physical_n % tile_n == 0,
              swap_ab ? "M" : "N", " is incompatible with the kernel tile");
  TORCH_CHECK(a.size(1) % tile_k == 0, "K is incompatible with the kernel tile");
}

void check_inputs_nt(const torch::Tensor& a, const torch::Tensor& b_nt,
                     int64_t tile_m, int64_t tile_n, int64_t tile_k,
                     int64_t cluster_m, bool swap_ab) {
  check_tensor(a, "A");
  check_tensor(b_nt, "B_nt");
  check_config(tile_m, tile_n, tile_k, cluster_m);
  TORCH_CHECK(a.dim() == 2 && b_nt.dim() == 2, "A and B_nt must be 2D");
  TORCH_CHECK(a.device() == b_nt.device(), "A and B_nt must be on the same device");
  TORCH_CHECK(a.size(1) == b_nt.size(1), "Inner dimensions must match");
  const int64_t physical_m = swap_ab ? b_nt.size(0) : a.size(0);
  const int64_t physical_n = swap_ab ? a.size(0) : b_nt.size(0);
  TORCH_CHECK(physical_m % (tile_m * cluster_m) == 0,
              swap_ab ? "N" : "M", " is incompatible with the kernel tile");
  TORCH_CHECK(physical_n % tile_n == 0,
              swap_ab ? "M" : "N", " is incompatible with the kernel tile");
  TORCH_CHECK(a.size(1) % tile_k == 0, "K is incompatible with the kernel tile");
}

void check_output(const torch::Tensor& out, int64_t m, int64_t n,
                  const c10::Device& device) {
  check_tensor(out, "Output");
  TORCH_CHECK(out.dim() == 2 && out.device() == device, "invalid output tensor");
  TORCH_CHECK(out.size(0) == m && out.size(1) == n, "invalid output shape");
}

torch::Tensor run_out(torch::Tensor a, torch::Tensor b, torch::Tensor out,
                      uint64_t launcher_address, int64_t tile_m,
                      int64_t tile_n, int64_t tile_k, int64_t cluster_m,
                      bool swap_ab, bool b_is_nt) {
  TORCH_CHECK(launcher_address != 0, "BF16 GEMM launcher is null");
  if (b_is_nt) {
    check_inputs_nt(a, b, tile_m, tile_n, tile_k, cluster_m, swap_ab);
    check_output(out, a.size(0), b.size(0), a.device());
  } else {
    check_inputs(a, b, tile_m, tile_n, tile_k, cluster_m, swap_ab);
    check_output(out, a.size(0), b.size(1), a.device());
  }
  c10::cuda::CUDAGuard device_guard(a.device());
  const auto launcher = reinterpret_cast<Bf16GemmLauncher>(
      static_cast<uintptr_t>(launcher_address));
  launcher(reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
           reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
           reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
           static_cast<int>(a.size(0)), static_cast<int>(b_is_nt ? b.size(0) : b.size(1)),
           static_cast<int>(a.size(1)), swap_ab,
           c10::cuda::getCurrentCUDAStream(a.device().index()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor bf16_gemm(torch::Tensor a, torch::Tensor b, uint64_t launcher,
                        int64_t tm, int64_t tn, int64_t tk, int64_t cm, bool swap) {
  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  return run_out(a, b, out, launcher, tm, tn, tk, cm, swap, false);
}
torch::Tensor bf16_gemm_out(torch::Tensor a, torch::Tensor b, torch::Tensor out,
                            uint64_t launcher, int64_t tm, int64_t tn,
                            int64_t tk, int64_t cm, bool swap) {
  return run_out(a, b, out, launcher, tm, tn, tk, cm, swap, false);
}
torch::Tensor bf16_gemm_nt(torch::Tensor a, torch::Tensor b, uint64_t launcher,
                           int64_t tm, int64_t tn, int64_t tk, int64_t cm, bool swap) {
  auto out = torch::empty({a.size(0), b.size(0)}, a.options());
  return run_out(a, b, out, launcher, tm, tn, tk, cm, swap, true);
}
torch::Tensor bf16_gemm_nt_out(torch::Tensor a, torch::Tensor b, torch::Tensor out,
                               uint64_t launcher, int64_t tm, int64_t tn,
                               int64_t tk, int64_t cm, bool swap) {
  return run_out(a, b, out, launcher, tm, tn, tk, cm, swap, true);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bf16_gemm", &bf16_gemm);
  m.def("bf16_gemm_out", &bf16_gemm_out);
  m.def("bf16_gemm_nt", &bf16_gemm_nt);
  m.def("bf16_gemm_nt_out", &bf16_gemm_nt_out);
}

