#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace sm110a::mxfp4_gemm_split_torch_binding {

using Mxfp4GemmLauncher = void (*)(const uint8_t*, const uint8_t*,
                                    const uint8_t*, const uint8_t*,
                                    __nv_bfloat16*, int, int, int,
                                    cudaStream_t);

void check_fp4(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat4_e2m1fn_x2,
              name, " must have dtype torch.float4_e2m1fn_x2");
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_scale(const torch::Tensor& tensor, const char* name,
                 int64_t expected_numel, const c10::Device& device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat8_e8m0fnu,
              name, " must have dtype torch.float8_e8m0fnu (UE8M0 bits)");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.device() == device, name, " must be on the same device as A");
  TORCH_CHECK(tensor.numel() == expected_numel, name, " must contain ",
              expected_numel, " blocked scale bytes, got ", tensor.numel());
}

void check_inputs(const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& sfa, const torch::Tensor& sfb,
                  int64_t tile_m, int64_t tile_n, int64_t tile_k,
                  int64_t cluster_m, int64_t split_k) {
  check_fp4(a, "A");
  check_fp4(b, "B");
  TORCH_CHECK(tile_m > 0 && tile_n > 0 && tile_k > 0 && cluster_m > 0 &&
              (split_k == 1 || split_k == 2 || split_k == 4 || split_k == 8) &&
              cluster_m * split_k <= 8,
              "invalid MXFP4 GEMM configuration");
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same device");
  const int64_t m = a.size(0);
  const int64_t n = b.size(0);
  const int64_t k = a.size(1) * 2;
  TORCH_CHECK(b.size(1) * 2 == k,
              "A and transposed-B packed K dimensions must match");
  TORCH_CHECK(m % (tile_m * cluster_m) == 0,
              "M must be a multiple of ", tile_m * cluster_m);
  TORCH_CHECK(n % tile_n == 0, "N must be a multiple of ", tile_n);
  TORCH_CHECK(n % 128 == 0,
              "N must be a multiple of 128 for blocked MXFP4 scales");
  TORCH_CHECK(k % (tile_k * split_k) == 0, "K must be a multiple of ", tile_k * split_k);
  check_scale(sfa, "SFA", m * (k / 16), a.device());
  check_scale(sfb, "SFB", n * (k / 16), a.device());
}

void check_output(const torch::Tensor& out, int64_t m, int64_t n,
                  const c10::Device& device) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16,
              "Output must be a CUDA torch.bfloat16 tensor");
  TORCH_CHECK(out.dim() == 2 && out.is_contiguous(),
              "Output must be a contiguous 2D tensor");
  TORCH_CHECK(out.device() == device, "Output must be on the same device as A");
  TORCH_CHECK(out.size(0) == m && out.size(1) == n,
              "Output must have shape (", m, ", ", n, ")");
}

torch::Tensor run_out(torch::Tensor a, torch::Tensor b, torch::Tensor sfa,
                      torch::Tensor sfb, torch::Tensor out,
                      uint64_t launcher_address, int64_t tile_m,
                      int64_t tile_n, int64_t tile_k, int64_t cluster_m,
                      int64_t split_k) {
  TORCH_CHECK(launcher_address != 0, "MXFP4 GEMM launcher is null");
  check_inputs(a, b, sfa, sfb, tile_m, tile_n, tile_k, cluster_m, split_k);
  check_output(out, a.size(0), b.size(0), a.device());
  c10::cuda::CUDAGuard device_guard(a.device());
  const auto launcher = reinterpret_cast<Mxfp4GemmLauncher>(
      static_cast<uintptr_t>(launcher_address));
  launcher(reinterpret_cast<const uint8_t*>(a.data_ptr()),
           reinterpret_cast<const uint8_t*>(b.data_ptr()),
           reinterpret_cast<const uint8_t*>(sfa.data_ptr()),
           reinterpret_cast<const uint8_t*>(sfb.data_ptr()),
           reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
           static_cast<int>(a.size(0)), static_cast<int>(b.size(0)),
           static_cast<int>(a.size(1) * 2),
           c10::cuda::getCurrentCUDAStream(a.device().index()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor mxfp4_gemm_out(torch::Tensor a, torch::Tensor b,
                             torch::Tensor sfa, torch::Tensor sfb,
                             torch::Tensor out, uint64_t launcher,
                             int64_t tile_m, int64_t tile_n, int64_t tile_k,
                             int64_t cluster_m, int64_t split_k) {
  return run_out(a, b, sfa, sfb, out, launcher, tile_m, tile_n, tile_k,
                 cluster_m, split_k);
}

torch::Tensor mxfp4_gemm(torch::Tensor a, torch::Tensor b,
                         torch::Tensor sfa, torch::Tensor sfb,
                         uint64_t launcher, int64_t tile_m, int64_t tile_n,
                         int64_t tile_k, int64_t cluster_m, int64_t split_k) {
  auto out = torch::empty({a.size(0), b.size(0)},
                          a.options().dtype(torch::kBFloat16));
  return run_out(a, b, sfa, sfb, out, launcher, tile_m, tile_n, tile_k,
                 cluster_m, split_k);
}

} // namespace sm110a::mxfp4_gemm_split_torch_binding

using namespace sm110a::mxfp4_gemm_split_torch_binding;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mxfp4_gemm", &mxfp4_gemm);
  m.def("mxfp4_gemm_out", &mxfp4_gemm_out);
}
