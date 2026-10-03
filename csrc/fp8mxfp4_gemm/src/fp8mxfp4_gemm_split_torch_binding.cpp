#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {
using Fp8Mxfp4GemmLauncher = void (*)(const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, __nv_bfloat16*, int, int, int, cudaStream_t);

void check_fp8(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat8_e4m3fn && x.dim() == 2 && x.is_contiguous(), name, " must be a contiguous CUDA torch.float8_e4m3fn tensor");
}
void check_fp4(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kByte && x.dim() == 2 && x.is_contiguous(), "B must be a contiguous CUDA torch.uint8 tensor");
}
void check_inputs(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfb, int64_t tm, int64_t tn, int64_t tk, int64_t cm, int64_t sk, int64_t compact_b_gmem) {
  check_fp8(a, "A"); check_fp4(b);
  TORCH_CHECK(tm == 128 && (tn == 64 || tn == 128 || tn == 256) && (tk == 64 || tk == 128) && (cm == 1 || cm == 2) && (sk == 1 || sk == 2 || sk == 4 || sk == 8) && cm * sk <= 8, "invalid FP8xMXFP4 GEMM configuration");
  TORCH_CHECK(a.device() == b.device() && sfb.device() == a.device(), "operands must share a CUDA device");
  TORCH_CHECK(compact_b_gmem == 0 || compact_b_gmem == 1, "compact_b_gmem must be 0 or 1");
  TORCH_CHECK(b.size(1) == (compact_b_gmem ? a.size(1) / 2 : a.size(1)), compact_b_gmem ? "compact B must have shape [N,K/2]" : "padded B must have shape [N,K]");
  if (compact_b_gmem) {
    TORCH_CHECK(tk == 128, "compact B requires tile_k=128 for TMA b4x16_p64");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(b.data_ptr()) % 32 == 0, "compact B must be 32-byte aligned for TMA b4x16_p64");
  }
  TORCH_CHECK(a.size(0) % (tm * cm) == 0 && b.size(0) % tn == 0 && b.size(0) % 128 == 0 && a.size(1) % (tk * sk) == 0, "operand shapes are incompatible with the selected configuration");
  TORCH_CHECK(sfb.scalar_type() == at::kFloat8_e8m0fnu && sfb.is_contiguous() && sfb.numel() == b.size(0) * (a.size(1) / 8), "SFB must be contiguous torch.float8_e8m0fnu in K=32 blocked layout");
}
torch::Tensor run_out(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor out, uint64_t address, int64_t tm, int64_t tn, int64_t tk, int64_t cm, int64_t sk, int64_t compact_b_gmem) {
  TORCH_CHECK(address != 0, "FP8xMXFP4 GEMM launcher is null"); check_inputs(a, b, sfb, tm, tn, tk, cm, sk, compact_b_gmem);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.is_contiguous() && out.device() == a.device() && out.size(0) == a.size(0) && out.size(1) == b.size(0), "out must be a contiguous BF16 [M,N] tensor on A's device");
  c10::cuda::CUDAGuard guard(a.device());
  auto launch = reinterpret_cast<Fp8Mxfp4GemmLauncher>(static_cast<uintptr_t>(address));
  TORCH_CHECK(sfa.scalar_type() == at::kFloat8_e8m0fnu && sfa.is_contiguous() && sfa.device() == a.device() && sfa.numel() == a.size(0) * (a.size(1) / 8), "internal SFA has an invalid K=32 blocked layout");
  launch(reinterpret_cast<const uint8_t*>(a.data_ptr()), reinterpret_cast<const uint8_t*>(b.data_ptr()), reinterpret_cast<const uint8_t*>(sfa.data_ptr()), reinterpret_cast<const uint8_t*>(sfb.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), static_cast<int>(a.size(0)), static_cast<int>(b.size(0)), static_cast<int>(a.size(1)), c10::cuda::getCurrentCUDAStream(a.device().index()));
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
torch::Tensor fp8mxfp4_gemm_out(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor out, uint64_t launcher, int64_t tm, int64_t tn, int64_t tk, int64_t cm, int64_t sk, int64_t compact_b_gmem) { return run_out(a,b,sfa,sfb,out,launcher,tm,tn,tk,cm,sk,compact_b_gmem); }
torch::Tensor fp8mxfp4_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, uint64_t launcher, int64_t tm, int64_t tn, int64_t tk, int64_t cm, int64_t sk, int64_t compact_b_gmem) { auto out=torch::empty({a.size(0),b.size(0)},a.options().dtype(torch::kBFloat16)); return run_out(a,b,sfa,sfb,out,launcher,tm,tn,tk,cm,sk,compact_b_gmem); }
}  // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fp8mxfp4_gemm", &fp8mxfp4_gemm); m.def("fp8mxfp4_gemm_out", &fp8mxfp4_gemm_out); }
