#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace sm110a::bf16tonvfp4_quant_torch_binding {

constexpr int kRowMultiple = 128;
constexpr int kColMultiple = 64;
constexpr int kFp4Block = 16;

using Nvfp4QuantLauncher = void (*)(const __nv_bfloat16*, uint8_t*, uint8_t*,
                                    int, int, cudaStream_t);

void check_input(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
              "input must have dtype torch.bfloat16");
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous(),
              "input must be a contiguous 2D tensor");
  TORCH_CHECK(input.size(0) > 0 && input.size(0) % kRowMultiple == 0,
              "tokens must be a positive multiple of ", kRowMultiple);
  TORCH_CHECK(input.size(1) > 0 && input.size(1) % kColMultiple == 0,
              "D must be a positive multiple of ", kColMultiple);
}

void check_packed(const torch::Tensor& packed, const torch::Tensor& input) {
  TORCH_CHECK(packed.is_cuda() && packed.is_contiguous() &&
                  packed.scalar_type() == at::kFloat4_e2m1fn_x2,
              "packed must be a contiguous CUDA torch.float4_e2m1fn_x2 tensor");
  TORCH_CHECK(packed.device() == input.device() && packed.dim() == 2 &&
                  packed.size(0) == input.size(0) &&
                  packed.size(1) == input.size(1) / 2,
              "packed must be on input's device with shape [tokens, D/2]");
}

void check_scales(const torch::Tensor& scales, const torch::Tensor& input) {
  TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() &&
                  scales.scalar_type() == at::kFloat8_e4m3fn,
              "scales must be a contiguous CUDA torch.float8_e4m3fn tensor");
  TORCH_CHECK(scales.device() == input.device() && scales.dim() == 1 &&
                  scales.numel() == input.numel() / kFp4Block,
              "scales must be on input's device with ",
              input.numel() / kFp4Block, " blocked-layout scale bytes");
}

std::tuple<torch::Tensor, torch::Tensor> bf16tonvfp4_quant_out(
    torch::Tensor input, torch::Tensor packed, torch::Tensor scales,
    uint64_t launcher_address) {
  TORCH_CHECK(launcher_address != 0, "BF16-to-NVFP4 launcher is null");
  check_input(input);
  check_packed(packed, input);
  check_scales(scales, input);
  c10::cuda::CUDAGuard guard(input.device());
  const auto launcher = reinterpret_cast<Nvfp4QuantLauncher>(
      static_cast<uintptr_t>(launcher_address));
  launcher(reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
           reinterpret_cast<uint8_t*>(packed.data_ptr()),
           reinterpret_cast<uint8_t*>(scales.data_ptr()),
           static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
           c10::cuda::getCurrentCUDAStream(input.device().index()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales};
}

std::tuple<torch::Tensor, torch::Tensor> bf16tonvfp4_quant(
    torch::Tensor input, uint64_t launcher_address) {
  check_input(input);
  auto packed = torch::empty({input.size(0), input.size(1) / 2},
                             input.options().dtype(torch::kFloat4_e2m1fn_x2));
  auto scales = torch::empty(input.numel() / kFp4Block,
                             input.options().dtype(torch::kFloat8_e4m3fn));
  return bf16tonvfp4_quant_out(input, packed, scales, launcher_address);
}

} // namespace sm110a::bf16tonvfp4_quant_torch_binding

using namespace sm110a::bf16tonvfp4_quant_torch_binding;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bf16tonvfp4_quant", &bf16tonvfp4_quant);
  m.def("bf16tonvfp4_quant_out", &bf16tonvfp4_quant_out);
}
