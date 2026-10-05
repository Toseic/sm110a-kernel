#include <cstdint>
#include <tuple>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kScaleModeUe8m0 = 0;
constexpr int kScaleModeFp32 = 1;
using QuantLauncher = void (*)(const __nv_bfloat16*, uint8_t*, uint8_t*, float*,
                               int, int, int, int, cudaStream_t);

bool valid_block_size(int64_t block_size) {
  return block_size == 64 || block_size == 128;
}

bool valid_scale_mode(int64_t scale_mode) {
  return scale_mode == kScaleModeUe8m0 || scale_mode == kScaleModeFp32;
}

void check_input(const torch::Tensor& input, int64_t block_size) {
  TORCH_CHECK(valid_block_size(block_size), "block_size must be 64 or 128");
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
              "input must have dtype torch.bfloat16");
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous(),
              "input must be a contiguous 2D tensor");
  TORCH_CHECK(input.size(0) > 0, "tokens must be positive");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) % block_size == 0,
              "D must be a positive multiple of ", block_size);
}

void check_output(const torch::Tensor& output, const torch::Tensor& scales,
                  const torch::Tensor& input, int64_t block_size,
                  int64_t scale_mode) {
  TORCH_CHECK(output.is_cuda() && output.is_contiguous() &&
                  output.scalar_type() == at::kFloat8_e4m3fn,
              "output must be a contiguous CUDA torch.float8_e4m3fn tensor");
  TORCH_CHECK(output.device() == input.device() && output.sizes() == input.sizes(),
              "output must be on input's device with shape [tokens, D]");
  const at::ScalarType expected_scales = scale_mode == kScaleModeUe8m0
      ? at::kFloat8_e8m0fnu : at::kFloat;
  TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() &&
                  scales.scalar_type() == expected_scales,
              "scales must be a contiguous CUDA ",
              (scale_mode == kScaleModeUe8m0 ? "torch.float8_e8m0fnu"
                                             : "torch.float32"),
              " tensor");
  TORCH_CHECK(scales.device() == input.device() && scales.dim() == 2 &&
                  scales.size(0) == input.size(0) &&
                  scales.size(1) == input.size(1) / block_size,
              "scales must have shape [tokens, D/", block_size, "]");
}

std::tuple<torch::Tensor, torch::Tensor> fp8_act_quant_out(
    torch::Tensor input, torch::Tensor output, torch::Tensor scales,
    int64_t block_size, int64_t scale_mode, uint64_t launcher_address) {
  TORCH_CHECK(valid_scale_mode(scale_mode), "scale_mode must be 0 (ue8m0) or 1 (fp32)");
  TORCH_CHECK(launcher_address != 0, "FP8 activation quant launcher is null");
  check_input(input, block_size);
  check_output(output, scales, input, block_size, scale_mode);
  c10::cuda::CUDAGuard guard(input.device());
  const auto launcher = reinterpret_cast<QuantLauncher>(
      static_cast<uintptr_t>(launcher_address));
  const uint8_t* ue8m0_scales =
      scale_mode == kScaleModeUe8m0
          ? reinterpret_cast<const uint8_t*>(scales.data_ptr())
          : nullptr;
  float* fp32_scales = scale_mode == kScaleModeFp32
      ? reinterpret_cast<float*>(scales.data_ptr())
      : nullptr;
  launcher(reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
           reinterpret_cast<uint8_t*>(output.data_ptr()),
           const_cast<uint8_t*>(ue8m0_scales), fp32_scales,
           static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
           static_cast<int>(block_size), static_cast<int>(scale_mode),
           c10::cuda::getCurrentCUDAStream(input.device().index()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}

std::tuple<torch::Tensor, torch::Tensor> fp8_act_quant(
    torch::Tensor input, int64_t block_size, int64_t scale_mode,
    uint64_t launcher_address) {
  check_input(input, block_size);
  auto output = torch::empty(input.sizes(), input.options().dtype(torch::kFloat8_e4m3fn));
  auto scales = torch::empty(
      {input.size(0), input.size(1) / block_size},
      input.options().dtype(scale_mode == kScaleModeUe8m0
                                ? torch::kFloat8_e8m0fnu
                                : torch::kFloat32));
  return fp8_act_quant_out(input, output, scales, block_size, scale_mode,
                           launcher_address);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_act_quant", &fp8_act_quant);
  m.def("fp8_act_quant_out", &fp8_act_quant_out);
}
