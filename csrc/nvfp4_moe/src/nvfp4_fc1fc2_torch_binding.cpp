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
    const uint8_t*, const uint8_t*, __nv_bfloat16*, int, cudaStream_t);

void check_fp4(const torch::Tensor& tensor, const char* name, int dimensions) {
    TORCH_CHECK(tensor.is_cuda() &&
        tensor.scalar_type() == at::kFloat4_e2m1fn_x2 &&
        tensor.dim() == dimensions && tensor.is_contiguous(),
        name, " must be contiguous CUDA float4_e2m1fn_x2");
}

void check_scale(const torch::Tensor& tensor, const char* name,
                 int64_t elements, const c10::Device& device) {
    TORCH_CHECK(tensor.is_cuda() &&
        tensor.scalar_type() == at::kFloat8_e4m3fn &&
        tensor.is_contiguous() && tensor.numel() == elements &&
        tensor.device() == device,
        name, " has invalid blocked scale storage");
}

torch::Tensor run_out(
    torch::Tensor a, torch::Tensor w1, torch::Tensor sfa,
    torch::Tensor sfw1, torch::Tensor w2, torch::Tensor sfw2,
    torch::Tensor output, uint64_t address) {
    check_fp4(a, "A", 2);
    check_fp4(w1, "W1", 3);
    check_fp4(w2, "W2", 3);
    const int64_t m = a.size(0);
    TORCH_CHECK(m > 0 && m % 512 == 0 && a.size(1) == 3072,
        "A must have packed shape [M,3072] with M divisible by 512");
    TORCH_CHECK(w1.sizes() == torch::IntArrayRef({1, 4096, 3072}),
        "W1 must have packed shape [1,4096,3072]");
    TORCH_CHECK(w2.sizes() == torch::IntArrayRef({1, 6144, 1024}),
        "W2 must have packed shape [1,6144,1024]");
    check_scale(sfa, "SFA", m * 384, a.device());
    check_scale(sfw1, "SFW1", 4096 * 384, a.device());
    check_scale(sfw2, "SFW2", 6144 * 128, a.device());
    TORCH_CHECK(a.device() == w1.device() && a.device() == w2.device(),
        "all inputs must share one CUDA device");
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kBFloat16 &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({m, 6144}) &&
        output.device() == a.device(),
        "output must be contiguous CUDA BF16 [M,6144]");
    TORCH_CHECK(address != 0, "launcher must be valid");

    c10::cuda::CUDAGuard guard(a.device());
    auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
    reinterpret_cast<Launcher>(static_cast<uintptr_t>(address))(
        reinterpret_cast<const uint8_t*>(a.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfa.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfw1.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfw2.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        static_cast<int>(m), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor run(
    torch::Tensor a, torch::Tensor w1, torch::Tensor sfa,
    torch::Tensor sfw1, torch::Tensor w2, torch::Tensor sfw2,
    uint64_t address) {
    auto output = torch::empty({a.size(0), 6144},
        a.options().dtype(torch::kBFloat16));
    return run_out(a, w1, sfa, sfw1, w2, sfw2, output, address);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("nvfp4_fc1fc2", &run);
    module.def("nvfp4_fc1fc2_out", &run_out);
}
