#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {
using Fp8GemmLauncher = void (*)(const uint8_t *,
    const uint8_t *,
    const uint8_t *,
    const uint8_t *,
    __nv_bfloat16 *,
    int,
    int,
    int,
    cudaStream_t);

void check_fp8(
    const torch::Tensor &x,
    const char *name) {
    TORCH_CHECK(
        x.is_cuda() && x.scalar_type() == at::kFloat8_e4m3fn && x.dim() == 2 && x.is_contiguous(),
        name,
        " must be a contiguous CUDA torch.float8_e4m3fn tensor");
}
void check_inputs(
    const torch::Tensor &a,
    const torch::Tensor &b,
    const torch::Tensor &sfa,
    const torch::Tensor &sfb,
    int64_t tm,
    int64_t tn,
    int64_t tk,
    int64_t cm,
    int64_t sk) {
    check_fp8(a, "A");
    check_fp8(b, "B");
    TORCH_CHECK(tn == 128 && sk == 1, "FP8 FC1 requires tile_n=128 and split_k=1");
    TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && b.size(0) > 0,
        "FP8 FC1 dimensions must be positive");
    TORCH_CHECK(tm == 128 && (tn == 64 || tn == 128 || tn == 256) && (tk == 64 || tk == 128) &&
                    (cm == 1 || cm == 2) && (sk == 1 || sk == 2 || sk == 4 || sk == 8) &&
                    cm * sk <= 8,
        "invalid FP8 FC1 configuration");
    TORCH_CHECK(a.device() == b.device(), "operands must share a CUDA device");
    TORCH_CHECK(b.size(1) == a.size(1), "A and B must share the K dimension");
    TORCH_CHECK(a.size(0) % (tm * cm) == 0 && b.size(0) % tn == 0 && b.size(0) % 128 == 0 &&
                    a.size(1) % (tk * sk) == 0,
        "operand shapes are incompatible with the selected configuration");
    auto check_scales = [&](const torch::Tensor &s, int64_t rows, const char *name) {
        TORCH_CHECK(s.scalar_type() == at::kFloat8_e8m0fnu && s.is_contiguous() &&
                        s.device() == a.device() && s.numel() == rows * (a.size(1) / 8),
            name,
            " must be contiguous torch.float8_e8m0fnu in the K=32 blocked layout");
    };
    check_scales(sfa, a.size(0), "SFA");
    check_scales(sfb, b.size(0), "SFB");
}
torch::Tensor run_out(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor out,
    uint64_t address,
    int64_t tm,
    int64_t tn,
    int64_t tk,
    int64_t cm,
    int64_t sk) {
    TORCH_CHECK(address != 0, "FP8 FC1 launcher is null");
    check_inputs(a, b, sfa, sfb, tm, tn, tk, cm, sk);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 2 &&
                    out.is_contiguous() && out.device() == a.device() && out.size(0) == a.size(0) &&
                    out.size(1) == b.size(0) / 2,
        "out must be a contiguous BF16 [M,N/2] tensor on A's device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
        "out must be 16-byte aligned");
    for (const auto &tensor : {a, b, sfa, sfb}) {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
            "FC1 inputs must be 16-byte aligned");
        TORCH_CHECK(!out.is_alias_of(tensor), "out must not alias FC1 inputs");
    }
    c10::cuda::CUDAGuard guard(a.device());
    auto launch = reinterpret_cast<Fp8GemmLauncher>(static_cast<uintptr_t>(address));
    launch(reinterpret_cast<const uint8_t *>(a.data_ptr()),
        reinterpret_cast<const uint8_t *>(b.data_ptr()),
        reinterpret_cast<const uint8_t *>(sfa.data_ptr()),
        reinterpret_cast<const uint8_t *>(sfb.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
        static_cast<int>(a.size(0)),
        static_cast<int>(b.size(0)),
        static_cast<int>(a.size(1)),
        c10::cuda::getCurrentCUDAStream(a.device().index()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
torch::Tensor fp8_shared_fc1_out(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor out,
    uint64_t launcher,
    int64_t tm,
    int64_t tn,
    int64_t tk,
    int64_t cm,
    int64_t sk) {
    return run_out(a, b, sfa, sfb, out, launcher, tm, tn, tk, cm, sk);
}
torch::Tensor fp8_shared_fc1(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    uint64_t launcher,
    int64_t tm,
    int64_t tn,
    int64_t tk,
    int64_t cm,
    int64_t sk) {
    auto out = torch::empty({a.size(0), b.size(0) / 2}, a.options().dtype(torch::kBFloat16));
    return run_out(a, b, sfa, sfb, out, launcher, tm, tn, tk, cm, sk);
}
} // namespace
PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m) {
    m.def("fp8_shared_fc1", &fp8_shared_fc1);
    m.def("fp8_shared_fc1_out", &fp8_shared_fc1_out);
}
