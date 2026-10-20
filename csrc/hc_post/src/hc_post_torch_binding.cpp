#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <limits>
#include <torch/extension.h>

namespace {
using Launcher = void (*)(const void *,
    const void *,
    const float *,
    const float *,
    void *,
    int64_t,
    int,
    cudaStream_t);

torch::Tensor hc_post_out(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor post,
    torch::Tensor comb,
    torch::Tensor output,
    int64_t d,
    uint64_t address) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() >= 1 && x.size(-1) == d,
        "x must be contiguous CUDA [..., D] matching compiled D");
    auto type = x.scalar_type();
    TORCH_CHECK(type == at::kBFloat16 || type == at::kHalf || type == at::kFloat,
        "x must be BF16, FP16 or FP32");
    auto rshape = x.sizes().vec();
    rshape.insert(rshape.end() - 1, 4);
    auto pshape = x.sizes().vec();
    pshape.back() = 4;
    auto cshape = pshape;
    cshape.push_back(4);
    auto matches = [&](const torch::Tensor &t, at::ScalarType dtype, const auto &shape) {
        return t.device() == x.device() && t.scalar_type() == dtype && t.is_contiguous() &&
               t.sizes().vec() == shape;
    };
    TORCH_CHECK(matches(residual, type, rshape) && matches(output, type, rshape),
        "residual/output must be contiguous [..., 4, D] matching x dtype/device");
    TORCH_CHECK(matches(post, at::kFloat, pshape) && matches(comb, at::kFloat, cshape),
        "post/comb must be contiguous FP32 [..., 4]/[..., 4, 4] on x device");
    for (const auto &t : {x, residual, post, comb}) {
        auto a = reinterpret_cast<uintptr_t>(t.data_ptr());
        auto b = reinterpret_cast<uintptr_t>(output.data_ptr());
        TORCH_CHECK(
            !t.nbytes() || !output.nbytes() || a >= b + output.nbytes() || b >= a + t.nbytes(),
            "HC post output must not overlap inputs");
    }
    auto rows = x.numel() / d;
    // Scalar fallback can have ceil(D/32) CTAs per token.
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() / ((d + 31) / 32),
        "HC post grid is too large");
    TORCH_CHECK(address, "HC post launcher is null");
    c10::cuda::CUDAGuard guard(x.device());
    reinterpret_cast<Launcher>(address)(x.data_ptr(),
        residual.data_ptr(),
        post.data_ptr<float>(),
        comb.data_ptr<float>(),
        output.data_ptr(),
        rows,
        type == at::kBFloat16 ? 0
        : type == at::kHalf   ? 1
                              : 2,
        c10::cuda::getCurrentCUDAStream(x.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
} // namespace
PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m) {
    m.def("hc_post_out", &hc_post_out);
}
