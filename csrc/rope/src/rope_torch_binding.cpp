#include "rope_params.h"
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <vector>

namespace sm110a::rope {
using Launcher = void (*)(void *, const void *, Layout, int, bool, cudaStream_t);

torch::Tensor rope_(
    torch::Tensor x,
    torch::Tensor freqs,
    bool inverse,
    int64_t d,
    uint64_t address) {
    TORCH_CHECK(x.is_cuda() && (x.dim() == 3 || x.dim() == 4),
        "x must be CUDA [B,S,D] or [B,S,H,D]");
    TORCH_CHECK(x.size(-1) == d && x.stride(-1) == 1,
        "x must have compiled rotary dimension and unit last stride");
    const auto type = x.scalar_type();
    TORCH_CHECK(type == at::kBFloat16 || type == at::kHalf || type == at::kFloat,
        "x must be bfloat16, float16 or float32");
    TORCH_CHECK(!x.is_neg(), "x must not be a lazy negative view");
    // A sorted-stride span check accepts permutations and sliced head tails,
    // while rejecting overlapping writes (including expanded tensors).
    std::vector<std::pair<int64_t, int64_t>> dimensions;
    for (int i = 0; i < x.dim(); ++i)
        if (x.size(i) > 1)
            dimensions.emplace_back(x.stride(i), x.size(i));
    std::sort(dimensions.begin(), dimensions.end());
    int64_t span = 1;
    for (auto [stride, size] : dimensions) {
        TORCH_CHECK(stride >= span, "x must have nonoverlapping positive strides");
        span += (size - 1) * stride;
    }
    TORCH_CHECK(freqs.device() == x.device() && freqs.scalar_type() == at::kComplexFloat &&
                    freqs.dim() == 2 && freqs.size(0) == x.size(1) && freqs.size(1) == d / 2 &&
                    freqs.stride(1) == 1 && freqs.stride(0) >= 0 && !freqs.is_neg(),
        "freqs must be complex64 [S,D/2] on x's device with unit last stride");
    const auto xp = reinterpret_cast<uintptr_t>(x.data_ptr());
    const auto fp = reinterpret_cast<uintptr_t>(freqs.data_ptr());
    const int64_t fspan = freqs.numel() ? (freqs.size(0) - 1) * freqs.stride(0) + d / 2 : 0;
    TORCH_CHECK(!x.numel() ||
                    !(xp < fp + fspan * freqs.element_size() && fp < xp + span * x.element_size()),
        "x must not overlap freqs");
    TORCH_CHECK(address, "RoPE launcher is null");
    c10::cuda::CUDAGuard guard(x.device());
    Layout layout{x.numel() / 2,
        x.size(1),
        x.dim() == 4 ? x.size(2) : 1,
        x.stride(0),
        x.stride(1),
        x.dim() == 4 ? x.stride(2) : 0,
        freqs.stride(0)};
    reinterpret_cast<Launcher>(address)(x.data_ptr(),
        freqs.data_ptr(),
        layout,
        type == at::kBFloat16 ? 0
        : type == at::kHalf   ? 1
                              : 2,
        inverse != freqs.is_conj(),
        c10::cuda::getCurrentCUDAStream(x.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return x;
}
} // namespace sm110a::rope

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m) {
    m.def("rope_", &sm110a::rope::rope_);
}
