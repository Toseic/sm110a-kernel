#include "compressor_pool_params.h"
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>
#include <torch/extension.h>
#include <vector>

namespace sm110a::compressor_pool {
using Launcher = void (*)(Params, cudaStream_t);
torch::Tensor pool_out(
    torch::Tensor kv,
    torch::Tensor score,
    torch::Tensor ape,
    torch::Tensor kv_state,
    torch::Tensor score_state,
    torch::Tensor output,
    int64_t start,
    int64_t ratio,
    uint64_t address) {
    TORCH_CHECK(ratio == 4 || ratio == 128, "ratio must be 4 or 128");
    TORCH_CHECK(start >= 0 && start < std::numeric_limits<int64_t>::max(), "invalid start_pos");
    TORCH_CHECK(kv.is_cuda() && kv.dim() == 3 && kv.scalar_type() == at::kBFloat16,
        "kv must be CUDA BF16 [B,S,W]");
    TORCH_CHECK(kv.size(1) > 0 && (start == 0 || kv.size(1) == 1), "decode requires S=1");
    const int64_t b = kv.size(0), s = kv.size(1), width = kv.size(2);
    const int64_t coeff = ratio == 4 ? 2 : 1, d = width / coeff;
    TORCH_CHECK(width % coeff == 0 && d >= 1 && d <= 32768, "invalid channel width");
    TORCH_CHECK(kv.stride(2) == 1 && kv.stride(1) >= width &&
                    kv.stride(0) >= (s - 1) * kv.stride(1) + width && !kv.is_neg(),
        "kv must have nonoverlapping positive row strides");
    TORCH_CHECK(score.device() == kv.device() && score.scalar_type() == kv.scalar_type() &&
                    score.sizes() == kv.sizes() && score.strides() == kv.strides() &&
                    !score.is_neg(),
        "score must match kv shape, dtype and strides");
    for (const auto &t : {ape, kv_state, score_state}) {
        TORCH_CHECK(t.device() == kv.device() && t.scalar_type() == at::kFloat &&
                        t.is_contiguous() && !t.is_neg(),
            "ape and states must be contiguous CUDA FP32 on kv device");
    }
    TORCH_CHECK(ape.dim() == 2 && ape.size(0) == ratio && ape.size(1) == width,
        "ape must be [ratio,W]");
    TORCH_CHECK(kv_state.dim() == 3 && kv_state.size(0) >= b && kv_state.size(1) == coeff * ratio &&
                    kv_state.size(2) == width && score_state.sizes() == kv_state.sizes(),
        "states must be [max_batch,coeff*ratio,W]");
    const int64_t groups = start == 0 ? s / ratio : ((start + 1) % ratio == 0);
    TORCH_CHECK(output.device() == kv.device() && output.scalar_type() == at::kBFloat16 &&
                    output.is_contiguous() && !output.is_neg() && output.dim() == 3 &&
                    output.size(0) == b && output.size(1) == groups && output.size(2) == d,
        "output must be contiguous CUDA BF16 [B,groups,D]");
    TORCH_CHECK(b <= 65535 && groups < 65535, "batch or group count exceeds launch limits");
    // Writers must be disjoint from every input and each other.
    std::vector<torch::Tensor> tensors{kv, score, ape, kv_state, score_state, output};
    auto span = [](const torch::Tensor &t) -> uint64_t {
        if (!t.numel())
            return 0;
        uint64_t n = 1;
        for (int i = 0; i < t.dim(); ++i)
            n += (t.size(i) - 1) * t.stride(i);
        return n * t.element_size();
    };
    for (int i = 3; i < 6; ++i)
        for (int j = 0; j < i; ++j) {
            const auto a = reinterpret_cast<uintptr_t>(tensors[i].data_ptr());
            const auto b = reinterpret_cast<uintptr_t>(tensors[j].data_ptr());
            const auto na = span(tensors[i]), nb = span(tensors[j]);
            TORCH_CHECK(!na || !nb || a >= b + nb || b >= a + na,
                "states/output must not overlap other tensors");
        }
    TORCH_CHECK(address, "null compressor pooling launcher");
    c10::cuda::CUDAGuard guard(kv.device());
    Params p{kv.data_ptr(),
        score.data_ptr(),
        ape.data_ptr<float>(),
        kv_state.data_ptr<float>(),
        score_state.data_ptr<float>(),
        output.data_ptr(),
        b,
        s,
        d,
        start,
        kv.stride(0),
        kv.stride(1)};
    reinterpret_cast<Launcher>(address)(p, c10::cuda::getCurrentCUDAStream(kv.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
} // namespace sm110a::compressor_pool
PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m) {
    m.def("pool_out", &sm110a::compressor_pool::pool_out);
}
