#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>
#include <torch/extension.h>

namespace {
using Launcher = cudaError_t (*)(const void *,
    const float *,
    const float *,
    const float *,
    void *,
    float *,
    float *,
    float *,
    int64_t,
    int,
    float,
    float,
    cudaStream_t);

void hc_pre_out(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor base,
    torch::Tensor output,
    std::optional<torch::Tensor> post,
    std::optional<torch::Tensor> comb,
    std::optional<torch::Tensor> scratch,
    double norm_eps,
    double hc_eps,
    int64_t d,
    bool head,
    uint64_t address) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() >= 2 &&
                    input.size(-2) == 4 && input.size(-1) == d,
        "input must be contiguous CUDA [..., 4, D] matching compiled D");
    auto type = input.scalar_type();
    TORCH_CHECK(type == at::kBFloat16 || type == at::kHalf || type == at::kFloat,
        "input must be BF16, FP16 or FP32");
    auto fp32 = [&](const torch::Tensor &t) {
        return t.device() == input.device() && t.scalar_type() == at::kFloat && t.is_contiguous();
    };
    TORCH_CHECK(
        fp32(weight) && weight.dim() == 2 && weight.size(0) == (head ? 4 : 24) && weight.size(1) == 4 * d,
        "weight must be FP32 [4, 4*D] for head or [24, 4*D] for pre");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
        "HC weight must have a 16-byte aligned base address");
    TORCH_CHECK(fp32(scale) && scale.dim() == 1 && scale.numel() == (head ? 1 : 3) && fp32(base) &&
                    base.dim() == 1 && base.numel() == (head ? 4 : 24),
        "scale/base must be FP32 [1]/[4] for head or [3]/[24] for pre");
    auto yshape = input.sizes().vec();
    yshape.erase(yshape.end() - 2);
    auto pshape = input.sizes().vec();
    pshape.pop_back();
    auto cshape = pshape;
    cshape.push_back(4);
    TORCH_CHECK(output.device() == input.device() && output.scalar_type() == type &&
                    output.is_contiguous() && output.sizes().vec() == yshape,
        "output must match input dtype/device and have shape [..., D]");
    std::vector<torch::Tensor> outputs{output};
    if (head) {
        TORCH_CHECK(!post && !comb && !scratch, "head mode does not use post, comb or scratch");
    } else {
        TORCH_CHECK(post && comb && scratch, "pre mode requires post, comb and scratch");
        TORCH_CHECK(fp32(*post) && post->sizes().vec() == pshape &&
                        fp32(*comb) && comb->sizes().vec() == cshape,
            "post/comb must be FP32 [..., 4]/[..., 4, 4] on input device");
        outputs.push_back(*post);
        outputs.push_back(*comb);
    }
    for (auto eps : {norm_eps, hc_eps})
        TORCH_CHECK(std::isfinite(eps) && eps > 0 && eps <= std::numeric_limits<float>::max() &&
                        static_cast<float>(eps) > 0,
            "eps values must be positive finite FP32 values");
    TORCH_CHECK(address, "HC pre launcher is null");
    auto overlaps = [](const torch::Tensor &a, const torch::Tensor &b) {
        auto ap = reinterpret_cast<uintptr_t>(a.data_ptr());
        auto bp = reinterpret_cast<uintptr_t>(b.data_ptr());
        return a.nbytes() && b.nbytes() && ap < bp + b.nbytes() && bp < ap + a.nbytes();
    };
    for (const auto &out : outputs) {
        for (const auto &in : {input, weight, scale, base})
            TORCH_CHECK(!overlaps(out, in), "HC pre outputs must not overlap inputs");
    }
    for (size_t i = 0; i < outputs.size(); ++i)
        for (size_t j = i + 1; j < outputs.size(); ++j)
            TORCH_CHECK(!overlaps(outputs[i], outputs[j]), "HC outputs must not overlap");
    int64_t rows = input.numel() / (4 * d);
    TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "too many token rows");
    if (!head) {
        TORCH_CHECK(fp32(*scratch) && scratch->numel() == rows * ((4 * d + 511) / 512) * 32,
            "invalid HC pre scratch");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(scratch->data_ptr()) % 16 == 0,
            "HC pre scratch must be 16-byte aligned");
        for (const auto &t : {input, weight, scale, base})
            TORCH_CHECK(!overlaps(*scratch, t), "HC pre scratch must not overlap tensors");
        for (const auto &t : outputs)
            TORCH_CHECK(!overlaps(*scratch, t), "HC pre scratch must not overlap outputs");
    }
    c10::cuda::CUDAGuard guard(input.device());
    C10_CUDA_CHECK(reinterpret_cast<Launcher>(address)(input.data_ptr(),
        weight.data_ptr<float>(),
        scale.data_ptr<float>(),
        base.data_ptr<float>(),
        output.data_ptr(),
        post ? post->data_ptr<float>() : nullptr,
        comb ? comb->data_ptr<float>() : nullptr,
        scratch ? scratch->data_ptr<float>() : nullptr,
        rows,
        type == at::kBFloat16 ? 0
        : type == at::kHalf   ? 1
                              : 2,
        float(norm_eps),
        float(hc_eps),
        c10::cuda::getCurrentCUDAStream(input.get_device())));
}
} // namespace

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m) {
    m.def("hc_pre_out", &hc_pre_out);
}
