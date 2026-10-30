#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <limits>
#include <torch/extension.h>
#include <vector>

namespace sm110a::hc_post_pre_norm_torch_binding {
using Launcher = cudaError_t (*)(const void *,
    const void *,
    const float *,
    const float *,
    const float *,
    const float *,
    const float *,
    const float *,
    void *,
    void *,
    float *,
    float *,
    float *,
    int64_t,
    float,
    float,
    float,
    cudaStream_t);

void hc_post_pre_norm_out(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor prev_post,
    torch::Tensor prev_comb,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor base,
    torch::Tensor norm_weight,
    torch::Tensor mixed,
    torch::Tensor output,
    torch::Tensor post,
    torch::Tensor comb,
    torch::Tensor scratch,
    double norm_eps,
    double hc_eps,
    double rms_eps,
    int64_t d,
    uint64_t address) {
    TORCH_CHECK(d >= 256 && d <= 4096 && d % 32 == 0, "D must be a multiple of 32 in [256, 4096]");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 &&
                    x.dim() >= 1 && x.size(-1) == d,
        "x must be contiguous CUDA BF16 [..., D]");
    auto bf16 = [&](const torch::Tensor &t) {
        return t.device() == x.device() && t.scalar_type() == at::kBFloat16 && t.is_contiguous();
    };
    auto fp32 = [&](const torch::Tensor &t) {
        return t.device() == x.device() && t.scalar_type() == at::kFloat && t.is_contiguous();
    };
    auto rshape = x.sizes().vec();
    rshape.insert(rshape.end() - 1, 4);
    auto pshape = x.sizes().vec();
    pshape.back() = 4;
    auto cshape = pshape;
    cshape.push_back(4);
    TORCH_CHECK(bf16(residual) && residual.sizes().vec() == rshape,
        "residual must be BF16 [..., 4, D] matching x");
    TORCH_CHECK(fp32(prev_post) && prev_post.sizes().vec() == pshape && fp32(prev_comb) &&
                    prev_comb.sizes().vec() == cshape,
        "invalid previous post/comb");
    TORCH_CHECK(
        fp32(weight) && weight.dim() == 2 && weight.size(0) == 24 && weight.size(1) == 4 * d,
        "weight must be FP32 [24, 4*D]");
    TORCH_CHECK(fp32(scale) && scale.dim() == 1 && scale.numel() == 3 && fp32(base) &&
                    base.dim() == 1 && base.numel() == 24,
        "scale/base must be FP32 [3]/[24]");
    TORCH_CHECK(fp32(norm_weight) && norm_weight.dim() == 1 && norm_weight.numel() == d,
        "norm_weight must be FP32 [D]");
    TORCH_CHECK(
        bf16(mixed) && mixed.sizes().vec() == rshape && bf16(output) && output.sizes() == x.sizes(),
        "invalid mixed/output shape or dtype");
    TORCH_CHECK(
        fp32(post) && post.sizes().vec() == pshape && fp32(comb) && comb.sizes().vec() == cshape,
        "invalid output post/comb");
    int64_t rows = x.numel() / d;
    TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "too many token rows");
    TORCH_CHECK(fp32(scratch) && scratch.numel() == rows * ((4 * d + 511) / 512) * 32,
        "invalid scratch size or dtype");
    for (const auto &t : {x, residual, weight, mixed, output, scratch})
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
            "x/residual/weight/mixed/output/scratch must be 16-byte aligned");
    for (auto eps : {norm_eps, hc_eps, rms_eps})
        TORCH_CHECK(std::isfinite(eps) && eps > 0 && eps <= std::numeric_limits<float>::max() &&
                        static_cast<float>(eps) > 0,
            "eps values must be positive finite FP32 values");
    auto overlaps = [](const torch::Tensor &a, const torch::Tensor &b) {
        auto ap = reinterpret_cast<uintptr_t>(a.data_ptr());
        auto bp = reinterpret_cast<uintptr_t>(b.data_ptr());
        return a.nbytes() && b.nbytes() && ap < bp + b.nbytes() && bp < ap + a.nbytes();
    };
    const std::vector<torch::Tensor>
        inputs{x, residual, prev_post, prev_comb, weight, scale, base, norm_weight};
    const std::vector<torch::Tensor> outputs{mixed, output, post, comb, scratch};
    for (size_t i = 0; i < outputs.size(); ++i) {
        for (const auto &in : inputs)
            TORCH_CHECK(!overlaps(outputs[i], in), "outputs/scratch must not overlap inputs");
        for (size_t j = i + 1; j < outputs.size(); ++j)
            TORCH_CHECK(!overlaps(outputs[i], outputs[j]), "outputs/scratch must not overlap");
    }
    TORCH_CHECK(address, "HC post-pre-norm launcher is null");
    c10::cuda::CUDAGuard guard(x.device());
    C10_CUDA_CHECK(reinterpret_cast<Launcher>(address)(x.data_ptr(),
        residual.data_ptr(),
        prev_post.data_ptr<float>(),
        prev_comb.data_ptr<float>(),
        weight.data_ptr<float>(),
        scale.data_ptr<float>(),
        base.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        mixed.data_ptr(),
        output.data_ptr(),
        post.data_ptr<float>(),
        comb.data_ptr<float>(),
        scratch.data_ptr<float>(),
        rows,
        float(norm_eps),
        float(hc_eps),
        float(rms_eps),
        c10::cuda::getCurrentCUDAStream(x.get_device())));
}
} // namespace sm110a::hc_post_pre_norm_torch_binding

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m) {
    m.def("hc_post_pre_norm_out", &sm110a::hc_post_pre_norm_torch_binding::hc_post_pre_norm_out);
}
