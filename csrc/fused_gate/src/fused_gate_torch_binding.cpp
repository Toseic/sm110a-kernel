#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <limits>
#include <torch/extension.h>

using Launcher = void (*)(const float *,
    const float *,
    const void *,
    float *,
    void *,
    int64_t,
    bool,
    float,
    cudaStream_t);

void fused_gate_out(
    torch::Tensor logits,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> hash,
    torch::Tensor weights,
    torch::Tensor indices,
    double scale,
    int64_t experts,
    int64_t topk,
    uint64_t address) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous() && logits.dim() == 2 &&
                    logits.scalar_type() == at::kFloat && logits.size(1) == experts,
        "logits must be contiguous CUDA float32 [tokens, experts]");
    auto check = [&](const torch::Tensor &t) {
        TORCH_CHECK(t.device() == logits.device() && t.is_contiguous(),
            "all tensors must be contiguous on logits device");
    };
    check(weights);
    check(indices);
    TORCH_CHECK(weights.scalar_type() == at::kFloat && weights.dim() == 2 &&
                    weights.size(0) == logits.size(0) && weights.size(1) == topk &&
                    indices.sizes() == weights.sizes(),
        "outputs must have shape [tokens, topk]");
    if (bias) {
        check(*bias);
        TORCH_CHECK(
            bias->scalar_type() == at::kFloat && bias->dim() == 1 && bias->numel() == experts,
            "bias must be float32 [experts]");
    }
    if (hash) {
        check(*hash);
        TORCH_CHECK(hash->sizes() == weights.sizes() &&
                        (hash->scalar_type() == at::kInt || hash->scalar_type() == at::kLong) &&
                        indices.scalar_type() == hash->scalar_type(),
            "hash and indices must have matching int32/int64 dtype and [tokens, topk] shape");
    } else {
        TORCH_CHECK(indices.scalar_type() == at::kLong, "topk indices must be int64");
    }
    auto overlaps = [](const torch::Tensor &a, const torch::Tensor &b) {
        auto ap = reinterpret_cast<uintptr_t>(a.data_ptr());
        auto bp = reinterpret_cast<uintptr_t>(b.data_ptr());
        return a.numel() && b.numel() && ap < bp + b.nbytes() && bp < ap + a.nbytes();
    };
    for (const auto &out : {weights, indices}) {
        TORCH_CHECK(!overlaps(out, logits) && (!bias || !overlaps(out, *bias)) &&
                        (!hash || !overlaps(out, *hash)),
            "outputs must not overlap inputs");
    }
    TORCH_CHECK(!overlaps(weights, indices), "outputs must not overlap each other");
    TORCH_CHECK(
        address && std::isfinite(scale) && std::abs(scale) <= std::numeric_limits<float>::max(),
        "launcher must be non-null and scale must be finite float32");
    c10::cuda::CUDAGuard guard(logits.device());
    reinterpret_cast<Launcher>(address)(logits.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
        hash ? hash->data_ptr() : nullptr,
        weights.data_ptr<float>(),
        indices.data_ptr(),
        logits.size(0),
        indices.scalar_type() == at::kInt,
        float(scale),
        c10::cuda::getCurrentCUDAStream(logits.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m) {
    m.def("fused_gate_out", &fused_gate_out);
}
