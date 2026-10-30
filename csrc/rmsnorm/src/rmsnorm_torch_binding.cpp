#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <torch/extension.h>

namespace sm110a::rmsnorm_torch_binding {
using Launcher =
    void (*)(const void *, const void *, void *, int64_t, int, int, float, cudaStream_t);

torch::Tensor rmsnorm_out(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    torch::Tensor output,
    double eps,
    int64_t d,
    uint64_t address) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() >= 1,
        "input must be a contiguous CUDA tensor with at least one dimension");
    TORCH_CHECK(input.size(-1) == d, "input last dimension must match compiled D");
    auto type = input.scalar_type();
    TORCH_CHECK(type == at::kBFloat16 || type == at::kHalf || type == at::kFloat,
        "input must be bfloat16, float16 or float32");
    if (weight) {
        TORCH_CHECK(weight->device() == input.device() &&
                        (weight->scalar_type() == type || weight->scalar_type() == at::kFloat) &&
                        weight->is_contiguous() && weight->dim() == 1 && weight->numel() == d,
            "weight must be contiguous [D] on input's device, with float32 or input dtype");
    }
    TORCH_CHECK(output.device() == input.device() && output.scalar_type() == type &&
                    output.is_contiguous() && output.sizes() == input.sizes(),
        "output must match input shape, device and dtype and be contiguous");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0 && eps <= std::numeric_limits<float>::max(),
        "eps must be finite, nonnegative and representable in float32");
    TORCH_CHECK(address, "RMSNorm launcher is null");
    if (d % (16 / input.element_size()) == 0) {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
                        (!weight || reinterpret_cast<uintptr_t>(weight->data_ptr()) % 16 == 0) &&
                        reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0,
            "vectorized RMSNorm requires 16-byte aligned tensors");
    }
    // Exact in-place input/output is safe. Reject shifted overlap and weight
    // overlap: other CTAs can still be reading those locations.
    auto overlaps = [](const torch::Tensor &a, const torch::Tensor &b) {
        auto ap = reinterpret_cast<uintptr_t>(a.data_ptr());
        auto bp = reinterpret_cast<uintptr_t>(b.data_ptr());
        return ap < bp + b.nbytes() && bp < ap + a.nbytes();
    };
    TORCH_CHECK((!weight || !overlaps(output, *weight)), "output must not overlap weight");
    TORCH_CHECK(output.data_ptr() == input.data_ptr() || !overlaps(output, input),
        "output may alias input exactly, but must not partially overlap it");
    c10::cuda::CUDAGuard guard(input.device());
    reinterpret_cast<Launcher>(address)(input.data_ptr(),
        weight ? weight->data_ptr() : nullptr,
        output.data_ptr(),
        input.numel() / d,
        type == at::kBFloat16 ? 0
        : type == at::kHalf   ? 1
                              : 2,
        weight && weight->scalar_type() == at::kFloat,
        static_cast<float>(eps),
        c10::cuda::getCurrentCUDAStream(input.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
} // namespace sm110a::rmsnorm_torch_binding

using namespace sm110a::rmsnorm_torch_binding;

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m) {
    m.def("rmsnorm_out", &rmsnorm_out);
}
