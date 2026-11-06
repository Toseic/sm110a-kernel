#include "compressor_store_params.h"
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <optional>
#include <torch/extension.h>
#include <vector>

namespace sm110a::compressor_store {
using Launcher = void (*)(Params, cudaStream_t);
void store_out(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor frequencies,
    torch::Tensor cv,
    torch::Tensor cs,
    torch::Tensor cr,
    int64_t position,
    double eps,
    std::optional<std::vector<torch::Tensor>> packed,
    bool e8m0,
    uint64_t address) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 && input.dim() == 3 &&
                    input.size(2) == 512 && input.is_contiguous() && !input.is_neg(),
        "input must be contiguous CUDA BF16 [B,S,512]");
    const int64_t b = input.size(0), s = input.size(1);
    TORCH_CHECK(std::isfinite(eps) && eps > 0, "eps must be finite and positive");
    TORCH_CHECK(weight.device() == input.device() && weight.scalar_type() == at::kFloat &&
                    weight.dim() == 1 && weight.size(0) == 512 && weight.is_contiguous() &&
                    !weight.is_neg(),
        "weight must be contiguous FP32 [512] on input device");
    TORCH_CHECK(
        frequencies.device() == input.device() && frequencies.scalar_type() == at::kComplexFloat &&
            frequencies.dim() == 2 && frequencies.size(0) == s && frequencies.size(1) == 32 &&
            frequencies.stride(1) == 1 && frequencies.stride(0) >= 0 && !frequencies.is_neg(),
        "frequencies must be complex64 [S,32] with unit last stride");
    const auto scale_type = e8m0 ? at::ScalarType::Float8_e8m0fnu : at::kFloat;
    auto check = [&](const torch::Tensor &t,
                     int64_t batches,
                     int64_t rows,
                     int64_t d,
                     at::ScalarType dtype) {
        TORCH_CHECK(t.device() == input.device() && t.scalar_type() == dtype && t.dim() == 3 &&
                        t.size(0) == batches && t.size(1) == rows && t.size(2) == d &&
                        t.is_contiguous() && !t.is_neg(),
            "invalid cache/packed tensor shape, dtype or layout");
    };
    TORCH_CHECK(cv.dim() == 3 && cv.size(0) >= b, "cache batch capacity too small");
    const int64_t capacity = cv.size(1);
    TORCH_CHECK(position >= 0 && position <= capacity && s <= capacity - position,
        "cache position out of range");
    check(cv, cv.size(0), capacity, 448, at::ScalarType::Float8_e4m3fn);
    check(cs, cv.size(0), capacity, 7, scale_type);
    check(cr, cv.size(0), capacity, 64, at::kBFloat16);
    std::vector<torch::Tensor> tensors{input, weight, frequencies, cv, cs, cr};
    if (packed) {
        TORCH_CHECK(packed->size() == 3, "packed must contain values, scales, RoPE");
        check((*packed)[0], b, s, 448, at::ScalarType::Float8_e4m3fn);
        check((*packed)[1], b, s, 7, scale_type);
        check((*packed)[2], b, s, 64, at::kBFloat16);
        tensors.insert(tensors.end(), packed->begin(), packed->end());
    }
    auto span = [](const torch::Tensor &t) -> uint64_t {
        if (!t.numel())
            return 0;
        uint64_t n = 1;
        for (int i = 0; i < t.dim(); ++i)
            n += (t.size(i) - 1) * t.stride(i);
        return n * t.element_size();
    };
    for (size_t i = 3; i < tensors.size(); ++i)
        for (size_t j = 0; j < i; ++j) {
            const auto a = reinterpret_cast<uintptr_t>(tensors[i].data_ptr());
            const auto b = reinterpret_cast<uintptr_t>(tensors[j].data_ptr());
            const auto na = span(tensors[i]), nb = span(tensors[j]);
            TORCH_CHECK(!na || !nb || a >= b + nb || b >= a + na,
                "cache/packed outputs must not overlap inputs or other outputs");
        }
    TORCH_CHECK(input.numel() / 512 <= 0x7fffffffLL, "too many rows");
    TORCH_CHECK(address, "null compressor store launcher");
    c10::cuda::CUDAGuard guard(input.device());
    Params p{input.data_ptr(),
        frequencies.data_ptr(),
        weight.data_ptr<float>(),
        cv.data_ptr(),
        cs.data_ptr(),
        cr.data_ptr(),
        packed ? (*packed)[0].data_ptr() : nullptr,
        packed ? (*packed)[1].data_ptr() : nullptr,
        packed ? (*packed)[2].data_ptr() : nullptr,
        b * s,
        s,
        capacity,
        position,
        frequencies.stride(0),
        float(eps),
        frequencies.is_conj()};
    reinterpret_cast<Launcher>(address)(p, c10::cuda::getCurrentCUDAStream(input.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace sm110a::compressor_store
PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m) {
    m.def("store_out", &sm110a::compressor_store::store_out);
}
