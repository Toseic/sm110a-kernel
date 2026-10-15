#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>
#include <cstdint>
#include <torch/extension.h>

using Launcher = void (*)(const void *, int32_t *, int, int, int, int, cudaStream_t);
torch::Tensor topk_out(
    torch::Tensor scores,
    torch::Tensor out,
    int64_t k,
    int64_t split_k,
    uint64_t address) {
    TORCH_CHECK(scores.is_cuda() && scores.dim() == 2 && scores.is_contiguous(),
        "scores must be contiguous CUDA [Qlen, KVlen]");
    TORCH_CHECK(scores.scalar_type() == at::kFloat || scores.scalar_type() == at::kBFloat16,
        "scores must be float32 or bfloat16");
    TORCH_CHECK(scores.size(0) <= INT_MAX && scores.size(1) <= INT_MAX,
        "dimensions must fit int32");
    TORCH_CHECK(k == 512 && scores.size(1) >= 512,
        "topk is fixed at 512 and KVlen must be >= 512");
    TORCH_CHECK(out.is_cuda() && out.device() == scores.device() && out.scalar_type() == at::kInt &&
                    out.is_contiguous() && out.dim() == 2 && out.size(0) == scores.size(0) &&
                    out.size(1) == k,
        "out must be contiguous CUDA int32 [Qlen, topk] on the scores device");
    TORCH_CHECK(!out.is_alias_of(scores), "out must not alias scores");
    TORCH_CHECK(split_k == 1 || split_k == 2 || split_k == 4 || split_k == 8,
        "split_k must be 1, 2, 4, or 8");
    TORCH_CHECK(scores.size(0) <= INT_MAX / split_k, "cluster grid must fit int32");
    TORCH_CHECK(address, "topk launcher is null");
    c10::cuda::CUDAGuard guard(scores.device());
    reinterpret_cast<Launcher>(address)(scores.data_ptr(),
        out.data_ptr<int32_t>(),
        scores.size(0),
        scores.size(1),
        scores.scalar_type() == at::kFloat ? 0 : 1,
        split_k,
        c10::cuda::getCurrentCUDAStream(scores.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
torch::Tensor topk(
    torch::Tensor scores,
    int64_t k,
    int64_t split_k,
    uint64_t address) {
    TORCH_CHECK(scores.dim() == 2 && k == 512 && scores.size(1) >= 512,
        "scores must be 2D and topk is fixed at 512 and KVlen must be >= 512");
    return topk_out(scores,
        torch::empty({scores.size(0), k}, scores.options().dtype(at::kInt)),
        k,
        split_k,
        address);
}
PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m) {
    m.def("topk", &topk);
    m.def("topk_out", &topk_out);
}
