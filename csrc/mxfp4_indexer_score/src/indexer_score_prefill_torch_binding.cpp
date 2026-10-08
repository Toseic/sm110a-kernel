#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

using Launcher = void (*)(const uint8_t*, const uint8_t*, const uint8_t*,
                          const uint8_t*, const half*, float*, int, int,
                          int, bool, cudaStream_t);

void check_packed(const torch::Tensor& value, const char* name,
                  const c10::Device& device) {
    TORCH_CHECK(value.is_cuda() && value.device() == device, name, " must be on the same CUDA device as Q");
    TORCH_CHECK(value.scalar_type() == at::kFloat4_e2m1fn_x2,
                name, " must have dtype torch.float4_e2m1fn_x2");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

void check_scales(const torch::Tensor& value, const char* name,
                  const c10::Device& device, int64_t bytes) {
    TORCH_CHECK(value.is_cuda() && value.device() == device, name, " must be on the same CUDA device as Q");
    TORCH_CHECK(value.scalar_type() == at::kFloat8_e8m0fnu,
                name, " must have dtype torch.float8_e8m0fnu");
    TORCH_CHECK(value.is_contiguous() && value.numel() == bytes,
                name, " must contain ", bytes, " blocked scale bytes");
}

torch::Tensor run(torch::Tensor q, torch::Tensor k,
                  torch::Tensor q_scales, torch::Tensor k_scales,
                  torch::Tensor head_weight, torch::Tensor out,
                  int cluster_m, bool causal,
                  uint64_t launcher_address) {
    TORCH_CHECK(launcher_address != 0, "indexer score launcher is null");
    TORCH_CHECK(q.is_cuda(), "Q must be a CUDA tensor");
    const auto device = q.device();
    check_packed(q, "Q", device);
    check_packed(k, "K", device);
    TORCH_CHECK(q.dim() == 3 && q.size(1) == 64 && q.size(2) == 64,
                "Q must have packed shape [Qs, 64, 64] (D=128)");
    TORCH_CHECK(k.dim() == 2 && k.size(1) == 64,
                "K must have packed shape [KVs, 64] (D=128)");
    const int64_t qs = q.size(0);
    const int64_t kvs = k.size(0);
    TORCH_CHECK(qs > 0 && kvs > 0 && qs <= INT32_MAX && kvs <= INT32_MAX,
                "Qs and KVs must be positive and fit in int32");
    TORCH_CHECK(cluster_m == 1 || cluster_m == 2, "cluster_m must be 1 or 2");
    check_scales(q_scales, "Q scales", device, ((qs + 1) / 2) * 1024);
    check_scales(k_scales, "K scales", device, ((kvs + 127) / 128) * 1024);
    TORCH_CHECK(head_weight.is_cuda() && head_weight.device() == device &&
                head_weight.scalar_type() == at::kHalf && head_weight.is_contiguous() &&
                head_weight.dim() == 2 && head_weight.size(0) == qs &&
                head_weight.size(1) == 64,
                "head_weight must be contiguous CUDA fp16 [Qs, 64]");
    TORCH_CHECK(out.is_cuda() && out.device() == device &&
                out.scalar_type() == at::kFloat && out.is_contiguous() &&
                out.dim() == 2 && out.size(0) == qs && out.size(1) == kvs,
                "out must be contiguous CUDA fp32 [Qs, KVs]");
    c10::cuda::CUDAGuard guard(device);
    const auto launcher = reinterpret_cast<Launcher>(static_cast<uintptr_t>(launcher_address));
    launcher(reinterpret_cast<const uint8_t*>(q.data_ptr()),
             reinterpret_cast<const uint8_t*>(k.data_ptr()),
             reinterpret_cast<const uint8_t*>(q_scales.data_ptr()),
             reinterpret_cast<const uint8_t*>(k_scales.data_ptr()),
             reinterpret_cast<const half*>(head_weight.data_ptr<at::Half>()),
             out.data_ptr<float>(), static_cast<int>(qs), static_cast<int>(kvs),
             cluster_m, causal,
             c10::cuda::getCurrentCUDAStream(device.index()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor indexer_score_prefill_out(
    torch::Tensor q, torch::Tensor k, torch::Tensor q_scales,
    torch::Tensor k_scales, torch::Tensor head_weight, torch::Tensor out,
    int cluster_m, bool causal,
    uint64_t launcher_address) {
    return run(q, k, q_scales, k_scales, head_weight, out, cluster_m, causal,
               launcher_address);
}

torch::Tensor indexer_score_prefill(
    torch::Tensor q, torch::Tensor k, torch::Tensor q_scales,
    torch::Tensor k_scales, torch::Tensor head_weight, int cluster_m, bool causal,
    uint64_t launcher_address) {
    auto out = torch::empty({q.size(0), k.size(0)},
                            q.options().dtype(torch::kFloat));
    return run(q, k, q_scales, k_scales, head_weight, out, cluster_m, causal,
               launcher_address);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("indexer_score_prefill", &indexer_score_prefill);
    module.def("indexer_score_prefill_out", &indexer_score_prefill_out);
}
