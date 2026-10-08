#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

using Launcher = void (*)(const uint8_t*, const uint8_t*, const uint8_t*,
                          const uint8_t*, const half*, float*, int, int, int,
                          cudaStream_t);

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
                  int64_t split_kv, uint64_t launcher_address) {
    TORCH_CHECK(launcher_address != 0, "indexer score decode launcher is null");
    TORCH_CHECK(q.is_cuda(), "Q must be a CUDA tensor");
    const auto device = q.device();
    check_packed(q, "Q", device);
    check_packed(k, "K", device);
    TORCH_CHECK(q.dim() == 3 && q.size(1) == 64 && q.size(2) == 64,
                "Q must have packed shape [B, 64, 64] (D=128)");
    TORCH_CHECK(k.dim() == 3 && k.size(0) == q.size(0) && k.size(2) == 64,
                "K must have packed shape [B, KVs, 64] (D=128)");
    const int64_t batch = q.size(0);
    const int64_t kvs = k.size(1);
    TORCH_CHECK(batch > 0 && kvs > 0 && batch <= 65535 && kvs <= INT32_MAX,
                "B must be in [1, 65535] and KVs must be positive and fit in int32");
    TORCH_CHECK((split_kv == 1 || split_kv == 2 || split_kv == 4 || split_kv == 8) &&
                split_kv <= (kvs + 255) / 256,
                "split_kv must be 1, 2, 4, or 8 and at most ceil(KVs/256)");
    check_scales(q_scales, "Q scales", device, batch * 1024);
    check_scales(k_scales, "K scales", device, batch * ((kvs + 127) / 128) * 1024);
    TORCH_CHECK(head_weight.is_cuda() && head_weight.device() == device &&
                head_weight.scalar_type() == at::kHalf && head_weight.is_contiguous() &&
                head_weight.dim() == 2 && head_weight.size(0) == batch &&
                head_weight.size(1) == 64,
                "head_weight must be contiguous CUDA fp16 [B, 64]");
    TORCH_CHECK(out.is_cuda() && out.device() == device &&
                out.scalar_type() == at::kFloat && out.is_contiguous() &&
                out.dim() == 2 && out.size(0) == batch && out.size(1) == kvs,
                "out must be contiguous CUDA fp32 [B, KVs]");
    c10::cuda::CUDAGuard guard(device);
    const auto launcher = reinterpret_cast<Launcher>(static_cast<uintptr_t>(launcher_address));
    launcher(reinterpret_cast<const uint8_t*>(q.data_ptr()),
             reinterpret_cast<const uint8_t*>(k.data_ptr()),
             reinterpret_cast<const uint8_t*>(q_scales.data_ptr()),
             reinterpret_cast<const uint8_t*>(k_scales.data_ptr()),
             reinterpret_cast<const half*>(head_weight.data_ptr<at::Half>()),
             out.data_ptr<float>(), static_cast<int>(batch), static_cast<int>(kvs),
             static_cast<int>(split_kv), c10::cuda::getCurrentCUDAStream(device.index()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor indexer_score_decode_out(
    torch::Tensor q, torch::Tensor k, torch::Tensor q_scales,
    torch::Tensor k_scales, torch::Tensor head_weight, torch::Tensor out,
    int64_t split_kv, uint64_t launcher_address) {
    return run(q, k, q_scales, k_scales, head_weight, out, split_kv, launcher_address);
}

torch::Tensor indexer_score_decode(
    torch::Tensor q, torch::Tensor k, torch::Tensor q_scales,
    torch::Tensor k_scales, torch::Tensor head_weight, int64_t split_kv, uint64_t launcher_address) {
    auto out = torch::empty({q.size(0), k.size(1)},
                            q.options().dtype(torch::kFloat));
    return run(q, k, q_scales, k_scales, head_weight, out, split_kv, launcher_address);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("indexer_score_decode", &indexer_score_decode);
    module.def("indexer_score_decode_out", &indexer_score_decode_out);
}
