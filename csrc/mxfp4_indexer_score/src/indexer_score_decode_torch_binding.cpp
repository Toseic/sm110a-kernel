#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace sm110a::indexer_score_decode_torch_binding {

using Launcher = void (*)(const uint8_t*, const uint8_t*, const uint8_t*,
                          const uint8_t*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int,
                          int, const int32_t*, int, const int32_t*, int,
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

void check_int32(const torch::Tensor& value, const char* name,
                 const c10::Device& device) {
    TORCH_CHECK(value.is_cuda() && value.device() == device &&
                value.scalar_type() == torch::kInt32 && value.is_contiguous(),
                name, " must be contiguous CUDA int32 on the Q device");
}

torch::Tensor run(torch::Tensor q, torch::Tensor k,
                  torch::Tensor q_scales, torch::Tensor k_scales,
                  torch::Tensor head_weight, torch::Tensor out,
                  int64_t split_kv, c10::optional<torch::Tensor> block_table,
                  c10::optional<torch::Tensor> kv_seqlens, int64_t max_kv_len,
                  uint64_t launcher_address) {
    TORCH_CHECK(launcher_address != 0, "indexer score decode launcher is null");
    TORCH_CHECK(q.is_cuda(), "Q must be a CUDA tensor");
    const auto device = q.device();
    check_packed(q, "Q", device);
    check_packed(k, "K", device);
    TORCH_CHECK(q.dim() == 3 && q.size(1) == 64 && q.size(2) == 64,
                "Q must have packed shape [B, 64, 64] (D=128)");
    const int64_t batch = q.size(0);
    const bool paged = block_table.has_value();
    int64_t page_size = 0;
    int64_t num_pages = 0;
    int64_t table_stride = 0;
    if (paged) {
        check_int32(*block_table, "block_table", device);
        TORCH_CHECK(block_table->dim() == 2 && block_table->size(0) == batch,
                    "block_table must have shape [B, max_pages]");
        table_stride = block_table->size(1);
        TORCH_CHECK(k.dim() == 3 && k.size(0) > 0 && k.size(2) == 64 &&
                    (k.size(1) == 128 || k.size(1) == 256),
                    "paged K must have shape [num_pages, page_size, 64]");
        page_size = k.size(1);
        num_pages = k.size(0);
        TORCH_CHECK(kv_seqlens.has_value(), "paged K requires kv_seqlens");
    } else {
        TORCH_CHECK(k.dim() == 3 && k.size(0) == batch && k.size(2) == 64,
                    "dense K must have packed shape [B, KVs, 64]");
    }
    const int64_t kvs = paged ? max_kv_len : k.size(1);
    TORCH_CHECK(batch > 0 && kvs > 0 && batch <= 65535 && kvs <= INT32_MAX,
                "B must be in [1, 65535] and KVs must be positive and fit in int32");
    TORCH_CHECK(num_pages <= INT32_MAX && table_stride <= INT32_MAX &&
                batch * kvs <= INT32_MAX,
                "paged dimensions must fit kernel indexing limits");
    if (paged)
        TORCH_CHECK(table_stride >= (kvs + page_size - 1) / page_size,
                    "block_table does not cover max_kv_len");
    if (kv_seqlens) {
        check_int32(*kv_seqlens, "kv_seqlens", device);
        TORCH_CHECK(kv_seqlens->dim() == 1 && kv_seqlens->numel() == batch,
                    "kv_seqlens must have B entries");
    }
    TORCH_CHECK((split_kv == 1 || split_kv == 2 || split_kv == 4 || split_kv == 8) &&
                split_kv <= (kvs + 127) / 128,
                "split_kv must be 1, 2, 4, or 8 and at most ceil(KVs/128)");
    check_scales(q_scales, "Q scales", device, batch * 1024);
    check_scales(k_scales, "K scales", device, paged
        ? num_pages * (page_size / 128) * 1024
        : batch * ((kvs + 127) / 128) * 1024);
    TORCH_CHECK(head_weight.is_cuda() && head_weight.device() == device &&
                head_weight.scalar_type() == at::kBFloat16 && head_weight.is_contiguous() &&
                head_weight.dim() == 2 && head_weight.size(0) == batch &&
                head_weight.size(1) == 64,
                "head_weight must be contiguous CUDA bf16 [B, 64]");
    TORCH_CHECK(out.is_cuda() && out.device() == device &&
                out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                out.dim() == 2 && out.size(0) == batch && out.size(1) == kvs,
                "out must be contiguous CUDA bf16 [B, KVs]");
    c10::cuda::CUDAGuard guard(device);
    const auto launcher = reinterpret_cast<Launcher>(static_cast<uintptr_t>(launcher_address));
    launcher(reinterpret_cast<const uint8_t*>(q.data_ptr()),
             reinterpret_cast<const uint8_t*>(k.data_ptr()),
             reinterpret_cast<const uint8_t*>(q_scales.data_ptr()),
             reinterpret_cast<const uint8_t*>(k_scales.data_ptr()),
             reinterpret_cast<const __nv_bfloat16*>(head_weight.data_ptr<at::BFloat16>()),
             reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), static_cast<int>(batch), static_cast<int>(kvs),
             static_cast<int>(num_pages), static_cast<int>(page_size),
             paged ? block_table->data_ptr<int32_t>() : nullptr,
             static_cast<int>(table_stride),
             kv_seqlens ? kv_seqlens->data_ptr<int32_t>() : nullptr,
             static_cast<int>(split_kv), c10::cuda::getCurrentCUDAStream(device.index()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor indexer_score_decode_out(
    torch::Tensor q, torch::Tensor k, torch::Tensor q_scales,
    torch::Tensor k_scales, torch::Tensor head_weight, torch::Tensor out,
    int64_t split_kv, c10::optional<torch::Tensor> block_table,
    c10::optional<torch::Tensor> kv_seqlens, int64_t max_kv_len,
    uint64_t launcher_address) {
    return run(q, k, q_scales, k_scales, head_weight, out, split_kv,
               block_table, kv_seqlens, max_kv_len, launcher_address);
}

torch::Tensor indexer_score_decode(
    torch::Tensor q, torch::Tensor k, torch::Tensor q_scales,
    torch::Tensor k_scales, torch::Tensor head_weight, int64_t split_kv,
    c10::optional<torch::Tensor> block_table,
    c10::optional<torch::Tensor> kv_seqlens, int64_t max_kv_len,
    uint64_t launcher_address) {
    auto out = torch::empty({q.size(0), block_table ? max_kv_len : k.size(1)},
                            q.options().dtype(torch::kBFloat16));
    return run(q, k, q_scales, k_scales, head_weight, out, split_kv,
               block_table, kv_seqlens, max_kv_len, launcher_address);
}

} // namespace sm110a::indexer_score_decode_torch_binding

using namespace sm110a::indexer_score_decode_torch_binding;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("indexer_score_decode", &indexer_score_decode);
    module.def("indexer_score_decode_out", &indexer_score_decode_out);
}
