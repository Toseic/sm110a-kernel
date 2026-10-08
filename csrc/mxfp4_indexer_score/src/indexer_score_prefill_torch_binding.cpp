#include <cstdint>
#include <limits>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

using Launcher = void (*)(const uint8_t*, const uint8_t*, const uint8_t*,
                          const uint8_t*, const __nv_bfloat16*, __nv_bfloat16*, int, int,
                          int, int, int, int, const int32_t*, int,
                          const int32_t*, const int32_t*, int, bool, cudaStream_t);

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
                  int cluster_m, bool causal, c10::optional<torch::Tensor> block_table,
                  c10::optional<torch::Tensor> cu_seqlens_q,
                  c10::optional<torch::Tensor> kv_seqlens,
                  int64_t max_q_len, int64_t max_kv_len,
                  uint64_t launcher_address) {
    TORCH_CHECK(launcher_address != 0, "indexer score launcher is null");
    TORCH_CHECK(q.is_cuda(), "Q must be a CUDA tensor");
    const auto device = q.device();
    check_packed(q, "Q", device);
    check_packed(k, "K", device);
    const bool paged = block_table.has_value();
    const bool packed_q = cu_seqlens_q.has_value();
    int64_t batch = 1;
    int64_t page_size = 0;
    int64_t num_pages = 0;
    int64_t table_stride = 0;
    if (paged) {
        check_int32(*block_table, "block_table", device);
        TORCH_CHECK(block_table->dim() == 2 && block_table->size(0) > 0,
                    "block_table must have shape [B, max_pages]");
        batch = block_table->size(0);
        table_stride = block_table->size(1);
        TORCH_CHECK(k.dim() == 3 && k.size(0) > 0 && k.size(2) == 64 &&
                    (k.size(1) == 128 || k.size(1) == 256),
                    "paged K must have shape [num_pages, page_size, 64], "
                    "page_size=128 or 256");
        page_size = k.size(1);
        num_pages = k.size(0);
        TORCH_CHECK(kv_seqlens.has_value(), "paged K requires kv_seqlens");
    } else if (k.dim() == 3) {
        TORCH_CHECK(k.size(0) > 0 && k.size(1) > 0 && k.size(2) == 64,
                    "batched dense K must have shape [B, KVs, 64]");
        batch = k.size(0);
    } else {
        TORCH_CHECK(k.dim() == 2 && k.size(0) > 0 && k.size(1) == 64,
                    "dense K must have shape [KVs, 64]");
    }
    if (packed_q) {
        check_int32(*cu_seqlens_q, "cu_seqlens_q", device);
        TORCH_CHECK(cu_seqlens_q->dim() == 1 && cu_seqlens_q->numel() == batch + 1,
                    "cu_seqlens_q must have B+1 entries");
        TORCH_CHECK(q.dim() == 3 && q.size(1) == 64 && q.size(2) == 64,
                    "packed Q must have shape [sum_q, 64, 64]");
        TORCH_CHECK(max_q_len > 0 && max_q_len <= INT32_MAX,
                    "max_q_len must be positive and fit int32");
    } else if (batch > 1 || q.dim() == 4) {
        TORCH_CHECK(q.dim() == 4 && q.size(0) == batch && q.size(1) > 0 &&
                    q.size(2) == 64 && q.size(3) == 64,
                    "batched Q must have shape [B, Qs, 64, 64]");
    } else {
        TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(1) == 64 &&
                    q.size(2) == 64, "Q must have shape [Qs, 64, 64]");
    }
    TORCH_CHECK(!paged || q.dim() == 4 || packed_q || batch == 1,
                "paged Q must be batched or use cu_seqlens_q");
    const int64_t qs = packed_q ? max_q_len : q.dim() == 4 ? q.size(1) : q.size(0);
    const int64_t total_q = q.dim() == 4 ? batch * qs : q.size(0);
    const int64_t kvs = paged ? max_kv_len : k.dim() == 3 ? k.size(1) : k.size(0);
    TORCH_CHECK(qs > 0 && kvs > 0 && qs <= INT32_MAX && kvs <= INT32_MAX &&
                total_q <= INT32_MAX && batch <= 65535 &&
                num_pages <= INT32_MAX && table_stride <= INT32_MAX &&
                batch * kvs <= INT32_MAX && total_q * 64 <= INT32_MAX,
                "dimensions must be positive and fit kernel indexing limits");
    if (paged)
        TORCH_CHECK(table_stride >= (kvs + page_size - 1) / page_size,
                    "block_table does not cover max_kv_len");
    if (packed_q)
        TORCH_CHECK(paged, "packed Q requires paged K");
    if (kv_seqlens) {
        check_int32(*kv_seqlens, "kv_seqlens", device);
        TORCH_CHECK(kv_seqlens->dim() == 1 && kv_seqlens->numel() == batch,
                    "kv_seqlens must have B entries");
    }
    TORCH_CHECK(cluster_m == 1 || cluster_m == 2, "cluster_m must be 1 or 2");
    check_scales(q_scales, "Q scales", device, batch * ((qs + 1) / 2) * 1024);
    check_scales(k_scales, "K scales", device, paged
        ? num_pages * (page_size / 128) * 1024
        : batch * ((kvs + 127) / 128) * 1024);
    TORCH_CHECK(head_weight.is_cuda() && head_weight.device() == device &&
                head_weight.scalar_type() == at::kBFloat16 && head_weight.is_contiguous() &&
                head_weight.dim() == q.dim() - 1 &&
                head_weight.size(-1) == 64 && head_weight.numel() == total_q * 64,
                "head_weight must be contiguous CUDA bf16 [Qs, 64], "
                "[B, Qs, 64], or [sum_q, 64]");
    TORCH_CHECK(out.is_cuda() && out.device() == device &&
                out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                out.dim() == q.dim() - 1 && out.size(-1) == kvs &&
                out.numel() == total_q * kvs,
                "out must be contiguous CUDA bf16 [Qs, KVs], [B, Qs, KVs], "
                "or [sum_q, max_kv_len]");
    c10::cuda::CUDAGuard guard(device);
    const auto launcher = reinterpret_cast<Launcher>(static_cast<uintptr_t>(launcher_address));
    launcher(reinterpret_cast<const uint8_t*>(q.data_ptr()),
             reinterpret_cast<const uint8_t*>(k.data_ptr()),
             reinterpret_cast<const uint8_t*>(q_scales.data_ptr()),
             reinterpret_cast<const uint8_t*>(k_scales.data_ptr()),
             reinterpret_cast<const __nv_bfloat16*>(head_weight.data_ptr<at::BFloat16>()),
             reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), static_cast<int>(qs), static_cast<int>(kvs),
             static_cast<int>(batch), static_cast<int>(total_q),
             static_cast<int>(num_pages), static_cast<int>(page_size),
             paged ? block_table->data_ptr<int32_t>() : nullptr,
             static_cast<int>(table_stride),
             packed_q ? cu_seqlens_q->data_ptr<int32_t>() : nullptr,
             kv_seqlens ? kv_seqlens->data_ptr<int32_t>() : nullptr,
             cluster_m, causal,
             c10::cuda::getCurrentCUDAStream(device.index()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor indexer_score_prefill_out(
    torch::Tensor q, torch::Tensor k, torch::Tensor q_scales,
    torch::Tensor k_scales, torch::Tensor head_weight, torch::Tensor out,
    int cluster_m, bool causal, c10::optional<torch::Tensor> block_table,
    c10::optional<torch::Tensor> cu_seqlens_q,
    c10::optional<torch::Tensor> kv_seqlens,
    int64_t max_q_len, int64_t max_kv_len,
    uint64_t launcher_address) {
    return run(q, k, q_scales, k_scales, head_weight, out, cluster_m, causal,
               block_table, cu_seqlens_q, kv_seqlens, max_q_len, max_kv_len,
               launcher_address);
}

torch::Tensor indexer_score_prefill(
    torch::Tensor q, torch::Tensor k, torch::Tensor q_scales,
    torch::Tensor k_scales, torch::Tensor head_weight, int cluster_m, bool causal,
    c10::optional<torch::Tensor> block_table,
    c10::optional<torch::Tensor> cu_seqlens_q,
    c10::optional<torch::Tensor> kv_seqlens,
    int64_t max_q_len, int64_t max_kv_len,
    uint64_t launcher_address) {
    const int64_t kvs = block_table ? max_kv_len : k.size(k.dim() - 2);
    auto out = q.dim() == 4
        ? torch::empty({q.size(0), q.size(1), kvs}, q.options().dtype(torch::kBFloat16))
        : torch::empty({q.size(0), kvs}, q.options().dtype(torch::kBFloat16));
    return run(q, k, q_scales, k_scales, head_weight, out, cluster_m, causal,
               block_table, cu_seqlens_q, kv_seqlens, max_q_len, max_kv_len,
               launcher_address);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("indexer_score_prefill", &indexer_score_prefill);
    module.def("indexer_score_prefill_out", &indexer_score_prefill_out);
}
