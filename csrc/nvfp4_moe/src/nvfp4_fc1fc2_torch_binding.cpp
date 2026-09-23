#include <algorithm>
#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

using Launcher = void (*)(
    const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
    const uint8_t*, const uint8_t*, const int32_t*, const int32_t*,
    const int32_t*, const int32_t*, const __nv_bfloat16*, __nv_bfloat16*,
    uint8_t*, int, int, int, int, int, int, cudaStream_t);

void check_fp4(const torch::Tensor& tensor, const char* name, int dimensions) {
    TORCH_CHECK(tensor.is_cuda() &&
        tensor.scalar_type() == at::kFloat4_e2m1fn_x2 &&
        tensor.dim() == dimensions && tensor.is_contiguous(),
        name, " must be contiguous CUDA float4_e2m1fn_x2");
}

void check_scale(const torch::Tensor& tensor, const char* name,
                 int64_t elements, const c10::Device& device) {
    TORCH_CHECK(tensor.is_cuda() &&
        tensor.scalar_type() == at::kFloat8_e4m3fn &&
        tensor.is_contiguous() && tensor.numel() == elements &&
        tensor.device() == device,
        name, " has invalid blocked scale storage");
}

torch::Tensor run_out(
    torch::Tensor a, torch::Tensor w1, torch::Tensor sfa,
    torch::Tensor sfw1, torch::Tensor w2, torch::Tensor sfw2,
    torch::Tensor output, torch::Tensor workspace, uint64_t address) {
    check_fp4(a, "A", 2);
    check_fp4(w1, "W1", 3);
    check_fp4(w2, "W2", 3);
    const int64_t m = a.size(0);
    TORCH_CHECK(m > 0 && m % 512 == 0 && a.size(1) == 3072,
        "A must have packed shape [M,3072] with M divisible by 512");
    TORCH_CHECK(w1.sizes() == torch::IntArrayRef({1, 4096, 3072}),
        "W1 must have packed shape [1,4096,3072]");
    TORCH_CHECK(w2.sizes() == torch::IntArrayRef({1, 6144, 1024}),
        "W2 must have packed shape [1,6144,1024]");
    check_scale(sfa, "SFA", m * 384, a.device());
    check_scale(sfw1, "SFW1", 4096 * 384, a.device());
    check_scale(sfw2, "SFW2", 6144 * 128, a.device());
    TORCH_CHECK(a.device() == w1.device() && a.device() == w2.device(),
        "all inputs must share one CUDA device");
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kBFloat16 &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({m, 6144}) &&
        output.device() == a.device(),
        "output must be contiguous CUDA BF16 [M,6144]");
    TORCH_CHECK(address != 0, "launcher must be valid");
    TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == torch::kUInt8 &&
        workspace.is_contiguous() && workspace.device() == a.device() &&
        workspace.numel() >= m * 1152 + std::max<int64_t>(8, m / 256 * 12),
        "workspace is too small for intermediates and synchronization state");

    c10::cuda::CUDAGuard guard(a.device());
    auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
    reinterpret_cast<Launcher>(static_cast<uintptr_t>(address))(
        reinterpret_cast<const uint8_t*>(a.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfa.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfw1.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfw2.data_ptr()),
        nullptr, nullptr, nullptr, nullptr, nullptr,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        workspace.data_ptr<uint8_t>(),
        static_cast<int>(m), static_cast<int>(m), 1, 1,
        static_cast<int>(m / 256), 0, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor run_routed_out(
    torch::Tensor a, torch::Tensor w1, torch::Tensor sfa,
    torch::Tensor sfw1, torch::Tensor w2, torch::Tensor sfw2,
    torch::Tensor source_rows, torch::Tensor routing, torch::Tensor tasks,
    torch::Tensor state, torch::Tensor routing_weights, int64_t topk,
    int64_t row_task_count, torch::Tensor output, torch::Tensor workspace,
    uint64_t address) {
    check_fp4(a, "A", 2);
    check_fp4(w1, "W1", 3);
    check_fp4(w2, "W2", 3);
    const int64_t tokens = a.size(0);
    const int64_t experts = w1.size(0);
    const int64_t m = source_rows.numel();
    TORCH_CHECK(tokens > 0 && tokens % 128 == 0 && topk > 0 &&
        m == tokens * topk && a.size(1) == 3072,
        "routed A must be [tokens,3072], tokens divisible by 128, and M=tokens*topk");
    TORCH_CHECK(w1.size(1) == 4096 && w1.size(2) == 3072 &&
        w2.sizes() == torch::IntArrayRef({experts, 6144, 1024}),
        "routed weights must be W1 [E,4096,3072] and W2 [E,6144,1024]");
    check_scale(sfa, "SFA", tokens * 384, a.device());
    check_scale(sfw1, "SFW1", experts * 4096 * 384, a.device());
    check_scale(sfw2, "SFW2", experts * 6144 * 128, a.device());
    TORCH_CHECK(source_rows.is_cuda() && source_rows.scalar_type() == torch::kInt32 &&
        source_rows.is_contiguous(), "source_rows must be contiguous CUDA int32");
    TORCH_CHECK(routing.is_cuda() && routing.scalar_type() == torch::kInt32 &&
        routing.is_contiguous() && routing.numel() == 2 * experts + 3 * m,
        "routing has invalid packed metadata");
    TORCH_CHECK(tasks.is_cuda() && tasks.scalar_type() == torch::kInt32 &&
        tasks.is_contiguous() && tasks.dim() == 2 && tasks.size(1) == 5,
        "tasks must be contiguous CUDA int32 [T,5]");
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kInt32 &&
        state.is_contiguous() && state.numel() == 2,
        "state must be contiguous CUDA int32[2]");
    TORCH_CHECK(routing_weights.is_cuda() &&
        routing_weights.scalar_type() == torch::kBFloat16 &&
        routing_weights.is_contiguous() &&
        routing_weights.sizes() == torch::IntArrayRef({tokens, topk}),
        "routing_weights must be contiguous CUDA BF16 [tokens,topk]");
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kBFloat16 &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({tokens, 6144}),
        "routed output must be contiguous CUDA BF16 [tokens,6144]");
    TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == torch::kUInt8 &&
        workspace.is_contiguous(), "workspace must be contiguous CUDA uint8");
    TORCH_CHECK(a.device() == w1.device() && a.device() == w2.device() &&
        a.device() == sfa.device() && a.device() == sfw1.device() &&
        a.device() == sfw2.device() && a.device() == source_rows.device() &&
        a.device() == routing.device() && a.device() == tasks.device() &&
        a.device() == state.device() && a.device() == routing_weights.device() &&
        a.device() == output.device() && a.device() == workspace.device(),
        "all routed tensors must share one CUDA device");
    const int64_t row_task_capacity = tasks.size(0) / 32;
    TORCH_CHECK(tasks.size(0) % 32 == 0 && workspace.numel() >=
        row_task_capacity * 2 * 147456 +
            std::max<int64_t>(8, row_task_capacity * 12),
        "routed workspace is too small for the FC1 task capacity");
    TORCH_CHECK(row_task_count > 0 && row_task_count <= row_task_capacity,
        "invalid routed row_task_count");
    TORCH_CHECK(address != 0, "launcher must be valid");

    c10::cuda::CUDAGuard guard(a.device());
    auto stream = c10::cuda::getCurrentCUDAStream(a.device().index());
    const auto* grouped_assignments = routing.data_ptr<int32_t>() +
        2 * experts + m;
    reinterpret_cast<Launcher>(static_cast<uintptr_t>(address))(
        reinterpret_cast<const uint8_t*>(a.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfa.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfw1.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfw2.data_ptr()),
        source_rows.data_ptr<int32_t>(), grouped_assignments,
        tasks.data_ptr<int32_t>(), state.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16*>(
            routing_weights.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        workspace.data_ptr<uint8_t>(), static_cast<int>(tokens),
        static_cast<int>(m), static_cast<int>(experts), static_cast<int>(topk),
        static_cast<int>(row_task_count),
        static_cast<int>(tasks.size(0)), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor run(
    torch::Tensor a, torch::Tensor w1, torch::Tensor sfa,
    torch::Tensor sfw1, torch::Tensor w2, torch::Tensor sfw2,
    torch::Tensor workspace, uint64_t address) {
    auto output = torch::empty({a.size(0), 6144},
        a.options().dtype(torch::kBFloat16));
    return run_out(a, w1, sfa, sfw1, w2, sfw2, output, workspace, address);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("nvfp4_fc1fc2", &run);
    module.def("nvfp4_fc1fc2_out", &run_out);
    module.def("nvfp4_fc1fc2_routed_out", &run_routed_out);
}
