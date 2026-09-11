#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

using Bf16GemmServingLauncher = void (*)(const __nv_bfloat16*, const __nv_bfloat16*,
                                   __nv_bfloat16*, int, int, int, bool,
                                   int, cudaStream_t);
using ServingQueueInitLauncher = cudaError_t (*)(
    void*, size_t, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);
using PersistentServingLauncher = cudaError_t (*)(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, void*,
    uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);
using ServingDrainLauncher = cudaError_t (*)(void*, cudaStream_t);
using ServingTestPublishLauncher = cudaError_t (*)(
    const __nv_bfloat16*, __nv_bfloat16*, void*, uint64_t, uint32_t, bool, uint64_t,
    bool, cudaStream_t);
using ServingTestConsumeLauncher = cudaError_t (*)(
    const __nv_bfloat16*, __nv_bfloat16*, void*, uint64_t, uint32_t, bool,
    cudaStream_t);

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_control(const torch::Tensor& control) {
  TORCH_CHECK(control.is_cuda(), "Serving queue control must be a CUDA tensor");
  TORCH_CHECK(control.scalar_type() == torch::kUInt8,
              "Serving queue control must have dtype torch.uint8");
  TORCH_CHECK(control.is_contiguous() && control.dim() == 1,
              "Serving queue control must be a contiguous 1D tensor");
}

void check_cuda_status(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation, ": ", cudaGetErrorString(status));
}

void check_config(int64_t tile_m, int64_t tile_n, int64_t tile_k,
                  int64_t cluster_m, int64_t max_active_clusters) {
  TORCH_CHECK(tile_m > 0 && tile_n > 0 && tile_k > 0 && cluster_m > 0 &&
                  max_active_clusters > 0,
              "invalid BF16 GEMM configuration");
}

void check_inputs(const torch::Tensor& a, const torch::Tensor& b,
                  int64_t tile_m, int64_t tile_n, int64_t tile_k,
                  int64_t cluster_m, int64_t max_active_clusters, bool swap_ab) {
  check_tensor(a, "A");
  check_tensor(b, "B");
  check_config(tile_m, tile_n, tile_k, cluster_m, max_active_clusters);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "A and B must be 2D");
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same device");
  TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions must match");
  const int64_t physical_m = swap_ab ? b.size(1) : a.size(0);
  const int64_t physical_n = swap_ab ? a.size(0) : b.size(1);
  TORCH_CHECK(physical_m % (tile_m * cluster_m) == 0,
              swap_ab ? "N" : "M", " is incompatible with the kernel tile");
  TORCH_CHECK(physical_n % tile_n == 0,
              swap_ab ? "M" : "N", " is incompatible with the kernel tile");
  TORCH_CHECK(a.size(1) % tile_k == 0, "K is incompatible with the kernel tile");
}

void check_inputs_nt(const torch::Tensor& a, const torch::Tensor& b_nt,
                     int64_t tile_m, int64_t tile_n, int64_t tile_k,
                     int64_t cluster_m, int64_t max_active_clusters, bool swap_ab) {
  check_tensor(a, "A");
  check_tensor(b_nt, "B_nt");
  check_config(tile_m, tile_n, tile_k, cluster_m, max_active_clusters);
  TORCH_CHECK(a.dim() == 2 && b_nt.dim() == 2, "A and B_nt must be 2D");
  TORCH_CHECK(a.device() == b_nt.device(), "A and B_nt must be on the same device");
  TORCH_CHECK(a.size(1) == b_nt.size(1), "Inner dimensions must match");
  const int64_t physical_m = swap_ab ? b_nt.size(0) : a.size(0);
  const int64_t physical_n = swap_ab ? a.size(0) : b_nt.size(0);
  TORCH_CHECK(physical_m % (tile_m * cluster_m) == 0,
              swap_ab ? "N" : "M", " is incompatible with the kernel tile");
  TORCH_CHECK(physical_n % tile_n == 0,
              swap_ab ? "M" : "N", " is incompatible with the kernel tile");
  TORCH_CHECK(a.size(1) % tile_k == 0, "K is incompatible with the kernel tile");
}

void check_output(const torch::Tensor& out, int64_t m, int64_t n,
                  const c10::Device& device) {
  check_tensor(out, "Output");
  TORCH_CHECK(out.dim() == 2 && out.device() == device, "invalid output tensor");
  TORCH_CHECK(out.size(0) == m && out.size(1) == n, "invalid output shape");
}

torch::Tensor run_out(torch::Tensor a, torch::Tensor b, torch::Tensor out,
                      uint64_t launcher_address, int64_t tile_m,
                      int64_t tile_n, int64_t tile_k, int64_t cluster_m,
                      int64_t max_active_clusters, bool swap_ab, bool b_is_nt) {
  TORCH_CHECK(launcher_address != 0, "BF16 GEMM launcher is null");
  if (b_is_nt) {
    check_inputs_nt(a, b, tile_m, tile_n, tile_k, cluster_m,
                    max_active_clusters, swap_ab);
    check_output(out, a.size(0), b.size(0), a.device());
  } else {
    check_inputs(a, b, tile_m, tile_n, tile_k, cluster_m,
                 max_active_clusters, swap_ab);
    check_output(out, a.size(0), b.size(1), a.device());
  }
  c10::cuda::CUDAGuard device_guard(a.device());
  const auto launcher = reinterpret_cast<Bf16GemmServingLauncher>(
      static_cast<uintptr_t>(launcher_address));
  launcher(reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
           reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
           reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
           static_cast<int>(a.size(0)), static_cast<int>(b_is_nt ? b.size(0) : b.size(1)),
           static_cast<int>(a.size(1)), swap_ab,
           static_cast<int>(max_active_clusters),
           c10::cuda::getCurrentCUDAStream(a.device().index()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor bf16_gemm_serving(torch::Tensor a, torch::Tensor b, uint64_t launcher,
                        int64_t tm, int64_t tn, int64_t tk, int64_t cm,
                        int64_t max_active_clusters, bool swap) {
  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  return run_out(a, b, out, launcher, tm, tn, tk, cm, max_active_clusters, swap, false);
}
torch::Tensor bf16_gemm_serving_out(torch::Tensor a, torch::Tensor b, torch::Tensor out,
                            uint64_t launcher, int64_t tm, int64_t tn,
                            int64_t tk, int64_t cm, int64_t max_active_clusters,
                            bool swap) {
  return run_out(a, b, out, launcher, tm, tn, tk, cm, max_active_clusters, swap, false);
}
torch::Tensor bf16_gemm_serving_nt(torch::Tensor a, torch::Tensor b, uint64_t launcher,
                           int64_t tm, int64_t tn, int64_t tk, int64_t cm,
                           int64_t max_active_clusters, bool swap) {
  auto out = torch::empty({a.size(0), b.size(0)}, a.options());
  return run_out(a, b, out, launcher, tm, tn, tk, cm, max_active_clusters, swap, true);
}
torch::Tensor bf16_gemm_serving_nt_out(torch::Tensor a, torch::Tensor b, torch::Tensor out,
                               uint64_t launcher, int64_t tm, int64_t tn,
                               int64_t tk, int64_t cm, int64_t max_active_clusters,
                               bool swap) {
  return run_out(a, b, out, launcher, tm, tn, tk, cm, max_active_clusters, swap, true);
}

void serving_queue_initialize(torch::Tensor control, uint64_t launcher_address,
                              int64_t slot_count, int64_t n, int64_t k,
                              int64_t launched_clusters) {
  check_control(control);
  TORCH_CHECK(launcher_address != 0, "Serving queue initializer is null");
  TORCH_CHECK(slot_count > 0 && n > 0 && k > 0 && launched_clusters > 0,
              "Invalid serving queue dimensions");
  c10::cuda::CUDAGuard device_guard(control.device());
  const auto launcher = reinterpret_cast<ServingQueueInitLauncher>(
      static_cast<uintptr_t>(launcher_address));
  check_cuda_status(launcher(control.data_ptr(), control.numel(),
                             static_cast<uint32_t>(slot_count),
                             static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                             static_cast<uint32_t>(launched_clusters),
                             c10::cuda::getCurrentCUDAStream(control.device().index())),
                    "Serving queue initialization failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void persistent_serving_start(torch::Tensor a_ring, torch::Tensor b_nt,
                              torch::Tensor c_ring, torch::Tensor control,
                              uint64_t launcher_address, int64_t slot_count,
                              int64_t n, int64_t k, int64_t launched_clusters) {
  check_tensor(a_ring, "A_ring");
  check_tensor(b_nt, "B_nt");
  check_tensor(c_ring, "C_ring");
  check_control(control);
  TORCH_CHECK(launcher_address != 0, "Persistent serving launcher is null");
  TORCH_CHECK(a_ring.dim() == 3 && b_nt.dim() == 2 && c_ring.dim() == 3,
              "A_ring/C_ring must be 3D and B_nt must be 2D");
  TORCH_CHECK(a_ring.size(0) == slot_count && a_ring.size(1) == 128 &&
                  a_ring.size(2) == k,
              "Invalid A_ring shape");
  TORCH_CHECK(b_nt.size(0) == n && b_nt.size(1) == k,
              "Invalid B_nt shape");
  TORCH_CHECK(c_ring.size(0) == slot_count && c_ring.size(1) == 128 &&
                  c_ring.size(2) == n,
              "Invalid C_ring shape");
  TORCH_CHECK(a_ring.device() == b_nt.device() && a_ring.device() == c_ring.device() &&
                  a_ring.device() == control.device(),
              "All serving tensors must be on the same device");
  c10::cuda::CUDAGuard device_guard(a_ring.device());
  const auto launcher = reinterpret_cast<PersistentServingLauncher>(
      static_cast<uintptr_t>(launcher_address));
  check_cuda_status(launcher(
                        reinterpret_cast<const __nv_bfloat16*>(
                            a_ring.data_ptr<at::BFloat16>()),
                        reinterpret_cast<const __nv_bfloat16*>(
                            b_nt.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__nv_bfloat16*>(
                            c_ring.data_ptr<at::BFloat16>()),
                        control.data_ptr(), static_cast<uint32_t>(slot_count),
                        static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                        static_cast<uint32_t>(launched_clusters),
                        c10::cuda::getCurrentCUDAStream(a_ring.device().index())),
                    "Persistent serving launch failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void serving_request_drain(torch::Tensor control, uint64_t launcher_address) {
  check_control(control);
  TORCH_CHECK(launcher_address != 0, "Serving drain launcher is null");
  c10::cuda::CUDAGuard device_guard(control.device());
  const auto launcher = reinterpret_cast<ServingDrainLauncher>(
      static_cast<uintptr_t>(launcher_address));
  check_cuda_status(launcher(
                        control.data_ptr(),
                        c10::cuda::getCurrentCUDAStream(control.device().index())),
                    "Serving drain request failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void serving_test_publish(torch::Tensor rows, torch::Tensor a_ring,
                          torch::Tensor control, uint64_t launcher_address,
                          int64_t group_ticket, bool flush,
                          int64_t delay_cycles, bool bypass) {
  check_tensor(rows, "rows");
  check_tensor(a_ring, "A_ring");
  check_control(control);
  TORCH_CHECK(launcher_address != 0, "Serving test producer is null");
  TORCH_CHECK(rows.dim() == 2 && a_ring.dim() == 3,
              "rows must be 2D and A_ring must be 3D");
  TORCH_CHECK(rows.size(0) > 0 && rows.size(0) <= 256 &&
                  rows.size(1) == a_ring.size(2),
              "Invalid producer row shape");
  TORCH_CHECK(rows.device() == a_ring.device() && rows.device() == control.device(),
              "Producer tensors must be on the same device");
  TORCH_CHECK(group_ticket >= 0, "group_ticket must be non-negative");
  TORCH_CHECK(delay_cycles >= 0, "delay_cycles must be non-negative");
  c10::cuda::CUDAGuard device_guard(rows.device());
  const auto launcher = reinterpret_cast<ServingTestPublishLauncher>(
      static_cast<uintptr_t>(launcher_address));
  check_cuda_status(launcher(
                        reinterpret_cast<const __nv_bfloat16*>(
                            rows.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__nv_bfloat16*>(
                            a_ring.data_ptr<at::BFloat16>()),
                        control.data_ptr(), static_cast<uint64_t>(group_ticket),
                        static_cast<uint32_t>(rows.size(0)), flush,
                        static_cast<uint64_t>(delay_cycles), bypass,
                        c10::cuda::getCurrentCUDAStream(rows.device().index())),
                    "Serving test publication failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void serving_test_consume(torch::Tensor c_ring, torch::Tensor rows,
                          torch::Tensor control, uint64_t launcher_address,
                          int64_t group_ticket, bool bypass) {
  check_tensor(c_ring, "C_ring");
  check_tensor(rows, "rows");
  check_control(control);
  TORCH_CHECK(launcher_address != 0, "Serving test consumer is null");
  TORCH_CHECK(c_ring.dim() == 3 && rows.dim() == 2,
              "C_ring must be 3D and rows must be 2D");
  TORCH_CHECK(rows.size(0) > 0 && rows.size(0) <= 256 &&
                  rows.size(1) == c_ring.size(2),
              "Invalid consumer output shape");
  TORCH_CHECK(c_ring.device() == rows.device() && rows.device() == control.device(),
              "Consumer tensors must be on the same device");
  TORCH_CHECK(group_ticket >= 0, "group_ticket must be non-negative");
  c10::cuda::CUDAGuard device_guard(rows.device());
  const auto launcher = reinterpret_cast<ServingTestConsumeLauncher>(
      static_cast<uintptr_t>(launcher_address));
  check_cuda_status(launcher(
                        reinterpret_cast<const __nv_bfloat16*>(
                            c_ring.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__nv_bfloat16*>(
                            rows.data_ptr<at::BFloat16>()),
                        control.data_ptr(), static_cast<uint64_t>(group_ticket),
                        static_cast<uint32_t>(rows.size(0)), bypass,
                        c10::cuda::getCurrentCUDAStream(rows.device().index())),
                    "Serving test consumption failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bf16_gemm_serving", &bf16_gemm_serving);
  m.def("bf16_gemm_serving_out", &bf16_gemm_serving_out);
  m.def("bf16_gemm_serving_nt", &bf16_gemm_serving_nt);
  m.def("bf16_gemm_serving_nt_out", &bf16_gemm_serving_nt_out);
  m.def("serving_queue_initialize", &serving_queue_initialize);
  m.def("persistent_serving_start", &persistent_serving_start);
  m.def("serving_request_drain", &serving_request_drain);
  m.def("serving_test_publish", &serving_test_publish);
  m.def("serving_test_consume", &serving_test_consume);
}
