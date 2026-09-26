#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstddef>
#include <string>

#include "nccl_lsa.h"

namespace py = pybind11;

namespace {

class NcclLsaAllReduce {
 public:
  NcclLsaAllReduce(py::bytes unique_id, int rank, int world_size, int device,
                   int64_t capacity)
      : rank_(rank),
        world_size_(world_size),
        device_(device),
        capacity_(capacity) {
    TORCH_CHECK(capacity > 0, "capacity must be positive");
    const std::string id = unique_id;
    c10::cuda::CUDAGuard device_guard(c10::Device(c10::kCUDA, device));
    state_ = sm110a_nccl_lsa_create(id.data(), id.size(), rank, world_size,
                                    device, capacity);
  }

  ~NcclLsaAllReduce() { close(); }

  NcclLsaAllReduce(const NcclLsaAllReduce&) = delete;
  NcclLsaAllReduce& operator=(const NcclLsaAllReduce&) = delete;

  torch::Tensor all_reduce_out(const torch::Tensor& input,
                               torch::Tensor output) {
    TORCH_CHECK(state_ != nullptr, "NCCL LSA communicator is closed");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "input must have dtype torch.float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32,
                "output must have dtype torch.float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(input.sizes() == output.sizes(),
                "input and output must have the same shape");
    TORCH_CHECK(input.device() == output.device(),
                "input and output must be on the same device");
    TORCH_CHECK(input.get_device() == device_,
                "tensor device does not match communicator device");
    TORCH_CHECK(input.numel() > 0 && input.numel() <= capacity_,
                "input element count must be in [1, ", capacity_, "]");

    c10::cuda::CUDAGuard device_guard(input.device());
    const cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(device_).stream();
    sm110a_nccl_lsa_all_reduce(
        state_, input.data_ptr<float>(), output.data_ptr<float>(),
        static_cast<std::size_t>(input.numel()), stream);
    return output;
  }

  torch::Tensor all_reduce(const torch::Tensor& input) {
    return all_reduce_out(input, torch::empty_like(input));
  }

  void close() noexcept {
    if (state_ != nullptr) {
      sm110a_nccl_lsa_destroy(state_);
      state_ = nullptr;
    }
  }

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  int device() const { return device_; }
  int64_t capacity() const { return capacity_; }
  bool closed() const { return state_ == nullptr; }

 private:
  Sm110aNcclLsaState* state_ = nullptr;
  int rank_ = 0;
  int world_size_ = 0;
  int device_ = 0;
  int64_t capacity_ = 0;
};

py::bytes nccl_unique_id() {
  std::string id(sm110a_nccl_lsa_unique_id_size(), '\0');
  sm110a_nccl_lsa_get_unique_id(id.data(), id.size());
  return py::bytes(id);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("nccl_unique_id", &nccl_unique_id,
             "Create an NCCL unique ID as opaque bytes");
  py::class_<NcclLsaAllReduce>(module, "NcclLsaAllReduce")
      .def(py::init<py::bytes, int, int, int, int64_t>(),
           py::arg("unique_id"), py::arg("rank"), py::arg("world_size"),
           py::arg("device"), py::arg("capacity"))
      .def("all_reduce", &NcclLsaAllReduce::all_reduce)
      .def("all_reduce_out", &NcclLsaAllReduce::all_reduce_out)
      .def("close", &NcclLsaAllReduce::close)
      .def_property_readonly("rank", &NcclLsaAllReduce::rank)
      .def_property_readonly("world_size", &NcclLsaAllReduce::world_size)
      .def_property_readonly("device", &NcclLsaAllReduce::device)
      .def_property_readonly("capacity", &NcclLsaAllReduce::capacity)
      .def_property_readonly("closed", &NcclLsaAllReduce::closed);
}
