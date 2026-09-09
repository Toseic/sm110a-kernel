#include <cuda.h>
#include <torch/extension.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

std::string driver_error(CUresult result) {
  const char* name = nullptr;
  const char* detail = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &detail);
  std::ostringstream stream;
  stream << (name ? name : "CUDA driver error") << " ("
         << static_cast<int>(result) << ")";
  if (detail) stream << ": " << detail;
  return stream.str();
}

void check_driver(CUresult result, const char* operation) {
  TORCH_CHECK(result == CUDA_SUCCESS, operation, " failed: ", driver_error(result));
}

class GreenContext {
 public:
  GreenContext(int device, unsigned int sm_count, CUgreenCtx context,
               CUstream stream)
      : device_(device), sm_count_(sm_count), context_(context), stream_(stream) {}

  ~GreenContext() { close_noexcept(); }
  GreenContext(const GreenContext&) = delete;
  GreenContext& operator=(const GreenContext&) = delete;

  int device() const { return device_; }
  unsigned int sm_count() const { return sm_count_; }
  uint64_t stream_ptr() const {
    TORCH_CHECK(!closed(), "GreenContext is closed");
    return reinterpret_cast<uint64_t>(stream_);
  }
  uint64_t handle() const {
    TORCH_CHECK(!closed(), "GreenContext is closed");
    return reinterpret_cast<uint64_t>(context_);
  }
  bool closed() const { return context_ == nullptr; }

  void synchronize() const {
    TORCH_CHECK(!closed(), "GreenContext is closed");
    py::gil_scoped_release release;
    check_driver(cuStreamSynchronize(stream_), "cuStreamSynchronize");
  }

  void close() {
    if (closed()) return;
    // A stream is owned by this helper.  CUDA requires it to be destroyed before
    // destroying the green context it is associated with.
    const CUresult stream_result = cuStreamDestroy(stream_);
    stream_ = nullptr;
    const CUresult context_result = cuGreenCtxDestroy(context_);
    context_ = nullptr;
    check_driver(stream_result, "cuStreamDestroy");
    check_driver(context_result, "cuGreenCtxDestroy");
  }

 private:
  void close_noexcept() noexcept {
    if (context_ == nullptr) return;
    if (stream_ != nullptr) cuStreamDestroy(stream_);
    cuGreenCtxDestroy(context_);
    stream_ = nullptr;
    context_ = nullptr;
  }

  int device_;
  unsigned int sm_count_;
  CUgreenCtx context_;
  CUstream stream_;
};

CUdevResource get_sm_resource(int device) {
  CUdevice cuda_device;
  check_driver(cuDeviceGet(&cuda_device, device), "cuDeviceGet");
  CUdevResource resource{};
  check_driver(cuDeviceGetDevResource(cuda_device, &resource, CU_DEV_RESOURCE_TYPE_SM),
               "cuDeviceGetDevResource(CU_DEV_RESOURCE_TYPE_SM)");
  return resource;
}

py::dict sm_resource_info(int device) {
  check_driver(cuInit(0), "cuInit");
  const CUdevResource resource = get_sm_resource(device);
  py::dict result;
  result["device"] = device;
  result["sm_count"] = resource.sm.smCount;
  result["min_sm_partition_size"] = resource.sm.minSmPartitionSize;
  result["sm_coscheduled_alignment"] = resource.sm.smCoscheduledAlignment;
  return result;
}

std::vector<std::shared_ptr<GreenContext>> create_green_contexts(
    const std::vector<unsigned int>& sm_counts, int device, int priority) {
  TORCH_CHECK(!sm_counts.empty(), "sm_counts must not be empty");
  check_driver(cuInit(0), "cuInit");

  CUdevice cuda_device;
  check_driver(cuDeviceGet(&cuda_device, device), "cuDeviceGet");
  const CUdevResource input = get_sm_resource(device);
  const unsigned int alignment = input.sm.smCoscheduledAlignment;
  uint64_t total = 0;
  for (const unsigned int count : sm_counts) {
    TORCH_CHECK(count > 0, "each requested SM count must be positive");
    TORCH_CHECK(count >= input.sm.minSmPartitionSize,
                "requested ", count, " SMs is below this device's minimum partition size ",
                input.sm.minSmPartitionSize);
    TORCH_CHECK(count % alignment == 0,
                "requested ", count, " SMs is not aligned to this device's co-scheduled SM alignment ",
                alignment, "; exact Green Context partitions must use a multiple of this value");
    total += count;
  }
  TORCH_CHECK(total <= input.sm.smCount, "requested ", total,
              " SMs but device ", device, " has only ", input.sm.smCount);

  std::vector<CUdevResource> resources(sm_counts.size());
  std::vector<CU_DEV_SM_RESOURCE_GROUP_PARAMS> params(sm_counts.size());
  for (size_t index = 0; index < sm_counts.size(); ++index) {
    params[index].smCount = sm_counts[index];
    params[index].coscheduledSmCount = alignment;
    params[index].preferredCoscheduledSmCount = alignment;
  }
  CUdevResource remainder{};
  check_driver(cuDevSmResourceSplit(resources.data(), resources.size(), &input,
                                    &remainder, 0, params.data()),
               "cuDevSmResourceSplit");

  std::vector<std::shared_ptr<GreenContext>> contexts;
  contexts.reserve(resources.size());
  try {
    for (const CUdevResource& resource : resources) {
      CUdevResourceDesc descriptor = nullptr;
      check_driver(cuDevResourceGenerateDesc(&descriptor,
                                             const_cast<CUdevResource*>(&resource), 1),
                   "cuDevResourceGenerateDesc");
      CUgreenCtx green_context = nullptr;
      check_driver(cuGreenCtxCreate(&green_context, descriptor, cuda_device,
                                    CU_GREEN_CTX_DEFAULT_STREAM),
                   "cuGreenCtxCreate");
      CUstream stream = nullptr;
      const CUresult stream_result = cuGreenCtxStreamCreate(
          &stream, green_context, CU_STREAM_NON_BLOCKING, priority);
      if (stream_result != CUDA_SUCCESS) {
        cuGreenCtxDestroy(green_context);
        check_driver(stream_result, "cuGreenCtxStreamCreate");
      }
      contexts.push_back(std::make_shared<GreenContext>(device, resource.sm.smCount,
                                                         green_context, stream));
    }
  } catch (...) {
    contexts.clear();
    throw;
  }
  return contexts;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<GreenContext, std::shared_ptr<GreenContext>>(m, "GreenContext")
      .def_property_readonly("device", &GreenContext::device)
      .def_property_readonly("sm_count", &GreenContext::sm_count)
      .def_property_readonly("stream_ptr", &GreenContext::stream_ptr)
      .def_property_readonly("handle", &GreenContext::handle)
      .def_property_readonly("closed", &GreenContext::closed)
      .def("synchronize", &GreenContext::synchronize)
      .def("close", &GreenContext::close)
      .def("__enter__", [](const std::shared_ptr<GreenContext>& self) { return self; })
      .def("__exit__", [](const std::shared_ptr<GreenContext>& self, py::object,
                            py::object, py::object) { self->close(); });
  m.def("sm_resource_info", &sm_resource_info, py::arg("device"));
  m.def("create_green_contexts", &create_green_contexts, py::arg("sm_counts"),
        py::arg("device"), py::arg("priority") = 0,
        "Create independent Green Contexts from exact, disjoint SM partitions.");
}
