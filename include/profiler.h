#pragma once

#include <cstdint>

enum class ProfilerTag : int64_t {
  Setup = 0,
  IssueTMA,
  IssueMMA,
  WaitTMA,
  WaitMMA,
  WaitMainloop,
  WaitEpilogue,
  Epilogue,
};

constexpr int kProfilerEntryStride = 4;

__device__ inline int64_t globaltimer() {
  int64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory");
  return t;
}

struct Profiler {
  int64_t* data_ptr_ = nullptr;
  int sm_id_ = 0;
  int cnt_ = 0;
  int capacity_ = 0;

  __device__ void init(int num_entries, int64_t* data_ptr, int slot) {
    capacity_ = num_entries;
    data_ptr_ = data_ptr + slot * (1 + num_entries * kProfilerEntryStride);
    asm volatile("mov.u32 %0, %smid;" : "=r"(sm_id_));
    cnt_ = 0;
  }

  __device__ void start(ProfilerTag tag) {
    if (cnt_ >= capacity_) {
      return;
    }
    data_ptr_[1 + cnt_ * kProfilerEntryStride + 0] = sm_id_;
    data_ptr_[1 + cnt_ * kProfilerEntryStride + 1] =
        static_cast<int64_t>(tag);
    data_ptr_[1 + cnt_ * kProfilerEntryStride + 2] = globaltimer();
  }

  __device__ void stop() {
    if (cnt_ >= capacity_) {
      return;
    }
    data_ptr_[1 + cnt_ * kProfilerEntryStride + 3] =
        globaltimer() - data_ptr_[1 + cnt_ * kProfilerEntryStride + 2];
    cnt_ += 1;
  }

  __device__ void flush() {
    if (data_ptr_ != nullptr) {
      data_ptr_[0] = cnt_;
    }
  }
};
