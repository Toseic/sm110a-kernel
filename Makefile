CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_110a
COMPUTE_CUDA_ARCH ?= compute_110a
BENCH ?= bf16_gemm
BENCH_ARGS ?=

PROJECT_DIR := csrc/bf16-gemm
SRC_DIR := $(PROJECT_DIR)/src
BENCH_DIR := benchmarks
INC_DIR := include
BUILD_DIR := build

NVCCFLAGS := -std=c++17 -O3 -lineinfo -gencode arch=$(COMPUTE_CUDA_ARCH),code=$(CUDA_ARCH) -I$(INC_DIR)
LDFLAGS := -lcublas -lcublasLt -lcuda

COMMON_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
BENCH_SOURCES := $(wildcard $(BENCH_DIR)/benchmark_*.cu)
BENCH_NAMES := $(patsubst $(BENCH_DIR)/benchmark_%.cu,%,$(BENCH_SOURCES))
BENCH_TARGET := $(BUILD_DIR)/benchmark_$(BENCH)
ALL_TARGETS := $(patsubst $(BENCH_DIR)/%.cu,$(BUILD_DIR)/%,$(BENCH_SOURCES))

.PHONY: all benchmark bench bench-list clean

all: $(ALL_TARGETS)

benchmark: $(BENCH_TARGET)

bench: $(BENCH_TARGET)
	./$(BENCH_TARGET) $(BENCH_ARGS)

bench-list:
	@printf '%s\n' $(BENCH_NAMES)

$(BUILD_DIR)/benchmark_%: $(COMMON_SOURCES) $(BENCH_DIR)/benchmark_%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
