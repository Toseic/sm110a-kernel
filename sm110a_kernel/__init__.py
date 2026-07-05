import os
from pathlib import Path


def _load_extension():
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sm110a_kernel requires PyTorch with CUDA support installed."
        ) from exc

    root = Path(__file__).resolve().parent.parent
    cuda_arch = os.environ.get("CUDA_ARCH", "sm_110a")
    compute_arch = os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a")

    return load(
        name=f"sm110a_kernel_ext_{cuda_arch}",
        sources=[
            str(root / "csrc/bf16-gemm/src/torch_binding.cpp"),
            str(root / "csrc/bf16-gemm/src/bf16_gemm_dynamic_schedule.cu"),
        ],
        extra_include_paths=[str(root / "include")],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            f"-gencode=arch={compute_arch},code={cuda_arch}",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda", "-lcublas", "-lcublasLt"],
        verbose=False,
    )


_EXT = _load_extension()

from .profiler import (
    PROFILER_WARPS_PER_BLOCK,
    make_profiler_buffer,
    profiler_events,
    write_trace,
)


def bf16_gemm_dynamic_schedule(a, b):
    return _EXT.bf16_gemm_dynamic_schedule(a, b)


def bf16_gemm_dynamic_schedule_out(a, b, out):
    return _EXT.bf16_gemm_dynamic_schedule_out(a, b, out)


def bf16_gemm_dynamic_schedule_profile(a, b, profiler, num_entries):
    return _EXT.bf16_gemm_dynamic_schedule_profile(a, b, profiler, num_entries)


def cublaslt_gemm(a, b):
    return _EXT.cublaslt_gemm(a, b)


def cublaslt_gemm_out(a, b, out, workspace):
    return _EXT.cublaslt_gemm_out(a, b, out, workspace)


def cublaslt_gemm_out_batched(a, b, out, workspace, iters):
    return _EXT.cublaslt_gemm_out_batched(a, b, out, workspace, iters)


def bf16_grouped_m_gemm(a, b, grouped_layout, config=None):
    from .jit import get_bf16_grouped_m_gemm_module

    module = get_bf16_grouped_m_gemm_module(a, b, config)
    return module.bf16_grouped_m_gemm(a, b, grouped_layout)


def bf16_grouped_m_gemm_out(a, b, grouped_layout, out, config=None):
    from .jit import get_bf16_grouped_m_gemm_module

    module = get_bf16_grouped_m_gemm_module(a, b, config)
    return module.bf16_grouped_m_gemm_out(a, b, grouped_layout, out)


from .jit import get_bf16_gemm_module as _get_bf16_gemm_module

_bf16_module_cache = {}


def _cached_bf16_module(a, b, config):
    # Use object identity of config (or its hash) as cache key.
    # Since Bf16GemmConfig is frozen, hash is stable.
    key = hash(config) if config is not None else 0
    mod = _bf16_module_cache.get(key)
    if mod is None:
        mod = _get_bf16_gemm_module(a, b, config)
        _bf16_module_cache[key] = mod
    return mod


def bf16_gemm(a, b, config=None):
    return _cached_bf16_module(a, b, config).bf16_gemm(a, b)


def bf16_gemm_out(a, b, out, config=None):
    return _cached_bf16_module(a, b, config).bf16_gemm_out(a, b, out)


from .jit import (
    Bf16GemmConfig,
    GroupedMConfig,
    compile_bf16_gemm,
    compile_bf16_grouped_m_gemm,
)
