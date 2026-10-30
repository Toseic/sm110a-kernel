"""Configuration-specialized standard BF16 GEMM kernel."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import BaseKernelRunner, PreparedKernel
from .pdl import validate_pdl

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16_gemm/src"
_INCLUDE_DIR = _ROOT / "include"


@dataclass(frozen=True)
class Bf16GemmConfig:
    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 128
    cluster_m: int = 2
    epilogue_block_n: int = 32
    split_k: int = 1
    output_fp32: bool = False

    pdl: bool = False

    def validate(self) -> None:
        validate_pdl(self.pdl)
        if not isinstance(self.output_fp32, bool):
            raise TypeError("output_fp32 must be a bool")
        if self.tile_m not in (64, 128):
            raise ValueError("tile_m must be 64 or 128")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
        if self.tile_m == 64 and self.cluster_m != 1:
            raise ValueError("tile_m=64 requires cluster_m=1")
        if self.split_k not in (1, 2, 4, 8) or self.cluster_m * self.split_k > 8:
            raise ValueError(
                "split_k must be 1, 2, 4, or 8 and cluster_m * split_k <= 8"
            )
        if self.tile_n <= 0:
            raise ValueError("tile_n must be positive")
        if self.epilogue_block_n not in (32, 64, 128):
            raise ValueError("epilogue_block_n must be one of 32, 64, or 128")
        if self.tile_n % self.epilogue_block_n:
            raise ValueError("tile_n must be divisible by epilogue_block_n")
        if self.tile_n % self.cluster_m:
            raise ValueError("tile_n must be divisible by cluster_m")
        if self.tile_k <= 0 or self.tile_k % 64:
            raise ValueError("tile_k must be a positive multiple of 64")
        epilogue_smem = (
            2 * self.tile_m * self.epilogue_block_n * (4 if self.output_fp32 else 2)
        )
        mainloop_stage_smem = (
            (self.tile_m + self.tile_n // self.cluster_m) * self.tile_k * 2
        )
        reduction_smem = (
            self.tile_m * self.epilogue_block_n * 4 if self.split_k > 1 else 0
        )
        if (227 * 1024 - epilogue_smem - reduction_smem) // mainloop_stage_smem < 2:
            raise ValueError(
                "configuration leaves room for fewer than two mainloop stages"
            )


def _default_config(a, b) -> Bf16GemmConfig:
    if a.ndim != 2 or b.ndim != 2:
        return Bf16GemmConfig()
    tile_m = 64 if a.size(0) <= 64 else 128
    return Bf16GemmConfig(tile_m=tile_m, cluster_m=1 if a.size(0) <= tile_m else 2)


class Bf16GemmModule:
    def __init__(self, binding, library, config: Bf16GemmConfig):
        self._binding, self._launcher, self._config = (
            binding,
            library.entry_address,
            config,
        )

    def _args(self, device, max_active_clusters=None):
        c = self._config
        if max_active_clusters is None:
            import torch

            sm_count = torch.cuda.get_device_properties(device).multi_processor_count
            max_active_clusters = sm_count // (c.cluster_m * c.split_k)
        if max_active_clusters <= 0:
            raise ValueError("max_active_clusters must be positive")
        return (
            self._launcher,
            c.tile_m,
            c.tile_n,
            c.tile_k,
            c.cluster_m,
            c.split_k,
            int(max_active_clusters),
        )

    def bf16_gemm(self, a, b, swap_ab=False, max_active_clusters=None):
        return self._binding.bf16_gemm(
            a,
            b,
            *self._args(a.device, max_active_clusters),
            swap_ab,
            self._config.output_fp32,
        )

    def bf16_gemm_out(self, a, b, out, swap_ab=False, max_active_clusters=None):
        return self._binding.bf16_gemm_out(
            a,
            b,
            out,
            *self._args(a.device, max_active_clusters),
            swap_ab,
            self._config.output_fp32,
        )

    def bf16_gemm_nt(self, a, b_nt, swap_ab=False, max_active_clusters=None):
        return self._binding.bf16_gemm_nt(
            a,
            b_nt,
            *self._args(a.device, max_active_clusters),
            swap_ab,
            self._config.output_fp32,
        )

    def bf16_gemm_nt_out(self, a, b_nt, out, swap_ab=False, max_active_clusters=None):
        return self._binding.bf16_gemm_nt_out(
            a,
            b_nt,
            out,
            *self._args(a.device, max_active_clusters),
            swap_ab,
            self._config.output_fp32,
        )


def _kernel_cflags(config, _variant):
    return (f"-DSM110A_PDL={int(config.pdl)}",) + tuple(
        f"-DSM110A_BF16_{name}={value}"
        for name, value in (
            ("TILE_M", config.tile_m),
            ("TILE_N", config.tile_n),
            ("TILE_K", config.tile_k),
            ("CLUSTER_M", config.cluster_m),
            ("SPLIT_K", config.split_k),
            ("OUTPUT_FP32", int(config.output_fp32)),
            ("EPILOGUE_BLOCK_N", config.epilogue_block_n),
        )
    )


_JIT = SplitCudaJit(
    name="sm110a_bf16_gemm",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16_gemm_split_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16_gemm_kernel.cu",),
    kernel_dependencies=(
        _INCLUDE_DIR / "bf16_gemm.h",
        _INCLUDE_DIR / "common.h",
        _SOURCE_DIR / "bf16_gemm_pipeline_delay.cuh",
    ),
    kernel_symbol="sm110a_bf16_gemm_launch",
    config_key=lambda config: config,
    kernel_cflags=_kernel_cflags,
    make_facade=Bf16GemmModule,
    include_paths=(_INCLUDE_DIR,),
)


def compile_bf16_gemm(config: Bf16GemmConfig | None = None) -> Bf16GemmModule:
    config = config or Bf16GemmConfig()
    if not isinstance(config, Bf16GemmConfig):
        raise TypeError("config must be a Bf16GemmConfig")
    config.validate()
    return _JIT.compile(config)


def compile_bf16_gemm_configs(configs: Iterable[Bf16GemmConfig], max_workers=None):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Bf16GemmConfig):
            raise TypeError("configs must contain Bf16GemmConfig instances")
        config.validate()
    return _JIT.compile_many(configs, max_workers)


def get_bf16_gemm_module(a, b, config=None):
    return compile_bf16_gemm(config or _default_config(a, b))


def bf16_gemm(a, b, config=None, swap_ab=False, max_active_clusters=None):
    return get_bf16_gemm_module(a, b, config).bf16_gemm(
        a, b, swap_ab, max_active_clusters
    )


def bf16_gemm_out(a, b, out, config=None, swap_ab=False, max_active_clusters=None):
    return get_bf16_gemm_module(a, b, config).bf16_gemm_out(
        a, b, out, swap_ab, max_active_clusters
    )


def bf16_gemm_nt_out(
    a, b_nt, out, config=None, swap_ab=False, max_active_clusters=None
):
    """Compute ``a @ b_nt.T`` for a contiguous, pretransposed ``b_nt(N, K)``."""
    return compile_bf16_gemm(config or _default_config(a, b_nt)).bf16_gemm_nt_out(
        a, b_nt, out, swap_ab, max_active_clusters
    )


def _bf16_gemm_configs(output_fp32=False):
    for tile_m in (64, 128):
        for tile_n in (128, 256):
            for tile_k in (64, 128):
                for cluster_m in (1, 2):
                    for epilogue_block_n in (32, 64, 128):
                        if tile_n % epilogue_block_n:
                            continue
                        for split_k in (1, 2, 4, 8):
                            config = Bf16GemmConfig(
                                tile_m=tile_m,
                                tile_n=tile_n,
                                tile_k=tile_k,
                                cluster_m=cluster_m,
                                epilogue_block_n=epilogue_block_n,
                                split_k=split_k,
                                output_fp32=output_fp32,
                            )
                            try:
                                config.validate()
                            except ValueError:
                                continue
                            yield config


BF16_GEMM_CONFIGS = tuple(_bf16_gemm_configs())


def bf16_candidates(m, n, k, output_fp32=False):
    return tuple(
        (config, swap_ab)
        for config in (_bf16_gemm_configs(True) if output_fp32 else BF16_GEMM_CONFIGS)
        for swap_ab in ((False, True) if config.split_k == 1 else (False,))
        if k % (config.tile_k * config.split_k) == 0
        and (
            (
                not swap_ab
                and (
                    m % (config.tile_m * config.cluster_m) == 0
                    or (config.cluster_m == 1 and 0 < m < config.tile_m)
                )
                and n % config.tile_n == 0
            )
            or (
                swap_ab
                and n % (config.tile_m * config.cluster_m) == 0
                and m % config.tile_n == 0
            )
        )
    )


def format_bf16_candidate(candidate):
    config, swap_ab = candidate
    return (
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
        f"/c{config.cluster_m}"
        f"/e{config.epilogue_block_n}"
        f"/sk{config.split_k}"
        f"/{'swap' if swap_ab else 'noswap'}"
        f"/{'fp32' if config.output_fp32 else 'bf16'}/pdl{int(config.pdl)}"
    )


class Bf16GemmRunner(BaseKernelRunner):
    namespace = "bf16_gemm_nt_v6"
    format_candidate = staticmethod(format_bf16_candidate)

    def __init__(self, a, b_nt, out, *, max_active_clusters=None, pdl=False):
        validate_pdl(pdl)
        super().__init__(a.device)
        self.pdl = pdl
        self.a, self.b, self.out = a, b_nt, out
        self.m, self.k, self.n = a.size(0), a.size(1), b_nt.size(0)
        self.max_active_clusters = max_active_clusters

    @property
    def workload_key(self):
        return (
            self.m,
            self.n,
            self.k,
            self.max_active_clusters,
            str(self.out.dtype),
            self.pdl,
        )

    def compatible_candidates(self):
        candidates = bf16_candidates(
            self.m, self.n, self.k, self.out.dtype == torch.float32
        )
        # Model projections may be narrower than the benchmark's N>=128 family.
        if not candidates:
            candidates = tuple(
                (
                    Bf16GemmConfig(
                        tile_m=tm,
                        tile_n=tn,
                        tile_k=tk,
                        cluster_m=1,
                        output_fp32=self.out.dtype == torch.float32,
                    ),
                    False,
                )
                for tm in (64, 128)
                for tn in (32, 64)
                for tk in (64, 128)
                if self.n % tn == 0
                and self.k % tk == 0
                and (self.m % tm == 0 or self.m < tm)
            )
        return tuple((replace(c, pdl=self.pdl), swap) for c, swap in candidates)

    def pre_jit(self, candidate):
        return compile_bf16_gemm(candidate[0])

    def prepare(self, candidate):
        module = self.pre_jit(candidate)
        return PreparedKernel(
            candidate,
            lambda: module.bf16_gemm_nt_out(
                self.a, self.b, self.out, candidate[1], self.max_active_clusters
            ),
        )
