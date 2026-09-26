"""Minimal NCCL Device API all-reduce over an LSA team."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/nccl_lsa/src"
_INCLUDE_DIR = _ROOT / "include"
_SOURCES = (
    _SOURCE_DIR / "nccl_lsa_torch_binding.cpp",
    _SOURCE_DIR / "nccl_lsa_kernel.cu",
)
_DEPENDENCIES = (_INCLUDE_DIR / "nccl_lsa.h",)


def _nccl_install() -> tuple[Path, str]:
    try:
        package = distribution("nvidia-nccl-cu13")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "NCCL LSA requires the nvidia-nccl-cu13 Python package"
        ) from exc
    root = Path(package.locate_file("nvidia/nccl"))
    if not (root / "include/nccl_device.h").is_file():
        raise RuntimeError(f"NCCL Device API headers not found under {root}")
    if not (root / "lib/libnccl.so.2").is_file():
        raise RuntimeError(f"NCCL shared library not found under {root}")
    return root, package.version


@lru_cache(maxsize=1)
def _extension():
    import torch
    from torch.utils.cpp_extension import load

    nccl_root, nccl_version = _nccl_install()
    cuda_arch = os.environ.get("CUDA_ARCH", "sm_110a")
    compute_arch = os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a")
    digest = hashlib.sha256()
    digest.update(torch.__version__.encode())
    digest.update(nccl_version.encode())
    digest.update(cuda_arch.encode())
    digest.update(compute_arch.encode())
    for path in (*_SOURCES, *_DEPENDENCIES):
        digest.update(path.read_bytes())
    name = f"sm110a_nccl_lsa_{digest.hexdigest()[:16]}"
    return load(
        name=name,
        sources=[str(path) for path in _SOURCES],
        extra_include_paths=[
            str(_INCLUDE_DIR),
            str(nccl_root / "include"),
        ],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-std=c++20",
            f"-gencode=arch={compute_arch},code={cuda_arch}",
        ],
        extra_ldflags=[
            f"-L{nccl_root / 'lib'}",
            "-l:libnccl.so.2",
            f"-Wl,-rpath,{nccl_root / 'lib'}",
            "-lcuda",
        ],
        verbose=False,
        with_cuda=True,
    )


def nccl_unique_id() -> bytes:
    """Return opaque bytes shared by every rank in one communicator."""
    return _extension().nccl_unique_id()


class NcclLsaAllReduce:
    """One-CTA float32 all-reduce implemented with NCCL LSA primitives.

    Every rank must construct the object collectively with the same unique ID,
    world size and capacity. Calls must use the same element count and order on
    every rank. Multiple processes sharing one GPU require MPS and
    ``NCCL_MULTI_RANK_GPU_ENABLE=1``.
    """

    def __init__(
        self,
        unique_id: bytes,
        rank: int,
        world_size: int,
        capacity: int,
        device: int = 0,
    ) -> None:
        self._native = _extension().NcclLsaAllReduce(
            unique_id, rank, world_size, device, capacity
        )

    @property
    def rank(self) -> int:
        return self._native.rank

    @property
    def world_size(self) -> int:
        return self._native.world_size

    @property
    def device(self) -> int:
        return self._native.device

    @property
    def capacity(self) -> int:
        return self._native.capacity

    @property
    def closed(self) -> bool:
        return self._native.closed

    def all_reduce(self, input):
        return self._native.all_reduce(input)

    def all_reduce_out(self, input, output):
        return self._native.all_reduce_out(input, output)

    def close(self) -> None:
        self._native.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
