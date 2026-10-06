"""PyTorch-friendly helpers for CUDA Green Context SM partitions.

Each :class:`GreenContext` owns a CUDA Driver Green Context and one nonblocking
stream.  Use :meth:`GreenContext.torch_stream` with ``torch.cuda.stream`` to
submit PyTorch work to that context's SM partition.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "csrc/green_context/green_context_torch_binding.cpp"


@lru_cache(maxsize=1)
def _binding():
    try:
        from torch.utils.cpp_extension import load
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Green Context helpers require PyTorch with CUDA support installed."
        ) from exc
    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    return load(
        name=f"sm110a_green_context_{digest}",
        sources=[str(_SOURCE)],
        extra_cflags=["-O3", "-std=c++20"],
        extra_ldflags=["-lcuda"],
        verbose=False,
        with_cuda=True,
    )


class GreenContext:
    """An exact SM partition and its owned CUDA stream.

    Do not call :meth:`close` until all work issued to :meth:`torch_stream`
    has completed.  The object can also be used as a context manager.
    """

    def __init__(self, native) -> None:
        self._native = native

    @property
    def device(self) -> int:
        return self._native.device

    @property
    def sm_count(self) -> int:
        return self._native.sm_count

    @property
    def closed(self) -> bool:
        return self._native.closed

    @property
    def stream_ptr(self) -> int:
        """Raw ``cudaStream_t`` address, for interop with native code."""
        return self._native.stream_ptr

    def torch_stream(self):
        """Return this context's stream as a ``torch.cuda.ExternalStream``."""
        import torch

        if self.closed:
            raise RuntimeError("GreenContext is closed")
        return torch.cuda.ExternalStream(self.stream_ptr, device=self.device)

    def synchronize(self) -> None:
        self._native.synchronize()

    def close(self) -> None:
        self._native.close()

    def __enter__(self) -> GreenContext:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def sm_resource_info(device: int | None = None) -> dict[str, int]:
    """Return Green Context SM partition constraints for a CUDA device."""
    import torch

    if device is None:
        device = torch.cuda.current_device()
    return dict(_binding().sm_resource_info(int(device)))


def create_green_contexts(
    sm_counts: Sequence[int], *, device: int | None = None, priority: int = 0
) -> tuple[GreenContext, ...]:
    """Split a device into exact, disjoint Green Context SM partitions.

    Every requested count must satisfy the device's reported partition size and
    co-scheduled alignment.  SMs not listed remain available to the primary
    PyTorch context.
    """
    import torch

    if device is None:
        device = torch.cuda.current_device()
    counts = tuple(int(count) for count in sm_counts)
    return tuple(
        GreenContext(context)
        for context in _binding().create_green_contexts(
            counts, int(device), int(priority)
        )
    )


def create_green_context(
    sm_count: int, *, device: int | None = None, priority: int = 0
) -> GreenContext:
    """Create one Green Context with exactly ``sm_count`` SMs."""
    return create_green_contexts((sm_count,), device=device, priority=priority)[0]


def split_green_contexts(
    sm_count: int,
    count: int,
    *,
    device: int | None = None,
    priority: int = 0,
) -> tuple[GreenContext, ...]:
    """Create ``count`` equal Green Contexts, each with ``sm_count`` SMs."""
    if count <= 0:
        raise ValueError("count must be positive")
    return create_green_contexts((sm_count,) * count, device=device, priority=priority)


__all__ = [
    "GreenContext",
    "create_green_context",
    "create_green_contexts",
    "sm_resource_info",
    "split_green_contexts",
]
