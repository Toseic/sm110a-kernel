#!/usr/bin/env python3
"""Compute the exact model-resident size without allocating CUDA memory."""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "inference"))
from model import ModelArgs, Transformer, precompute_freqs_cis  # noqa: E402


def gib(num_bytes: int) -> float:
    return num_bytes / 2**30


def model_storage_bytes(config: dict) -> tuple[int, int]:
    """Return parameter and registered-buffer storage for an official model.

    PyTorch's meta device creates only tensor metadata, so this is exact for
    parameter dtypes and quantization-scale tensors but consumes no GPU memory.
    """
    # The official runtime caches RoPE frequency tables. Do not retain the
    # meta-device table: the subsequently constructed CUDA model must create a
    # CUDA table of its own.
    precompute_freqs_cis.cache_clear()
    try:
        args = ModelArgs(**config)
        with torch.device("meta"):
            model = Transformer(args)
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    finally:
        precompute_freqs_cis.cache_clear()
    return parameter_bytes, buffer_bytes


def main() -> None:
    config = json.loads((ROOT / "benchmark_config_32g.json").read_text())
    parameter_bytes, buffer_bytes = model_storage_bytes(config)
    resident_bytes = parameter_bytes + buffer_bytes
    runtime_reserve_bytes = 6 * 2**30

    print(f"parameters: {gib(parameter_bytes):.2f} GiB")
    print(f"buffers (including KV cache): {gib(buffer_bytes):.2f} GiB")
    print(f"model-resident total: {gib(resident_bytes):.2f} GiB")
    print(f"resident + 6 GiB runtime reserve: {gib(resident_bytes + runtime_reserve_bytes):.2f} GiB")
    if resident_bytes + runtime_reserve_bytes > 32 * 2**30:
        raise SystemExit("configuration exceeds the 32-GiB budget")


if __name__ == "__main__":
    main()
