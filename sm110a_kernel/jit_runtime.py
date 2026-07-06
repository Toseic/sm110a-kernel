"""Reusable runtime for split-binding, configuration-specialized CUDA JITs."""

from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Generic, Hashable, Iterable, Optional, TypeVar


__all__ = [
    "LoadedCudaLibrary",
    "SplitCudaJit",
    "jit_build_directory",
    "source_digest",
]


ConfigT = TypeVar("ConfigT")
FacadeT = TypeVar("FacadeT")


@dataclass(frozen=True)
class LoadedCudaLibrary:
    """A loaded config-specific CUDA library and its exported entry point."""

    name: str
    path: str
    handle: ctypes.CDLL
    entry_address: int


def source_digest(paths: Iterable[Path], root: Path) -> str:
    """Hash source contents and root-relative paths."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def jit_build_directory(
    name: str,
    cache_root_env: str = "SM110A_JIT_CACHE_DIR",
) -> Optional[str]:
    """Return a persistent per-module build directory when configured."""
    cache_root = os.environ.get(cache_root_env)
    if not cache_root:
        return None
    build_path = Path(cache_root).expanduser().resolve() / name
    build_path.mkdir(parents=True, exist_ok=True)
    return str(build_path)


class SplitCudaJit(Generic[ConfigT, FacadeT]):
    """Compile one stable Torch binding plus specialized CUDA libraries.

    The binding and CUDA libraries are built concurrently. Batch compilation
    also builds independent CUDA configurations concurrently while preserving
    per-module locks and both in-process and torch-extension disk caches.
    """

    def __init__(
        self,
        *,
        name: str,
        root: Path,
        binding_sources: Iterable[Path],
        binding_dependencies: Iterable[Path] = (),
        kernel_sources: Iterable[Path],
        kernel_dependencies: Iterable[Path],
        kernel_symbol: str,
        config_key: Callable[[ConfigT], Hashable],
        kernel_cflags: Callable[[ConfigT, Hashable], Iterable[str]],
        make_facade: Callable[
            [ModuleType, LoadedCudaLibrary, ConfigT], FacadeT
        ],
        variant_key: Callable[[], Hashable] = lambda: None,
        include_paths: Iterable[Path] = (),
        binding_cflags: Iterable[str] = ("-O3", "-std=c++17"),
        binding_ldflags: Iterable[str] = ("-lcuda",),
        kernel_ldflags: Iterable[str] = ("-lcuda",),
        cache_root_env: str = "SM110A_JIT_CACHE_DIR",
        verbose_env: str = "SM110A_JIT_VERBOSE",
        workers_env: str = "SM110A_JIT_MAX_WORKERS",
    ) -> None:
        self._name = name
        self._root = root
        self._binding_sources = tuple(binding_sources)
        self._binding_dependencies = tuple(binding_dependencies)
        self._kernel_sources = tuple(kernel_sources)
        self._kernel_dependencies = tuple(kernel_dependencies)
        self._kernel_symbol = kernel_symbol
        self._config_key = config_key
        self._kernel_cflags = kernel_cflags
        self._make_facade = make_facade
        self._variant_key = variant_key
        self._include_paths = tuple(include_paths)
        self._binding_cflags = tuple(binding_cflags)
        self._binding_ldflags = tuple(binding_ldflags)
        self._kernel_ldflags = tuple(kernel_ldflags)
        self._cache_root_env = cache_root_env
        self._verbose_env = verbose_env
        self._workers_env = workers_env

        self._binding: Optional[ModuleType] = None
        self._binding_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._build_locks: dict[str, threading.Lock] = {}
        self._kernel_cache: dict[str, LoadedCudaLibrary] = {}
        self._facade_cache: dict[str, FacadeT] = {}
        self._fast_cache: dict[tuple[Hashable, str, str, Hashable], FacadeT] = {}

    def _verbose(self) -> bool:
        return os.environ.get(self._verbose_env, "0") == "1"

    def _build_directory(self, name: str) -> Optional[str]:
        return jit_build_directory(name, self._cache_root_env)

    def _binding_name(self, torch) -> str:
        signature = repr(
            (
                torch.__version__,
                torch.version.cuda,
                sys.implementation.cache_tag,
                tuple(str(path) for path in self._include_paths),
                self._binding_cflags,
                self._binding_ldflags,
                source_digest(
                    self._binding_sources + self._binding_dependencies,
                    self._root,
                ),
            )
        )
        suffix = hashlib.sha256(signature.encode()).hexdigest()[:16]
        return f"{self._name}_binding_{suffix}"

    def _kernel_name(
        self,
        config: ConfigT,
        variant: Hashable,
        cuda_arch: str,
        compute_arch: str,
        torch,
        extra_cuda_cflags: tuple[str, ...],
    ) -> str:
        signature = repr(
            (
                self._config_key(config),
                variant,
                cuda_arch,
                compute_arch,
                torch.__version__,
                torch.version.cuda,
                sys.implementation.cache_tag,
                self._kernel_symbol,
                tuple(str(path) for path in self._include_paths),
                extra_cuda_cflags,
                self._kernel_ldflags,
                source_digest(
                    self._kernel_sources + self._kernel_dependencies,
                    self._root,
                ),
            )
        )
        suffix = hashlib.sha256(signature.encode()).hexdigest()[:16]
        return f"{self._name}_kernel_{suffix}"

    def _fast_key(
        self,
        config: ConfigT,
    ) -> tuple[Hashable, str, str, Hashable]:
        return (
            self._config_key(config),
            os.environ.get("CUDA_ARCH", "sm_110a"),
            os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a"),
            self._variant_key(),
        )

    def _load_binding(self) -> ModuleType:
        with self._binding_lock:
            if self._binding is not None:
                return self._binding

            import torch
            from torch.utils.cpp_extension import load

            name = self._binding_name(torch)
            self._binding = load(
                name=name,
                sources=[str(path) for path in self._binding_sources],
                extra_include_paths=[
                    str(path) for path in self._include_paths
                ],
                extra_cflags=list(self._binding_cflags),
                extra_ldflags=list(self._binding_ldflags),
                build_directory=self._build_directory(name),
                verbose=self._verbose(),
                with_cuda=True,
            )
            return self._binding

    def _load_kernel(self, config: ConfigT) -> LoadedCudaLibrary:
        import torch
        from torch.utils.cpp_extension import load

        cuda_arch = os.environ.get("CUDA_ARCH", "sm_110a")
        compute_arch = os.environ.get(
            "COMPUTE_CUDA_ARCH", "compute_110a"
        )
        variant = self._variant_key()
        extra_cuda_cflags = tuple(self._kernel_cflags(config, variant))
        name = self._kernel_name(
            config,
            variant,
            cuda_arch,
            compute_arch,
            torch,
            extra_cuda_cflags,
        )
        with self._cache_lock:
            cached = self._kernel_cache.get(name)
            if cached is not None:
                return cached
            build_lock = self._build_locks.setdefault(
                name, threading.Lock()
            )

        with build_lock:
            with self._cache_lock:
                cached = self._kernel_cache.get(name)
                if cached is not None:
                    return cached
            try:
                library_path = load(
                    name=name,
                    sources=[str(path) for path in self._kernel_sources],
                    extra_include_paths=[
                        str(path) for path in self._include_paths
                    ],
                    extra_cuda_cflags=[
                        "-O3",
                        "-lineinfo",
                        f"-gencode=arch={compute_arch},code={cuda_arch}",
                        *extra_cuda_cflags,
                    ],
                    extra_ldflags=list(self._kernel_ldflags),
                    build_directory=self._build_directory(name),
                    verbose=self._verbose(),
                    is_python_module=False,
                )
                handle = ctypes.CDLL(library_path)
                entry = getattr(handle, self._kernel_symbol)
                entry_address = ctypes.cast(
                    entry, ctypes.c_void_p
                ).value
                if entry_address is None:
                    raise RuntimeError(
                        f"failed to resolve CUDA symbol {self._kernel_symbol}"
                    )
                library = LoadedCudaLibrary(
                    name, library_path, handle, entry_address
                )
            except Exception:
                with self._cache_lock:
                    self._build_locks.pop(name, None)
                raise

            with self._cache_lock:
                self._kernel_cache[name] = library
                self._build_locks.pop(name, None)
            return library

    def _facade(
        self,
        config: ConfigT,
        binding: ModuleType,
        library: LoadedCudaLibrary,
    ) -> FacadeT:
        with self._cache_lock:
            cached = self._facade_cache.get(library.name)
            if cached is not None:
                return cached
            facade = self._make_facade(binding, library, config)
            self._facade_cache[library.name] = facade
            return facade

    def compile(self, config: ConfigT) -> FacadeT:
        """Compile or load one configuration."""
        fast_key = self._fast_key(config)
        with self._cache_lock:
            cached = self._fast_cache.get(fast_key)
            if cached is not None:
                return cached

        with ThreadPoolExecutor(max_workers=2) as executor:
            binding_future = executor.submit(self._load_binding)
            kernel_future = executor.submit(self._load_kernel, config)
            binding = binding_future.result()
            library = kernel_future.result()
        facade = self._facade(config, binding, library)
        with self._cache_lock:
            self._fast_cache[fast_key] = facade
        return facade

    def compile_many(
        self,
        configs: Iterable[ConfigT],
        max_workers: Optional[int] = None,
    ) -> tuple[FacadeT, ...]:
        """Compile configurations and their shared binding concurrently."""
        configs = tuple(configs)
        if not configs:
            return ()
        if max_workers is None:
            configured_workers = int(
                os.environ.get(self._workers_env, "4")
            )
            max_workers = min(len(configs), configured_workers)
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")

        with ThreadPoolExecutor(max_workers=max_workers + 1) as executor:
            binding_future = executor.submit(self._load_binding)
            kernel_futures = [
                executor.submit(self._load_kernel, config)
                for config in configs
            ]
            binding = binding_future.result()
            libraries = [
                future.result() for future in kernel_futures
            ]
        facades = tuple(
            self._facade(config, binding, library)
            for config, library in zip(configs, libraries)
        )
        with self._cache_lock:
            for config, facade in zip(configs, facades):
                self._fast_cache[self._fast_key(config)] = facade
        return facades
