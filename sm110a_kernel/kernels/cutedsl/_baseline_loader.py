"""Load CuTe DSL baseline modules from baselinedir/TensorRT-LLM.

The baseline lives at
    baselinedir/TensorRT-LLM/tensorrt_llm/_torch/cute_dsl_kernels
and is consumed by sm110a_kernel via the
``baselinedir/cute_dsl_kernels -> TensorRT-LLM/tensorrt_llm/_torch/cute_dsl_kernels``
symlink.

Recent baseline kernels (e.g. ``blackwell/dense_gemm_persistent.py`` added in
the Rubin SM107 PR) use intra-trtllm relative imports such as
``from ...cute_dsl_utils import ...`` which only resolve when the module is
loaded with the full ``tensorrt_llm._torch.cute_dsl_kernels.blackwell``
package context. Loading the directory directly on ``sys.path`` breaks those
imports with ``ImportError: attempted relative import beyond top-level
package``.

We rebuild that package context here without executing the real
``tensorrt_llm/__init__.py`` (which would launch OpenMPI's ORTE and abort on
hosts without a working ``mpirun``/``orterun``). The stubs only set
``__path__``; nothing in ``tensorrt_llm`` or ``tensorrt_llm._torch`` runs.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRTLLM_DIR = _REPO_ROOT / "baselinedir" / "TensorRT-LLM" / "tensorrt_llm"
_TORCH_DIR = _REPO_ROOT / "baselinedir" / "TensorRT-LLM" / "tensorrt_llm" / "_torch"
_KERNEL_ROOT = (
    _REPO_ROOT
    / "baselinedir"
    / "TensorRT-LLM"
    / "tensorrt_llm"
    / "_torch"
    / "cute_dsl_kernels"
)
_BLACKWELL_DIR = _KERNEL_ROOT / "blackwell"

_STUB_PACKAGES = (
    ("tensorrt_llm", _TRTLLM_DIR),
    ("tensorrt_llm._torch", _TORCH_DIR),
    ("tensorrt_llm._torch.cute_dsl_kernels", _KERNEL_ROOT),
    ("tensorrt_llm._torch.cute_dsl_kernels.blackwell", _BLACKWELL_DIR),
)

# Modules that must be pre-loaded (because other baseline modules reach for
# them via intra-trtllm relative imports) so their relative imports resolve
# through the stubs above.
_PREREQ_LOADERS = (
    ("tensorrt_llm.logger", _TRTLLM_DIR / "logger.py"),
    ("tensorrt_llm._torch.cute_dsl_utils", _TORCH_DIR / "cute_dsl_utils.py"),
)


def _ensure_stub_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _ensure_prereqs() -> None:
    for name, path in _PREREQ_LOADERS:
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot build import spec for {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


def install_baseline_packages() -> None:
    """Stub the ``tensorrt_llm`` package tree so baseline relative imports work.

    Safe to call multiple times.
    """
    for name, path in _STUB_PACKAGES:
        _ensure_stub_package(name, path)
    _ensure_prereqs()
