"""Fixed-shape two-CTA NVFP4 FC1+FC2 kernel."""

from __future__ import annotations

from pathlib import Path

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "csrc/nvfp4_moe/src"
_INC = _ROOT / "include"


class Nvfp4Fc1Fc2Module:
    def __init__(self, binding, library, _config):
        self._binding = binding
        self._address = library.entry_address
        self.library_path = library.path

    def run(self, a, w1, sfa, sfw1, w2, sfw2):
        return self._binding.nvfp4_fc1fc2(
            a, w1, sfa, sfw1, w2, sfw2, self._address
        )

    def run_out(self, a, w1, sfa, sfw1, w2, sfw2, output):
        return self._binding.nvfp4_fc1fc2_out(
            a, w1, sfa, sfw1, w2, sfw2, output, self._address
        )


_JIT = SplitCudaJit(
    name="sm110a_nvfp4_fc1fc2",
    root=_ROOT,
    binding_sources=(_SRC / "nvfp4_fc1fc2_torch_binding.cpp",),
    kernel_sources=(_SRC / "nvfp4_fc1fc2_kernel.cu",),
    kernel_dependencies=(_INC / "nvfp4_fc1fc2.h", _INC / "common.h"),
    kernel_symbol="sm110a_nvfp4_fc1fc2_launch",
    config_key=lambda config: config,
    kernel_cflags=lambda _config, _variant: (),
    make_facade=Nvfp4Fc1Fc2Module,
    include_paths=(_INC,),
)


def compile_nvfp4_fc1fc2():
    return _JIT.compile(None)


def nvfp4_fc1fc2(a, w1, sfa, sfw1, w2, sfw2):
    return compile_nvfp4_fc1fc2().run(a, w1, sfa, sfw1, w2, sfw2)


def nvfp4_fc1fc2_out(a, w1, sfa, sfw1, w2, sfw2, output):
    return compile_nvfp4_fc1fc2().run_out(
        a, w1, sfa, sfw1, w2, sfw2, output
    )
