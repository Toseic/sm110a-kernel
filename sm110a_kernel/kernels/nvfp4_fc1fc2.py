"""Fixed-shape two-CTA NVFP4 FC1+FC2 streaming kernel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..jit_runtime import SplitCudaJit

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "csrc/nvfp4_moe/src"
_INC = _ROOT / "include"


@dataclass(frozen=True)
class Nvfp4Fc1Fc2Config:
    """Compile-time streaming-buffer configuration.

    ``buffer_i`` is the number of logical FC1 output columns retained in
    shared memory before switching to FC2.  It must tile the fixed I=2048
    dimension in 256-column mainloop stages.
    """

    buffer_i: int = 2048

    def validate(self) -> None:
        if self.buffer_i not in (256, 512, 1024, 2048):
            raise ValueError("buffer_i must be one of 256, 512, 1024, or 2048")


class Nvfp4Fc1Fc2Module:
    def __init__(self, binding, library, config):
        self._binding = binding
        self._address = library.entry_address
        self.config = config
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
    kernel_cflags=lambda config, _variant: (
        f"-DSM110A_NVFP4_FC1FC2_BUFFER_I={config.buffer_i}",
    ),
    make_facade=Nvfp4Fc1Fc2Module,
    include_paths=(_INC,),
)


def compile_nvfp4_fc1fc2(config: Optional[Nvfp4Fc1Fc2Config] = None):
    config = config or Nvfp4Fc1Fc2Config()
    if not isinstance(config, Nvfp4Fc1Fc2Config):
        raise TypeError("config must be an Nvfp4Fc1Fc2Config")
    config.validate()
    return _JIT.compile(config)


def compile_nvfp4_fc1fc2_configs(
    configs: Iterable[Nvfp4Fc1Fc2Config], max_workers=None
):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Nvfp4Fc1Fc2Config):
            raise TypeError("configs must contain Nvfp4Fc1Fc2Config values")
        config.validate()
    return _JIT.compile_many(configs, max_workers) if configs else ()


def nvfp4_fc1fc2(a, w1, sfa, sfw1, w2, sfw2, config=None):
    return compile_nvfp4_fc1fc2(config).run(a, w1, sfa, sfw1, w2, sfw2)


def nvfp4_fc1fc2_out(a, w1, sfa, sfw1, w2, sfw2, output, config=None):
    return compile_nvfp4_fc1fc2(config).run_out(
        a, w1, sfa, sfw1, w2, sfw2, output
    )
