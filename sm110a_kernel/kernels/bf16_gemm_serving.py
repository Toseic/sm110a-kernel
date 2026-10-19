"""Configuration-specialized standard BF16 GEMM kernel."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from ..jit_runtime import SplitCudaJit
from ..runner import BaseKernelRunner, PreparedKernel

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "csrc/bf16_gemm_serving/src"
_INCLUDE_DIR = _ROOT / "include"


@dataclass(frozen=True)
class Bf16GemmServingConfig:
    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 128
    cluster_m: int = 2
    epilogue_block_n: int = 32

    def validate(self) -> None:
        if self.tile_m != 128:
            raise ValueError("tile_m must be 128 for the current four-warp mainloop")
        if self.cluster_m not in (1, 2):
            raise ValueError("cluster_m must be 1 or 2")
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
        epilogue_smem = 2 * self.tile_m * self.epilogue_block_n * 2
        mainloop_stage_smem = (
            (self.tile_m + self.tile_n // self.cluster_m) * self.tile_k * 2
        )
        if (227 * 1024 - epilogue_smem) // mainloop_stage_smem < 2:
            raise ValueError(
                "configuration leaves room for fewer than two mainloop stages"
            )


def _default_config(a, b) -> Bf16GemmServingConfig:
    if a.ndim != 2 or b.ndim != 2:
        return Bf16GemmServingConfig()
    return Bf16GemmServingConfig(cluster_m=1 if a.size(0) <= 128 else 2)


class Bf16GemmServingModule:
    def __init__(self, binding, library, config: Bf16GemmServingConfig):
        self._binding, self._library = binding, library
        self._launcher, self._config = library.entry_address, config
        self._queue_init = self._symbol_address(
            "sm110a_bf16_gemm_serving_queue_initialize"
        )
        self._persistent_launch = self._symbol_address(
            "sm110a_bf16_gemm_persistent_serving_launch"
        )
        self._request_drain = self._symbol_address(
            "sm110a_bf16_gemm_serving_request_drain"
        )
        self._test_publish = self._symbol_address(
            "sm110a_bf16_gemm_serving_test_publish"
        )
        self._test_consume = self._symbol_address(
            "sm110a_bf16_gemm_serving_test_consume"
        )

    def _symbol_address(self, name: str) -> int:
        symbol = getattr(self._library.handle, name)
        address = ctypes.cast(symbol, ctypes.c_void_p).value
        if address is None:
            raise RuntimeError(f"failed to resolve CUDA symbol {name}")
        return address

    def _args(self, device, max_active_clusters=None):
        c = self._config
        if max_active_clusters is None:
            import torch

            sm_count = torch.cuda.get_device_properties(device).multi_processor_count
            max_active_clusters = sm_count // c.cluster_m
        if max_active_clusters <= 0:
            raise ValueError("max_active_clusters must be positive")
        return (
            self._launcher,
            c.tile_m,
            c.tile_n,
            c.tile_k,
            c.cluster_m,
            int(max_active_clusters),
        )

    def bf16_gemm_serving(self, a, b, swap_ab=False, max_active_clusters=None):
        return self._binding.bf16_gemm_serving(
            a, b, *self._args(a.device, max_active_clusters), swap_ab
        )

    def bf16_gemm_serving_out(self, a, b, out, swap_ab=False, max_active_clusters=None):
        return self._binding.bf16_gemm_serving_out(
            a, b, out, *self._args(a.device, max_active_clusters), swap_ab
        )

    def bf16_gemm_serving_nt(self, a, b_nt, swap_ab=False, max_active_clusters=None):
        return self._binding.bf16_gemm_serving_nt(
            a, b_nt, *self._args(a.device, max_active_clusters), swap_ab
        )

    def bf16_gemm_serving_nt_out(
        self, a, b_nt, out, swap_ab=False, max_active_clusters=None
    ):
        return self._binding.bf16_gemm_serving_nt_out(
            a, b_nt, out, *self._args(a.device, max_active_clusters), swap_ab
        )

    def create_queue(
        self,
        b_nt,
        slot_count: int,
        max_active_clusters=None,
        *,
        persistent_stream=None,
        control_stream=None,
    ):
        return Bf16GemmPersistentServingQueue(
            self,
            b_nt,
            slot_count,
            max_active_clusters,
            persistent_stream=persistent_stream,
            control_stream=control_stream,
        )


class Bf16GemmPersistentServingQueue:
    """Owns the tensors, stream, and lifetime of one persistent serving launch."""

    _CONTROL_BYTES = 640
    _GROUP_META_BYTES = 32

    def __init__(
        self,
        module: Bf16GemmServingModule,
        b_nt,
        slot_count: int,
        max_active_clusters=None,
        *,
        persistent_stream=None,
        control_stream=None,
    ):

        config = module._config
        if config != Bf16GemmServingConfig():
            raise ValueError(
                "persistent serving v1 requires the fixed 128x256x128/c2/e32 tactic"
            )
        if (
            not isinstance(slot_count, int)
            or slot_count <= 0
            or slot_count % config.cluster_m
        ):
            raise ValueError("slot_count must be a positive multiple of cluster_m")
        if (
            not b_nt.is_cuda
            or b_nt.dtype != torch.bfloat16
            or not b_nt.is_contiguous()
            or b_nt.ndim != 2
        ):
            raise ValueError("B_nt must be a contiguous 2D CUDA bfloat16 tensor")
        n, k = b_nt.shape
        if n <= 0 or n % config.tile_n or k <= 0 or k % config.tile_k:
            raise ValueError("N and K must be positive and divisible by tile_n/tile_k")

        if max_active_clusters is None:
            sm_count = torch.cuda.get_device_properties(
                b_nt.device
            ).multi_processor_count
            max_active_clusters = sm_count // config.cluster_m
        if not isinstance(max_active_clusters, int) or max_active_clusters <= 0:
            raise ValueError("max_active_clusters must be positive")

        self.module = module
        self.b_nt = b_nt
        self.slot_count = slot_count
        self.group_capacity = slot_count // config.cluster_m
        self.n, self.k = n, k
        self.max_active_clusters = max_active_clusters
        self.a_ring = torch.empty(
            (slot_count, config.tile_m, k), dtype=torch.bfloat16, device=b_nt.device
        )
        self.c_ring = torch.empty(
            (slot_count, config.tile_m, n), dtype=torch.bfloat16, device=b_nt.device
        )
        self.control = torch.empty(
            self._CONTROL_BYTES + self.group_capacity * self._GROUP_META_BYTES,
            dtype=torch.uint8,
            device=b_nt.device,
        )
        self.stream = persistent_stream or torch.cuda.Stream(device=b_nt.device)
        self.control_stream = control_stream or torch.cuda.Stream(device=b_nt.device)
        for name, stream in (
            ("persistent_stream", self.stream),
            ("control_stream", self.control_stream),
        ):
            if stream.device != b_nt.device:
                raise ValueError(f"{name} must be on the B_nt device")
        self._started = False
        self._drain_requested = False
        self._stopped = False
        self._next_ticket = 0
        self._next_consume_ticket = 0
        self._published_valid_rows = {}

        # Initialize on the caller stream. start() makes the persistent stream
        # wait for this stream, while a subsequent producer/drain operation on
        # the caller stream is therefore ordered after initialization.
        module._binding.serving_queue_initialize(
            self.control, module._queue_init, slot_count, n, k, max_active_clusters
        )

    def start(self):

        if self._started:
            raise RuntimeError("serving queue has already been started")
        caller_stream = torch.cuda.current_stream(self.b_nt.device)
        self.stream.wait_stream(caller_stream)
        self.stream.wait_stream(self.control_stream)
        with torch.cuda.stream(self.stream):
            self.module._binding.persistent_serving_start(
                self.a_ring,
                self.b_nt,
                self.c_ring,
                self.control,
                self.module._persistent_launch,
                self.slot_count,
                self.n,
                self.k,
                self.max_active_clusters,
            )
        for tensor in (self.a_ring, self.b_nt, self.c_ring, self.control):
            tensor.record_stream(self.stream)
        self._started = True
        return self

    def request_drain(self):

        if self._stopped:
            return
        if self._drain_requested:
            return
        self.control_stream.wait_stream(torch.cuda.current_stream(self.b_nt.device))
        with torch.cuda.stream(self.control_stream):
            self.module._binding.serving_request_drain(
                self.control, self.module._request_drain
            )
        self._drain_requested = True

    def test_publish(self, rows, *, flush=False, delay_cycles=0, bypass=False):
        """Publish one group with the test producer (not a production enqueue API).

        With ``bypass=True`` the producer skips the A-ring payload copy and
        only runs the queue state machine, so timing reflects the serving
        kernel and queue rather than the test I/O path.
        """

        if self._drain_requested:
            raise RuntimeError("cannot publish after drain has been requested")
        if (
            not rows.is_cuda
            or rows.dtype != torch.bfloat16
            or not rows.is_contiguous()
            or rows.ndim != 2
            or rows.device != self.b_nt.device
            or rows.shape[1] != self.k
        ):
            raise ValueError("rows must be contiguous CUDA bfloat16 [valid_rows, K]")
        if not 0 < rows.shape[0] <= 256:
            raise ValueError("valid_rows must be in [1, 256]")
        if rows.shape[0] != 256 and not flush:
            raise ValueError("a partial group requires flush=True")
        if not isinstance(delay_cycles, int) or delay_cycles < 0:
            raise ValueError("delay_cycles must be a non-negative integer")
        self.module._binding.serving_test_publish(
            rows,
            self.a_ring,
            self.control,
            self.module._test_publish,
            self._next_ticket,
            flush,
            delay_cycles,
            bool(bypass),
        )
        ticket = self._next_ticket
        self._published_valid_rows[ticket] = rows.shape[0]
        self._next_ticket += 1
        return ticket

    def test_consume(self, ticket=None, out=None, bypass=False):
        """Consume the next OUTPUT_READY group and recycle its ring slot.

        With ``bypass=True`` the consumer skips the C-ring payload copy and
        only waits for OUTPUT_READY plus the recycle step.
        """

        if ticket is None:
            ticket = self._next_consume_ticket
        if ticket != self._next_consume_ticket:
            raise ValueError("test consumer must recycle tickets in order")
        try:
            valid_rows = self._published_valid_rows[ticket]
        except KeyError as exc:
            raise ValueError("ticket was not published by this test handle") from exc
        if out is None:
            rows = torch.empty(
                (valid_rows, self.n), dtype=torch.bfloat16, device=self.b_nt.device
            )
        else:
            rows = out
            if (
                not rows.is_cuda
                or rows.dtype != torch.bfloat16
                or not rows.is_contiguous()
                or rows.device != self.b_nt.device
                or tuple(rows.shape) != (valid_rows, self.n)
            ):
                raise ValueError("out must be contiguous CUDA bfloat16 [valid_rows, N]")
        self.module._binding.serving_test_consume(
            self.c_ring,
            rows,
            self.control,
            self.module._test_consume,
            ticket,
            bool(bypass),
        )
        self._next_consume_ticket += 1
        del self._published_valid_rows[ticket]
        return rows

    def wait_stopped(self):

        if not self._started:
            raise RuntimeError("serving queue has not been started")
        self.stream.synchronize()
        lifecycle = int(self.control[512:516].view(torch.int32).item())
        if lifecycle != 3:
            raise RuntimeError(
                f"serving kernel exited with lifecycle state {lifecycle}"
            )
        self._stopped = True

    def close(self):
        if self._started and not self._stopped:
            self.request_drain()
            self.wait_stopped()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _kernel_cflags(config, _variant):
    return tuple(
        f"-DSM110A_BF16_{name}={value}"
        for name, value in (
            ("TILE_M", config.tile_m),
            ("TILE_N", config.tile_n),
            ("TILE_K", config.tile_k),
            ("CLUSTER_M", config.cluster_m),
            ("EPILOGUE_BLOCK_N", config.epilogue_block_n),
        )
    )


_JIT = SplitCudaJit(
    name="sm110a_bf16_gemm_serving",
    root=_ROOT,
    binding_sources=(_SOURCE_DIR / "bf16_gemm_serving_split_torch_binding.cpp",),
    kernel_sources=(_SOURCE_DIR / "bf16_gemm_serving_kernel.cu",),
    kernel_dependencies=(
        _INCLUDE_DIR / "bf16_gemm_serving.h",
        _INCLUDE_DIR / "bf16_gemm_serving_queue.h",
        _INCLUDE_DIR / "common.h",
    ),
    kernel_symbol="sm110a_bf16_gemm_serving_launch",
    config_key=lambda config: config,
    kernel_cflags=_kernel_cflags,
    make_facade=Bf16GemmServingModule,
    include_paths=(_INCLUDE_DIR,),
)


def compile_bf16_gemm_serving(
    config: Bf16GemmServingConfig | None = None,
) -> Bf16GemmServingModule:
    config = config or Bf16GemmServingConfig()
    if not isinstance(config, Bf16GemmServingConfig):
        raise TypeError("config must be a Bf16GemmServingConfig")
    config.validate()
    return _JIT.compile(config)


def compile_bf16_gemm_serving_configs(
    configs: Iterable[Bf16GemmServingConfig], max_workers=None
):
    configs = tuple(configs)
    for config in configs:
        if not isinstance(config, Bf16GemmServingConfig):
            raise TypeError("configs must contain Bf16GemmServingConfig instances")
        config.validate()
    return _JIT.compile_many(configs, max_workers)


def get_bf16_gemm_serving_module(a, b, config=None):
    return compile_bf16_gemm_serving(config or _default_config(a, b))


def bf16_gemm_serving(a, b, config=None, swap_ab=False, max_active_clusters=None):
    return get_bf16_gemm_serving_module(a, b, config).bf16_gemm_serving(
        a, b, swap_ab, max_active_clusters
    )


def bf16_gemm_serving_out(
    a, b, out, config=None, swap_ab=False, max_active_clusters=None
):
    return get_bf16_gemm_serving_module(a, b, config).bf16_gemm_serving_out(
        a, b, out, swap_ab, max_active_clusters
    )


def bf16_gemm_serving_nt_out(
    a, b_nt, out, config=None, swap_ab=False, max_active_clusters=None
):
    """Compute ``a @ b_nt.T`` for a contiguous, pretransposed ``b_nt(N, K)``."""
    return compile_bf16_gemm_serving(config).bf16_gemm_serving_nt_out(
        a, b_nt, out, swap_ab, max_active_clusters
    )


class Bf16GemmServingRunner(BaseKernelRunner):
    namespace = "bf16_gemm_serving_runner_v1"

    def __init__(self, a, b_nt, out, *, max_active_clusters=None):
        super().__init__(a.device)
        self.a, self.b, self.out = a, b_nt, out
        self.max_active_clusters = max_active_clusters

    @property
    def workload_key(self):
        return (tuple(self.a.shape), tuple(self.b.shape), self.max_active_clusters)

    def compatible_candidates(self):
        candidates = []
        m, k = self.a.shape
        n = self.b.size(0)
        for tn in (64, 128, 256):
            for tk in (64, 128):
                for cm in (1, 2):
                    for en in (32, 64, 128):
                        config = Bf16GemmServingConfig(
                            tile_n=tn, tile_k=tk, cluster_m=cm, epilogue_block_n=en
                        )
                        try:
                            config.validate()
                        except ValueError:
                            continue
                        if n % tn == 0 and k % tk == 0 and m % (128 * cm) == 0:
                            candidates.append(config)
        return tuple(candidates)

    def pre_jit(self, candidate):
        return compile_bf16_gemm_serving(candidate)

    def prepare(self, candidate):
        module = self.pre_jit(candidate)
        return PreparedKernel(
            candidate,
            lambda: module.bf16_gemm_serving_nt_out(
                self.a, self.b, self.out, False, self.max_active_clusters
            ),
        )

    def create_queue(self, **kwargs):
        """Create the persistent queue, whose native ABI supports one fixed tactic."""
        return compile_bf16_gemm_serving(Bf16GemmServingConfig()).create_queue(
            self.b, **kwargs
        )
