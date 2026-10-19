#!/usr/bin/env python3
"""MXFP4 FC1+FC2 correctness/performance benchmark (configurable hidden dim)."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    BenchmarkBase,
    SolReport,
    benchmark_with_stats,
    powerlaw_router_scores,
    quick_probe,
    sol_bounds,
)
from sm110a_kernel import (  # noqa: E402
    Mxfp4Fc1Config,
    Mxfp4Fc1Fc2Config,
    Mxfp4Fc2Config,
    bf16tofp8_quant,
    compact_mxfp4_fc1fc2_weight_scales,
    compile_mxfp4_fc1,
    compile_mxfp4_fc1fc2,
    compile_mxfp4_fc2,
    mxfp4_fc1_out,
    mxfp4_fc2_out,
    prepare_mxfp4_fc1_metadata,
    prepare_mxfp4_fc2_metadata,
    quantize_fp8mxfp4_weights,
    to_blocked_mxfp4_sfa,
)
from sm110a_kernel.runner import (  # noqa: E402
    AutotuneRunner,
    PreparedKernel,
    TuneOptions,
)

H = 6144
INTERMEDIATE = 2048
ROUTER_POWERLAW_ALPHA = 0.8


@dataclass(frozen=True)
class StandaloneConfig:
    """Serializable pair of independently tuned baseline configurations."""

    fc1: Mxfp4Fc1Config
    fc2: Mxfp4Fc2Config

    def __post_init__(self):
        if isinstance(self.fc1, dict):
            object.__setattr__(self, "fc1", Mxfp4Fc1Config(**self.fc1))
        if isinstance(self.fc2, dict):
            object.__setattr__(self, "fc2", Mxfp4Fc2Config(**self.fc2))


def fused_tflops(m, elapsed_ms):
    """Two FC1 projections plus FC2: 6*M*H*I floating-point operations."""
    flops = 6.0 * m * H * INTERMEDIATE
    return flops / (elapsed_ms * 1.0e9)


def format_config(config):
    return (
        f"gather={config.gather_warps},tiled={int(config.tiled_schedule)},cluster_m={config.cluster_m},h_split={config.h_split},stages={config.stages},acc={config.accumulator_stages},"
        f"compact_sf={int(config.compact_weight_scales)},reduce_n={config.reduce_tile_n},fuse_routing={int(config.fuse_routing)},"
        f"weight={config.routing_weight_position},bf16={int(config.activation_bf16_rounding)}"
    )


def row_tasks_for_cluster(args, cluster):
    rows = 128 * cluster
    if args.fuse_routing:
        return sum((count + rows - 1) // rows for count in args._expert_counts)
    return args.m // rows


def compatible_configs(m, args):
    splits = (args.h_split,) if args.h_split else (1, 2, 4)
    stages = (args.stages,) if args.stages else (2, 3, 4, 5)
    accumulators = (args.accumulator_stages,) if args.accumulator_stages else (1, 2)
    clusters = (args.cluster_m,) if args.cluster_m else (1, 2)
    schedules = (
        (args.schedule == "tiled",) if args.schedule != "auto" else (False, True)
    )
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    gathers = (
        (args.gather_warps,)
        if args.gather_warps
        else ((2, 4) if args.fuse_routing else (4,))
    )
    reductions = (args.reduce_tile_n,) if args.reduce_tile_n else (64, 128)
    compact_formats = (
        (False, True)
        if args.weight_scales == "auto"
        else (args.weight_scales == "compact",)
    )
    configs = tuple(
        Mxfp4Fc1Fc2Config(
            hidden_dim=H,
            compact_weight_scales=compact,
            cluster_m=cluster,
            tiled_schedule=tiled,
            gather_warps=gather,
            reduce_tile_n=reduce,
            h_split=split,
            stages=stage,
            accumulator_stages=accumulator,
            fuse_routing=args.fuse_routing,
            routing_weight_position=args.routing_weight_position,
            activation_bf16_rounding=args.activation_bf16_rounding,
        )
        for compact in compact_formats
        for reduce in reductions
        for gather in gathers
        for tiled in schedules
        for cluster in clusters
        for split in splits
        for stage in stages
        for accumulator in accumulators
        if (args.fuse_routing or m % (128 * cluster) == 0)
        and (cluster == 2 or stage <= 3)
        and split <= sms // cluster
        and H % (256 * split) == 0
        and (not tiled or split == 1)
    )
    if not configs:
        raise SystemExit("no resident fused configurations match the filters")
    return configs


def fused_sol_bounds(m, config, *, tokens=None, experts=1, row_tasks=None):
    tokens = m if tokens is None else tokens
    row_tasks = math.ceil(m / 256) if row_tasks is None else row_tasks
    a_bytes = tokens * H * (1 + 4 / 32)
    weights = (
        experts
        * 3
        * H
        * INTERMEDIATE
        * (0.5 + (1 if config.compact_weight_scales else 4) / 32)
    )
    out_bytes = tokens * H * 2
    image_bytes = row_tasks * 256 * INTERMEDIATE * (1 + 4 / 32)
    # FC2 reloads the image once per 256-column H tile; workspace is initially
    # produced within the timed fused call. These are L2 tile-read estimates.
    tile_reads = (
        row_tasks * 256 * H * (1 + 4 / 32) * (INTERMEDIATE // 128)
        + row_tasks * weights / experts
        + image_bytes * (H // 256)
        + image_bytes
        + out_bytes
    )
    return sol_bounds(
        6.0 * m * H * INTERMEDIATE,
        a_bytes + weights + out_bytes + image_bytes,
        a_bytes + weights,
        tile_reads,
        mma_peak_tflops=500.0,
    )


def quantize_fc1_weight(n, k):
    gate = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    up = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    interleaved = torch.stack(
        (gate.view(n // 64, 64, k), up.view(n // 64, 64, k)), dim=1
    ).reshape(2 * n, k)
    packed, scales = quantize_fp8mxfp4_weights(interleaved)
    return packed.view(torch.float4_e2m1fn_x2).unsqueeze(0), scales


def quantize_expert_weights(experts, h, intermediate):
    w1q = sfw1 = w2q = sfw2 = None
    for expert in range(experts):
        w1, sfw1 = quantize_fc1_weight(intermediate, h)
        w2, sfw2 = quantize_fp8mxfp4_weights(
            torch.randn((h, intermediate), device="cuda", dtype=torch.bfloat16)
        )
        w2 = w2.view(torch.float4_e2m1fn_x2)
        if w1q is None:
            w1q = torch.empty((experts, *w1.shape[1:]), device="cuda", dtype=w1.dtype)
            sfw1_all = torch.empty(
                experts * sfw1.numel(), device="cuda", dtype=sfw1.dtype
            )
            w2q = torch.empty((experts, *w2.shape), device="cuda", dtype=w2.dtype)
            sfw2_all = torch.empty(
                experts * sfw2.numel(), device="cuda", dtype=sfw2.dtype
            )
        w1q[expert].copy_(w1[0])
        sfw1_all[expert * sfw1.numel() : (expert + 1) * sfw1.numel()].copy_(sfw1)
        w2q[expert].copy_(w2)
        sfw2_all[expert * sfw2.numel() : (expert + 1) * sfw2.numel()].copy_(sfw2)
        del w1, sfw1, w2, sfw2
    return w1q, sfw1_all, w2q, sfw2_all


def validate_scale_encoding():
    import ctypes
    import hashlib
    import os

    from torch.utils.cpp_extension import load

    from sm110a_kernel.jit_runtime import jit_build_directory

    # This diagnostic is built only by --validate, outside production modules.
    header = ROOT / "csrc/mxfp4_moe/src/mxfp4_fc1fc2_scale.cuh"
    source = r"""
#include <cuda_fp8.h>
#include "mxfp4_fc1fc2_scale.cuh"
__global__ void scale_check(const float *values, uint8_t *actual,
                            uint8_t *expected, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        actual[index] = encode_activation_scale(values[index]);
        expected[index] = __nv_fp8_e8m0(values[index] / 448.0f).__x;
    }
}
extern "C" cudaError_t launch_scale_check(const float *values, uint8_t *actual,
    uint8_t *expected, int count, cudaStream_t stream) {
    scale_check<<<(count + 255) / 256, 256, 0, stream>>>(values, actual, expected, count);
    return cudaGetLastError();
}
"""
    arch = os.environ.get("CUDA_ARCH", "sm_110a")
    compute_arch = os.environ.get("COMPUTE_CUDA_ARCH", "compute_110a")
    digest = hashlib.sha256(
        header.read_bytes()
        + source.encode()
        + f"{arch}/{compute_arch}/{torch.__version__}/{torch.version.cuda}".encode()
    ).hexdigest()[:16]
    name = f"sm110a_bench_mxfp4_scale_check_{digest}"
    directory = Path(jit_build_directory(name))
    source_path = directory / "scale_check.cu"
    if not source_path.exists() or source_path.read_text() != source:
        source_path.write_text(source)
    library_path = load(
        name=name,
        sources=[str(source_path)],
        is_python_module=False,
        extra_include_paths=[str(header.parent)],
        extra_cuda_cflags=["-O3", f"-gencode=arch={compute_arch},code={arch}"],
        build_directory=str(directory),
    )
    library = ctypes.CDLL(library_path)
    launch = library.launch_scale_check
    launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int
    thresholds = [(exponent << 23) | 0x600000 for exponent in range(1, 255)]
    bits = [value + delta for value in thresholds for delta in range(-4, 5)]
    bits.extend(
        (
            0,
            1,
            0x003FFFFF,
            0x00400000,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0x7F800000,
            0x7FC00000,
        )
    )
    boundary = torch.tensor(bits, device="cuda", dtype=torch.int32)
    random_bits = torch.randint(
        0, 0x7F800000, (500_000,), device="cuda", dtype=torch.int32
    )
    values = torch.cat((boundary, random_bits)).view(torch.float32)
    actual = torch.empty_like(values, dtype=torch.uint8)
    expected = torch.empty_like(actual)
    error = launch(
        values.data_ptr(),
        actual.data_ptr(),
        expected.data_ptr(),
        values.numel(),
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"scale check launch failed with CUDA error {error}")
    if not torch.equal(actual, expected):
        bad = torch.nonzero(actual != expected).flatten()[:10]
        details = [
            (
                hex(int(values.view(torch.int32)[i].item())),
                int(actual[i]),
                int(expected[i]),
            )
            for i in bad
        ]
        raise AssertionError(
            f"E8M0 scale encoding differs from FP32 division: {details}"
        )
    print(
        f"[validate] {values.numel()} scale encodings match FP32 division/E8M0; thresholds/extremes/random",
        flush=True,
    )


def run_validation(group="all"):
    """Run correctness and graph regressions through this benchmark entry."""
    from sm110a_kernel import (
        compile_mxfp4_fc1fc2_configs,
        mxfp4_fc1fc2,
        mxfp4_fc1fc2_out,
        mxfp4_fc1fc2_routed_out,
        prepare_mxfp4_fc1fc2_metadata,
    )

    torch.manual_seed(917)
    validate_scale_encoding()
    if group == "scale":
        return
    activation, scales = bf16tofp8_quant(
        torch.randn(3072, H, device="cuda", dtype=torch.bfloat16)
    )
    blocked_sfa = to_blocked_mxfp4_sfa(scales)
    weights = quantize_expert_weights(4, H, INTERMEDIATE)
    capacity = (
        torch.cuda.get_device_properties(activation.device).multi_processor_count // 2
    )
    grouped = [
        (
            f"grouped-M{m}-split{split}-stages{stages}",
            m,
            1,
            1,
            Mxfp4Fc1Fc2Config(h_split=split, stages=stages),
            7.0,
            False,
        )
        for m, split, stages in ((256, 1, 2), (512, 4, 5), (1024, 2, 3), (3072, 1, 5))
    ]
    routed = [
        (
            f"routed-top{topk}-split{split}-{position}-bf16{int(rounding)}",
            128,
            4,
            topk,
            Mxfp4Fc1Fc2Config(
                h_split=split,
                fuse_routing=True,
                routing_weight_position=position,
                activation_bf16_rounding=rounding,
            ),
            7.0,
            False,
        )
        for topk, split in ((1, 1), (1, 2), (2, 1), (4, 1))
        for position, rounding in (
            ("fc2", False),
            ("fc2", True),
            ("fc1", False),
            ("fc1", True),
        )
    ]
    routed.append(
        (
            "routed-multiple-waves",
            1536,
            4,
            2,
            Mxfp4Fc1Fc2Config(fuse_routing=True),
            7.0,
            False,
        )
    )
    grouped.append(
        (
            "grouped-single-accumulator",
            512,
            1,
            1,
            Mxfp4Fc1Fc2Config(h_split=4, accumulator_stages=1),
            7.0,
            False,
        )
    )
    grouped.append(
        ("grouped-stages4", 512, 1, 1, Mxfp4Fc1Fc2Config(stages=4), 7.0, False)
    )
    grouped.extend(
        (
            f"grouped-cta1-M{m}-split{split}-acc{acc}",
            m,
            1,
            1,
            Mxfp4Fc1Fc2Config(
                cluster_m=1, stages=stages, h_split=split, accumulator_stages=acc
            ),
            7.0,
            False,
        )
        for m, split, stages, acc in ((128, 4, 3, 2), (512, 4, 2, 1), (3072, 1, 3, 2))
    )
    routed.extend(
        (
            f"routed-cta1-top{topk}-split{split}-{position}-bf16{int(rounding)}",
            128,
            4,
            topk,
            Mxfp4Fc1Fc2Config(
                cluster_m=1,
                stages=3,
                h_split=split,
                fuse_routing=True,
                routing_weight_position=position,
                activation_bf16_rounding=rounding,
            ),
            7.0,
            False,
        )
        for topk, split in ((1, 2), (2, 4), (4, 1))
        for position, rounding in (
            ("fc1", False),
            ("fc2", False),
            ("fc1", True),
            ("fc2", True),
        )
    )
    grouped.extend(
        (
            f"grouped-multiwave-split{split}",
            3072,
            1,
            1,
            Mxfp4Fc1Fc2Config(h_split=split),
            7.0,
            False,
        )
        for split in (2, 4)
    )
    routed.extend(
        (
            f"routed-multiwave-cta{cluster}-split{split}",
            1536,
            4,
            2,
            Mxfp4Fc1Fc2Config(
                cluster_m=cluster, stages=3, h_split=split, fuse_routing=True
            ),
            7.0,
            False,
        )
        for cluster in (1, 2)
        for split in (2, 4)
    )
    grouped.extend(
        (
            f"grouped-tiled-cta{cluster}-M{m}-acc{acc}",
            m,
            1,
            1,
            Mxfp4Fc1Fc2Config(
                cluster_m=cluster,
                stages=stages,
                tiled_schedule=True,
                accumulator_stages=acc,
            ),
            7.0,
            False,
        )
        for cluster, m, stages, acc in (
            (1, 128, 3, 2),
            (1, 3072, 2, 1),
            (2, 512, 5, 2),
            (2, 3072, 3, 1),
        )
    )
    routed.extend(
        (
            f"routed-tiled-cta{cluster}-top{topk}-{position}-bf16{int(rounding)}",
            128,
            4,
            topk,
            Mxfp4Fc1Fc2Config(
                cluster_m=cluster,
                stages=3,
                tiled_schedule=True,
                fuse_routing=True,
                routing_weight_position=position,
                activation_bf16_rounding=rounding,
            ),
            7.0,
            False,
        )
        for cluster in (1, 2)
        for topk in (1, 2, 4)
        for position, rounding in (("fc1", False), ("fc2", True))
    )
    routed.extend(
        (
            f"routed-gather2-cta{cluster}-tiled{int(tiled)}",
            128,
            4,
            2,
            Mxfp4Fc1Fc2Config(
                cluster_m=cluster,
                stages=3,
                gather_warps=2,
                tiled_schedule=tiled,
                fuse_routing=True,
                routing_weight_position="fc1",
                activation_bf16_rounding=True,
            ),
            7.0,
            False,
        )
        for cluster in (1, 2)
        for tiled in (False, True)
    )
    # Wider reduction slabs exercise alias protection with both TMEM
    # accumulator layouts, both cluster sizes and row/tiled scheduling.
    grouped.extend(
        (
            f"grouped-reduce128-cta{cluster}-tiled{int(tiled)}-acc{acc}",
            512,
            1,
            1,
            Mxfp4Fc1Fc2Config(
                cluster_m=cluster,
                stages=3,
                reduce_tile_n=128,
                tiled_schedule=tiled,
                accumulator_stages=acc,
            ),
            7.0,
            False,
        )
        for cluster, tiled, acc in (
            (1, False, 1),
            (1, True, 2),
            (2, False, 2),
            (2, True, 1),
        )
    )
    routed.extend(
        (
            f"routed-reduce128-cta{cluster}-tiled{int(tiled)}-acc{acc}",
            128,
            4,
            2,
            Mxfp4Fc1Fc2Config(
                cluster_m=cluster,
                stages=3,
                reduce_tile_n=128,
                gather_warps=2,
                tiled_schedule=tiled,
                accumulator_stages=acc,
                fuse_routing=True,
                routing_weight_position="fc2",
            ),
            7.0,
            False,
        )
        for cluster, tiled, acc in (
            (1, False, 2),
            (1, True, 1),
            (2, False, 1),
            (2, True, 2),
        )
    )
    edges = [
        (
            f"zero-activation-bf16{int(rounding)}",
            256,
            1,
            1,
            Mxfp4Fc1Fc2Config(activation_bf16_rounding=rounding),
            0.0,
            True,
        )
        for rounding in (False, True)
    ]
    compact = []
    # Every consumer role, both schedules, split synchronization, rounding and
    # routing positions. The baseline retains its padded scale storage.
    for cluster, tiled, gather in (
        (1, False, 2),
        (1, True, 4),
        (2, False, 2),
        (2, True, 2),
        (2, False, 4),
        (2, True, 4),
    ):
        for routed_mode in (False, True):
            compact.append(
                (
                    f"compact-cta{cluster}-tiled{int(tiled)}-g{gather}-routed{int(routed_mode)}",
                    512 if not routed_mode else 128,
                    4 if routed_mode else 1,
                    2 if routed_mode else 1,
                    Mxfp4Fc1Fc2Config(
                        cluster_m=cluster,
                        stages=3 if cluster == 1 else 5,
                        tiled_schedule=tiled,
                        gather_warps=gather,
                        accumulator_stages=1 if gather == 4 else 2,
                        reduce_tile_n=64 if gather == 4 else 128,
                        fuse_routing=routed_mode,
                        compact_weight_scales=True,
                        routing_weight_position="fc1"
                        if routed_mode and gather == 4
                        else "fc2",
                        activation_bf16_rounding=gather == 4,
                    ),
                    7.0,
                    False,
                )
            )
    compact.extend(
        (
            f"compact-split{split}-cta{cluster}",
            1536,
            4,
            2,
            Mxfp4Fc1Fc2Config(
                cluster_m=cluster,
                stages=3,
                h_split=split,
                fuse_routing=True,
                compact_weight_scales=True,
            ),
            7.0,
            False,
        )
        for cluster in (1, 2)
        for split in (2, 4)
    )
    compact.append(
        (
            "compact-zero",
            256,
            1,
            1,
            Mxfp4Fc1Fc2Config(stages=3, compact_weight_scales=True),
            0.0,
            True,
        )
    )
    cases = {"grouped": grouped, "routed": routed, "edge": edges, "compact": compact}
    chosen = grouped + routed + edges + compact if group == "all" else cases[group]
    compile_mxfp4_fc1fc2_configs(tuple({case[4] for case in chosen}))
    passed = skipped = 0
    for name, tokens, experts, topk, config, limit, zero in chosen:
        a = activation[:tokens]
        if zero:
            a = a.clone()
            a.view(torch.uint8).zero_()
        sfa = blocked_sfa[: tokens * 768]
        w1, sfw1, w2, sfw2 = weights
        w1, sfw1 = w1[:experts], sfw1[: experts * 4096 * 768]
        w2, sfw2 = w2[:experts], sfw2[: experts * H * 256]
        c1 = Mxfp4Fc1Config(
            cluster_m=config.cluster_m,
            direct_gmem=True,
            routing_weight_position=config.routing_weight_position,
            activation_bf16_rounding=config.activation_bf16_rounding,
        )
        c2 = Mxfp4Fc2Config(
            cluster_m=config.cluster_m,
            routing_weight_position=config.routing_weight_position,
        )
        router = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16)
        if topk == 1 and experts > 1:
            router[:, -1] = -100  # Include an empty expert.
        metadata = prepare_mxfp4_fc1_metadata(router, topk, 4096, c1)
        row_tasks = int(metadata.state[0].item()) // 16
        if config.h_split > capacity * 2 // config.cluster_m:
            print(
                f"[validate] SKIP {name}: resident clusters {row_tasks * config.h_split}>{capacity}",
                flush=True,
            )
            skipped += 1
            continue
        metadata2 = prepare_mxfp4_fc2_metadata(
            metadata.routing.narrow(0, 2 * experts, tokens * topk), experts, H, c2
        )
        dtype = (
            torch.float32 if config.routing_weight_position == "fc1" else torch.bfloat16
        )
        routing_weights = torch.softmax(
            torch.topk(router.float(), topk, dim=1).values, dim=1
        ).to(dtype)
        if config.routing_weight_position == "fc1":
            routing_weights[0] = 0
            routing_weights[1] = 1e-8
            routing_weights[2] = -0.5
        intermediate = torch.empty(
            tokens * topk, INTERMEDIATE, device="cuda", dtype=torch.float8_e4m3fn
        )
        sf_intermediate = torch.empty(
            tokens * topk * 256, device="cuda", dtype=torch.float8_e8m0fnu
        )
        expected = torch.empty(tokens, H, device="cuda", dtype=torch.bfloat16)
        mxfp4_fc1_out(
            a,
            w1,
            sfa,
            sfw1,
            metadata,
            intermediate,
            sf_intermediate,
            limit,
            c1,
            routing_weights=routing_weights
            if config.routing_weight_position == "fc1"
            else None,
        )
        mxfp4_fc2_out(
            intermediate,
            w2,
            sf_intermediate,
            sfw2,
            metadata.routing,
            routing_weights.to(torch.bfloat16),
            metadata2,
            topk,
            expected,
            c2,
        )
        output = torch.empty_like(expected)
        module = compile_mxfp4_fc1fc2(config)

        fused_sfw1, fused_sfw2 = sfw1, sfw2
        if config.compact_weight_scales:
            fused_sfw1 = compact_mxfp4_fc1fc2_weight_scales(sfw1)
            fused_sfw2 = compact_mxfp4_fc1fc2_weight_scales(sfw2)
            for source, packed in ((sfw1, fused_sfw1), (sfw2, fused_sfw2)):
                if packed.numel() * 4 != source.numel() or not torch.equal(
                    packed.view(torch.uint8),
                    source.view(torch.uint8).reshape(-1, 4)[:, 0],
                ):
                    raise AssertionError(
                        "compact weight scale pack changed live scale bits"
                    )

        def run():
            if config.fuse_routing:
                return module.run_routed_out(
                    a,
                    w1,
                    sfa,
                    fused_sfw1,
                    w2,
                    fused_sfw2,
                    metadata,
                    routing_weights,
                    topk,
                    output,
                    limit,
                    row_task_count=row_tasks,
                )
            return module.run_out(a, w1, sfa, fused_sfw1, w2, fused_sfw2, output, limit)

        def check(label):
            torch.cuda.synchronize()
            if not torch.isfinite(output).all():
                raise AssertionError(f"{name}/{label}: non-finite output")
            if topk <= 2:
                if not torch.equal(output, expected):
                    raise AssertionError(f"{name}/{label}: expected bitwise equality")
                return 0.0
            relative_l2 = float(
                (
                    torch.linalg.vector_norm(output.float() - expected.float())
                    / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
                ).item()
            )
            if relative_l2 > 5e-3:
                raise AssertionError(
                    f"{name}/{label}: relative L2={relative_l2} > 0.005"
                )
            return relative_l2

        run()
        relative_l2 = check("eager")
        # Repeated output writes must clear token-major reductions each launch.
        run()
        check("repeat")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        check("graph")
        graph.replay()
        check("graph-repeat")
        # Exercise reading current metadata.state outside graph capture.
        if config.fuse_routing:
            module.run_routed_out(
                a,
                w1,
                sfa,
                fused_sfw1,
                w2,
                fused_sfw2,
                metadata,
                routing_weights,
                topk,
                output,
                limit,
            )
            check("live-state")
        if name == "grouped-M512-split4-stages5":
            allocated = mxfp4_fc1fc2(a, w1, sfa, sfw1, w2, sfw2, limit)
            torch.cuda.synchronize()
            if not torch.equal(allocated, expected):
                raise AssertionError("allocating convenience API differs from baseline")
            mxfp4_fc1fc2_out(a, w1, sfa, sfw1, w2, sfw2, output, limit)
            check("convenience-out")
        if name == "routed-top1-split1-fc2-bf160":
            # Top-1 avoids tied scores and any top-k summation-order ambiguity.
            prepared_metadata = prepare_mxfp4_fc1fc2_metadata(router, topk, config)
            mxfp4_fc1fc2_routed_out(
                a,
                w1,
                sfa,
                sfw1,
                w2,
                sfw2,
                prepared_metadata,
                routing_weights,
                topk,
                output,
                limit,
            )
            check("convenience-routed-metadata")
        print(
            f"[validate] PASS {name}; eager/repeat/graph; relative_l2={relative_l2:.6g}",
            flush=True,
        )
        passed += 1
    config = Mxfp4Fc1Fc2Config(stages=2)
    if compile_mxfp4_fc1fc2(config) is not compile_mxfp4_fc1fc2(config):
        raise AssertionError("JIT in-process cache did not reuse the facade")
    if compile_mxfp4_fc1fc2_configs((config, config), max_workers=2) != (
        compile_mxfp4_fc1fc2(config),
        compile_mxfp4_fc1fc2(config),
    ):
        raise AssertionError("batch JIT cache did not reuse the facade")
    narrow = compile_mxfp4_fc1fc2(config)
    wide = compile_mxfp4_fc1fc2(replace(config, reduce_tile_n=128))
    if narrow is wide or narrow.library_path == wide.library_path:
        raise AssertionError("JIT cache aliased different reduction slab sizes")
    packed_module = compile_mxfp4_fc1fc2(replace(config, compact_weight_scales=True))
    if narrow is packed_module or narrow.library_path == packed_module.library_path:
        raise AssertionError("JIT cache aliased different weight scale formats")
    # Both bindings must reject the opposite storage size before launching.
    for candidate, scale1, scale2 in (
        (
            narrow,
            compact_mxfp4_fc1fc2_weight_scales(weights[1][: 4096 * 768]),
            compact_mxfp4_fc1fc2_weight_scales(weights[3][: H * 256]),
        ),
        (packed_module, weights[1][: 4096 * 768], weights[3][: H * 256]),
    ):
        try:
            candidate.run(
                activation[:256],
                weights[0][:1],
                blocked_sfa[: 256 * 768],
                scale1,
                weights[2][:1],
                scale2,
            )
        except RuntimeError as error:
            if "invalid blocked scale storage" not in str(error):
                raise
        else:
            raise AssertionError("opposite weight scale storage was accepted")
    for bad_scales in (
        torch.empty(512),
        torch.empty(512, device="cuda", dtype=torch.uint8),
        weights[1][::2],
        weights[1][:511],
    ):
        try:
            compact_mxfp4_fc1fc2_weight_scales(bad_scales)
        except ValueError:
            continue
        raise AssertionError("invalid weight scale pack input was accepted")
    for bad in (
        replace(config, compact_weight_scales=1),
        replace(config, h_split=3),
        replace(config, cluster_m=0),
        replace(config, cluster_m=1, stages=4),
        replace(config, accumulator_stages=3),
        replace(config, gather_warps=3),
        replace(config, reduce_tile_n=32),
        replace(config, tiled_schedule=True, h_split=2),
        replace(config, stages=6),
        replace(config, routing_weight_position="invalid"),
    ):
        try:
            compile_mxfp4_fc1fc2(bad)
        except ValueError:
            continue
        raise AssertionError(f"invalid config was accepted: {bad}")
    print(
        f"[validate] {passed} cases passed; {skipped} skipped; JIT cache/config checks passed"
    )


def run_sweep(args):
    """Serial workload/config sweep; each case uses this benchmark entry point."""
    import hashlib
    import json
    import os
    import subprocess
    from datetime import datetime, timezone

    source_paths = sorted(
        {
            Path(__file__).resolve(),
            ROOT / "benchmarks/common.py",
            ROOT / "sm110a_kernel/jit_runtime.py",
            ROOT / "include/common.h",
            *ROOT.glob("include/mxfp4_fc*.h"),
            *ROOT.glob("csrc/mxfp4_moe/src/mxfp4_fc*"),
            *ROOT.glob("sm110a_kernel/kernels/mxfp4_fc*.py"),
        }
    )

    def source_hashes():
        return {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources_sha256": source_hashes(),
        "argv": sys.argv[1:],
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key in ("CUDA_ARCH", "COMPUTE_CUDA_ARCH")
            or key.startswith("SM110A_PIPELINE_DELAY")
        },
    }

    workloads = []
    if args.sweep in ("all", "grouped"):
        workloads.extend(
            (f"grouped_m{m}", ["--m", str(m)])
            for m in (256, 512, 1024, 2048, 4096, 8192)
        )
    if args.sweep in ("all", "routed"):
        workloads.extend(
            (
                f"routed_t{t}_e{e}_k{k}",
                [
                    "--fuse-routing",
                    "--tokens",
                    str(t),
                    "--experts",
                    str(e),
                    "--topk",
                    str(k),
                ],
            )
            for e, k in ((2, 2), (8, 6), (32, 6))
            for t in (128, 512, 2048)
        )
    forwarded = []
    raw = iter(sys.argv[1:])
    for arg in raw:
        if arg in (
            "--sweep",
            "--sweep-output",
            "--json-output",
            "--m",
            "--tokens",
            "--experts",
            "--topk",
        ):
            next(raw)
        elif any(
            arg.startswith(option + "=")
            for option in (
                "--sweep",
                "--sweep-output",
                "--json-output",
                "--m",
                "--tokens",
                "--experts",
                "--topk",
            )
        ):
            continue
        elif arg not in ("--fuse-routing", "--validate"):
            forwarded.append(arg)
    args.sweep_output.mkdir(parents=True, exist_ok=True)
    (args.sweep_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    results = []
    for name, workload in workloads:
        if source_hashes() != manifest["sources_sha256"]:
            raise RuntimeError(
                "sources changed during sweep; use a new output directory"
            )
        output = args.sweep_output / f"{name}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *forwarded,
            *workload,
            "--force-sweep",
            "--json-output",
            str(output),
        ]
        print(f"[sweep] {name}", flush=True)
        with (args.sweep_output / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(output.read_text())
        results.append({"case": name, **result})
        (args.sweep_output / "summary.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(
            f"[sweep] {name}: fused={result['fused_ms']:.6f} ms; baseline={result['standalone_ms']:.6f} ms; speedup={result['speedup']:.4f}x",
            flush=True,
        )
    if source_hashes() != manifest["sources_sha256"]:
        raise RuntimeError(
            "sources changed during sweep; results have mixed provenance"
        )
    print(f"[sweep] {len(results)} workloads complete; results={args.sweep_output}")


def main():
    global H
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate", action="store_true", help="run the correctness matrix and exit"
    )
    parser.add_argument(
        "--validation-group",
        choices=("all", "grouped", "routed", "edge", "scale", "compact"),
        default="all",
    )
    parser.add_argument("--sweep", choices=("all", "grouped", "routed"))
    parser.add_argument(
        "--sweep-output", type=Path, default=ROOT / "profiles/mxfp4_fc1fc2"
    )
    parser.add_argument(
        "--baseline-weight-scales", choices=("padded", "compact"), default="padded"
    )
    parser.add_argument("--hidden-dim", type=int, default=6144)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--fuse-routing", action="store_true")
    parser.add_argument(
        "--tokens",
        type=int,
        default=0,
        help="token count in routed mode; 0 derives it from M/topk",
    )
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--gather-warps", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--reduce-tile-n", type=int, choices=(0, 64, 128), default=0)
    parser.add_argument(
        "--weight-scales",
        choices=("auto", "padded", "compact"),
        default="auto",
        help="static weight scale format; auto tunes both",
    )
    parser.add_argument("--schedule", choices=("auto", "row", "tiled"), default="auto")
    parser.add_argument("--cluster-m", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--h-split", type=int, choices=(0, 1, 2, 4), default=0)
    parser.add_argument("--accumulator-stages", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--stages", type=int, choices=(0, 2, 3, 4, 5), default=0)
    parser.add_argument("--swiglu-limit", type=float, default=7.0)
    parser.add_argument(
        "--routing-weight-position", choices=("fc1", "fc2"), default="fc2"
    )
    parser.add_argument("--activation-bf16-rounding", action="store_true")
    parser.add_argument("--clean-l2", action="store_true", help="evict L2 by reading")
    parser.add_argument(
        "--baseline-cluster-m",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="0 tunes standalone cluster sizes; 1/2 fixes both",
    )
    parser.add_argument("--graph-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--force-sweep", action="store_true")
    parser.add_argument("--cache-path")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sol", action="store_true")
    parser.add_argument(
        "--pre-parallel-jit",
        dest="pre_parallel_jit",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-pre-parallel-jit", dest="pre_parallel_jit", action="store_false"
    )
    parser.add_argument("--no-accuracy", action="store_true")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--max-relative-rmse", type=float, default=5e-3)
    args = parser.parse_args()
    if args.hidden_dim <= 0 or args.hidden_dim % 256:
        parser.error("--hidden-dim must be positive and divisible by 256")
    H = args.hidden_dim
    if args.hidden_dim != 6144 and (args.sweep or args.validate):
        parser.error("--sweep and --validate use the fixed 6144 validation suite")
    if args.sweep:
        run_sweep(args)
        return
    if args.validate:
        if args.no_accuracy:
            parser.error("--validate requires accuracy checking")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required")
        run_validation(args.validation_group)
        return
    if args.m <= 0:
        parser.error("--m must be positive")
    if args.experts <= 0 or args.topk <= 0 or args.topk > args.experts:
        parser.error("--topk must be in [1, experts]")
    if args.fuse_routing:
        if args.tokens:
            args.m = args.tokens * args.topk
        elif args.m % args.topk:
            parser.error("routed --m must be divisible by --topk")
        args.tokens = args.m // args.topk
        if args.tokens % 128 or args.m % 128:
            parser.error("routed tokens and M=tokens*topk must be divisible by 128")
    elif args.m % 128:
        parser.error("grouped --m must be a multiple of 128")
    if args.warmup < 0 or args.iters <= 0 or args.samples <= 0:
        parser.error("invalid warmup/iteration count")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    BenchmarkBase().configure_cold_l2(args)
    cold_l2_flush = args._cold_l2_flush
    if cold_l2_flush is None:
        raise RuntimeError("cold L2 timing requires a reported L2 size")
    h, intermediate = H, INTERMEDIATE
    torch.manual_seed(1234)
    tokens = args.tokens if args.fuse_routing else args.m
    experts = args.experts if args.fuse_routing else 1
    topk = args.topk if args.fuse_routing else 1
    aq, logical_sfa = bf16tofp8_quant(
        torch.randn((tokens, h), device="cuda", dtype=torch.bfloat16)
    )
    sfa = to_blocked_mxfp4_sfa(logical_sfa)
    w1q, sfw1, w2q, sfw2 = quantize_expert_weights(experts, h, intermediate)

    output = torch.empty((tokens, h), device="cuda", dtype=torch.bfloat16)

    fc1_config = Mxfp4Fc1Config(
        tile_n=256,
        tile_k=128,
        epilogue_tile_n=128,
        cluster_m=2,
        direct_gmem=True,
        routing_weight_position=args.routing_weight_position,
        activation_bf16_rounding=args.activation_bf16_rounding,
        compact_weight_scales=args.baseline_weight_scales == "compact",
    )
    fc2_config = Mxfp4Fc2Config(
        tile_n=256,
        tile_k=128,
        epilogue_tile_n=32,
        cluster_m=2,
        routing_weight_position=args.routing_weight_position,
        compact_weight_scales=args.baseline_weight_scales == "compact",
    )
    compile_mxfp4_fc1(fc1_config)
    compile_mxfp4_fc2(fc2_config)
    if args.fuse_routing:
        router = powerlaw_router_scores(
            tokens,
            experts,
            ROUTER_POWERLAW_ALPHA,
            device="cuda",
            dtype=torch.bfloat16,
        )
        topk_result = torch.topk(router.float(), topk, dim=1)
        routing_weights = torch.softmax(topk_result.values, dim=1).float().contiguous()
        expert_counts = torch.bincount(
            topk_result.indices.reshape(-1), minlength=experts
        )
        row_task_count = int(
            torch.div(expert_counts + 255, 256, rounding_mode="floor").sum().item()
        )
    else:
        router = torch.ones((args.m, 1), device="cuda", dtype=torch.bfloat16)
        routing_weights = torch.ones((args.m, 1), device="cuda", dtype=torch.float32)
        row_task_count = args.m // 256
    if args.routing_weight_position == "fc2":
        routing_weights = routing_weights.to(torch.bfloat16)
    args._expert_counts = (
        expert_counts.cpu().tolist() if args.fuse_routing else [args.m]
    )
    args._row_task_count = row_task_count
    fc1_metadata = prepare_mxfp4_fc1_metadata(
        router, topk, 2 * intermediate, fc1_config
    )
    intermediate_packed = torch.empty(
        (args.m, intermediate),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    intermediate_scales = torch.empty(
        (args.m * (intermediate // 32 * 4),),
        device="cuda",
        dtype=torch.float8_e8m0fnu,
    )
    reference = torch.empty_like(output)

    candidates = compatible_configs(args.m, args)

    compact_sfw1 = compact_sfw2 = None
    weight_scale_pack_ms = 0.0
    if any(config.compact_weight_scales for config in candidates):
        pack_start = torch.cuda.Event(enable_timing=True)
        pack_end = torch.cuda.Event(enable_timing=True)
        pack_start.record()
        compact_sfw1 = compact_mxfp4_fc1fc2_weight_scales(sfw1)
        compact_sfw2 = compact_mxfp4_fc1fc2_weight_scales(sfw2)
        pack_end.record()
        pack_end.synchronize()
        weight_scale_pack_ms = pack_start.elapsed_time(pack_end)
        print(
            f"[weight-scale-pack] one-time={weight_scale_pack_ms:.6f} ms; "
            f"padded={sfw1.numel() + sfw2.numel()} bytes; "
            f"compact={compact_sfw1.numel() + compact_sfw2.numel()} bytes",
            flush=True,
        )

    fused_metadata = {2: fc1_metadata}

    def prepare_fused(config):
        fused_sfw1 = compact_sfw1 if config.compact_weight_scales else sfw1
        fused_sfw2 = compact_sfw2 if config.compact_weight_scales else sfw2
        module = compile_mxfp4_fc1fc2(config)
        local_rows = row_tasks_for_cluster(args, config.cluster_m)
        if config.cluster_m not in fused_metadata:
            fused_metadata[config.cluster_m] = prepare_mxfp4_fc1_metadata(
                router,
                topk,
                2 * intermediate,
                replace(fc1_config, cluster_m=config.cluster_m),
            )
        metadata = fused_metadata[config.cluster_m]
        if args.fuse_routing:
            run = lambda: module.run_routed_out(
                aq,
                w1q,
                sfa,
                fused_sfw1,
                w2q,
                fused_sfw2,
                metadata,
                routing_weights,
                topk,
                output,
                args.swiglu_limit,
                row_task_count=local_rows,
            )
        else:
            run = lambda: module.run_out(
                aq, w1q, sfa, fused_sfw1, w2q, fused_sfw2, output, args.swiglu_limit
            )
        run()
        torch.cuda.synchronize()
        return PreparedKernel(config, run)

    tune_options = TuneOptions(
        warmup=args.warmup,
        iters=args.iters,
        samples=args.samples,
        cache_path=args.cache_path,
        force_sweep=args.force_sweep,
        debug=args.debug,
        pre_parallel_jit=args.pre_parallel_jit,
    )
    measurements = {"fused": [], "standalone": []}

    def measure(prepared, kind):
        from dataclasses import asdict

        stats = benchmark_with_stats(
            prepared.run, args.warmup, args.iters, args.samples, cold_l2_flush
        )
        config = prepared.candidate
        config = (
            [asdict(c) for c in config] if isinstance(config, tuple) else asdict(config)
        )
        measurements[kind].append({"config": config, "stats": stats})
        return stats

    selection = AutotuneRunner(tune_options).resolve(
        candidates,
        namespace="mxfp4_fc1fc2",
        key=(
            "v10-compact-weight-scales-w4a8-cold-l2",
            args.m,
            H,
            INTERMEDIATE,
            experts,
            topk,
            row_task_count,
            args.fuse_routing,
            args.cluster_m,
            args.accumulator_stages,
            args.reduce_tile_n,
            args.weight_scales,
            args.swiglu_limit,
            args.routing_weight_position,
            args.activation_bf16_rounding,
            args.clean_l2,
        ),
        prepare=prepare_fused,
        benchmark=lambda prepared: measure(prepared, "fused"),
        probe=None
        if args.force_sweep
        else lambda prepared: quick_probe(prepared.run, args.iters, cold_l2_flush),
        format_candidate=format_config,
        metric_formatter=lambda _config, stats: (
            f"{fused_tflops(args.m, stats['median_ms']):.1f} TF/s"
        ),
        pre_jit=compile_mxfp4_fc1fc2,
    )
    fused_config = selection.candidate
    row_task_count = row_tasks_for_cluster(args, fused_config.cluster_m)

    def run_fused():
        return selection.prepared.run()

    fc2_routing_weights = routing_weights.to(torch.bfloat16)

    baseline_sfw1, baseline_sfw2 = sfw1, sfw2
    if args.baseline_weight_scales == "compact":
        if compact_sfw1 is None:
            compact_sfw1 = compact_mxfp4_fc1fc2_weight_scales(sfw1)
            compact_sfw2 = compact_mxfp4_fc1fc2_weight_scales(sfw2)
        baseline_sfw1, baseline_sfw2 = compact_sfw1, compact_sfw2

    def prepare_baseline(candidate):
        config1, config2 = candidate.fc1, candidate.fc2
        metadata1 = prepare_mxfp4_fc1_metadata(router, topk, 2 * intermediate, config1)
        metadata2 = prepare_mxfp4_fc2_metadata(
            metadata1.routing.narrow(0, 2 * experts, args.m), experts, h, config2
        )

        def fc1():
            return mxfp4_fc1_out(
                aq,
                w1q,
                sfa,
                baseline_sfw1,
                metadata1,
                intermediate_packed,
                intermediate_scales,
                args.swiglu_limit,
                config1,
                routing_weights=routing_weights
                if args.routing_weight_position == "fc1"
                else None,
            )

        def fc2():
            return mxfp4_fc2_out(
                intermediate_packed,
                w2q,
                intermediate_scales,
                baseline_sfw2,
                metadata1.routing,
                fc2_routing_weights,
                metadata2,
                topk,
                reference,
                config2,
            )

        def run():
            fc1()
            return fc2()

        run()
        torch.cuda.synchronize()
        prepared = PreparedKernel(candidate, run)
        prepared.fc1, prepared.fc2 = fc1, fc2
        return prepared

    clusters = (args.baseline_cluster_m,) if args.baseline_cluster_m else (1, 2)
    baseline_candidates = tuple(
        StandaloneConfig(
            replace(fc1_config, cluster_m=c1),
            replace(fc2_config, cluster_m=c2, epilogue_tile_n=epi),
        )
        for c1 in clusters
        for c2 in clusters
        for epi in (32, 64)
    )
    baseline = AutotuneRunner(tune_options).resolve(
        baseline_candidates,
        namespace="mxfp4_fc1fc2_standalone",
        key=(
            "v4-weight-scale-format-cold-l2",
            args.baseline_weight_scales,
            H,
            tokens,
            experts,
            topk,
            args.swiglu_limit,
            args.routing_weight_position,
            args.activation_bf16_rounding,
            args.clean_l2,
        ),
        prepare=prepare_baseline,
        benchmark=lambda prepared: measure(prepared, "standalone"),
        probe=None
        if args.force_sweep
        else lambda prepared: quick_probe(prepared.run, args.iters, cold_l2_flush),
        format_candidate=lambda candidate: (
            f"fc1_c={candidate.fc1.cluster_m},fc2_c={candidate.fc2.cluster_m},epi={candidate.fc2.epilogue_tile_n},compact={int(candidate.fc1.compact_weight_scales)}"
        ),
        pre_jit=lambda candidate: (
            compile_mxfp4_fc1(candidate.fc1),
            compile_mxfp4_fc2(candidate.fc2),
        ),
    )
    run_standalone = baseline.prepared.run
    run_fc1 = baseline.prepared.fc1
    run_fc2 = baseline.prepared.fc2

    accuracy = None
    if not args.no_accuracy:
        run_fused()
        run_standalone()
        torch.cuda.synchronize()
        if not torch.isfinite(output).all() or not torch.isfinite(reference).all():
            raise AssertionError("non-finite FC1+FC2 output")
        difference = output.float() - reference.float()
        within_tolerance = torch.isclose(
            output.float(), reference.float(), atol=args.atol, rtol=args.rtol
        )
        outside_tolerance = int((~within_tolerance).sum().item())
        rmse = difference.square().mean().sqrt()
        reference_rms = reference.float().square().mean().sqrt()
        relative_rmse = float((rmse / reference_rms.clamp_min(1e-12)).item())
        accuracy = {
            "max_abs": difference.abs().max().item(),
            "rmse": rmse.item(),
            "relative_rmse": relative_rmse,
            "mismatches": int((output != reference).sum().item()),
            "outside_tolerance": outside_tolerance,
        }
        print(
            f"[accuracy] max_abs={difference.abs().max().item():.6g}; "
            f"mean_abs={difference.abs().mean().item():.6g}; "
            f"rmse={rmse.item():.6g}; relative_rmse={relative_rmse:.6g}; "
            f"mismatches={int((output != reference).sum().item())}; "
            f"outside_tol={outside_tolerance}"
        )
        if topk == 1 and int((output != reference).sum().item()):
            raise AssertionError("topk=1 fused output must be bitwise equal to FC1+FC2")
        if topk > 1 and relative_rmse > args.max_relative_rmse:
            raise AssertionError(
                "fused output exceeds the FC1+FC2 relative-RMSE bound: "
                f"{relative_rmse} > {args.max_relative_rmse}"
            )

    if args.graph_check:
        # Workspace and compiled modules are already initialized before capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_fused()
        graph.replay()
        torch.cuda.synchronize()
        if not args.no_accuracy:
            if topk == 1:
                if not torch.equal(output, reference):
                    raise AssertionError("graph output differs from FC1+FC2")
            else:
                relative_l2 = torch.linalg.vector_norm(
                    output.float() - reference.float()
                ) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
                if (
                    not torch.isfinite(output).all()
                    or relative_l2.item() > args.max_relative_rmse
                ):
                    raise AssertionError("graph output exceeds the relative-RMSE bound")
        print("[graph] capture/replay passed")

    fused_ms = selection.stats["median_ms"]
    standalone_stats = baseline.stats
    standalone_ms = standalone_stats["median_ms"]
    fc1_ms = benchmark_with_stats(
        run_fc1, args.warmup, args.iters, args.samples, cold_l2_flush
    )["median_ms"]
    fc2_ms = benchmark_with_stats(
        run_fc2, args.warmup, args.iters, args.samples, cold_l2_flush
    )["median_ms"]
    print(f"[config] {format_config(fused_config)}; cache_hit={selection.cache_hit}")
    print(
        f"[baseline-config] fc1={baseline.candidate.fc1}; fc2={baseline.candidate.fc2}"
    )
    print(
        f"[workload] tokens={tokens}; M={args.m}; experts={experts}; "
        f"topk={topk}; input={'token-major' if args.fuse_routing else 'grouped'}; "
        f"output={'token-major-finalized' if args.fuse_routing else 'grouped'}"
    )
    print(f"[fused] {fused_ms:.6f} ms")
    print(f"[fc1+fc2] {standalone_ms:.6f} ms")
    print(f"[fc1] {fc1_ms:.6f} ms")
    print(f"[fc2] {fc2_ms:.6f} ms")
    print(
        f"[throughput] fused={fused_tflops(args.m, fused_ms):.3f} TF/s; "
        f"fc1+fc2={fused_tflops(args.m, standalone_ms):.3f} TF/s"
    )
    print(f"[ratio] {standalone_ms / fused_ms:.4f}x")
    if args.json_output:
        import json
        from dataclasses import asdict

        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "h": H,
                    "intermediate": INTERMEDIATE,
                    "row_tasks": row_task_count,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "samples": args.samples,
                    "swiglu_limit": args.swiglu_limit,
                    "baseline_configs": [
                        asdict(baseline.candidate.fc1),
                        asdict(baseline.candidate.fc2),
                    ],
                    "accuracy": accuracy,
                    "graph_check": args.graph_check,
                    "measurements": measurements,
                    "tokens": tokens,
                    "m": args.m,
                    "experts": experts,
                    "topk": topk,
                    "cold_l2": "clean" if args.clean_l2 else "dirty",
                    "config": asdict(fused_config),
                    "weight_scale_pack_ms": weight_scale_pack_ms,
                    "weight_scale_padded_bytes": sfw1.numel() + sfw2.numel(),
                    "weight_scale_compact_bytes": (sfw1.numel() + sfw2.numel()) // 4,
                    "kernel_library": str(
                        compile_mxfp4_fc1fc2(fused_config).library_path
                    ),
                    "fused_ms": fused_ms,
                    "standalone_ms": standalone_ms,
                    "fc1_ms": fc1_ms,
                    "fc2_ms": fc2_ms,
                    "speedup": standalone_ms / fused_ms,
                },
                indent=2,
            )
            + "\n"
        )
    if args.sol:
        report = SolReport(debug=args.debug)
        report.add(
            fused_sol_bounds(
                args.m,
                fused_config,
                tokens=tokens,
                experts=experts,
                row_tasks=row_task_count,
            ),
            fused_ms,
            label=f"M={args.m} fused",
        )
        report.print()


if __name__ == "__main__":
    main()
