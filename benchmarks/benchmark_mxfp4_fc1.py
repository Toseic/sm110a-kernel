#!/usr/bin/env python3
"""Config and token sweep for MXFP4 (W4A8) fused-permute SwiGLU FC1.

Times the MXFP4 W4A8 kernel against the NVFP4 W4A4 kernel on the same routed
shapes (the performance target is parity with NVFP4), and checks the MXFP4
result against an in-file PyTorch reference: dequantize the FP8 activations
and MXFP4 weights, run the grouped GEMM, apply the limited SwiGLU, and
quantize the reference output with the same per-128 E8M0 block semantics.
This is a GEMM feeding an FP8 quantizer, so bitwise equality is not expected:
values whose FP32 accumulation order straddles an E4M3 rounding boundary may
differ by one code, which the sparse same-sign fallback accepts.
"""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    BenchmarkBase,
    SolReport,
    TablePrinter,
    benchmark_with_stats,
    quick_probe,
    sol_bounds,
    tflops_from_ms,
)
from sm110a_kernel import (
    Mxfp4Fc1Config,
    Nvfp4Fc1Config,
    bf16tofp8_quant,
    compile_mxfp4_fc1,
    compile_nvfp4_fc1,
    from_blocked_fp8mxfp4_scales,
    mxfp4_fc1_out,
    nvfp4_fc1_out,
    prepare_mxfp4_fc1_metadata,
    prepare_nvfp4_fc1_metadata,
    quantize_mxfp4_fc1_weights,
    quantize_nvfp4_rows,
    to_blocked_mxfp4_sfa,
)
from sm110a_kernel.runner import AutotuneRunner, PreparedKernel


DEFAULT_EXPERTS = 256
DEFAULT_N = 2048
DEFAULT_K = 6144
DEFAULT_TOKEN_VALUES = (64 * 32, 128 * 32, 256 * 32, 512 * 32, 1024 * 32)
# DeepSeek-V4-Flash routes each token to its top-6 experts.
DEFAULT_TOPK = 6
DEFAULT_SWIGLU_LIMIT = 7.0
ROUTER_POWERLAW_ALPHA = 0.8
NVFP4_MMA_PEAK_TFLOPS = 4.0 * 250.0
MXFP4_W4A8_MMA_PEAK_TFLOPS = 2.0 * 250.0
# An FP8 quantizer fed by two FP32 accumulations of the same sum can flip the
# E4M3 code of values sitting on a rounding boundary.  The fallback accepts
# only a very sparse set of same-sign, near-identical dequantized values.
MAX_ADJACENT_CODE_MISMATCH_RATE = 4e-3
MIN_ADJACENT_CODE_MISMATCHES = 64

MXFP4_FC1_CONFIGS = tuple(
    Mxfp4Fc1Config(
        tile_k=128,
        cluster_m=cluster_m,
        direct_gmem=direct_gmem,
    )
    for cluster_m in (1, 2)
    for direct_gmem in (False, True)
)

# NVFP4 parity reference: tile_n=256 candidates (the measured winner family in
# benchmark_nvfp4_fc1.py) across cluster and epilogue variants.
NVFP4_FC1_PARITY_CONFIGS = tuple(
    Nvfp4Fc1Config(
        tile_n=256,
        tile_k=256,
        epilogue_tile_n=128,
        cluster_m=cluster_m,
        direct_gmem=direct_gmem,
    )
    for cluster_m in (1, 2)
    for direct_gmem in (False, True)
)


def format_candidate(config):
    epilogue = "dg" if config.direct_gmem else "tma"
    return "/".join([
        f"{config.tile_m}x{config.tile_n}x{config.tile_k}",
        f"e{config.epilogue_tile_n}",
        f"c{config.cluster_m}",
        epilogue,
    ])


def mxfp4_candidates(args):
    candidates = MXFP4_FC1_CONFIGS
    if args.tile_k is not None:
        candidates = tuple(c for c in candidates if c.tile_k == args.tile_k)
    if args.cluster_m is not None:
        candidates = tuple(c for c in candidates if c.cluster_m == args.cluster_m)
    if args.direct_gmem:
        candidates = tuple(c for c in candidates if c.direct_gmem)
    if args.tma_epilogue:
        candidates = tuple(c for c in candidates if not c.direct_gmem)
    return candidates


def fc1_tflops_from_ms(m, n, k, elapsed_ms):
    """FC1 computes both gate and up projections: 2 GEMMs, or 4MNK FLOPs."""
    return 2.0 * tflops_from_ms(m, n, k, elapsed_ms)


def unpack_e2m1(q, rows, k):
    raw = q.view(torch.uint8)
    codes = torch.stack((raw & 0xF, raw >> 4), dim=-1).reshape(rows, k)
    magnitudes = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device=q.device, dtype=torch.float32,
    )
    values = magnitudes[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -values, values)


def blocked_to_logical_scales(scales, rows, k):
    tiles = scales.view(rows // 128, k // 32, 32, 4, 4).view(torch.uint8)
    blocks = tiles.transpose(2, 3).reshape(rows // 128, k // 32, 128, 4)
    logical = blocks.permute(0, 2, 1, 3)[..., 0].reshape(rows, k // 32)
    return logical.view(torch.float8_e8m0fnu)


def quant_fp8_e8m0(x):
    """Mirror the hardware quantizer: E8M0 byte = round(log2(amax/448)) + 127."""
    rows, cols = x.shape
    blocks = x.view(rows, cols // 128, 128)
    amax = blocks.abs().amax(-1)
    exponent = torch.round(torch.log2(amax / 448.0))
    exponent = torch.where(amax > 0, exponent, torch.full_like(exponent, -127.0))
    byte = (exponent + 127.0).clamp(0, 254)
    scale = torch.ldexp(torch.ones_like(exponent), byte - 127.0)
    q = (blocks / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(rows, cols), byte.to(torch.uint8).view(torch.float8_e8m0fnu), scale


def mxfp4_fc1_reference(aq, sfa128, bq, sfb, metadata, experts, n, k, swiglu_limit):
    m = metadata.source_rows.numel()
    a = aq.float() * sfa128.float().repeat_interleave(128, dim=1)
    grouped_a = a[metadata.source_rows.long()]
    grouped = torch.empty((m, 2 * n), device=aq.device, dtype=torch.float32)
    scale_elems = (2 * n) * (k // 32) * 4
    layout = metadata.routing[2 * experts:2 * experts + m]
    for expert in range(experts):
        rows = (layout == expert).nonzero().flatten()
        if rows.numel():
            sfb_e = blocked_to_logical_scales(
                sfb[expert * scale_elems:(expert + 1) * scale_elems], 2 * n, k)
            b = unpack_e2m1(bq[expert], 2 * n, k) * sfb_e.float().repeat_interleave(32, dim=1)
            grouped[rows] = grouped_a[rows] @ b.t()

    # Physical [gate:64, up:64] blocks -> logical SwiGLU columns.
    g_blocks = grouped.view(m, 2 * n // 128, 2, 64)
    gate_v = g_blocks[:, :, 0, :].reshape(m, n)
    up_v = g_blocks[:, :, 1, :].reshape(m, n)
    if swiglu_limit > 0:
        up_v = torch.clamp(up_v, min=-swiglu_limit, max=swiglu_limit)
        gate_v = torch.clamp(gate_v, max=swiglu_limit)
    act = torch.nn.functional.silu(gate_v) * up_v
    ref_q, ref_scale_raw, ref_scale = quant_fp8_e8m0(act)
    return ref_q.float() * ref_scale.repeat_interleave(128, dim=1).view(m, n), ref_scale_raw


def mxfp4_fc1_outputs_close(out, out_scale, ref, ref_scale_raw, m, n):
    """Dequantized tolerance check plus a sparse one-code fallback."""
    ks_tiles = from_blocked_fp8mxfp4_scales(out_scale, m, n).view(torch.uint8)
    ks_tiles = ks_tiles.view(m, n // 128, 4)
    replicas_agree = bool(torch.equal(
        ks_tiles, ks_tiles[:, :, :1].expand_as(ks_tiles)))
    ks128 = ks_tiles[:, :, 0].view(torch.float8_e8m0fnu)
    got = out.float() * (ks128.view(torch.uint8).to(torch.int32) << 23
                         ).view(torch.float32).repeat_interleave(128, dim=1)
    max_diff = (got - ref).abs().max().item()
    if not replicas_agree:
        return False, max_diff
    if torch.isclose(got, ref, atol=2e-2, rtol=2e-2).all():
        return True, max_diff
    mismatched = ~torch.isclose(got, ref, atol=2e-2, rtol=2e-2)
    mismatch = int(mismatched.sum().item())
    limit = max(MIN_ADJACENT_CODE_MISMATCHES,
                int(got.numel() * MAX_ADJACENT_CODE_MISMATCH_RATE))
    if mismatch > limit:
        return False, max_diff
    got_bad, ref_bad = got[mismatched], ref[mismatched]
    rel = (got_bad - ref_bad).abs() / ref_bad.abs().clamp_min(1e-6)
    same_sign = torch.sign(got_bad) == torch.sign(ref_bad)
    return bool((same_sign & (rel <= 0.26)).all().item()), max_diff


def make_interleaved_weights(experts, n, k):
    """MXFP4 weights in the physical ``[gate:64, up:64]`` block order."""
    gate = torch.randn((experts, n, k), device="cuda", dtype=torch.bfloat16)
    up = torch.randn((experts, n, k), device="cuda", dtype=torch.bfloat16)
    return quantize_mxfp4_fc1_weights(gate, up)


def mxfp4_fc1_sol_bounds(router, topk, n, k, config):
    """Model main-kernel traffic for the W4A8 formats; routing excluded."""
    experts = router.size(1)
    selected = torch.argsort(
        torch.nan_to_num(router.float(), nan=-float("inf")),
        dim=1, descending=True, stable=True,
    )[:, :topk]
    expert_rows = torch.bincount(selected.reshape(-1), minlength=experts)
    active_experts = int((expert_rows > 0).sum().item())
    m = selected.numel()
    m_tasks = sum(
        math.ceil(rows / (config.cluster_m * config.tile_m))
        for rows in expert_rows.tolist() if rows
    )
    task_count = m_tasks * ((2 * n) // (config.tile_n * config.cluster_n))

    flops = 4 * m * n * k
    routed_a_bytes = m * (k + k / 8)
    b_bytes = active_experts * (2 * n) * (k / 2 + k / 8)
    c_bytes = m * (n + n / 8)
    staged_sfb_n = max(config.tile_n, 128)
    tile_read_bytes = task_count * (
        config.cluster_m * config.tile_m * (k + k / 8)
        + config.tile_n * k / 2
        + staged_sfb_n * k / 8
    )
    return sol_bounds(
        flops,
        routed_a_bytes + b_bytes + c_bytes,
        routed_a_bytes + b_bytes,
        tile_read_bytes,
        mma_peak_tflops=MXFP4_W4A8_MMA_PEAK_TFLOPS,
    )


class Mxfp4Fc1Benchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
        parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
        parser.add_argument("--n", type=int, default=DEFAULT_N)
        parser.add_argument("--k", type=int, default=DEFAULT_K)
        parser.add_argument("--swiglu-limit", type=float, default=DEFAULT_SWIGLU_LIMIT)
        parser.add_argument(
            "--token", "--tokens", dest="token", type=int, default=0,
            help="run one token count; default runs the standard token sweep",
        )
        self.add_timing_args(parser, warmup=2, iters=5, samples=3)
        self.add_cache_args(parser)
        self.add_pre_parallel_jit_arg(parser)
        self.add_sol_arg(parser)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--seed", type=int, default=self.seed)
        parser.add_argument("--tile-k", type=int, choices=(128,))
        parser.add_argument("--cluster-m", type=int, choices=(1, 2))
        parser.add_argument("--direct-gmem", action="store_true")
        parser.add_argument("--tma-epilogue", action="store_true")
        return parser

    def validate_args(self, args):
        self.validate_timing_args(args, allow_zero_warmup=True)
        if args.experts <= 0 or args.topk <= 0 or args.topk > args.experts:
            raise SystemExit("experts must be positive and topk must be in [1, experts]")
        # TileN=256 physical: N logical must be a multiple of 128.
        if args.n <= 0 or args.n % 128 or args.k <= 0 or args.k % 64:
            raise SystemExit("N must be a positive multiple of 128 and K of 64")
        if args.token < 0:
            raise SystemExit("token must be non-negative")
        if args.direct_gmem and args.tma_epilogue:
            raise SystemExit("--direct-gmem and --tma-epilogue are mutually exclusive")
        token_values = (args.token,) if args.token else DEFAULT_TOKEN_VALUES
        if any(tokens <= 0 or tokens % 128 for tokens in token_values):
            raise SystemExit("every token count must be a positive multiple of 128")
        args.token_values = token_values
        if not mxfp4_candidates(args):
            raise SystemExit("no compatible MXFP4 FC1 configs match the filters")
        if 2 * args.n % 256:
            raise SystemExit("physical N (2*n) must be divisible by the 256 tile")
        return args

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        routed_m = None
        print(
            f"[config] experts={args.experts} topk={args.topk} "
            f"n={args.n} k={args.k} swiglu_limit={args.swiglu_limit} "
            f"tokens={args.token_values} routing=powerlaw(alpha={ROUTER_POWERLAW_ALPHA})"
        )
        print("[parity] NVFP4 W4A4 kernel on the same shapes (tile_n=256 family)")
        table = TablePrinter((
            ("Tokens", 9), ("M", 10), ("TopK", 7), ("E", 6),
            ("N", 8), ("K", 8), ("MXFP4(ms)", 12), ("MXFP4 TF/s", 13),
            ("NVFP4(ms)", 12), ("NVFP4 TF/s", 13),
            ("NVFP4/MXFP4", 12), ("MaxDiff", 10),
            ("Cache", 8), ("MXFP4Config", 28),
        ), debug=args.debug)
        table.print_header()
        sol_report = SolReport(debug=args.debug) if args.sol else None

        bq, sfb = make_interleaved_weights(args.experts, args.n, args.k)
        # NVFP4 parity weights, interleaved the same way.
        nv_bq = None
        nv_sfb = None
        for expert in range(args.experts):
            gate = torch.randn((args.n, args.k), device="cuda", dtype=torch.bfloat16)
            up = torch.randn((args.n, args.k), device="cuda", dtype=torch.bfloat16)
            b_expert = torch.stack(
                (gate.view(args.n // 64, 64, args.k), up.view(args.n // 64, 64, args.k)),
                dim=1).reshape(2 * args.n, args.k)
            bq_expert, sfb_expert = quantize_nvfp4_rows(b_expert)
            if nv_bq is None:
                nv_bq = torch.empty((args.experts, 2 * args.n, args.k // 2),
                                    device="cuda", dtype=bq_expert.dtype)
                nv_sfb = torch.empty((args.experts * sfb_expert.numel(),),
                                     device="cuda", dtype=sfb_expert.dtype)
            nv_bq[expert].copy_(bq_expert)
            nv_sfb[expert * sfb_expert.numel():(expert + 1) * sfb_expert.numel()] \
                .copy_(sfb_expert)
            del gate, up, b_expert, bq_expert, sfb_expert

        nv_candidates = NVFP4_FC1_PARITY_CONFIGS

        for tokens in args.token_values:
            router = (lambda logits: (
                torch.randn((tokens, args.experts), device="cuda")
                * logits.unsqueeze(0)).to(torch.bfloat16))(
                torch.arange(1, args.experts + 1, device="cuda",
                             dtype=torch.float32).pow(-ROUTER_POWERLAW_ALPHA))
            a = torch.randn((tokens, args.k), device="cuda", dtype=torch.bfloat16)
            aq, sfa128 = bf16tofp8_quant(a)
            del a
            sfa = to_blocked_mxfp4_sfa(sfa128)
            nv_aq, nv_sfa = quantize_nvfp4_rows(
                torch.randn((tokens, args.k), device="cuda", dtype=torch.bfloat16))

            m = tokens * args.topk
            out = torch.empty((m, args.n), device="cuda", dtype=torch.float8_e4m3fn)
            out_scale = torch.empty((m * (args.n // 32) * 4,), device="cuda",
                                    dtype=torch.float8_e8m0fnu)
            nv_out = torch.empty((m, args.n // 2), device="cuda",
                                 dtype=torch.float4_e2m1fn_x2)
            nv_out_scale = torch.empty((m * (args.n // 16),), device="cuda",
                                       dtype=torch.float8_e4m3fn)

            metadata = prepare_mxfp4_fc1_metadata(
                router, args.topk, 2 * args.n)
            nv_metadata = prepare_nvfp4_fc1_metadata(
                router, args.topk, 2 * args.n)
            routed_m = m

            def build_mxfp4(config):
                return lambda: mxfp4_fc1_out(
                    aq, bq, sfa, sfb, metadata, out, out_scale,
                    swiglu_limit=args.swiglu_limit, config=config)

            def build_nvfp4(config):
                return lambda: nvfp4_fc1_out(
                    nv_aq, nv_bq, nv_sfa, nv_sfb, nv_metadata, nv_out,
                    nv_out_scale, config=config)

            options = self.make_tune_options(args, role="ours",
                                             pre_parallel_jit=args.pre_parallel_jit)
            nv_options = self.make_tune_options(args, role="baseline",
                                                pre_parallel_jit=args.pre_parallel_jit)
            ours_selection = AutotuneRunner(options).resolve(
                mxfp4_candidates(args),
                namespace="mxfp4_fc1",
                key=("v1-w4a8", tokens, args.experts, args.topk, args.n, args.k),
                prepare=lambda config: (lambda run: (
                    run(), torch.cuda.synchronize(),
                    PreparedKernel(config, run))[2])(
                    build_mxfp4(config)),
                benchmark=lambda prepared: benchmark_with_stats(
                    prepared.run, args.warmup, args.iters, args.samples,
                    cold_l2_flush=args._cold_l2_flush),
                probe=lambda prepared: quick_probe(
                    prepared.run, args.iters, cold_l2_flush=args._cold_l2_flush),
                format_candidate=format_candidate,
                metric_formatter=lambda _c, stats: (
                    f"{fc1_tflops_from_ms(routed_m, args.n, args.k, stats['median_ms']):.1f} TF/s"),
                pre_jit=compile_mxfp4_fc1,
            )
            nv_selection = AutotuneRunner(nv_options).resolve(
                nv_candidates,
                namespace="nvfp4_fc1",
                key=("mxfp4-parity", tokens, args.experts, args.topk, args.n, args.k),
                prepare=lambda config: (lambda run: (
                    run(), torch.cuda.synchronize(),
                    PreparedKernel(config, run))[2])(
                    build_nvfp4(config)),
                benchmark=lambda prepared: benchmark_with_stats(
                    prepared.run, args.warmup, args.iters, args.samples,
                    cold_l2_flush=args._cold_l2_flush),
                probe=lambda prepared: quick_probe(
                    prepared.run, args.iters, cold_l2_flush=args._cold_l2_flush),
                format_candidate=format_candidate,
                metric_formatter=lambda _c, stats: (
                    f"{fc1_tflops_from_ms(routed_m, args.n, args.k, stats['median_ms']):.1f} TF/s"),
                pre_jit=compile_nvfp4_fc1,
            )

            max_diff = float("nan")
            if not args.no_accuracy:
                ours_selection.prepared.run()
                torch.cuda.synchronize()
                ref, ref_scale_raw = mxfp4_fc1_reference(
                    aq, sfa128, bq, sfb, metadata, args.experts, args.n, args.k,
                    args.swiglu_limit)
                close, max_diff = mxfp4_fc1_outputs_close(
                    out, out_scale, ref, ref_scale_raw, m, args.n)
                del ref
                if not close:
                    raise AssertionError(
                        f"MXFP4 FC1 mismatch: max diff {max_diff}")

            kernel_ms = ours_selection.stats["median_ms"]
            nv_ms = nv_selection.stats["median_ms"]
            table.print_row(
                tokens, m, args.topk, args.experts, args.n, args.k,
                f"{kernel_ms:.4f}",
                f"{fc1_tflops_from_ms(m, args.n, args.k, kernel_ms):.2f}",
                f"{nv_ms:.4f}",
                f"{fc1_tflops_from_ms(m, args.n, args.k, nv_ms):.2f}",
                f"{kernel_ms / nv_ms:.2f}x" if nv_ms > 0 else "N/A",
                f"{max_diff:.4f}",
                "hit" if ours_selection.cache_hit else "miss",
                format_candidate(ours_selection.candidate),
            )
            del out, out_scale, nv_out, nv_out_scale, aq, sfa, sfa128, nv_aq, nv_sfa
            torch.cuda.empty_cache()
            if args.sol:
                sol_report.add(
                    mxfp4_fc1_sol_bounds(
                        router, args.topk, args.n, args.k, ours_selection.candidate),
                    kernel_ms,
                    label=f"MXFP4 FC1 kernel (tokens={tokens}, routing excluded)",
                )

        if sol_report:
            sol_report.print()


if __name__ == "__main__":
    Mxfp4Fc1Benchmark().run()
