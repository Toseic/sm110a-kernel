#!/usr/bin/env python3
"""Benchmark the BF16 FA4 MHA kernel against external attention kernels.

The default suite uses 32 attention heads and head dimension 128:

* causal prefill: B=1, Q=KV in {1K, 2K, 4K, 8K}
* decode: Q=1, KV=8K, B in {32, 64, 128}

Decode is deliberately non-causal. The kernel's causal mask is upper-left,
so a one-token query would otherwise only attend KV position zero; non-causal
Q=1 models the final token attending its complete KV cache.

The official FlashAttention-4 baseline requires ``flash-attn-4``::

    pip install "flash-attn-4[cu13]"

The TRTLLM-Gen baseline uses FlashInfer's standalone binding to the official
TRTLLM-Gen FMHA cubins and requires ``flashinfer-python>=0.6.18``::

    pip install "flashinfer-python[cu13]>=0.6.18"
"""

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    BenchmarkBase,
    TablePrinter,
    benchmark_with_stats,
    sol_bounds,
)
from sm110a_kernel import Bf16Fa4MhaTactic, bf16_fa4_mha_out


PREFILL_LENGTHS = (1024, 2048, 4096, 8192, 16384)
DECODE_BATCH_SIZES = (32, 64, 128)
DEFAULT_HEADS = 32
DEFAULT_HEAD_DIM = 128
TRTLLM_GEN_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))


def load_official_fa4():
    """Load the official CuTe DSL forward entry point only when requested."""
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
    except ImportError as exc:
        raise SystemExit(
            "the official FA4 baseline is not installed; run "
            f"'{sys.executable} -m pip install \"flash-attn-4[cu13]\"'"
        ) from exc

    try:
        package_version = version("flash-attn-4")
    except PackageNotFoundError:
        package_version = "unknown"

    # FA4 4.0.0 beta 29 emits this CuTe adapter warning while compiling even
    # though AuxData is registered and the resulting kernel launches correctly.
    warnings.filterwarnings(
        "ignore",
        message=r"Argument aux_data .*cannot be converted to a JitArgument.*",
        category=UserWarning,
    )
    return _flash_attn_fwd, package_version


def load_trtllm_gen():
    """Load FlashInfer's standalone TRTLLM-Gen context entry point."""
    capability = torch.cuda.get_device_capability()
    if capability not in TRTLLM_GEN_SUPPORTED_CAPABILITIES:
        supported = ", ".join(f"SM{major}{minor}" for major, minor in TRTLLM_GEN_SUPPORTED_CAPABILITIES)
        actual = f"SM{capability[0]}{capability[1]}"
        raise SystemExit(
            f"the TRTLLM-Gen FMHA cubins do not support {actual}; the current "
            f"standalone distribution supports {supported}. On Thor/SM110 use "
            "--baseline fa4 until an SM110 TRTLLM-Gen cubin is published"
        )

    try:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek
    except ImportError as exc:
        raise SystemExit(
            "the TRTLLM-Gen baseline is not installed; run "
            f"'{sys.executable} -m pip install \"flashinfer-python[cu13]>=0.6.18\"'"
        ) from exc

    try:
        package_version = version("flashinfer-python")
    except PackageNotFoundError:
        package_version = "unknown"
    return trtllm_ragged_attention_deepseek, package_version


@dataclass(frozen=True)
class AttentionCase:
    name: str
    batch: int
    heads: int
    q_len: int
    kv_len: int
    head_dim: int
    causal: bool


def attention_valid_score_count(case):
    """Return the number of logical QK scores visible after masking."""
    if not case.causal:
        return case.q_len * case.kv_len

    # The kernel uses the upper-left causal convention: query i sees keys
    # 0..min(i, KV - 1). This also handles Q and KV lengths that differ.
    ramp_rows = min(case.q_len, case.kv_len)
    return ramp_rows * (ramp_rows + 1) // 2 + (case.q_len - ramp_rows) * case.kv_len


def attention_flops(case):
    # QK^T and PV each cost 2 FLOPs per multiply-accumulate. Count only
    # score/value pairs visible through the requested attention mask.
    return 4 * case.batch * case.heads * attention_valid_score_count(case) * case.head_dim


def attention_tflops(case, elapsed_ms):
    return attention_flops(case) / (elapsed_ms * 1e9)


def attention_sol_bounds(case, *, tile_q=128, tile_kv=128):
    """Return a traffic-aware SOL model for this kernel's tiled dataflow.

    The compulsory model reads Q/K/V and writes O once.  The tiled model adds
    the K/V reloads caused by assigning one K/V mainloop to every pair of Q
    tiles.  For causal prefill it counts only the prefix visible to each pair.
    """
    element_bytes = 2  # BF16 Q, K, V, and O
    bh = case.batch * case.heads
    q_bytes = element_bytes * bh * case.q_len * case.head_dim
    kv_input_bytes = 2 * element_bytes * bh * case.kv_len * case.head_dim
    output_bytes = element_bytes * bh * case.q_len * case.head_dim
    q_pair_count = math.ceil(case.q_len / (2 * tile_q))
    kv_tile_count = math.ceil(case.kv_len / tile_kv)
    kv_tiles_per_head = 0
    for q_pair_idx in range(q_pair_count):
        if case.causal:
            q_pair_end = min((q_pair_idx + 1) * 2 * tile_q, case.q_len)
            kv_tiles_per_head += min(math.ceil(q_pair_end / tile_kv), kv_tile_count)
        else:
            kv_tiles_per_head += kv_tile_count

    # K and V are each TMA-loaded for every visited KV tile. Q is loaded once
    # per task, so it has no inter-Q-pair reload in this dual-Q-tile design.
    kv_tile_bytes = 2 * element_bytes * tile_kv * case.head_dim
    tile_read_bytes = q_bytes + bh * kv_tiles_per_head * kv_tile_bytes
    return sol_bounds(
        attention_flops(case),
        q_bytes + kv_input_bytes + output_bytes,
        q_bytes + kv_input_bytes,
        tile_read_bytes,
    )


def reference_attention(q, k, v, *, causal, scale):
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        mask = torch.ones((q.size(-2), k.size(-2)), device=q.device, dtype=torch.bool).triu(1)
        scores.masked_fill_(mask, float("-inf"))
    return torch.softmax(scores, dim=-1).matmul(v.float()).to(torch.bfloat16)


class Bf16Fa4MhaBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--single",
            action="store_true",
            help="benchmark one explicit shape instead of the default prefill/decode suite",
        )
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
        parser.add_argument("--q-len", type=int, default=1024)
        parser.add_argument("--kv-len", type=int, default=1024)
        parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
        parser.add_argument("--causal", action="store_true")
        parser.add_argument("--scale", type=float, default=None)
        parser.add_argument(
            "--baseline",
            choices=("fa4", "trtllm-gen"),
            default="fa4",
            help=(
                "attention baseline: official FlashAttention-4 CuTe DSL or "
                "TRTLLM-Gen FMHA via FlashInfer (default: fa4)"
            ),
        )
        parser.add_argument(
            "--tactic",
            choices=("16warp", "12warp"),
            default="16warp",
            help=(
                "kernel CTA schedule: 16-warp split softmax/rescale/TMA-store "
                "warpgroups or 12-warp fused per-Q-slot warpgroups (default: 16warp)"
            ),
        )
        self.add_timing_args(parser, warmup=5, iters=20, samples=7)
        self.add_sol_arg(parser)
        parser.add_argument("--debug", action="store_true", help="print benchmark records line by line")
        parser.add_argument(
            "--no-accuracy",
            action="store_true",
            help="skip the FP32 reference check in --single mode",
        )
        parser.add_argument("--atol", type=float, default=2e-2)
        parser.add_argument("--rtol", type=float, default=2e-2)
        return parser

    def validate_args(self, args):
        if min(args.batch, args.heads, args.q_len, args.kv_len, args.head_dim) <= 0:
            raise SystemExit("all dimensions must be positive")
        self.validate_timing_args(args, allow_zero_warmup=True)
        if args.scale is not None and (not math.isfinite(args.scale) or args.scale <= 0):
            raise SystemExit("--scale must be finite and positive")
        return args

    def cases(self, args):
        if args.single:
            yield AttentionCase(
                "single", args.batch, args.heads, args.q_len, args.kv_len,
                args.head_dim, args.causal,
            )
            return

        for seq_len in PREFILL_LENGTHS:
            yield AttentionCase(
                "prefill", 1, DEFAULT_HEADS, seq_len, seq_len,
                DEFAULT_HEAD_DIM, True,
            )
        for batch in DECODE_BATCH_SIZES:
            yield AttentionCase(
                "decode", batch, DEFAULT_HEADS, 1, 8192,
                DEFAULT_HEAD_DIM, False,
            )

    @staticmethod
    def make_tensors(case):
        # Our binding consumes BHSD while official FA4 consumes BSHD. Build the
        # baseline inputs once, outside the timed region, with identical values
        # and the contiguous layout expected by each implementation.
        q = torch.empty(
            (case.batch, case.heads, case.q_len, case.head_dim),
            device="cuda", dtype=torch.bfloat16,
        ).uniform_(-1, 1)
        k = torch.empty(
            (case.batch, case.heads, case.kv_len, case.head_dim),
            device="cuda", dtype=torch.bfloat16,
        ).uniform_(-1, 1)
        v = torch.empty_like(k).uniform_(-1, 1)
        q_fa4 = q.permute(0, 2, 1, 3).contiguous()
        k_fa4 = k.permute(0, 2, 1, 3).contiguous()
        v_fa4 = v.permute(0, 2, 1, 3).contiguous()
        return (
            q, k, v, torch.empty_like(q),
            q_fa4, k_fa4, v_fa4, torch.empty_like(q_fa4),
        )

    def run_case(self, case, args):
        q, k, v, out, q_fa4, k_fa4, v_fa4, out_fa4 = self.make_tensors(case)
        scale = args.scale if args.scale is not None else 1.0 / math.sqrt(case.head_dim)
        tactic = (
            Bf16Fa4MhaTactic.WARP_12
            if args.tactic == "12warp"
            else Bf16Fa4MhaTactic.WARP_16
        )

        def run_kernel():
            bf16_fa4_mha_out(q, k, v, out, causal=case.causal, scale=scale, tactic=tactic)

        if args.baseline == "fa4":
            def run_baseline():
                # Use FA4's official forward implementation directly so output is
                # preallocated and allocation/autograd bookkeeping is not part of
                # the comparison.
                self._official_fa4_fwd(
                    q_fa4,
                    k_fa4,
                    v_fa4,
                    softmax_scale=scale,
                    causal=case.causal,
                    out=out_fa4,
                )
        else:
            if case.causal and case.q_len != case.kv_len:
                raise SystemExit(
                    "TRTLLM-Gen and this kernel use different causal alignment for "
                    "Q != KV; use equal lengths or a non-causal workload"
                )

            # TRTLLM-Gen SeparateQkv context consumes packed variable-length
            # [sum(S), H, D] tensors. The BSHD tensors are already contiguous,
            # so flattening B/S is a view and is performed outside the timer.
            q_trtllm = q_fa4.view(-1, case.heads, case.head_dim)
            k_trtllm = k_fa4.view(-1, case.heads, case.head_dim)
            v_trtllm = v_fa4.view(-1, case.heads, case.head_dim)
            out_trtllm = out_fa4.view(-1, case.heads, case.head_dim)
            seq_lens = torch.full(
                (case.batch,), case.kv_len, device="cuda", dtype=torch.int32,
            )
            cum_seq_lens_q = torch.arange(
                case.batch + 1, device="cuda", dtype=torch.int32,
            ) * case.q_len
            cum_seq_lens_kv = torch.arange(
                case.batch + 1, device="cuda", dtype=torch.int32,
            ) * case.kv_len
            workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)

            def run_baseline():
                # All tensors, output, and metadata are preallocated. The
                # all-rows-active promise avoids a synchronization in the
                # generic ragged wrapper on every timed invocation.
                self._trtllm_gen_fwd(
                    q_trtllm,
                    k_trtllm,
                    v_trtllm,
                    workspace_buffer=workspace,
                    seq_lens=seq_lens,
                    max_q_len=case.q_len,
                    max_kv_len=case.kv_len,
                    bmm1_scale=scale,
                    bmm2_scale=1.0,
                    o_sf_scale=1.0,
                    batch_size=case.batch,
                    window_left=-1,
                    cum_seq_lens_q=cum_seq_lens_q,
                    cum_seq_lens_kv=cum_seq_lens_kv,
                    enable_pdl=False,
                    is_causal=case.causal,
                    return_lse=False,
                    out=out_trtllm,
                    backend="trtllm-gen",
                    skip_all_rows_active_check=True,
                )

        # Compile both JIT paths and validate their launches before timing.
        run_kernel()
        run_baseline()
        torch.cuda.synchronize()

        out_fa4_bhsd = out_fa4.permute(0, 2, 1, 3)
        torch.testing.assert_close(out, out_fa4_bhsd, atol=args.atol, rtol=args.rtol)
        max_diff = (out.float() - out_fa4_bhsd.float()).abs().max().item()

        if args.single and not args.no_accuracy:
            reference = reference_attention(q, k, v, causal=case.causal, scale=scale)
            torch.testing.assert_close(out, reference, atol=args.atol, rtol=args.rtol)
            torch.testing.assert_close(
                out_fa4_bhsd, reference, atol=args.atol, rtol=args.rtol
            )
            kernel_ref_diff = (out.float() - reference.float()).abs().max().item()
            baseline_ref_diff = (
                out_fa4_bhsd.float() - reference.float()
            ).abs().max().item()
            print(
                "[accuracy] PASS "
                f"(kernel_ref={kernel_ref_diff:.6g}, "
                f"{self._baseline_key}_ref={baseline_ref_diff:.6g}, "
                f"kernel_{self._baseline_key}={max_diff:.6g})"
            )

        kernel_stats = benchmark_with_stats(
            run_kernel,
            args.warmup,
            args.iters,
            args.samples,
            cold_l2_flush=args._cold_l2_flush,
        )
        baseline_stats = benchmark_with_stats(
            run_baseline,
            args.warmup,
            args.iters,
            args.samples,
            cold_l2_flush=args._cold_l2_flush,
        )
        return kernel_stats, baseline_stats, max_diff

    def run_with_args(self, args):
        if args.baseline == "fa4":
            self._official_fa4_fwd, baseline_version = load_official_fa4()
            self._baseline_key = "fa4"
            baseline_label = "FA4"
            print(f"[baseline] official FlashAttention-4 CuTe DSL {baseline_version}")
        else:
            self._trtllm_gen_fwd, baseline_version = load_trtllm_gen()
            self._baseline_key = "trtllm_gen"
            baseline_label = "TRTLLM-Gen"
            print(f"[baseline] official TRTLLM-Gen FMHA via FlashInfer {baseline_version}")
        if args.single:
            print("[mode] single shape (use no arguments for the default workload suite)")
        else:
            print(
                "[mode] default suite: 32 heads, head_dim=128; "
                "prefill 1K-8K and decode KV=8K"
            )
        print(f"[tactic] {args.tactic}")

        columns = (
            ("mode", 9),
            ("shape [B,H,Q,KV,D]", 24),
            ("mask", 7),
            ("tactic", 10),
            ("Kernel ms", 13),
            ("Kernel TF/s", 14),
            (f"{baseline_label} ms", 15),
            (f"vs {baseline_label}", 12),
            ("Diff", 12),
        ) + ((("SOL selected(ms)", 17), ("SOL attainment", 15)) if args.sol else ())
        table = TablePrinter(columns, debug=args.debug)
        table.print_header()
        for case in self.cases(args):
            kernel_stats, baseline_stats, max_diff = self.run_case(case, args)
            kernel_ms = kernel_stats["median_ms"]
            baseline_ms = baseline_stats["median_ms"]
            values = (
                case.name,
                f"{case.batch}x{case.heads}x{case.q_len}x{case.kv_len}x{case.head_dim}",
                "causal" if case.causal else "noncausal",
                args.tactic,
                f"{kernel_ms:.4f}",
                f"{attention_tflops(case, kernel_ms):.3f}",
                f"{baseline_ms:.4f}",
                f"{baseline_ms / kernel_ms:.3f}x",
                f"{max_diff:.6g}",
            )
            if args.sol:
                bounds = attention_sol_bounds(case)
                values += (
                    f"{bounds['sol_ms']:.4f}",
                    f"{100.0 * bounds['sol_ms'] / kernel_ms:.1f}%",
                )
            table.print_row(*values)


if __name__ == "__main__":
    Bf16Fa4MhaBenchmark().run()
