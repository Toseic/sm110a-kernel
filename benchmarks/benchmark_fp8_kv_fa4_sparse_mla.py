#!/usr/bin/env python3
"""Correctness and cold-L2 timing for sparse MLA with in-kernel FP8 KV dequant."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_bf16_fa4_sparse_mla_prefill_h64 import (
    PrefillCase,
    SparseMlaPrefillBenchmark,
    make_inputs,
    reference,
)

from common import (
    BenchmarkBase,
    SolReport,
    TablePrinter,
    benchmark_with_stats,
    sol_bounds,
)
from sm110a_kernel import (
    Bf16Fa4SparseMlaPrefillRunner,
    Bf16Fa4SparseMlaRunner,
    Fp8ActQuantRunner,
    Fp8KvFa4SparseMlaRunner,
)


class Fp8KvSparseMlaBenchmark(BenchmarkBase):
    def build_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--pdl", action="store_true", help="enable programmatic dependent launch"
        )
        parser.add_argument(
            "--single", action="store_true", help="run one custom shape"
        )
        parser.add_argument(
            "--mode", choices=("all", "prefill", "decode"), default="all"
        )
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--q-len", type=int, default=16)
        parser.add_argument("--kv-len", type=int, default=None)
        parser.add_argument("--decode-kv-len", type=int, default=8192)
        parser.add_argument(
            "--json-output", type=Path, help="save sweep measurements and errors"
        )
        parser.add_argument("--selected-len", type=int, default=640)
        parser.add_argument("--invalid", type=int, default=0)
        parser.add_argument("--out-of-range", type=int, default=0)
        parser.add_argument("--scale-dtype", choices=("e8m0", "fp32"), default="e8m0")
        parser.add_argument("--scale", type=float, default=512**-0.5)
        parser.add_argument("--baseline", choices=("bf16", "none"), default="bf16")
        parser.add_argument("--no-accuracy", action="store_true")
        parser.add_argument("--atol", type=float, default=3e-2)
        parser.add_argument("--rtol", type=float, default=3e-2)
        self.add_timing_args(parser, warmup=5, iters=20, samples=7)
        self.add_sol_arg(parser)
        return parser

    def validate_args(self, args):
        if min(args.batch, args.q_len, args.decode_kv_len, args.selected_len) <= 0:
            raise ValueError("batch and lengths must be positive")
        if args.kv_len is not None and args.kv_len <= 0:
            raise ValueError("KV length must be positive")
        if args.single and args.kv_len is not None and args.kv_len < args.q_len:
            raise ValueError("KV length must be at least Q length")
        if not args.single and args.kv_len is not None:
            raise ValueError(
                "--kv-len requires --single; use --decode-kv-len for the sweep"
            )
        if (
            min(args.invalid, args.out_of_range) < 0
            or args.invalid + args.out_of_range > args.selected_len
        ):
            raise ValueError("invalid and out-of-range counts must fit selected length")
        if not math.isfinite(args.scale) or args.scale <= 0:
            raise ValueError("scale must be finite and positive")
        self.validate_timing_args(args, allow_zero_warmup=True)
        return args

    def cases(self, args):
        if args.single:
            kv_len = args.kv_len if args.kv_len is not None else args.q_len
            mode = (
                "decode"
                if args.q_len == 1
                else "append"
                if kv_len != args.q_len
                else "single"
            )
            yield PrefillCase(mode, args.batch, args.q_len, kv_len, args.selected_len)
            return
        if args.mode in ("all", "prefill"):
            # Exactly the legacy prefill/batched/append sweep.
            yield from SparseMlaPrefillBenchmark().cases(args)
        if args.mode in ("all", "decode"):
            for batch in (1, 4, 16, 32, 64, 128):
                yield PrefillCase(
                    "decode", batch, 1, args.decode_kv_len, args.selected_len
                )

    def run_case(self, case, args):
        q, original_kv, indices, sink = make_inputs(
            case.batch,
            case.q_len,
            case.kv_len,
            case.selected_len,
            args.invalid,
            args.out_of_range,
        )
        flat = original_kv[..., :448].contiguous().view(-1, 448)
        values = torch.empty_like(flat, dtype=torch.float8_e4m3fn)
        scale_dtype = (
            torch.float8_e8m0fnu if args.scale_dtype == "e8m0" else torch.float32
        )
        scales = torch.empty((flat.size(0), 7), device="cuda", dtype=scale_dtype)
        Fp8ActQuantRunner(
            flat, values, scales, 64, "ue8m0" if args.scale_dtype == "e8m0" else "fp32"
        ).run()
        values = values.view(case.batch, case.kv_len, 448)
        scales = scales.view(case.batch, case.kv_len, 7)
        rope = original_kv[..., 448:].contiguous()
        decoded = torch.cat(
            (
                (values.float() * scales.float().repeat_interleave(64, -1)).to(q.dtype),
                rope,
            ),
            dim=-1,
        )
        # Exercise both public index layouts.
        kernel_indices = indices[:, 0].contiguous() if case.q_len == 1 else indices
        out = torch.empty_like(q)
        kernel = Fp8KvFa4SparseMlaRunner(
            q,
            values,
            scales,
            rope,
            kernel_indices,
            sink,
            out,
            scale=args.scale,
            pdl=args.pdl,
        ).run
        baseline_out = torch.empty_like(q)
        baseline_runner = (
            Bf16Fa4SparseMlaRunner if case.q_len == 1 else Bf16Fa4SparseMlaPrefillRunner
        )
        baseline = baseline_runner(
            q, decoded, kernel_indices, sink, baseline_out, scale=args.scale
        ).run
        kernel()
        torch.cuda.synchronize()
        diff = None
        bitwise = None
        if not args.no_accuracy:
            expected = reference(q, decoded, indices, sink, args.scale)
            torch.testing.assert_close(out, expected, atol=args.atol, rtol=args.rtol)
            baseline()
            torch.cuda.synchronize()
            # Same BF16 operands and MMA schedule: dequant must add no error.
            torch.testing.assert_close(out, baseline_out, atol=0, rtol=0)
            diff = (out.float() - expected.float()).abs().max().item()
            bitwise = True
        timings = {}
        for name, run in (("fp8", kernel), ("bf16", baseline)):
            if name == "bf16" and args.baseline == "none":
                continue
            stats = benchmark_with_stats(
                run,
                args.warmup,
                args.iters,
                args.samples,
                cold_l2_flush=args._cold_l2_flush,
            )
            timings[name] = stats["median_ms"]
        speedup = timings["bf16"] / timings["fp8"] if "bf16" in timings else None
        valid = (indices >= 0) & (indices < case.kv_len)
        valid_rows = int(valid.sum().item())
        bounds = None
        if args.sol:
            unique_rows = sum(
                torch.unique(indices[b][valid[b]]).numel() for b in range(case.batch)
            )
            entry_bytes = 448 + 7 * scales.element_size() + 128
            q_bytes = q.numel() * q.element_size()
            index_bytes = indices.numel() * indices.element_size()
            sink_bytes = sink.numel() * sink.element_size()
            input_bytes = q_bytes + unique_rows * entry_bytes + index_bytes + sink_bytes
            tile_bytes = (
                q_bytes
                + valid_rows * entry_bytes
                + index_bytes
                + case.batch * case.q_len * sink_bytes
            )
            bounds = sol_bounds(
                4 * 64 * 512 * valid_rows,
                input_bytes + q_bytes,
                input_bytes,
                tile_bytes,
            )
        result = {
            "mode": case.mode,
            "batch": case.batch,
            "q_len": case.q_len,
            "kv_len": case.kv_len,
            "selected_len": case.selected_len,
            "valid_per_query": valid_rows / (case.batch * case.q_len),
            "fp8_ms": timings["fp8"],
            "bf16_ms": timings.get("bf16"),
            "speedup": speedup,
            "max_diff": diff,
            "bf16_bitwise_equal": bitwise,
        }
        return result, bounds

    def run_with_args(self, args):
        torch.manual_seed(0)
        table = TablePrinter(
            (
                ("mode", 9),
                ("B", 5),
                ("Q", 6),
                ("KV", 8),
                ("K", 7),
                ("Valid/Q", 9),
                ("FP8 ms", 11),
                ("BF16 ms", 11),
                ("Speedup", 10),
                ("Max diff", 12),
            )
        )
        table.print_header()
        results = []
        report = SolReport() if args.sol else None
        for case in self.cases(args):
            result, bounds = self.run_case(case, args)
            results.append(result)
            table.print_row(
                result["mode"],
                result["batch"],
                result["q_len"],
                result["kv_len"],
                result["selected_len"],
                f"{result['valid_per_query']:.1f}",
                f"{result['fp8_ms']:.4f}",
                f"{result['bf16_ms']:.4f}" if result["bf16_ms"] is not None else "-",
                f"{result['speedup']:.3f}x" if result["speedup"] is not None else "-",
                f"{result['max_diff']:.6g}"
                if result["max_diff"] is not None
                else "skipped",
            )
            if report is not None:
                report.add(
                    bounds,
                    result["fp8_ms"],
                    label=f"{case.mode} B={case.batch} Q={case.q_len}",
                )
        if report is not None:
            report.print()
        if not args.no_accuracy:
            print(
                "All cases: Torch reference passed; BF16 native output bitwise equal."
            )
        print(
            "BF16 baseline uses the same decoded KV outside timing; FP8 includes gather/dequant."
        )
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "cold_l2": args.cold_l2,
                "scale_dtype": args.scale_dtype,
                "invalid": args.invalid,
                "out_of_range": args.out_of_range,
                "scale": args.scale,
                "gpu": torch.cuda.get_device_name(),
                "lock_clocks": not args.no_lock_clocks,
                "command": sys.argv,
                "warmup": args.warmup,
                "iters": args.iters,
                "samples": args.samples,
                "results": results,
            }
            args.json_output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    Fp8KvSparseMlaBenchmark().run()
