#!/usr/bin/env python3
"""Cold-L2 exact top-k: naive vs Torch, TRT-LLM GVR v2, and DeepSelect."""

import argparse
import importlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import BenchmarkBase, TablePrinter, benchmark_with_stats
from sm110a_kernel import compile_topk


def check_topk(scores, indices, k):
    """Exact selected-value multiset, distinct indices, and valid range (ties allowed)."""
    assert indices.shape == (scores.size(0), k)
    assert indices.dtype in (torch.int32, torch.int64)
    if not k or not scores.size(0):
        return
    idx = indices.long()
    assert ((idx >= 0) & (idx < scores.size(1))).all().item(), "index out of range"
    ordered = idx.sort(dim=1).values
    assert (ordered[:, 1:] != ordered[:, :-1]).all().item(), "duplicate indices"
    actual = scores.gather(1, idx).float().sort(dim=1).values
    expected = scores.topk(k, dim=1, sorted=False).values.float().sort(dim=1).values
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


def run_contract_checks(module):
    """Untimed naive-kernel checks, including shapes outside GVR's contract."""
    cases = 0
    for dtype in (torch.float32, torch.bfloat16):
        for n, k in (
            (1, 1),
            (31, 7),
            (257, 1),
            (513, 129),
            (4096, 1024),
            (1025, 1025),
            (131072, 512),
            (0, 0),
            (11, 0),
        ):
            scores = torch.randn(3, n, device="cuda", dtype=dtype)
            before = scores.clone()
            check_topk(scores, module.topk(scores, k), k)
            torch.testing.assert_close(scores, before)
            cases += 1
        for kind in ("ties", "constant", "masked", "special", "ascending"):
            scores = make_scores(
                3, 1025, dtype, kind if kind != "special" else "normal"
            )
            if kind == "masked":
                scores[:, 10:] = -torch.inf
            if kind == "special":
                scores[:, :7] = torch.tensor(
                    [torch.nan, torch.inf, -torch.inf, -0.0, 0.0, 1e-38, -1e-38],
                    device="cuda",
                    dtype=dtype,
                )
            out = torch.full((3, 257), -1, device="cuda", dtype=torch.int32)
            assert module.topk_out(scores, out, 257).data_ptr() == out.data_ptr()
            check_topk(scores, out, 257)
            if kind == "constant":
                torch.testing.assert_close(
                    out,
                    torch.arange(257, device="cuda", dtype=torch.int32).expand(3, -1),
                )
            cases += 1
    check_topk(
        torch.empty(0, 10, device="cuda"),
        module.topk(torch.empty(0, 10, device="cuda"), 3),
        3,
    )
    cases += 1
    scores = torch.randn(5, 513, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(5, 67, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module.topk_out(scores, out, 67)
    torch.cuda.current_stream().wait_stream(stream)
    check_topk(scores, out, 67)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        module.topk_out(scores, out, 67)
    scores.normal_()
    graph.replay()
    check_topk(scores, out, 67)
    cases += 1

    def rejected(fn):
        try:
            fn()
        except RuntimeError:
            return
        raise AssertionError("invalid input was accepted")

    x = torch.randn(3, 17, device="cuda")
    for bad in (x.cpu(), x.to(torch.float16), x.t(), x.flatten()):
        rejected(lambda: module.topk(bad, 3))
    for k in (-1, 18):
        rejected(lambda: module.topk(x, k))
    for bad_out in (
        torch.empty(3, 3, device="cuda", dtype=torch.int64),
        torch.empty(3, 4, device="cuda", dtype=torch.int32),
        torch.empty(3, 3, dtype=torch.int32),
    ):
        rejected(lambda: module.topk_out(x, bad_out, 3))
    rejected(lambda: module.topk_out(x, x.view(torch.int32), 17))
    torch.cuda.synchronize()
    print(
        f"Contract checks: {cases + 1} groups passed (boundaries, ties, special values, stream, graph, validation)"
    )


def load_gvr(root):
    directory = root / "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k"
    if not directory.is_dir():
        raise RuntimeError(f"GVR v2 source missing: {directory}")
    # Upstream explicitly supports standalone-dir imports; avoid importing TRT runtime.
    sys.path.insert(0, str(directory))
    return importlib.import_module("gvr_topk_decode_self_sampling_host")


def load_deepselect(root):
    root = root.resolve()
    if not (root / "deep_select/interface.py").is_file():
        raise RuntimeError(f"DeepSelect source missing: {root}")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("deep_select")
    except ImportError as error:
        raise RuntimeError(
            f"Build DeepSelect in {root} first; see csrc/topk/README.md"
        ) from error
    if Path(module.__file__).resolve().parent != root / "deep_select":
        raise RuntimeError("deep_select was imported from a different checkout")
    return module


def prepare_deepselect(module, scores, k):
    """Preallocate aligned indices; account separately for input padding copies."""
    rows, n = scores.shape
    input_alignment, output_alignment = module.get_stride_requirement()
    output_multiple = output_alignment // torch.int32.itemsize
    output_stride = (k + output_multiple - 1) // output_multiple * output_multiple
    out = torch.empty((rows, output_stride), device=scores.device, dtype=torch.int32)[
        :, :k
    ]
    needs_padding = scores.stride(0) * scores.element_size() % input_alignment != 0
    if needs_padding:
        multiple = input_alignment // scores.element_size()
        stride = (n + multiple - 1) // multiple * multiple
        # Logical width stays n; padding is excluded by the upstream row contract.
        aligned = torch.full(
            (rows, stride), -torch.inf, device=scores.device, dtype=scores.dtype
        )[:, :n]
        aligned.copy_(scores)
    else:
        aligned = scores

    def select():
        module.topk(
            aligned,
            k,
            sorted=False,
            sorted_index=False,
            indices_type=torch.int32,
            return_value=False,
            output_idx=out,
        )

    cases = [("deepselect-kernel", select, out)]
    if needs_padding:

        def padded_select():
            aligned.copy_(scores)
            select()

        cases.append(("pad+deepselect", padded_select, out))
    return cases


def make_scores(rows, n, dtype, distribution):
    scores = torch.randn(rows, n, device="cuda", dtype=torch.float32)
    if distribution == "ties":
        scores = scores.round()
    elif distribution == "constant":
        scores.zero_()
    elif distribution == "ascending":
        scores = (
            torch.arange(n, device="cuda", dtype=torch.float32).expand(rows, n).clone()
        )
    elif distribution == "masked":
        # Keep at least K finite entries in benchmark cases; tests cover fewer.
        scores[:, n * 3 // 4 :] = -torch.inf
    elif distribution == "spikes":
        scores[:, ::127] += 100
    return scores.to(dtype)


def prepare_indexer(args, rows, n, scores):
    from benchmark_mxfp4_indexer_score_prefill import (
        DIM,
        HEADS,
        make_blocked_scales,
        make_packed_rows,
    )

    from sm110a_kernel import (
        compile_indexer_score_decode,
        compile_indexer_score_prefill,
    )

    q = make_packed_rows(rows * HEADS).view(rows, HEADS, DIM // 2)
    weight = torch.randn(rows, HEADS, device="cuda", dtype=torch.bfloat16)
    if args.score_source == "indexer-decode":
        module = compile_indexer_score_decode()
        k = make_packed_rows(rows * n).view(rows, n, DIM // 2)
        qs = torch.cat([make_blocked_scales(HEADS) for _ in range(rows)])
        ks = torch.cat([make_blocked_scales(n) for _ in range(rows)])

        def run():
            module.indexer_score_decode_out(q, k, qs, ks, weight, scores, args.split_kv)
    else:
        module = compile_indexer_score_prefill()
        k = make_packed_rows(n)
        qs, ks = make_blocked_scales(rows * HEADS), make_blocked_scales(n)

        def run():
            module.indexer_score_prefill_out(
                q, k, qs, ks, weight, scores, args.cluster_m, causal=False
            )

    run()
    return run


class TopKBenchmark(BenchmarkBase):
    def build_parser(self):
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("--q-lens", type=int, nargs="+", default=(1, 8, 64, 256))
        p.add_argument(
            "--kv-lens", type=int, nargs="+", default=(4096, 16384, 65536, 131072)
        )
        p.add_argument("--topks", type=int, nargs="+", default=(128, 512, 1024))
        p.add_argument(
            "--dtypes", nargs="+", choices=("bf16", "fp32"), default=("bf16", "fp32")
        )
        p.add_argument(
            "--distributions",
            nargs="+",
            choices=("normal", "ties", "constant", "ascending", "masked", "spikes"),
            default=("normal",),
        )
        p.add_argument(
            "--baseline",
            choices=("none", "torch", "gvr-v2", "deepselect", "all"),
            default="all",
        )
        p.add_argument(
            "--score-source",
            choices=("random", "indexer-decode", "indexer-prefill"),
            default="random",
            help="indexer modes also time the complete score→topk chain",
        )
        p.add_argument("--split-kv", type=int, choices=(1, 2, 4, 8), default=1)
        p.add_argument("--cluster-m", type=int, choices=(1, 2), default=1)
        p.add_argument(
            "--trtllm-root", type=Path, default=ROOT / "baselinedir/TensorRT-LLM"
        )
        p.add_argument(
            "--deepselect-root", type=Path, default=ROOT / "baselinedir/DeepSelect"
        )
        p.add_argument("--json", type=Path, help="save results and source revision")
        p.add_argument("--seed", type=int, default=1234)
        p.add_argument("--no-accuracy", action="store_true")
        p.add_argument(
            "--check-only",
            action="store_true",
            help="run all naive contract checks without timing or baselines",
        )
        self.add_timing_args(p, warmup=3, iters=10, samples=5)
        return p

    def validate_args(self, args):
        if args.check_only and args.no_accuracy:
            raise SystemExit("--check-only cannot be combined with --no-accuracy")
        if any(x <= 0 for x in (*args.q_lens, *args.kv_lens, *args.topks)):
            raise SystemExit("dimensions and topks must be positive")
        if any(k > n for k, n in itertools.product(args.topks, args.kv_lens)):
            raise SystemExit("every topk must be <= every KV length")
        if args.baseline in ("gvr-v2", "all") and any(n % 4 for n in args.kv_lens):
            raise SystemExit("GVR v2 requires KV length divisible by 4")
        if args.baseline in ("deepselect", "all"):
            if any(k > 4096 for k in args.topks):
                raise SystemExit("DeepSelect requires topk <= 4096")
            if any(n >= 1 << 23 for n in args.kv_lens):
                raise SystemExit("DeepSelect requires KV length < 2^23")
        if args.score_source != "random":
            if list(args.dtypes) != ["bf16"] or list(args.distributions) != ["normal"]:
                raise SystemExit(
                    "indexer source requires --dtypes bf16 --distributions normal"
                )
            if args.score_source == "indexer-decode" and any(
                args.split_kv > (n + 127) // 128 for n in args.kv_lens
            ):
                raise SystemExit("split-kv exceeds available KV tiles")
        self.validate_timing_args(args, allow_zero_warmup=True)

    def run_with_args(self, args):
        torch.manual_seed(args.seed)
        module = compile_topk()
        if not args.no_accuracy:
            run_contract_checks(module)
        if args.check_only:
            return
        gvr = load_gvr(args.trtllm_root) if args.baseline in ("gvr-v2", "all") else None
        deepselect = (
            load_deepselect(args.deepselect_root)
            if args.baseline in ("deepselect", "all")
            else None
        )
        revision = None
        deepselect_revision = None
        if gvr:
            revision = subprocess.check_output(
                ["git", "-C", str(args.trtllm_root), "rev-parse", "HEAD"], text=True
            ).strip()
            print(f"TRT-LLM GVR v2 revision: {revision} (run_varlen, hint-free)")
        if deepselect:
            deepselect_revision = subprocess.check_output(
                ["git", "-C", str(args.deepselect_root), "rev-parse", "HEAD"], text=True
            ).strip()
            print(
                f"DeepSelect revision: {deepselect_revision} (unsorted int32, indices only)"
            )
        table = TablePrinter(
            (
                ("Qlen", 7),
                ("KVlen", 9),
                ("K", 6),
                ("dtype", 6),
                ("input", 11),
                ("implementation", 26),
                ("median(us)", 13),
                ("min(us)", 12),
                ("speedup", 10),
            )
        )
        table.print_header()
        results = []
        for rows, n, k, dtype_name, dist in itertools.product(
            args.q_lens, args.kv_lens, args.topks, args.dtypes, args.distributions
        ):
            dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
            scores = make_scores(rows, n, dtype, dist)
            score_runner = (
                prepare_indexer(args, rows, n, scores)
                if args.score_source != "random"
                else None
            )
            out = torch.empty((rows, k), device="cuda", dtype=torch.int32)
            cases = [("naive", lambda: module.topk_out(scores, out, k), out)]
            if args.baseline in ("torch", "all"):
                values = torch.empty((rows, k), device="cuda", dtype=dtype)
                torch_out = torch.empty((rows, k), device="cuda", dtype=torch.int64)
                cases.append(
                    (
                        "torch",
                        lambda: torch.topk(
                            scores, k, dim=1, sorted=False, out=(values, torch_out)
                        ),
                        torch_out,
                    )
                )
            if gvr:
                logits = scores.float()
                gvr_out = torch.empty_like(out)
                lengths = torch.full((rows,), n, device="cuda", dtype=torch.int32)
                workspace = torch.zeros(
                    gvr.workspace_bytes(), device="cuda", dtype=torch.uint8
                )

                def run_gvr():
                    gvr.run_varlen(
                        logits, lengths, gvr_out, max_seq_len=n, workspace=workspace
                    )

                cases.append(("gvr-v2-kernel", run_gvr, gvr_out))
                if dtype != torch.float32:

                    def converted_gvr():
                        logits.copy_(scores)
                        run_gvr()

                    cases.append(("cast+gvr-v2", converted_gvr, gvr_out))
            if deepselect:
                cases.extend(prepare_deepselect(deepselect, scores, k))
            if score_runner:
                # Select the complete adapter path for each baseline.
                has_padding = any(case[0] == "pad+deepselect" for case in cases)
                chain_cases = [
                    case
                    for case in cases
                    if case[0] != "gvr-v2-kernel"
                    and not (has_padding and case[0] == "deepselect-kernel")
                ]
                for name, select, indices in chain_cases:

                    def chain(select=select):
                        score_runner()
                        select()

                    cases.append(("score+" + name, chain, indices))
            naive_us = None
            for name, fn, indices in cases:
                # Compile, allocate lazy state, and validate outside timed regions.
                fn()
                torch.cuda.synchronize()
                if not args.no_accuracy:
                    check_topk(scores, indices, k)
                stats = benchmark_with_stats(
                    fn,
                    args.warmup,
                    args.iters,
                    args.samples,
                    cold_l2_flush=args._cold_l2_flush,
                )
                us = stats["median_ms"] * 1000
                if name in ("naive", "score+naive"):
                    naive_us = us
                row = dict(
                    q_len=rows,
                    kv_len=n,
                    topk=k,
                    dtype=dtype_name,
                    distribution=dist,
                    score_source=args.score_source,
                    implementation=name,
                    accuracy="skipped" if args.no_accuracy else "passed",
                    **stats,
                )
                results.append(row)
                table.print_row(
                    rows,
                    n,
                    k,
                    dtype_name,
                    dist,
                    name,
                    f"{us:.3f}",
                    f"{stats['min_ms'] * 1000:.3f}",
                    f"{naive_us / us:.2f}x",
                )
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(
                    dict(
                        device=torch.cuda.get_device_name(),
                        trtllm_revision=revision,
                        deepselect_revision=deepselect_revision,
                        cold_l2=args.cold_l2,
                        clean_l2=args.clean_l2,
                        no_lock_clocks=args.no_lock_clocks,
                        warmup=args.warmup,
                        iters=args.iters,
                        samples=args.samples,
                        seed=args.seed,
                        score_source=args.score_source,
                        split_kv=args.split_kv,
                        cluster_m=args.cluster_m,
                        results=results,
                    ),
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    TopKBenchmark().run()
