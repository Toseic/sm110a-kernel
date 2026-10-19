#!/usr/bin/env python3
"""Cold-L2 FC1 -> FC2 PDL comparison with fixed kernel configurations.

Example: python benchmarks/benchmark_pdl.py --kernel1 nvfp4_fc1 --kernel2 nvfp4_fc2
Use nsys profile --trace=cuda,nvtx ... to inspect actual kernel interval overlap.
"""

import argparse
import importlib
import json
import sqlite3
import statistics
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sm110a_kernel as sk
from common import benchmark_once, benchmark_with_stats, powerlaw_router_scores
from sm110a_kernel.kernels.pdl import PDL_TRIGGER_POINTS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel1", choices=("nvfp4_fc1", "mxfp4_fc1"), default="nvfp4_fc1")
    p.add_argument("--kernel2", choices=("nvfp4_fc2", "mxfp4_fc2"), default="nvfp4_fc2")
    p.add_argument(
        "--config1", default="{}", help="FC1 config overrides as a JSON object"
    )
    p.add_argument(
        "--config2", default="{}", help="FC2 config overrides as a JSON object"
    )
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--intermediate", type=int, default=2048)
    p.add_argument("--max-active-clusters1", type=int)
    p.add_argument("--max-active-clusters2", type=int)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--no-accuracy", action="store_true")
    p.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="time CUDA graph replay (default); disable only for eager launch debugging",
    )
    p.add_argument(
        "--trigger-points",
        nargs="+",
        choices=tuple(PDL_TRIGGER_POINTS),
        default=["entry"],
        help="FC1 PDL trigger points to compare on identical inputs",
    )
    p.add_argument(
        "--wait-points", nargs="+", choices=("init", "before_a"), default=["init"]
    )
    p.add_argument(
        "--prepared-control",
        action="store_true",
        help="also measure serial execution with FC2 preparation moved ahead of FC1",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        help="save configuration and interleaved timing samples",
    )
    p.add_argument(
        "--timeline",
        metavar="PREFIX",
        help="run nsys and report measured kernel interval overlap; save report to PREFIX",
    )
    p.add_argument(
        "--profile-iters",
        type=int,
        default=0,
        help="extra NVTX-marked serial/PDL graph replays for nsys",
    )
    a = p.parse_args()
    if a.kernel1.split("_")[0] != a.kernel2.split("_")[0]:
        p.error("FC1/FC2 must use the same format: their intermediate layouts differ")
    if (
        min(a.tokens, a.experts, a.topk, a.hidden, a.intermediate, a.iters, a.samples)
        <= 0
    ):
        p.error("dimensions, iterations and samples must be positive")
    if a.tokens % 128 or a.hidden % 256 or a.intermediate % 256 or a.topk > a.experts:
        p.error(
            "tokens must be divisible by 128; hidden/intermediate by 256; topk <= experts"
        )
    if a.warmup < 0 or a.profile_iters < 0:
        p.error("warmup/profile-iters must be nonnegative")
    for name in ("max_active_clusters1", "max_active_clusters2"):
        if getattr(a, name) is not None and getattr(a, name) <= 0:
            p.error(f"{name} must be positive")
    for name in ("config1", "config2"):
        try:
            value = json.loads(getattr(a, name))
            if (
                not isinstance(value, dict)
                or {"pdl", "pdl_trigger", "pdl_wait", "external_prepare"} & value.keys()
            ):
                raise ValueError(
                    "expected an object without pdl/pdl_trigger; use --trigger-points"
                )
            setattr(a, name, value)
        except (ValueError, TypeError) as e:
            p.error(f"{name}: {e}")
    a.trigger_points = list(dict.fromkeys(a.trigger_points))
    a.wait_points = list(dict.fromkeys(a.wait_points))
    return a


def variants(args):
    result = {"serial": (False, False, "entry", "init")}
    if args.prepared_control:
        result["serial_signaled"] = (False, False, "entry", "init")
        result["serial_prepared"] = (False, True, "entry", "init")
    for trigger in args.trigger_points:
        for wait in args.wait_points:
            label = trigger if args.wait_points == ["init"] else f"{trigger}/{wait}"
            result[label] = (True, True, trigger, wait)
    return result


class EagerExecution:
    def __init__(self, fn):
        self.replay = fn


def capture(fn, cuda_graph=True):
    fn()
    torch.cuda.synchronize()
    if not cuda_graph:
        return EagerExecution(fn)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def measure_timeline(args):
    """Profile a fresh process so the parent does not retain a GPU context."""
    prefix = Path(args.timeline).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    child_args = sys.argv[1:]
    index = next(
        i
        for i, arg in enumerate(child_args)
        if arg == "--timeline" or arg.startswith("--timeline=")
    )
    count = 1 if "=" in child_args[index] else 2
    del child_args[index : index + count]
    repetitions = args.profile_iters or 5
    if not args.profile_iters:
        child_args += ["--profile-iters", str(repetitions)]
    subprocess.run(
        [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-graph-trace=node",
            "--force-overwrite=true",
            "-o",
            str(prefix),
            sys.executable,
            str(Path(__file__).resolve()),
            *child_args,
        ],
        check=True,
    )
    database = str(prefix) + ".sqlite"
    subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "--output",
            database,
            str(prefix) + ".nsys-rep",
        ],
        check=True,
    )
    kind = args.kernel1.split("_")[0]
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.demangledName = s.id "
            "WHERE s.value LIKE ? OR s.value LIKE ? ORDER BY k.start",
            (f"%{kind}_fc1_kernel<%", f"%{kind}_fc2_fused_kernel<%"),
        ).fetchall()
    modes = list(variants(args))
    expected = 2 * len(modes) * repetitions
    rows = rows[-expected:]
    if len(rows) != expected:
        raise RuntimeError("nsys did not record all FC1/FC2 kernel intervals")
    overlaps = {mode: [] for mode in modes}
    ratios = {mode: [] for mode in modes}
    for index in range(0, len(rows), 2):
        first, second = rows[index : index + 2]
        if (
            f"{kind}_fc1_kernel<" not in first[2]
            or f"{kind}_fc2_fused_kernel<" not in second[2]
        ):
            raise RuntimeError("unexpected profiling kernel order")
        enabled = modes[(index // 2) % len(modes)]
        overlap = max(0, min(first[1], second[1]) - max(first[0], second[0]))
        overlaps[enabled].append(overlap / 1000)
        ratios[enabled].append(
            100 * overlap / min(first[1] - first[0], second[1] - second[0])
        )
    for enabled in modes:
        label = enabled
        print(
            f"{label} actual interval overlap: {statistics.median(overlaps[enabled]):.3f} us "
            f"({statistics.median(ratios[enabled]):.2f}% of shorter kernel; "
            f"{repetitions} pairs)"
        )
    print(f"Timeline: {prefix}.nsys-rep")


def main():
    args = parse_args()
    if args.timeline:
        return measure_timeline(args)
    torch.manual_seed(1234)
    props = torch.cuda.get_device_properties(0)
    if props.major < 9:
        raise SystemExit("PDL requires compute capability >= 9.0")
    kind = args.kernel1.split("_")[0]
    prefix = "Nvfp4" if kind == "nvfp4" else "Mxfp4"
    configs = [
        getattr(sk, f"{prefix}Fc{i}Config")(**getattr(args, f"config{i}"))
        for i in (1, 2)
    ]
    for config in configs:
        config.validate()
    if (
        kind == "mxfp4"
        and configs[0].routing_weight_position != configs[1].routing_weight_position
    ):
        raise ValueError("FC1/FC2 routing_weight_position must match")
    weight_helper = importlib.import_module(f"benchmark_{kind}_fc1fc2")
    x = torch.randn((args.tokens, args.hidden), device="cuda", dtype=torch.bfloat16)
    if kind == "nvfp4":
        aq, sfa = sk.quantize_nvfp4_rows(x)
    else:
        aq, logical_scales = sk.bf16tofp8_quant(x)
        sfa = sk.to_blocked_mxfp4_sfa(logical_scales)
    w1, sfw1, w2, sfw2 = weight_helper.quantize_expert_weights(
        args.experts, args.hidden, args.intermediate
    )
    router = powerlaw_router_scores(
        args.tokens, args.experts, 0.8, device="cuda", dtype=torch.bfloat16
    )
    weights = torch.softmax(torch.topk(router.float(), args.topk, dim=1).values, dim=1)
    weights = weights.to(torch.bfloat16).contiguous()
    m = args.tokens * args.topk
    hidden = torch.empty(
        (m, args.intermediate // 2 if kind == "nvfp4" else args.intermediate),
        device="cuda",
        dtype=torch.float4_e2m1fn_x2 if kind == "nvfp4" else torch.float8_e4m3fn,
    )
    scales = torch.empty(
        m
        * (args.intermediate // 16 if kind == "nvfp4" else args.intermediate // 32 * 4),
        device="cuda",
        dtype=torch.float8_e4m3fn if kind == "nvfp4" else torch.float8_e8m0fnu,
    )
    out = torch.empty((args.tokens, args.hidden), device="cuda", dtype=torch.bfloat16)
    graphs = {}
    launches = {}
    # Preparation/compilation are outside capture and timing. Routing and tile
    # configuration are identical for serial and PDL; there is no autotuning.
    modes = list(variants(args))
    for mode in modes:
        pdl, prepared, trigger, wait = variants(args)[mode]
        c1 = replace(configs[0], pdl=mode != "serial", pdl_trigger=trigger)
        c2 = replace(configs[1], pdl=pdl, external_prepare=prepared, pdl_wait=wait)
        mod1 = getattr(sk, f"compile_{kind}_fc1")(c1)
        mod2 = getattr(sk, f"compile_{kind}_fc2")(c2)
        meta1 = mod1.prepare_metadata(router, args.topk, 2 * args.intermediate)
        layout = meta1.routing.narrow(0, 2 * args.experts, m)
        meta2 = mod2.prepare_metadata(layout, args.experts, args.hidden)
        kwargs1 = {}
        if kind == "mxfp4" and c1.routing_weight_position == "fc1":
            kwargs1["routing_weights"] = weights.float().contiguous()

        # Bind defaults so graph callbacks and standalone timing retain the
        # correct module/metadata after the next loop iteration.
        def fc1(mod=mod1, meta=meta1, kwargs=kwargs1):
            return getattr(mod, f"{kind}_fc1_out")(
                aq,
                w1,
                sfa,
                sfw1,
                meta,
                hidden,
                scales,
                max_active_clusters=args.max_active_clusters1,
                **kwargs,
            )

        def fc2(mod=mod2, meta=meta2, routing=meta1.routing):
            return getattr(mod, f"{kind}_fc2_out")(
                hidden,
                w2,
                scales,
                sfw2,
                routing,
                weights,
                meta,
                args.topk,
                out,
                max_active_clusters=args.max_active_clusters2,
            )

        def chain(first=fc1, second=fc2, mod=mod2, meta=meta2, enabled=prepared):
            if enabled:
                mod.prepare_pdl_out(out, meta)
            first()
            second()

        graphs[mode] = capture(chain, args.cuda_graph)
        launches[mode] = (fc1, fc2)
        graphs[mode].replay()
        torch.cuda.synchronize()
        if not args.no_accuracy:
            if mode == "serial":
                reference = out.clone()
                reference_hidden, reference_scales = hidden.clone(), scales.clone()
            else:
                # FC1 is deterministic; BF16 atomic FC2 reduction can round in
                # different orders. Compare multiple replays to catch stale state.
                for _ in range(3):
                    graphs[mode].replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        hidden.view(torch.uint8),
                        reference_hidden.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        scales.view(torch.uint8),
                        reference_scales.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(out, reference, rtol=0.03, atol=2.0)
                print(
                    f"accuracy [{mode}]: FC1 exact; FC2 matches serial (3 replays)",
                    flush=True,
                )

    # Capture individual serial kernels too, keeping Python dispatch out of all
    # measured intervals. Each pair starts cold; FC2 naturally sees FC1 output.
    singles = [capture(fn, args.cuda_graph) for fn in launches["serial"]]
    eviction = torch.empty(
        max(32 * 1024 * 1024, props.L2_cache_size * 2), device="cuda", dtype=torch.uint8
    )

    def measure(graph):
        return benchmark_with_stats(
            graph.replay,
            args.warmup,
            args.iters,
            args.samples,
            lambda: eviction.zero_(),
        )

    print(
        f"{args.kernel1} -> {args.kernel2}; cuda_graph={args.cuda_graph}; cold L2; tokens={args.tokens}, "
        f"experts={args.experts}, topk={args.topk}, H={args.hidden}, I={args.intermediate}"
    )
    for i, config in enumerate(configs, 1):
        print(f"config{i}={json.dumps(asdict(config), sort_keys=True)}", flush=True)
    t1, t2 = [measure(g)["median_ms"] for g in singles]
    timings = {mode: [] for mode in modes}
    # Rotate sample order; all JIT compilation and accuracy work have already
    # finished. Keep routing/config fixed while comparing trigger positions.
    for sample in range(args.samples):
        order = modes[sample % len(modes) :] + modes[: sample % len(modes)]
        for mode in order:
            timings[mode].append(
                benchmark_once(
                    graphs[mode].replay,
                    args.warmup if sample == 0 else 0,
                    args.iters,
                    lambda: eviction.zero_(),
                )
            )
    serial = statistics.median(timings["serial"])
    print(f"FC1 alone: {t1 * 1000:.3f} us; FC2 alone: {t2 * 1000:.3f} us")
    print(
        "trigger                 median_us     min_us     max_us   saved_us  saved_pct   speedup"
    )
    for mode in modes:
        values = timings[mode]
        elapsed = statistics.median(values)
        print(
            f"{mode:24s} {elapsed * 1000:9.3f} {min(values) * 1000:10.3f} "
            f"{max(values) * 1000:10.3f} {(serial - elapsed) * 1000:+10.3f} "
            f"{100 * (serial - elapsed) / serial:+9.2f}% {serial / elapsed:9.4f}x"
        )
    print(
        "Saved time is net PDL benefit; --timeline measures actual interval overlap.",
        flush=True,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "device": props.name,
                    "torch_version": str(torch.__version__),
                    "cuda_version": torch.version.cuda,
                    "seed": 1234,
                    "routing_alpha": 0.8,
                    "cuda_graph": args.cuda_graph,
                    "cold_l2": True,
                    "tokens": args.tokens,
                    "experts": args.experts,
                    "topk": args.topk,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "kernel1": args.kernel1,
                    "kernel2": args.kernel2,
                    "config1": asdict(configs[0]),
                    "config2": asdict(configs[1]),
                    "max_active_clusters1": args.max_active_clusters1,
                    "max_active_clusters2": args.max_active_clusters2,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "samples": args.samples,
                    "sample_ms": timings,
                },
                indent=2,
            )
            + "\n"
        )
    for _ in range(args.profile_iters):
        for enabled in modes:
            label = f"{enabled}_pair"
            eviction.zero_()
            with torch.cuda.nvtx.range(label):
                graphs[enabled].replay()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
