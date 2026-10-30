#!/usr/bin/env python3
"""Benchmark the SM110A DSV4-Flash runtime with graphed decoding."""

# ruff: noqa: E402, I001

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

BENCHMARK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_ROOT.parent
OFFICIAL_MODEL_ROOT = REPO_ROOT / "models" / "deepseek_v4_flash"
SM110A_MODEL_ROOT = REPO_ROOT / "models" / "deepseek_v4_flash_sm110a"
MODEL_CONFIG = SM110A_MODEL_ROOT / "benchmark_config_32g.json"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(OFFICIAL_MODEL_ROOT))
sys.path.insert(0, str(OFFICIAL_MODEL_ROOT / "inference"))

from benchmark_utils import gib, model_storage_bytes
from common import GMEM_BW_GBPS, B300Sim, BenchmarkBase
from models.deepseek_v4_flash_sm110a.model import (
    ModelArgs as Sm110aModelArgs,
    Transformer as Sm110aTransformer,
    backend_stats,
    clear_backend_cache,
    reset_backend_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    BenchmarkBase.add_b300_sim_args(parser)
    parser.add_argument(
        "--checkpoint",
        "--ckpt-path",
        type=Path,
        help="Directory containing model0-mp1.safetensors; omit for random weights",
    )
    parser.add_argument(
        "--batch-size", "--bs", type=int, default=128, help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--seq-len",
        "--seq",
        type=int,
        default=8192,
        help="Model context capacity (default: 8192)",
    )
    parser.add_argument(
        "--prefill-len",
        "--prefill",
        "--p",
        type=int,
        default=1024,
        help="Prompt tokens per request",
    )
    parser.add_argument(
        "--decode-len",
        "--decode",
        "--d",
        type=int,
        default=1,
        help="Generated tokens per request (default: 1)",
    )
    parser.add_argument(
        "--decode-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "benchmark decode against zero-initialized KV cache (default: on); "
            "use --no-decode-only to include prefill"
        ),
    )
    parser.add_argument(
        "--layers", type=int, help="Override the default two transformer layers"
    )
    parser.add_argument(
        "--routing-weight-position",
        choices=("fc1", "fc2"),
        default="fc1",
        help="compile-time expert weighting mode (default: before FC1 activation quantization)",
    )
    parser.add_argument(
        "--moe-weight-scales",
        choices=("padded", "compact"),
        default="compact",
        help="static expert weight scale format (default: compact)",
    )
    parser.add_argument(
        "--moe-backend",
        choices=("unfused", "fused"),
        default="unfused",
        help="routed MoE backend (default: separate FC1/FC2 kernels)",
    )
    parser.add_argument(
        "--fc2-reduction",
        choices=("deterministic", "atomic"),
        default=None,
        help="expert accumulation (default: deterministic for unfused, atomic for fused)",
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Unmeasured iterations (default: 2)"
    )
    parser.add_argument(
        "--iters", type=int, default=5, help="Measured iterations (default: 5)"
    )
    parser.add_argument(
        "--pdl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable native-kernel PDL in graphed decode (default: on)",
    )
    parser.add_argument(
        "--kernel-configs",
        type=Path,
        help="Record selected kernel configurations, or reuse an existing manifest ignoring PDL fields",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        help="Write individual cold-L2 decode samples and output hashes",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--profile-phase",
        choices=("prefill", "decode"),
        help="collect warmed cold-L2 passes using CUDA profiler start/stop",
    )
    parser.add_argument("--memory-limit-gib", type=float, default=32.0)
    parser.add_argument(
        "--runtime-reserve-gib",
        type=float,
        default=6.0,
        help="Reserve for CUDA workspaces and activations (default: 6)",
    )
    parser.add_argument(
        "--sol",
        action="store_true",
        help="report HBM SOL using model-weight and KV-cache-read traffic",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print derived config and memory budget without loading weights",
    )
    args = parser.parse_args()
    if args.fc2_reduction is None:
        args.fc2_reduction = (
            "atomic" if args.moe_backend == "fused" else "deterministic"
        )
    if args.moe_backend == "fused" and args.fc2_reduction != "atomic":
        parser.error("--moe-backend fused requires --fc2-reduction atomic")
    if args.moe_backend == "fused" and args.pdl:
        parser.error("--moe-backend fused requires --no-pdl")
    if args.profile_phase == "prefill" and args.decode_only:
        parser.error("--profile-phase prefill requires --no-decode-only")
    return args


def _comparison_value(value):
    if dataclasses.is_dataclass(value):
        return {
            k: v
            for k, v in dataclasses.asdict(value).items()
            if not k.startswith("pdl")
        }
    if isinstance(value, bool):
        return "<bool>"
    if isinstance(value, (tuple, list)):
        return [_comparison_value(v) for v in value]
    if isinstance(value, dict):
        return {
            k: _comparison_value(v) for k, v in value.items() if not k.startswith("pdl")
        }
    return value


@contextlib.contextmanager
def fixed_kernel_configs(path):
    """Keep shape and kernel configuration fixed across separate PDL runs."""
    if path is None:
        yield
        return
    from sm110a_kernel.runner import BaseKernelRunner

    existing = path.exists()
    manifest = json.loads(path.read_text()) if existing else {}
    original = BaseKernelRunner.tune

    def tune(runner, *args, **kwargs):
        key = json.dumps(
            [runner.namespace, _comparison_value(runner.workload_key)], sort_keys=True
        )
        if key in manifest:
            candidates = kwargs.get("candidates")
            if candidates is None:
                candidates = runner.compatible_candidates()
            matches = [c for c in candidates if _comparison_value(c) == manifest[key]]
            if not matches:
                raise RuntimeError(
                    f"No matching fixed configuration for {key}: {manifest[key]}"
                )
            kwargs["candidates"] = matches[:1]
        elif existing:
            raise RuntimeError(f"Missing fixed configuration for {key}")
        selection = original(runner, *args, **kwargs)
        manifest[key] = _comparison_value(selection.candidate)
        if not existing:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return selection

    BaseKernelRunner.tune = tune
    try:
        yield
    finally:
        BaseKernelRunner.tune = original


def benchmark_config(args: argparse.Namespace) -> dict:
    config = json.loads(MODEL_CONFIG.read_text())
    if (
        min(args.batch_size, args.prefill_len, args.decode_len, args.warmup, args.iters)
        < 0
    ):
        raise ValueError("batch size, lengths, warmup, and iters must be non-negative")
    if (
        args.batch_size == 0
        or args.prefill_len == 0
        or args.decode_len == 0
        or args.iters == 0
    ):
        raise ValueError(
            "batch size, prefill/decode lengths, and iters must be positive"
        )
    if args.decode_only:
        if args.seq_len is None:
            raise ValueError("--decode-only requires --seq-len")
        if args.decode_len == 0:
            raise ValueError("--decode-only requires a positive --decode-len")
        seq_len = args.seq_len
        if seq_len < args.decode_len:
            raise ValueError("--seq-len must be at least --decode-len")
    else:
        total_len = args.prefill_len + args.decode_len
        seq_len = args.seq_len if args.seq_len is not None else total_len
        if seq_len < total_len:
            raise ValueError("--seq-len must be at least --prefill-len + --decode-len")
    config["max_batch_size"] = args.batch_size
    config["max_seq_len"] = seq_len
    if args.layers is not None:
        if args.layers < 1:
            raise ValueError("--layers must be positive")
        config["n_layers"] = args.layers
        config["compress_ratios"] = config["compress_ratios"][: args.layers]
        if len(config["compress_ratios"]) != args.layers:
            raise ValueError(
                f"--layers may not exceed {len(json.loads(MODEL_CONFIG.read_text())['compress_ratios'])}"
            )
        config["n_hash_layers"] = min(config["n_hash_layers"], args.layers)
    return config


def budget_check(config: dict, args: argparse.Namespace) -> tuple[int, int]:
    parameter_bytes, buffer_bytes = model_storage_bytes(config)
    # The official estimator widens checkpoint BF16 LM-head weights to FP32;
    # our runtime retains BF16, including the single head shared with MTP.
    parameter_bytes -= config["vocab_size"] * config["dim"] * 2
    # Main attention retains E4M3 non-RoPE + block-64 scales + BF16 RoPE.
    # The separate MXFP4/QAT indexer cache is unchanged.
    scale_bytes = 1 if config.get("scale_dtype", "fp8") == "fp8" else 4
    nope_dim = config["head_dim"] - config["rope_head_dim"]
    cache_entry_bytes = (
        nope_dim + nope_dim // 64 * scale_bytes + config["rope_head_dim"] * 2
    )
    for ratio in config["compress_ratios"]:
        entries = config["window_size"] + (
            config["max_seq_len"] // ratio if ratio else 0
        )
        buffer_bytes -= (
            config["max_batch_size"]
            * entries
            * (config["head_dim"] * 2 - cache_entry_bytes)
        )
    resident_bytes = parameter_bytes + buffer_bytes
    # Both backends can retain padded 4-byte scales or pack static scales
    # once into the original one-byte-per-K32 scale footprint.
    logical_expert_scales = (
        config["n_layers"]
        * config["n_routed_experts"]
        * (
            2 * config["moe_inter_dim"] * config["dim"] // 32
            + config["dim"] * config["moe_inter_dim"] // 32
        )
    )
    sm110a_extra_bytes = (
        3 if args.moe_weight_scales == "padded" else 0
    ) * logical_expert_scales
    sm110a_resident_bytes = resident_bytes + sm110a_extra_bytes
    budget_bytes = int(args.memory_limit_gib * 2**30)
    reserve_bytes = int(args.runtime_reserve_gib * 2**30)
    print(
        "config:",
        json.dumps(
            {
                key: config[key]
                for key in ("n_layers", "max_batch_size", "max_seq_len", "n_mtp_layers")
            },
            sort_keys=True,
        ),
    )
    print(
        f"SM110A resident: {gib(sm110a_resident_bytes):.2f} GiB "
        f"(model tensors {gib(resident_bytes):.2f}, "
        f"blocked expert-scale overhead {gib(sm110a_extra_bytes):.2f} GiB)"
    )
    print(
        f"SM110A + runtime reserve: {gib(sm110a_resident_bytes + reserve_bytes):.2f} / "
        f"{args.memory_limit_gib:.2f} GiB"
    )
    if sm110a_resident_bytes + reserve_bytes > budget_bytes:
        raise RuntimeError("requested shape exceeds the configured GPU memory budget")
    return parameter_bytes + sm110a_extra_bytes, buffer_bytes


def decode_kv_cache_load_bytes(config: dict, prompt_len: int, decode_len: int) -> int:
    """Return persistent KV-cache bytes read by the SM110A decode path.

    The FP8-cache path gathers indices in ``topk_idxs`` rather than
    reading the entire cache. Ratio-4 layers also read their indexer's
    compressed KV cache to choose those sparse indices. Cache writes and the
    temporary prefill KV tensor are deliberately excluded: this is a read-only
    persistent-KV-cache model. Invalid/padded gathers, repeated accesses and
    temporary dequantization traffic are excluded; this is a logical lower bound.
    """
    batch_size = config["max_batch_size"]
    window_size = config["window_size"]
    head_dim = config["head_dim"]
    index_head_dim = config["index_head_dim"]
    index_topk = config["index_topk"]
    bf16_bytes = 2
    nope_dim = head_dim - config["rope_head_dim"]
    scale_bytes = 1 if config.get("scale_dtype", "fp8") == "fp8" else 4
    entry_bytes = nope_dim + nope_dim // 64 * scale_bytes + config["rope_head_dim"] * 2
    total_bytes = 0

    for position in range(prompt_len, prompt_len + decode_len):
        # At decode position p, the cache includes the KV written for p.
        window_entries = min(position + 1, window_size)
        for ratio in config["compress_ratios"]:
            compressed_entries = 0
            if ratio:
                completed_groups = (position + 1) // ratio
                compressed_entries = (
                    min(index_topk, completed_groups)
                    if ratio == 4
                    else completed_groups
                )
            total_bytes += (
                batch_size * (window_entries + compressed_entries) * entry_bytes
            )
            if ratio == 4:
                # Indexer.forward scores every completed compressed entry.
                total_bytes += (
                    batch_size * ((position + 1) // ratio) * index_head_dim * bf16_bytes
                )
    return total_bytes


def print_memory_sol(
    config: dict,
    parameter_bytes: int,
    prompt_len: int,
    decode_len: int,
    *,
    include_prefill: bool = True,
    prefill_seconds: float | None = None,
    decode_seconds: float | None = None,
) -> None:
    """Print HBM SOL from model weights and persistent KV-cache reads.

    This omits activations, KV writes, input/output tensors, and compute. A
    prefill pass reads model weights once; each decode step reads weights once
    plus only the sparse KV entries selected by the official runtime.
    """

    def hbm_ms(weight_bytes: int) -> float:
        return weight_bytes / (GMEM_BW_GBPS * 1.0e9) * 1.0e3

    def report(
        stage: str, weight_bytes: int, kv_bytes: int, measured_seconds: float | None
    ) -> None:
        traffic_bytes = weight_bytes + kv_bytes
        sol_ms = hbm_ms(traffic_bytes)
        message = (
            f"[SOL HBM] {stage}: weights={gib(weight_bytes):.2f} GiB; "
            f"KV-load={gib(kv_bytes):.2f} GiB; SOL={sol_ms:.3f} ms"
        )
        if measured_seconds is not None:
            measured_ms = measured_seconds * 1.0e3
            message += f"; measured={measured_ms:.3f} ms; attainment={100.0 * sol_ms / measured_ms:.1f}%"
        print(message)

    decode_kv_bytes = decode_kv_cache_load_bytes(config, prompt_len, decode_len)
    print(
        f"[SOL HBM] bandwidth={GMEM_BW_GBPS:.1f} GB/s; model weights={gib(parameter_bytes):.2f} GiB"
    )
    if include_prefill:
        report("prefill", parameter_bytes, 0, prefill_seconds)
    if decode_len:
        report("decode", parameter_bytes * decode_len, decode_kv_bytes, decode_seconds)


@torch.no_grad()
def initialize_random_weights(model: torch.nn.Module, n_routed_experts: int) -> None:
    """Initialize every official-runtime parameter, including packed FP4 tensors.

    FP4 weights are represented as packed nibbles, so they cannot be populated
    with ``torch.randn_like``.  Random int8 bytes provide two random FP4 values
    per storage element.  E8M0 scale tensors must be positive; a power-of-two
    scale keeps their values representable.
    """
    fp4 = torch.float4_e2m1fn_x2
    fp8_scale = torch.float8_e8m0fnu
    for name, parameter in model.named_parameters():
        if parameter.dtype == fp4:
            packed = torch.randint(
                -128, 128, parameter.shape, dtype=torch.int8, device=parameter.device
            )
            parameter.copy_(packed.view(fp4))
        elif parameter.dtype == fp8_scale:
            scale = torch.full(
                parameter.shape, 0.03125, dtype=torch.float32, device=parameter.device
            )
            parameter.copy_(scale.to(fp8_scale))
        elif parameter.dtype in (torch.int32, torch.int64):
            if name.endswith("tid2eid"):
                # Hash routing expects distinct expert ids per token.  Keep
                # random initialization deterministic without introducing
                # duplicate assignments that a top-k metadata builder cannot
                # represent.
                base = torch.randint(
                    0,
                    n_routed_experts,
                    parameter.shape[:-1],
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                offsets = torch.arange(
                    parameter.size(-1),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                parameter.copy_((base.unsqueeze(-1) + offsets) % n_routed_experts)
            else:
                parameter.copy_(
                    torch.randint(
                        0,
                        n_routed_experts,
                        parameter.shape,
                        dtype=parameter.dtype,
                        device=parameter.device,
                    )
                )
        elif name.endswith("norm.weight"):
            parameter.fill_(1.0)
        else:
            values = (
                torch.randn(
                    parameter.shape, dtype=torch.float32, device=parameter.device
                )
                * 0.02
            )
            parameter.copy_(values.to(parameter.dtype))


class DecodeCudaGraph:
    """Fixed-address CUDA Graph containing a complete autoregressive decode."""

    def __init__(
        self,
        model: torch.nn.Module,
        sample_tokens: torch.Tensor,
        start_pos: int,
        decode_len: int,
    ) -> None:
        if decode_len <= 0:
            raise ValueError("a decode CUDA Graph requires at least one step")
        self.static_tokens = torch.empty_like(sample_tokens)
        self.static_tokens.copy_(sample_tokens)
        self.cache_state = [
            (buffer, buffer.clone())
            for name, buffer in model.named_buffers()
            if name.rsplit(".", 1)[-1]
            in (
                "kv_cache",
                "kv_cache_scale",
                "kv_cache_rope",
                "kv_state",
                "score_state",
            )
        ]

        def restore_cache():
            self.reset_cache()

        # CUDA Graph capture requires all lazy initialization and allocations
        # to have run on a non-capturing stream first. This pass is setup only
        # and is deliberately excluded from --warmup and measured iterations.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            warmup_token = self.static_tokens
            for position in range(start_pos, start_pos + decode_len):
                logits = model(warmup_token, position)
                if position == start_pos:
                    expected_first_logits = logits.clone()
                warmup_token = logits.argmax(dim=-1, keepdim=True)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        expected_output = warmup_token.clone()
        expected_final_logits = logits.clone()

        restore_cache()
        self.static_tokens.copy_(sample_tokens)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            graph_token = self.static_tokens
            for position in range(start_pos, start_pos + decode_len):
                logits = model(graph_token, position)
                if position == start_pos:
                    self.static_first_logits = logits
                graph_token = logits.argmax(dim=-1, keepdim=True)
            self.static_output = graph_token
            self.static_logits = logits
        torch.cuda.synchronize()

        # Capture records operations without producing valid outputs. Compare
        # actual replays against eager decode, starting from identical caches.
        # Compare the first step's logits: BF16 atomic reductions can change
        # near-tied argmax choices, so later autoregressive inputs may differ.
        similarities = []
        tokens_match = True
        for _ in range(2):
            restore_cache()
            self.static_tokens.copy_(sample_tokens)
            self.graph.replay()
            torch.cuda.synchronize()
            if not torch.isfinite(self.static_logits).all().item():
                raise RuntimeError("decode CUDA Graph produced non-finite logits")
            cosine = torch.nn.functional.cosine_similarity(
                self.static_first_logits.float().flatten(),
                expected_first_logits.float().flatten(),
                dim=0,
            ).item()
            if cosine < 0.999:
                raise RuntimeError(
                    f"decode graph first-step logits differ from eager: cosine={cosine}"
                )
            similarities.append(cosine)
            tokens_match &= torch.equal(self.static_output, expected_output)
            if model.deterministic_fc2:
                torch.testing.assert_close(
                    self.static_first_logits, expected_first_logits, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    self.static_logits, expected_final_logits, atol=0, rtol=0
                )
                if not tokens_match:
                    raise RuntimeError(
                        "deterministic decode graph changed its final token"
                    )
        restore_cache()
        torch.cuda.synchronize()
        print(
            f"[graph] eager/replay first-step logits cosine={min(similarities):.8f}; "
            f"final tokens equal={tokens_match} "
            f"(FC2 reduction={'deterministic' if model.deterministic_fc2 else 'atomic'})"
        )

    def reset_cache(self) -> None:
        for buffer, initial in self.cache_state:
            buffer.copy_(initial)


@torch.inference_mode()
def run_iteration(
    model: torch.nn.Module, prompt: torch.Tensor, decode_graph: DecodeCudaGraph
) -> tuple[float, float]:
    decode_graph.reset_cache()
    torch.cuda.synchronize()
    start = time.perf_counter()
    logits = model(prompt, 0)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - start

    next_token = logits.argmax(dim=-1, keepdim=True)
    decode_graph.static_tokens.copy_(next_token)
    torch.cuda.synchronize()
    start = time.perf_counter()
    decode_graph.graph.replay()
    torch.cuda.synchronize()
    return prefill_seconds, time.perf_counter() - start


@torch.inference_mode()
def run_decode_only_iteration(
    decode_graph: DecodeCudaGraph, tokens: torch.Tensor
) -> float:
    """Time one replay of a decode-only CUDA Graph."""
    decode_graph.reset_cache()
    decode_graph.static_tokens.copy_(tokens)
    decode_graph.eviction.zero_()
    torch.cuda.synchronize()
    start = time.perf_counter()
    decode_graph.graph.replay()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _load_model(config, checkpoint, args):
    torch.manual_seed(args.seed)
    with torch.device("cuda"):
        model = Sm110aTransformer(
            Sm110aModelArgs(**config),
            moe_backend=args.moe_backend,
            compact_moe_scales=args.moe_weight_scales == "compact",
            routing_weight_position=args.routing_weight_position,
            deterministic_fc2=args.fc2_reduction == "deterministic",
            pdl=args.pdl,
        ).eval()
    if checkpoint is None:
        initialize_random_weights(model, config["n_routed_experts"])
    else:
        from safetensors.torch import load_model

        load_model(model, checkpoint, strict=False)
    model.prepare_for_inference()
    return model


def _measure_sm110a(config, checkpoint, args, prompt, tokens, decode_start_pos):
    print("[sm110a] loading model")
    torch.cuda.reset_peak_memory_stats()
    clear_backend_cache()
    reset_backend_stats()
    model = _load_model(config, checkpoint, args)
    if args.decode_only:
        graph_seed = tokens
    else:
        # Seed capture after a real prefill so all KV/cache state and lazy
        # model paths match normal autoregressive execution.
        graph_seed = model(prompt, 0).argmax(dim=-1, keepdim=True)
    print(f"[sm110a] capturing fixed decode CUDA Graph ({args.decode_len} steps)")
    print(
        f"[moe] backend={args.moe_backend}; scales={args.moe_weight_scales}; routing-weight-position={args.routing_weight_position}; "
        f"FC2 reduction={args.fc2_reduction}; decode PDL={args.pdl}"
    )
    decode_graph = DecodeCudaGraph(model, graph_seed, decode_start_pos, args.decode_len)
    decode_graph.eviction = torch.empty(
        max(64 << 20, 2 * torch.cuda.get_device_properties(0).L2_cache_size),
        dtype=torch.uint8,
        device="cuda",
    )

    if args.decode_only:
        for _ in range(args.warmup):
            run_decode_only_iteration(decode_graph, tokens)
        decode_samples = [
            run_decode_only_iteration(decode_graph, tokens) for _ in range(args.iters)
        ]
        decode_seconds = statistics.median(decode_samples)
        print(f"[decode samples ms] {[round(t * 1000, 4) for t in decode_samples]}")
        prefill_seconds = None
    else:
        for _ in range(args.warmup):
            run_iteration(model, prompt, decode_graph)
        results = [
            run_iteration(model, prompt, decode_graph) for _ in range(args.iters)
        ]
        prefill_seconds = sum(item[0] for item in results) / len(results)
        decode_seconds = sum(item[1] for item in results) / len(results)

    if args.profile_phase is not None:
        # Each profiler range contains only one model phase. Cache restoration
        # and dirty L2 eviction run before collection starts.
        eviction = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        for iteration in range(args.iters):
            decode_graph.reset_cache()
            decode_graph.static_tokens.copy_(graph_seed)
            eviction.fill_(iteration % 255 + 1)
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            try:
                if args.profile_phase == "prefill":
                    model(prompt, 0)
                else:
                    decode_graph.graph.replay()
                torch.cuda.synchronize()
            finally:
                torch.cuda.profiler.stop()
        print(f"[profile] collected {args.iters} cold-L2 {args.profile_phase} passes")

    if args.results_json is not None:

        def digest(tensor):
            return hashlib.sha256(
                tensor.detach().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()

        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(
            json.dumps(
                {
                    "args": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                    "decode_samples_ms": [t * 1000 for t in decode_samples]
                    if args.decode_only
                    else [],
                    "first_logits_sha256": digest(decode_graph.static_first_logits),
                    "final_logits_sha256": digest(decode_graph.static_logits),
                    "final_tokens": decode_graph.static_output.tolist(),
                },
                indent=2,
            )
        )
    result = {
        "label": "sm110a",
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "peak_bytes": torch.cuda.max_memory_allocated(),
        "kernel_stats": backend_stats(),
        "cuda_graph": True,
        "pdl": args.pdl,
        "moe_backend": args.moe_backend,
        "moe_weight_scales": args.moe_weight_scales,
    }
    del decode_graph
    del model
    gc.collect()
    clear_backend_cache()
    torch.cuda.empty_cache()
    return result


def _print_results(results, args):
    prefill_tokens = args.batch_size * args.prefill_len
    decode_tokens = args.batch_size * args.decode_len
    print("\nDecode execution: fixed CUDA Graph replay")
    print(
        "\nBackend        Prefill(ms)   Prefill tok/s   Decode(ms)   Decode tok/s   Peak GiB"
    )
    print("-" * 85)
    for result in results:
        prefill = result["prefill_seconds"]
        decode = result["decode_seconds"]
        print(
            f"{result['label']:<14} "
            f"{prefill * 1e3 if prefill is not None else float('nan'):>11.2f}   "
            f"{prefill_tokens / prefill if prefill is not None else float('nan'):>13.2f}   "
            f"{decode * 1e3:>10.2f}   "
            f"{decode_tokens / decode if decode_tokens else float('nan'):>12.2f}   "
            f"{gib(result['peak_bytes']):>8.2f}"
        )
    ours = results[-1]
    if ours["kernel_stats"]:
        print("SM110A eager/capture dispatch counts (replays do not re-enter Python):")
        for name, count in ours["kernel_stats"].items():
            print(f"  {name}: {count}")


def run_benchmark(args: argparse.Namespace) -> None:
    config = benchmark_config(args)
    sm110a_parameter_bytes, _ = budget_check(config, args)
    print(
        f"workload: batch={args.batch_size}, context capacity={config['max_seq_len']}, "
        f"decode steps={args.decode_len}, decode-only={args.decode_only}"
    )
    decode_start_pos = (
        config["max_seq_len"] - args.decode_len
        if args.decode_only
        else args.prefill_len
    )
    if args.dry_run:
        if args.sol:
            print_memory_sol(
                config,
                sm110a_parameter_bytes,
                decode_start_pos,
                args.decode_len,
                include_prefill=not args.decode_only,
            )
        return
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint / "model0-mp1.safetensors"
        if not checkpoint.is_file():
            raise SystemExit(f"missing converted single-GPU checkpoint: {checkpoint}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    torch.cuda.set_device(0)
    if hasattr(torch._C, "_accelerator_setAllocatorSettings"):
        torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
    else:
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    prompt = torch.randint(
        0,
        config["vocab_size"],
        (args.batch_size, args.prefill_len),
        device="cuda",
        generator=generator,
    )
    tokens = torch.randint(
        0,
        config["vocab_size"],
        (args.batch_size, 1),
        device="cuda",
        generator=generator,
    )
    print(
        f"weights: {checkpoint if checkpoint is not None else 'deterministic random initialization'}"
    )
    print(
        "[cache] full model passes exceed L2 capacity; reported comparisons are cold-L2"
    )

    results = [
        _measure_sm110a(
            config,
            checkpoint,
            args,
            prompt,
            tokens,
            decode_start_pos,
        )
    ]
    _print_results(results, args)
    if args.sol:
        for result in results:
            print(f"[{result['label']}]")
            print_memory_sol(
                config,
                sm110a_parameter_bytes,
                decode_start_pos,
                args.decode_len,
                include_prefill=not args.decode_only,
                prefill_seconds=result["prefill_seconds"],
                decode_seconds=result["decode_seconds"],
            )


def main() -> None:
    args = parse_args()
    with B300Sim(args), fixed_kernel_configs(args.kernel_configs):
        run_benchmark(args)


if __name__ == "__main__":
    main()
