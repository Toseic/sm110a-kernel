#!/usr/bin/env python3
"""Single-GPU prefill/decode benchmark for the reduced official DSV4-Flash runtime."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

BENCHMARK_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = BENCHMARK_ROOT.parent / "models" / "deepseek_v4_flash"
MODEL_CONFIG = MODEL_ROOT / "benchmark_config_32g.json"
sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(MODEL_ROOT / "inference"))

from common import GMEM_BW_GBPS  # noqa: E402
from benchmark_utils import gib, model_storage_bytes  # noqa: E402
from model import ModelArgs, Transformer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt-path", type=Path,
                        help="Directory containing model0-mp1.safetensors; omit for random weights")
    parser.add_argument("--batch-size", "--bs", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--seq-len", "--seq", type=int, help="Model context capacity; defaults to prefill + decode")
    parser.add_argument("--prefill-len", "--prefill", "--p", type=int, default=1024, help="Prompt tokens per request")
    parser.add_argument("--decode-len", "--decode", "--d", type=int, default=128, help="Generated tokens per request")
    parser.add_argument("--decode-only", action="store_true",
                        help="skip prefill and benchmark decode against zero-initialized KV cache at --seq-len")
    parser.add_argument("--layers", type=int, help="Override the default six transformer layers")
    parser.add_argument("--warmup", type=int, default=2, help="Unmeasured iterations (default: 2)")
    parser.add_argument("--iters", type=int, default=5, help="Measured iterations (default: 5)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--memory-limit-gib", type=float, default=32.0)
    parser.add_argument("--runtime-reserve-gib", type=float, default=6.0,
                        help="Reserve for CUDA workspaces and activations (default: 6)")
    parser.add_argument("--sol", action="store_true",
                        help="report HBM SOL using model-weight and KV-cache-read traffic")
    parser.add_argument("--dry-run", action="store_true", help="Print derived config and memory budget without loading weights")
    return parser.parse_args()


def benchmark_config(args: argparse.Namespace) -> dict:
    config = json.loads(MODEL_CONFIG.read_text())
    if min(args.batch_size, args.prefill_len, args.decode_len, args.warmup, args.iters) < 0:
        raise ValueError("batch size, lengths, warmup, and iters must be non-negative")
    if args.batch_size == 0 or args.prefill_len == 0 or args.iters == 0:
        raise ValueError("batch size, prefill length, and iters must be positive")
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
        config["compress_ratios"] = config["compress_ratios"][:args.layers]
        if len(config["compress_ratios"]) != args.layers:
            raise ValueError(f"--layers may not exceed {len(json.loads(MODEL_CONFIG.read_text())['compress_ratios'])}")
        config["n_hash_layers"] = min(config["n_hash_layers"], args.layers)
    return config


def budget_check(config: dict, args: argparse.Namespace) -> tuple[int, int]:
    parameter_bytes, buffer_bytes = model_storage_bytes(config)
    resident_bytes = parameter_bytes + buffer_bytes
    budget_bytes = int(args.memory_limit_gib * 2**30)
    reserve_bytes = int(args.runtime_reserve_gib * 2**30)
    print("config:", json.dumps({key: config[key] for key in ("n_layers", "max_batch_size", "max_seq_len", "n_mtp_layers")}, sort_keys=True))
    print(f"model resident: {gib(resident_bytes):.2f} GiB (parameters {gib(parameter_bytes):.2f}, buffers {gib(buffer_bytes):.2f})")
    print(f"with runtime reserve: {gib(resident_bytes + reserve_bytes):.2f} / {args.memory_limit_gib:.2f} GiB")
    if resident_bytes + reserve_bytes > budget_bytes:
        raise RuntimeError("requested shape exceeds the configured GPU memory budget")
    return parameter_bytes, buffer_bytes


def decode_kv_cache_load_bytes(config: dict, prompt_len: int, decode_len: int) -> int:
    """Return persistent KV-cache bytes read by the official decode path.

    ``sparse_attn`` gathers only valid indices in ``topk_idxs`` rather than
    reading the entire cache. Ratio-4 layers also read their indexer's
    compressed KV cache to choose those sparse indices. Cache writes and the
    temporary prefill KV tensor are deliberately excluded: this is a read-only
    persistent-KV-cache model.
    """
    batch_size = config["max_batch_size"]
    window_size = config["window_size"]
    head_dim = config["head_dim"]
    index_head_dim = config["index_head_dim"]
    index_topk = config["index_topk"]
    bf16_bytes = 2
    total_bytes = 0

    for position in range(prompt_len, prompt_len + decode_len):
        # At decode position p, the cache includes the KV written for p.
        window_entries = min(position + 1, window_size)
        for ratio in config["compress_ratios"]:
            compressed_entries = 0
            if ratio:
                completed_groups = (position + 1) // ratio
                compressed_entries = min(index_topk, completed_groups) if ratio == 4 else completed_groups
            total_bytes += batch_size * (window_entries + compressed_entries) * head_dim * bf16_bytes
            if ratio == 4:
                # Indexer.forward scores every completed compressed entry.
                total_bytes += batch_size * ((position + 1) // ratio) * index_head_dim * bf16_bytes
    return total_bytes


def print_memory_sol(config: dict, parameter_bytes: int, prompt_len: int, decode_len: int,
                     *, include_prefill: bool = True, prefill_seconds: float | None = None,
                     decode_seconds: float | None = None) -> None:
    """Print HBM SOL from model weights and persistent KV-cache reads.

    This omits activations, KV writes, input/output tensors, and compute. A
    prefill pass reads model weights once; each decode step reads weights once
    plus only the sparse KV entries selected by the official runtime.
    """
    def hbm_ms(weight_bytes: int) -> float:
        return weight_bytes / (GMEM_BW_GBPS * 1.0e9) * 1.0e3

    def report(stage: str, weight_bytes: int, kv_bytes: int, measured_seconds: float | None) -> None:
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
    print(f"[SOL HBM] bandwidth={GMEM_BW_GBPS:.1f} GB/s; model weights={gib(parameter_bytes):.2f} GiB")
    if include_prefill:
        report("prefill", parameter_bytes, 0, prefill_seconds)
    if decode_len:
        report("decode", parameter_bytes * decode_len, decode_kv_bytes, decode_seconds)


@torch.no_grad()
def initialize_random_weights(model: Transformer, n_routed_experts: int) -> None:
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
            packed = torch.randint(-128, 128, parameter.shape, dtype=torch.int8, device=parameter.device)
            parameter.copy_(packed.view(fp4))
        elif parameter.dtype == fp8_scale:
            scale = torch.full(parameter.shape, 0.03125, dtype=torch.float32, device=parameter.device)
            parameter.copy_(scale.to(fp8_scale))
        elif parameter.dtype in (torch.int32, torch.int64):
            parameter.copy_(torch.randint(0, n_routed_experts, parameter.shape, dtype=parameter.dtype, device=parameter.device))
        elif name.endswith("norm.weight"):
            parameter.fill_(1.0)
        else:
            values = torch.randn(parameter.shape, dtype=torch.float32, device=parameter.device) * 0.02
            parameter.copy_(values.to(parameter.dtype))


@torch.inference_mode()
def run_iteration(model: Transformer, prompt: torch.Tensor, decode_len: int) -> tuple[float, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    logits = model(prompt, 0)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - start

    next_token = logits.argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for position in range(prompt.size(1), prompt.size(1) + decode_len):
        logits = model(next_token, position)
        next_token = logits.argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    return prefill_seconds, time.perf_counter() - start


@torch.inference_mode()
def run_decode_only_iteration(model: Transformer, tokens: torch.Tensor, start_pos: int, decode_len: int) -> float:
    """Time decode from an already allocated, zero-initialized KV cache."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for position in range(start_pos, start_pos + decode_len):
        logits = model(tokens, position)
        tokens = logits.argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    config = benchmark_config(args)
    parameter_bytes, _ = budget_check(config, args)
    decode_start_pos = config["max_seq_len"] - args.decode_len if args.decode_only else args.prefill_len
    if args.dry_run:
        if args.sol:
            print_memory_sol(config, parameter_bytes, decode_start_pos, args.decode_len,
                             include_prefill=not args.decode_only)
        return
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint / "model0-mp1.safetensors"
        if not checkpoint.is_file():
            raise SystemExit(f"missing converted single-GPU checkpoint: {checkpoint}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    torch.cuda.set_device(0)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(args.seed)
    with torch.device("cuda"):
        model = Transformer(ModelArgs(**config)).eval()
    if checkpoint is None:
        initialize_random_weights(model, config["n_routed_experts"])
        print("weights: deterministic random initialization")
    else:
        from safetensors.torch import load_model

        load_model(model, checkpoint, strict=False)
        print(f"weights: {checkpoint}")
    # The official kernel helpers construct cached index tensors with
    # torch.arange/full; match generate.py so those tensors are CUDA-resident.
    torch.set_default_device("cuda")
    if args.decode_only:
        tokens = torch.randint(0, config["vocab_size"], (args.batch_size, 1), device="cuda")
        for _ in range(args.warmup):
            run_decode_only_iteration(model, tokens, decode_start_pos, args.decode_len)
        decode_seconds = sum(run_decode_only_iteration(model, tokens, decode_start_pos, args.decode_len)
                             for _ in range(args.iters)) / args.iters
        prefill_seconds = None
    else:
        prompt = torch.randint(0, config["vocab_size"], (args.batch_size, args.prefill_len), device="cuda")
        for _ in range(args.warmup):
            run_iteration(model, prompt, args.decode_len)
        results = [run_iteration(model, prompt, args.decode_len) for _ in range(args.iters)]
        prefill_seconds = sum(item[0] for item in results) / len(results)
        decode_seconds = sum(item[1] for item in results) / len(results)
        prefill_tokens = args.batch_size * args.prefill_len
        print(f"prefill: {prefill_seconds * 1e3:.2f} ms, {prefill_tokens / prefill_seconds:.2f} tokens/s")
    decode_tokens = args.batch_size * args.decode_len
    if decode_tokens:
        print(f"decode: {decode_seconds * 1e3:.2f} ms, {decode_tokens / decode_seconds:.2f} tokens/s")
    if args.sol:
        print_memory_sol(config, parameter_bytes, decode_start_pos, args.decode_len,
                         include_prefill=not args.decode_only, prefill_seconds=prefill_seconds,
                         decode_seconds=decode_seconds)
    print(f"peak allocated: {gib(torch.cuda.max_memory_allocated()):.2f} GiB")


if __name__ == "__main__":
    main()
