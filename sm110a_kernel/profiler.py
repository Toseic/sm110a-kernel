from pathlib import Path
import gzip
import json

import torch


PROFILER_WARPS_PER_BLOCK = 6
PROFILER_ENTRY_STRIDE = 4


def make_profiler_buffer(num_blocks: int, num_entries: int, device="cuda"):
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_entries <= 0:
        raise ValueError("num_entries must be positive")
    return torch.zeros(
        (num_blocks * PROFILER_WARPS_PER_BLOCK, 1 + num_entries * PROFILER_ENTRY_STRIDE),
        dtype=torch.int64,
        device=device,
    )


def profiler_events(profiler: torch.Tensor, tags=None):
    if profiler.dim() != 2:
        raise ValueError("profiler buffer must be 2D")

    events = []
    rows = profiler.cpu().tolist()
    for slot, data in enumerate(rows):
        count = data[0]
        for i in range(count):
            base = 1 + i * PROFILER_ENTRY_STRIDE
            sm_id, tag, start, duration = data[base : base + PROFILER_ENTRY_STRIDE]
            if tags is None:
                tag_name = f"TAG_{tag}"
            else:
                tag_name = tags[tag]
            events.append(
                {
                    "name": tag_name,
                    "ph": "X",
                    "ts": start,
                    "dur": duration,
                    "pid": sm_id,
                    "tid": sm_id + slot,
                }
            )

    if not events:
        return events

    offset = min(event["ts"] for event in events)
    for event in events:
        event["ts"] -= offset
    return events


def write_trace(profiler: torch.Tensor, path, tags=None):
    trace = {"traceEvents": profiler_events(profiler, tags=tags)}
    trace_path = Path(path)
    payload = json.dumps(trace).encode("utf-8")
    if trace_path.suffix == ".gz":
        with gzip.open(trace_path, "wb") as f:
            f.write(payload)
    else:
        trace_path.write_bytes(payload)
