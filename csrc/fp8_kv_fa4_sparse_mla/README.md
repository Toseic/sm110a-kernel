# Sparse MLA with FP8 KV and BF16 Q

A separate H=64, D=512 kernel supporting both one-token decode and prefill.
It derives the MMA/softmax/output schedule from BF16 sparse MLA; the existing
BF16 implementation is unchanged. Each CTA owns one query token and all heads.

Contiguous CUDA inputs:

- Q and output: BF16 `[B,64,Q,512]` (BHSD).
- Non-RoPE KV: E4M3 FP8 `[B,N,448]`.
- Block-64 KV scales: E8M0 or FP32 `[B,N,7]`.
- RoPE KV: BF16 `[B,N,64]`.
- Indices: int32 `[B,Q,K]`, or `[B,K]` for Q=1. Negative and out-of-range
  entries are masked; duplicates remain separate softmax entries.
- Attention sink: FP32 `[64]`, included in the denominator with no value vector.

Q and cache lengths are independent. Callers provide causal selection indices.
The kernel does not read unselected cache rows or create a BF16 global cache.

## Warp ownership and dequantization

| Warps | Role |
| --- | --- |
| 0 | Q TMA load |
| 1 | Output TMA store |
| 2 | BF16 QK/PV `tcgen05.mma` issue |
| 3 | Trim fully invalid trailing tiles during prefill |
| **4–7 (WG1)** | **Sparse KV gather, scale load, register dequant, BF16 shared stores** |
| 8–11 (WG2) | FP32 online softmax and sink |
| 12–15 (WG3) | Output rescale and epilogue |

WG1 retains 96 registers per thread; WG0 retains 32, WG2 192, and WG3 96.
Each gather warp handles 16 of a tile's 64 selected rows. Four lanes cooperate
on one 64-feature chunk; each lane asynchronously loads 16 FP8 values into one
of its two BF16 shared-memory slots. It expands the values in place without
allocating extra FP8 shared storage. E8M0 scales use the native
`cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2` conversion, combining decode, scale and
BF16 rounding. FP32 scales use FP32 multiplication followed by BF16 rounding.
Both paths preserve the old QAT cache operands. Four peer lanes load all seven
scales once per selected row and broadcast the current feature scale with
`shfl`. The final feature chunk copies BF16 RoPE without dequantization.
Invalid rows are zero-filled without cache reads.

The async-copy pipeline keeps two feature chunks in flight. It publishes each
converted chunk to MMA before waiting for later chunks, preserving overlap with
PV retirement of the preceding tile. The two BF16 KV stages and per-chunk free
barriers prevent overwriting data still used by QK/PV. Each writer executes
`fence.proxy.async.shared::cta` before its release arrival on the chunk-ready
barrier (128 arrivals). Warp 2 acquires the barrier before shared MMA reads.
KV is decoded once and reused by both QK and PV. The full 64-KiB Q tile uses a
single TMA transfer/barrier; `mbarrier.try_wait` uses a 64-ns suspension hint.

Warp 3 scans backward for the last valid selected position during prefill,
allowing fully invalid trailing tiles to be skipped. It makes no prefix-order
assumption and retains one tile for an all-invalid list. Decode does not pay
for that scan. The existing BF16 kernel stays unchanged. Q remains BF16;
this kernel does not use FP8 MMA.

## Python and benchmark

```python
from sm110a_kernel import Fp8KvFa4SparseMlaRunner

Fp8KvFa4SparseMlaRunner(q, kv_fp8, scales, rope, indices, sink, out).run()
```

```bash
# Full sweep: the legacy eight prefill/batched/append cases, plus decode B=1..128.
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_fp8_kv_fa4_sparse_mla.py
# Restrict the sweep or write structured results.
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_fp8_kv_fa4_sparse_mla.py \
  --mode prefill --json-output /tmp/fp8-mla-prefill.json
# One custom shape.
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_fp8_kv_fa4_sparse_mla.py \
  --single --batch 128 --q-len 1 --kv-len 2176 --selected-len 640 --invalid 16
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_fp8_kv_fa4_sparse_mla.py \
  --single --batch 2 --q-len 129 --kv-len 257 --selected-len 193 \
  --invalid 5 --out-of-range 3 --scale-dtype fp32
```

The benchmark checks a Torch FP32 attention reference and bitwise equality
against native BF16 sparse MLA using the same reconstructed BF16 KV. It times
cold L2 only by default. The optional BF16 baseline reconstructs its KV outside
the timed region; FP8 kernel timing includes indexed loads and dequantization.
Q=1 compares the existing BF16 decode kernel; other cases compare the BF16
prefill kernel. `--sol` models the mixed cache's logical byte footprint;
`--json-output` includes measurements, correctness, arguments and clock settings.

Performance remains workload dependent: the retained version is faster on
shorter causal prefills and decode, while long prefill and full-selection
append still need tuning. A small or unstable win is not treated as an
across-the-board speedup. Further performance tuning is deferred.


## Retained performance snapshot

Thor, GPC 1575 MHz / EMC 4266 MHz, dirty cold-L2, E8M0 scales, K=640,
2 warmups, 5 iterations and 3 samples. The full 14-case sweep passed the
Torch reference and bitwise BF16-native comparison. Representative medians:

| Workload | FP8 ms | BF16 ms | BF16 / FP8 |
| --- | ---: | ---: | ---: |
| Prefill B1 Q1024 | 1.7117 | 2.1241 | 1.241x |
| Prefill B1 Q2048 | 3.6090 | 3.9387 | 1.091x |
| Prefill B1 Q4096 | 7.4338 | 7.4576 | 1.003x |
| Prefill B4 Q1024 | 6.4988 | 7.9687 | 1.226x |
| Append B1 Q128 N1024 | 0.3536 | 0.3413 | 0.965x |
| Decode B128 N8192 | 0.4827 | 0.5946 | 1.232x |

The Q4096 result is effectively a tie, and full-selection append can remain
slower. This snapshot does not establish a win for every workload or a full
model speedup. Reproduce with:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_fp8_kv_fa4_sparse_mla.py \
  --warmup 2 --iters 5 --samples 3 --json-output /tmp/fp8-mla-sweep.json
```
