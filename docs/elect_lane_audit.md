# Lane-zero election audit

Measured on NVIDIA Thor (`sm_110a`), 2026-10-15, starting at `2217c6f`.
Indexer, attention, quantization, routing, and the final NVFP4 comparison
locked GPC/EMC to 1575/4266 MHz. MXFP4 fused screening used its script's
normal clock policy. All results below use the default cold-L2 dirty write
flush (64 MiB).

## Retained change

Only the **paged indexer decode MMA issuer** benefits clearly enough to keep.
Use `elect_one()` for both dense and paged layouts at that issue point.
All other attempted source changes were restored.

For B=1, KV=8192, the following values are medians across two independent
benchmark invocations per version; each invocation itself reports the median
of seven samples. Speedup is baseline time divided by elected time.

| Page rows | SplitKV | Lane-zero baseline (us) | Elected issuer (us) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 35.43 | 27.39 | 1.294x |
| 128 | 4 | 15.00 | 14.24 | 1.053x |
| 256 | 1 | 35.75 | 26.07 | 1.371x |
| 256 | 4 | 14.77 | 14.05 | 1.052x |

The broader validation crossed B=1/8/32, KV=1024/8192/32768,
SplitKV=1/2/4/8 and page rows=128/256, with the benchmark's default balanced
ragged KV lengths. All **72 cases matched the reference exactly** (max
difference zero). Geometric mean speedups over each page size's 36 cases were
1.065x and 1.063x, respectively. These are unweighted averages over this
explicit workload grid, not an application-level speedup claim.

Some initial short/high-split measurements regressed. Rechecking the affected
page-256 cases with 10 warmups, 100 iterations and 15 samples gave:

| B | KV | SplitKV | Baseline (us) | Elected (us) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 4 | 8.78 | 8.60 |
| 1 | 1024 | 8 | 8.78 | 8.55 |
| 1 | 32768 | 4 | 40.10 | 36.94 |
| 1 | 32768 | 8 | 29.83 | 30.08 |

The last case remains approximately 0.8% slower; the change does not improve
every shape. All four rechecked cases also matched the reference exactly.

`cuobjdump --dump-sass` on the final binaries confirms that the paged
SplitKV=1, page-128 instance (four load stages, two result stages) goes from
53 to 30 static `ELECT` instructions and from 154 to 130 `BRA` instructions.
All four dense SplitKV instances have instruction-identical SASS before and
after. Counts refer to static instructions, not dynamic execution counts.

Raw case timings, commands, accuracy results, routing repeats, and rejected
MLA/NVFP4 comparisons are in [elect_lane_audit_results.json](elect_lane_audit_results.json).

## Rejected changes

- MHA's broad replacement slowed causal prefill by about 1%; producer-only
  changes also did not consistently improve both schedules and workloads.
- GQA/MQA initially showed small gains, but repeat measurements were mixed
  or below 1%. GQA prefill also regressed. These do not establish a reliable
  improvement against the observed measurement variation.
- MLA's long-KV improvements came with short-KV regressions. For example,
  1CTA B=1 improved 0.9891 to 0.9762 ms at KV=32768, but regressed 0.0455 to
  0.0463 ms at KV=1024. Both implementations were restored.
- Sparse MLA, quantizer barrier notifications, and routing grouping did not
  show consistent benefits. Lane-dependent reduction/broadcast logic was
  never substituted.
- Indexer prefill's broad replacement slowed the dense single-CTA cases
  (0.0658 to 0.0730 ms at Q=1024; 0.5012 to 0.5213 ms at Q=4096).
- MXFP4 fused row scheduling slowed 1.080621 to 1.113538 ms. The fixed tiled
  comparison slowed 0.317413 to 0.329189 ms. Both were restored.
- NVFP4 fused cold-L2 A/B repeats had medians 0.476725 ms for lane-zero and
  0.477424 ms for elected issuers, with overlapping run variation. It was
  restored. Its benchmark needed temporary cold-L2 and clock-lock plumbing
  for this experiment; the temporary instrumentation was also restored.

## Scope

The audit found 117 lane-zero expressions in 20 CUDA source/header files,
including `lane_idx == 0`, `issuer_lane == 0`, `(tid & 31) == 0`, and
`lane != 0` early returns. Three expressions require fixed lane identity:
the lane-zero reduction and subsequent shuffle broadcast in the two FP8
quantizers, and the lower-lane mask expression in BF16 FC2 routing. These
were left intact. CTA-wide `tid == 0` operations are a separate category
and were not mechanically converted into warp elections.

| Component | Files | Expressions | Experiment |
| --- | ---: | ---: | --- |
| MHA, GQA, MQA | 3 | 46 | All eligible sites, then TMA producers/stores alone |
| MLA, 1CTA and 2CTA | 2 | 16 | All eligible sites, then TMA producers/stores alone |
| Sparse MLA decode and prefill | 2 | 16 | All eligible sites, then TMA producers/stores alone |
| BF16-to-FP4/FP8/NVFP4 and FP8 activation quantization | 4 | 10 | Warp barrier notifications; preserve subgroup reductions |
| Indexer decode and prefill | 2 | 10 | All eligible sites, then isolate decode MMA issue |
| MXFP4/NVFP4/BF16 FC1/BF16 FC2 routing | 4 | 5 | Isolated stable grouping kernels, alternating A/B runs |
| MXFP4 row/tiled and NVFP4 fused FC1+FC2 | 3 | 14 | Single-thread copies and barrier notifications |

An election uses the existing kernel-local helper and executes with a full
warp mask at warp-uniform control flow. PTX specifies deterministic leader
selection for a fixed mask; consecutive bulk-copy issue/commit/wait operations
must use that same leader. Lane-specific values cannot simply be moved to an
arbitrary elected thread.

## Measurements

Screening used `--warmup 5 --iters 20 --samples 7`, with accuracy enabled.
Attention tested causal B=1, Q=KV=1024 and noncausal B=32, Q=1, KV=8192.
Both 16warp/12warp tactics were fixed for MHA/GQA. MQA used its fixed schedule.
MLA tested B=1/32, KV=8192, both implementations. Sparse MLA tested B=32
decode and B=1, Q=128 append over KV=8192, each selecting 640 positions.
Quantizers tested 128/1024/8192 tokens (also one token where supported),
D=4096, and D=7168 for NVFP4; FP8 activation quantization tested both block
sizes and both scale modes. Indexer prefill tested Q=KV=1024/4096, B=1,
cluster M=1/2, dense and paged KV. Decode tested B=1/32, KV=8192,
SplitKV=1/4, dense and 128/256-row pages.

Routing measurements extracted each production grouping kernel without
changing its body or 256-thread launch configuration. They tested 32 experts,
top-k=8, 1024/8192/65536 assignments, 10 warmups, 40 iterations, nine samples,
and three alternating A/B rounds. Both variants matched stable reference
grouping, including assignment order and source-row mapping. There was no
consistent improvement; the isolated measurements do not claim end-to-end
MoE improvements.

The fused NVFP4 screening configuration with `buffer_i=512` failed accuracy
before modification. It was replaced with the valid fixed `buffer_i=2048`,
`h_split=1`, `cluster_pairs=1`, `cache_source_rows=0` configuration; both
versions then matched FC1+FC2 bitwise. MXFP4 used M=512,
cluster M=2, H split=1, two accumulator stages, two pipeline stages,
and compact weight scales. Its final comparison fixed gather warps=4 and
reduction tile N=128 as well, avoiding autotuner configuration changes.
Both fused scripts used grouped input, which makes the effective workload
one expert with top-k=1; their `--experts`/`--topk` options apply to routed
input only. These are representative file-level experiments, not coverage
of every compile-time branch or routing mode.

## Reproduction

Representative commands, with the same flags on both source versions:

```bash
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_mxfp4_indexer_score_decode.py --batches 1 32 --kv-lens 8192 --split-kvs 1 4 --no-auto --uniform --paged-page-size 128 --warmup 5 --iters 20 --samples 7
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_gqa.py --single --batch 1 --q-len 1024 --kv-len 1024 --causal --tactic 16warp --warmup 5 --iters 20 --samples 7
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_bf16_fa4_mla.py --single --batch 1 --kv-len 8192 --impl 1cta --warmup 5 --iters 20 --samples 7
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_fp8_act_quant.py --tokens 1 128 1024 8192 --baseline none --warmup 5 --iters 20 --samples 7
/home/bifrost/miniforge3/bin/python3 benchmarks/benchmark_mxfp4_fc1fc2.py --m 512 --experts 2 --topk 2 --schedule tiled --cluster-m 2 --h-split 1 --accumulator-stages 2 --stages 2 --weight-scales compact --baseline-cluster-m 1 --gather-warps 4 --reduce-tile-n 128 --warmup 5 --iters 20 --samples 7
```

The preceding indexer optimization commits are `7bd5268` (prefill elected
issue) and `94be3b5` (dense decode elected issue). Later paged TMEM load
changes are present in this audit's baseline; historical conclusions about
lane-index predicates were remeasured against that baseline.
