# Search frame

## Question

Which mechanism can remove the largest Qwen3.8 decode inefficiency on P150x2 without weakening exact-output or fp32-state requirements?

## Fixed facts

- Full traced decode median: 55.3 ms.
- GDN weighted time: 28.5 ms, or 47.1%.
- MLP weighted time: 24.0 ms, or 39.5%.
- One GDN layer median: 594 microseconds.
- One GDN layer weight-read floor: about 113 microseconds.
- One MLP layer median: 374 microseconds.
- One MLP layer weight-read floor: about 261 microseconds.
- GDN efficiency is about 19%; MLP efficiency is about 70%.

## Candidate portfolio

| Candidate | Diversity | Evidence | Decision |
|---|---|---|---|
| Full TT-Lang decode kernel | One custom launch | Independent P150 code replaces about 30 TTNN launches | Blocked by missing co-versioned compiler and invalid fp32 carry on current stack |
| Conv add-chain | State-layout change | Independent code measured about 10% decode gain | Already present in local TP2 code |
| DRAM-sharded projections | Matmul-layout change | Prior local physical A/B | Rejected as slower than wide 1D |
| TTNN norm and state fusion | Existing exact-precision kernels | Matched physical roots | Rejected: GDN improved, full-step gate missed |
| Grouped-head recurrence | Matmul batch broadcast | Physical execution | Rejected: TTNN 0.74 rejects the batch broadcast |
| Package-matched fp32 recurrence kernel | One device program per GDN recurrence | Existing fp32 kernel APIs plus matched model evidence | Selected for bounded physical test |
| Wider service batch | Scheduling change | B=4 already admitted | Does not fix single-step GDN latency |

## Adversarial checks

- The independent fused kernel defaults to a state path that does not meet this repository's fp32 contract.
- TT-Lang 1.1.6 documents TF32 truncation in its attempted fp32 carry.
- The attempted SFPU fp32 route produces invalid values.
- A lower operation count can still lose when it introduces expensive concat or reduction operations.
- The local conv path already avoids those operations.
- The custom recurrence is not assumed faster or exact; fixed physical gates decide.

## Selected bounded experiment

Test `off`, `norm`, `state`, and `full` under one package closure.
Require exact outputs and state checks before any speed result is considered.
Require both GDN-local and full-step improvement before B=4 admission.
