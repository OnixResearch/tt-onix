# Tasks: Investigate MLP projection weight streaming

## Phase 1: Device-free characterization

- [ ] [serial] P1.1 Characterize per-layer gate/up/down weight rereads and residency from the packaged source, with no measurement; declare payload/tiled traffic, source identities and locations, assumptions, findings, and coverage bounds in the source-audit evidence style. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]
- [ ] [serial] P1.2 Fix the numerical-retention screen before implementation: name the NLL/perplexity floor, corpus, and baseline/candidate comparison method; retain the `QWEN36_MLP_DOWN_BF4` NLL rejection and `matmul_reduce_scatter_decode` decode-M=1 rejection as hurdles any variant must beat. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]
- [ ] [serial] P1.3 Establish legality and the baseline-relative traffic/reread model for candidate bank-matched DRAM-sharded weight residency: exactly 8 DRAM reader banks, compute grid bound to the 1×8 shard grid, and `per_core_N` derived from it; keep dtype, equations, compute fidelity, accumulation, collective order, the separate matmul/reduce-scatter contract, and every other projection's grid unchanged. Reject unsupported layouts or unchanged-bytes bandwidth above 512 GB/s. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]

## Phase 2: Device-free prototype

- [ ] [serial] P2.1 After defining the numerical-retention screen, prototype the bank-matched residency candidate with a device-free fail-closed legality/rejection plan module behind a default-off admission flag and the screen wired; record device-free evidence, exact source locations, method, findings, residual assumptions, and coverage bounds without a performance or silicon claim. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]

## Phase 3: Bounded hardware probe (deferred)

- [ ] [serial] P3.1 After devices 0 and 1 are released, predeclare the bounded `mlp-projection-streaming-p3` window: exact commands, duration, baseline/candidate identities, retention comparison, and stop conditions; require the traffic model, retention screen, and device-free evidence before execution. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]
- [ ] [serial] P3.2 Execute only that declared window and record results separately from source evidence and arithmetic; reject above-ceiling unchanged-byte interpretations and any retention failure. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]

## Phase 4: Promotion

- [ ] [serial] P4.1 Consider promotion only under the existing admission rules, with no admission on a bandwidth figure above the packaged ceiling; preserve the distinction between broad MLP attribution and recoverable projection time. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]
