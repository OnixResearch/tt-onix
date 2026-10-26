# Tasks: Investigate MLP projection weight streaming

## Phase 1: Device-free characterization

- [x] [serial] P1.1 Characterize per-layer gate/up/down weight rereads and residency from the packaged source, with no measurement; declare payload/tiled traffic, source identities and locations, assumptions, findings, and coverage bounds in the source-audit evidence style. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]
  - Completed negative characterization: exact packaged 0.77 selected-reader extraction and installed configuration builder cover all B1/B8 gate/up/down cases. Each of 43,520 weight pages per projection is requested once; total 97,484,800 B/device/layer. Persistent GDDR ownership is distinct from bounded L1 input CBs. [Reader evidence](evidence/p1-reader-transactions-2026-10-25.md).
- [x] [serial] P1.2 Fix the numerical-retention screen before implementation: name the NLL/perplexity floor, corpus, and baseline/candidate comparison method; retain the `QWEN36_MLP_DOWN_BF4` NLL rejection and `matmul_reduce_scatter_decode` decode-M=1 rejection as hurdles any variant must beat. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]
  - Closed at the authorized original six-passage B1 scope: matched equal-six-passage mean delta NLL ≤ +0.010 nats, equivalent perplexity ratio ≤ exp(0.010). Historical 0.7451044766709086 NLL is not a global floor. Frozen corpus, provenance rules, executable offline score/compare recipe, and passing/failing synthetic protocol proof are recorded in the [retention screen](evidence/p1-retention-screen-2026-10-25.md). No new model retention result or hardware authorization.
- [x] [serial] P1.3 Establish legality and the baseline-relative traffic/reread model for candidate bank-matched DRAM-sharded weight residency: exactly 8 DRAM reader banks, compute grid bound to the 1×8 shard grid, and `per_core_N` derived from it; keep dtype, equations, compute fidelity, accumulation, collective order, the separate matmul/reduce-scatter contract, and every other projection's grid unchanged. Reject unsupported layouts or unchanged-bytes bandwidth above 512 GB/s. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]
  - Completed **rejection closure**, not a legal/admitted candidate: the actual selected baseline has no duplicate weight-page requests to eliminate, and bank placement alone removes no mandatory unchanged-dtype matrix payload. Candidate request lengths/coalescing and full implementation legality are not established or claimed by this rejection. [Reader evidence](evidence/p1-reader-transactions-2026-10-25.md), [design verdict](design.md#p1-verdict-2026-10-25). P2 remains blocked; no prototype, successor, or hardware probe is introduced.

## Phase 2: Device-free prototype

- [ ] [serial] P2.1 After defining the numerical-retention screen, prototype the bank-matched residency candidate with a device-free fail-closed legality/rejection plan module behind a default-off admission flag and the screen wired; record device-free evidence, exact source locations, method, findings, residual assumptions, and coverage bounds without a performance or silicon claim. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]
  - Blocked on a successor mechanism that changes aggregate weight bytes or actual rereads, not placement alone. P2.1 remains unchecked; no prototype is authorized by the rejected bank-matching mechanism.

## Phase 3: Bounded hardware probe (deferred)

- [ ] [serial] P3.1 After devices 0 and 1 are released, predeclare the bounded `mlp-projection-streaming-p3` window: exact commands, duration, baseline/candidate identities, retention comparison, and stop conditions; require the traffic model, retention screen, and device-free evidence before execution. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]
- [ ] [serial] P3.2 Execute only that declared window and record results separately from source evidence and arithmetic; reject above-ceiling unchanged-byte interpretations and any retention failure. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]

## Phase 4: Promotion

- [ ] [serial] P4.1 Consider promotion only under the existing admission rules, with no admission on a bandwidth figure above the packaged ceiling; preserve the distinction between broad MLP attribution and recoverable projection time. r[tenstorrent.qwen38_p150x2.mlp_projection_streaming]
