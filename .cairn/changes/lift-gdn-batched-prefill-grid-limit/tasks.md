# Tasks: Lift the GDN batched-prefill grid limit

## Phase 1: Attribution

- [ ] [serial] A1 Confirm the `BH <= ncores` factory guard, its file and line, and the exact `BH` values at TP=2 for B=4, B=8, and B=16. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]
- [ ] [serial] A2 Reproduce the failure device-free and record the exact diagnostic and the two fallback paths that hide it today. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]
- [ ] [serial] A3 Reconcile the guard with the rejected B=8 receipt root cause and record whether the guard can explain both missed bounds. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]

## Phase 2: Implementation

- [ ] [serial] I1 Add the pure instance-assignment core with contiguous chunking and closed range validation. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]
- [ ] [serial] I2 Vendor the factory change that lifts the guard and passes `(bh_start, n_inst)` as runtime arguments. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]
- [ ] [serial] I3 Keep the per-core instance loop bounded and the copy width dtype-generic. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]
- [ ] [serial] I4 Keep the per-user fallback reachable and fail-closed for invalid plans. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]
- [ ] [serial] I5 Bind the factory change and the assignment core into the package execution identity. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]

## Phase 3: Verification

- [ ] [parallel] V1 Add positive device-free probes for the overflow plan, the chunk boundaries, and the unchanged in-range mapping. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid.valid]
- [ ] [parallel] V2 Add negative probes for every rejected plan shape. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid.invalid]
- [ ] [serial] V3 Prove the in-range program mapping is unchanged against the pre-change build. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid.unchanged]
- [ ] [serial] V4 Run the package build and the device-free check. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]
- [ ] [serial] V5 Run the bounded Blackhole kernel hazard audits over the changed factory. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]
- [ ] [serial] V6 Run a fresh Blackhole simulator smoke before hardware. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]
- [ ] [serial] V7 Run matched physical prefill on devices 0 and 1 and require exact per-position parity against the serialized reference. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid.valid]

## Phase 4: Completion

- [ ] [serial] D1 Record the mechanism result, the parity table, and every non-claim; do not move any admission binding. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]
- [ ] [serial] D2 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]

## Verification Coverage

- `Scenario: Batched prefill overflows the core count with exact parity` -> V1, V3, V4, V5, V6, V7
- `Scenario: The instance count stays within the core count` -> V1, V3
- `Scenario: A malformed instance assignment is rejected` -> V2, V4, D1
