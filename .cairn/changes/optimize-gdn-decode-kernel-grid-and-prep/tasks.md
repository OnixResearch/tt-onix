# Tasks: Optimize GDN decode occupancy and host preparation

## Phase 1: Attribution

- [ ] [serial] A1 Re-measure the admitted GDN decode layer cost and the per-layer instance count, and record the derived occupancy at TP=2. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] A2 Reconcile the local per-layer cost with the upstream composite and fused figures and record any mismatch. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] A3 Enumerate the host preparation ops in the admitted decode path and record each launch cost. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] A4 Confirm from the local profile that recurrent-state bandwidth is not the binding constraint. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]

## Phase 2: Implementation

- [ ] [serial] I1 Add the pure occupancy core: per-core state split, head-group schedule, and closed validation. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.plan]
- [ ] [serial] I2 Split the 128x128 recurrent state across cores per value head. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] I3 Add head pipelining only where the measured occupancy stays below the floor. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] I4 Fold the fp32 typecasts, the L2 norms, and the query scale into the reader or compute kernel. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] I5 Preserve the exact reader and compute numerics when the preparation ops move; verify the fold does not change the recurrence order. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] I6 Bind the kernel variant, the occupancy plan, and the profile into the package execution identity. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]

## Phase 3: Verification

- [ ] [parallel] V1 Add positive device-free probes for valid state splits and head-group schedules. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]
- [ ] [parallel] V2 Add negative probes for every malformed plan shape. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.plan]
- [ ] [serial] V3 Run the package build and the device-free check. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] V4 Run the bounded Blackhole kernel hazard audits over the changed reader and compute kernels. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] V5 Run a fresh Blackhole simulator smoke before hardware. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] V6 Run the matched physical screen and require exact decode and state parity before reading any timing. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]
- [ ] [serial] V7 Measure derived occupancy and the step-time ratio against the admitted kernel. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]
- [ ] [serial] V8 Verify the disabled state keeps the admitted kernel as the served route. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.invalid]

## Phase 4: Completion

- [ ] [serial] D1 Record the parity table, the occupancy derivation, the step-time result, and every non-claim. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] D2 Reject the candidate on any failed gate and keep the admitted kernel. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.invalid]
- [ ] [serial] D3 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]

## Verification Coverage

- `Scenario: The optimized kernel passes exact parity and both bounds` -> V1, V3, V4, V5, V6, V7
- `Scenario: Parity or occupancy fails and the admitted kernel stays` -> V6, V7, V8, D2
- `Scenario: A malformed occupancy or head-group plan is rejected` -> V1, V2, V3
