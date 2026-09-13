# Tasks: Optimize GDN decode occupancy and host preparation

## Phase 1: Attribution

- [x] [serial] A1 Re-measure the admitted GDN decode layer cost and the per-layer instance count, and record the derived occupancy at TP=2. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (evidence/a1-a4-attribution.md)
- [x] [serial] A2 Reconcile the local per-layer cost with the upstream composite and fused figures and record any mismatch. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [x] [serial] A3 Enumerate the host preparation ops in the admitted decode path and record each launch cost. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [x] [serial] A4 Confirm from the local profile that recurrent-state bandwidth is not the binding constraint. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]

## Phase 2: Implementation

> **Attribution correction before I2/I3.** At TP=2 the admitted decode is idle at
> B=1 (24 instances, 21.8 percent) and looping at B>=8 (2 and 4 instances per
> core). The state split (I2) addresses only the idle case: once instances exceed
> cores, splitting multiplies the instance count without reducing per-core work,
> and it multiplies the per-instance fixed cost (the `K x K` transpose plus
> reader, CB, and launch overhead). The per-user retention floor that blocks B=8
> and B=16 needs the instance loop addressed, which is I3 (pipelining) or a lower
> per-instance cost. See `evidence/a1-a4-attribution.md`.

- [x] [serial] I1 Add the pure occupancy core: per-core state split, head-group schedule, and closed validation. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.plan]
- [ ] [serial] I2 Split the 128x128 recurrent state across cores per value head. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (also the discriminating experiment for the loop's cost nature: see evidence/loop-cost-nature.md)
- [ ] [serial] I3 Add head pipelining only where the measured occupancy stays below the floor. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency] (Reopened: within-instance Dst batching is not head pipelining. The original Dst candidate violated capacity; the bounded replacement has host-only evidence. See evidence/i3-dst-batch-candidate.md.)
- [ ] [serial] I4 Fold the fp32 typecasts, the L2 norms, and the query scale into the reader or compute kernel. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (de-prioritised, not rejected: 0.470 ms per step measured against a 1.4 ms gap, in a loop the evidence says is not arithmetic-bound -- evidence/loop-cost-nature.md)
- [ ] [serial] I5 Preserve the exact reader and compute numerics when the preparation ops move; verify the fold does not change the recurrence order. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [x] [serial] I6 Bind the kernel variant, the occupancy plan, and the profile into the package execution identity. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (occupancy policy is an identity input and fail-closed at not-measured)

## Phase 3: Verification

- [x] [parallel] V1 Add positive device-free probes for valid state splits and head-group schedules. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]
- [x] [parallel] V2 Add negative probes for every malformed plan shape. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.plan]
- [x] [serial] V3 Run the package build and the device-free check. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
  The bounded replacement and installed host checks pass. See evidence/reliability-checks.md.
- [ ] [serial] V4 Run the bounded Blackhole kernel hazard audits over the changed reader and compute kernels. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
  The audit awaits the source-coverage choice. See evidence/hazard-source-preflight.md.
- [ ] [serial] V5 Run a fresh Blackhole simulator smoke before hardware. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
  The fresh run compiles the kernel, then rejects unmodeled SrcB denormal behavior. No numerical pass. See evidence/reliability-checks.md.
- [ ] [serial] V6 Run the matched physical screen and require exact decode and state parity before reading any timing. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid] (Reopened: old token receipts did not compare recurrent tensor contents. The bounded Dst replacement has no physical result.)
- [ ] [serial] V7 Measure derived occupancy and the step-time ratio against the admitted kernel. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid] (Reopened: old paired timings belong to the rejected, unbounded candidate. No measured improvement transfers to the replacement.)
- [x] [serial] V8 Verify the disabled state keeps the admitted kernel as the served route. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.invalid]
- [ ] [serial] V9 Verify measurement integrity under the unchanged ABBA order, profile bounds, and TTFT statistic. Reject unsupported measurement widths before device effects; retain all attempts and root-level diagnostics. Harden client width and identity checks, then require repeatable passing receipts. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.measurement_preflight] (The service preflight and its 60 device-free cases pass. The remaining campaign checks are open; see evidence/protocol-measurement-power.md.)

## Phase 4: Completion

- [x] [serial] D1 Record the parity table, the occupancy derivation, the step-time result, and every non-claim. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [x] [serial] D2 Reject the candidate on any failed gate and keep the admitted kernel. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.invalid]
- [ ] [serial] D3 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
  Cairn and Qwen gates pass. Existing artifact, Nix format, and training-platform gates fail. See evidence/reliability-checks.md.

## Verification Coverage

- `Scenario: The optimized kernel passes exact parity and both bounds` -> V1, V3, V4, V5, V6, V7
- `Scenario: Parity or occupancy fails and the admitted kernel stays` -> V6, V7, V8, D2
- `Scenario: A malformed occupancy or head-group plan is rejected` -> V1, V2, V3
- `Scenario: An unsupported measurement width has no device effects` -> V3, V9
- `Scenario: Valid and production widths retain their execution behavior` -> V3, V9
