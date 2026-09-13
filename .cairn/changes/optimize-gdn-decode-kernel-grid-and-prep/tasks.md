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
- [ ] [serial] I3 Add head pipelining only where the measured occupancy stays below the floor. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (reader-prefetch variant implemented, exact parity at B=8/B=16, neutral-to-negative step time, rejected: evidence/i3-candidate-verdict.md. The loop is latency-bound, not arithmetic-bound, so the untried form is a two-stage hoist of the next instance's independent opening stages inside the compute kernel -- evidence/loop-cost-nature.md)
- [ ] [serial] I4 Fold the fp32 typecasts, the L2 norms, and the query scale into the reader or compute kernel. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (de-prioritised, not rejected: 0.470 ms per step measured against a 1.4 ms gap, in a loop the evidence says is not arithmetic-bound -- evidence/loop-cost-nature.md)
- [ ] [serial] I5 Preserve the exact reader and compute numerics when the preparation ops move; verify the fold does not change the recurrence order. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [x] [serial] I6 Bind the kernel variant, the occupancy plan, and the profile into the package execution identity. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (occupancy policy is an identity input and fail-closed at not-measured)

## Phase 3: Verification

- [x] [parallel] V1 Add positive device-free probes for valid state splits and head-group schedules. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]
- [x] [parallel] V2 Add negative probes for every malformed plan shape. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.plan]
- [x] [serial] V3 Run the package build and the device-free check. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] V4 Run the bounded Blackhole kernel hazard audits over the changed reader and compute kernels. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [ ] [serial] V5 Run a fresh Blackhole simulator smoke before hardware. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [x] [serial] V6 Run the matched physical screen and require exact decode and state parity before reading any timing. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]  (exact parity at B=8 and B=16 for the rejected reader-prefetch candidate)
- [ ] [serial] V7 Measure derived occupancy and the step-time ratio against the admitted kernel. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]  (measured for the rejected candidate: no ratio win to claim. Acceptance target at width 8 is a 78.0 ms step or below, from the 0.75 retention bound; `scripts/qwen38-step-probe.sh` measures it with 480 samples in about three minutes)
- [x] [serial] V8 Verify the disabled state keeps the admitted kernel as the served route. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.invalid]

## Phase 4: Completion

- [x] [serial] D1 Record the parity table, the occupancy derivation, the step-time result, and every non-claim. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [x] [serial] D2 Reject the candidate on any failed gate and keep the admitted kernel. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.invalid]
- [ ] [serial] D3 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]

## Verification Coverage

- `Scenario: The optimized kernel passes exact parity and both bounds` -> V1, V3, V4, V5, V6, V7
- `Scenario: Parity or occupancy fails and the admitted kernel stays` -> V6, V7, V8, D2
- `Scenario: A malformed occupancy or head-group plan is rejected` -> V1, V2, V3
