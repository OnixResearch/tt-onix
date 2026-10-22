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
- [x] [serial] I3 Add head pipelining only where the measured occupancy stays below the floor. (entered as destination-register batching, the latency form the cost analysis implied) r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (reader-prefetch variant implemented, exact parity at B=8/B=16, neutral-to-negative step time, rejected: evidence/i3-candidate-verdict.md. The loop is latency-bound, not arithmetic-bound, so the untried form is a two-stage hoist of the next instance's independent opening stages inside the compute kernel -- evidence/loop-cost-nature.md)
- [ ] [serial] I4 Fold the fp32 typecasts, the L2 norms, and the query scale into the reader or compute kernel. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (de-prioritised, not rejected: 0.470 ms per step measured against a 1.4 ms gap, in a loop the evidence says is not arithmetic-bound -- evidence/loop-cost-nature.md)
- [ ] [serial] I5 Preserve the exact reader and compute numerics when the preparation ops move; verify the fold does not change the recurrence order. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [x] [serial] I6 Bind the kernel variant, the occupancy plan, and the profile into the package execution identity. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (occupancy policy is an identity input and fail-closed at not-measured)

> **Served-route correction (2026-10-21).** The service now decodes GDN layers through the
> `step` kernel in `pkgs/qwen36/gdn-decode-step.patch`, not the TT-Metal fused op this
> change's proposal names. The instance-loop question was re-measured on that kernel.
> Destination-register capacity is 4 tiles per sync round without full sync and 8 with it;
> full sync costs more than the batching saves, and double-buffered prefetch is numerically
> wrong at two instances per core. The one kept lever is the reader's staged scalar reads.
> See `evidence/step-kernel-scalar-staging.md`.

- [x] [serial] I7 Stage the four per-instance scalar reads behind the reader's existing NOC barrier in the served step kernel, leaving values and pages unchanged. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]  (evidence/step-kernel-scalar-staging.md)
- [x] [serial] I8 Reject destination-register batching under full sync and double-buffered I/O prefetch with their measured reasons. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.invalid]  (evidence/step-kernel-scalar-staging.md)

## Phase 3: Verification

- [x] [parallel] V1 Add positive device-free probes for valid state splits and head-group schedules. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]
- [x] [parallel] V2 Add negative probes for every malformed plan shape. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.plan]
- [x] [serial] V3 Run the package build and the device-free check. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [x] [serial] V4 Run the bounded Blackhole kernel hazard audits over the changed reader and compute kernels. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (source audit only, no silicon: every applicable class passes and the scratch page's reader-private ownership is recorded — evidence/v4-kernel-hazard-audit.md)
- [x] [serial] V5 Run a fresh Blackhole simulator smoke before hardware. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (simulated 1x1 Blackhole: three conv_prep + recurrence steps pass the physical harness's float64-reference bounds; docs/evidence/tt-metal-0.77/step-kernel-simulator-2026-10-22.json)
- [x] [serial] V6 Run the matched physical screen and require exact decode and state parity before reading any timing. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]  (exact parity at B=8 for the reader-prefetch and batched-destination candidates; exact in the contaminated round as well)
- [x] [serial] V7 Measure derived occupancy and the step-time ratio against the admitted kernel.  (batched-destination candidate: width-8 step 2.6% and 6.4% below the reference in two paired rounds; two of four frozen-protocol rounds admitted width 8, and the protocol spread exceeds the margin it decides -- evidence/protocol-measurement-power.md) r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]  (measured for the rejected candidate: no ratio win to claim. Acceptance target at width 8 is a 78.0 ms step or below, from the 0.75 retention bound; `scripts/qwen38-step-probe.sh` measures it with 480 samples in about three minutes)
- [x] [serial] V8 Verify the disabled state keeps the admitted kernel as the served route. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.invalid]
- [x] [serial] V11 Prove the staged scalar reads numerically transparent against the float64 reference at widths 1, 4, 3-of-8, 8, and 16; time the recurrence in two paired rounds; and compare 16 greedy tokens end to end. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]  (evidence/step-kernel-scalar-staging.md)

## Phase 4: Completion

- [x] [serial] D1 Record the parity table, the occupancy derivation, the step-time result, and every non-claim. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]
- [x] [serial] V9 Widen the admission protocol so it resolves a two percent difference: interleaved root order, per-user first-token statistics instead of the group maximum, and enough roots for a stable median. Blocks any further width-8 admission verdict; evidence/protocol-measurement-power.md. The interleaved protocol's spread across four pairs is 0.07 % on the aggregate decode speedup and on the per-user retention, against the four-root protocol's 0.070 band on retention; the per-user first-token ratio and the tail ratio remain noisy at 6 % and 18 %. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]  (evidence/protocol-v3-validation.md)
- [x] [serial] D2 Reject the candidate on any failed gate and keep the admitted kernel. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.invalid]
- [x] [serial] D3 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]  (`repository-quality-artifacts`, `repository-quality-cairn`, and `qwen38` all pass on the cleaned change; the artifact manifest carries the new evidence file)

## Verification Coverage

- `Scenario: The optimized kernel passes exact parity and both bounds` -> V1, V3, V4, V5, V6, V7
- `Scenario: Parity or occupancy fails and the admitted kernel stays` -> V6, V7, V8, D2
- `Scenario: A malformed occupancy or head-group plan is rejected` -> V1, V2, V3
