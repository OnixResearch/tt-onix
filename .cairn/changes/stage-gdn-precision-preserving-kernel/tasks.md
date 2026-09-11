# Tasks: Stage the GDN recurrence to preserve composite precision boundaries

## Phase 1: Falsification record and recovery

- [x] [serial] A1 Record the upstream source authority and the review state for PR 53482, PR 53587, and issue 50475. r[tenstorrent.qwen38_p150x2.gdn_precision_staging]
- [x] [serial] A2 Inspect the pinned 0.77 runtime for both named fused ops and record the exact binding counts. r[tenstorrent.qwen38_p150x2.gdn_precision_staging]
- [x] [serial] A3 Record that `ttnn.transformer.gated_delta_decode` is already vendored, built, wired, and admitted for sampled mode and B=4; write the falsification to `evidence/upstream-authority.md`. r[tenstorrent.qwen38_p150x2.gdn_precision_staging]
- [ ] [serial] A4 Re-read the archived parity root cause and restate every closed approach family, its state, and the named repair. r[tenstorrent.qwen38_p150x2.gdn_precision_staging]
- [ ] [serial] A5 Enumerate the composite's precision boundaries from package-matched source and the generated JIT descriptors. r[tenstorrent.qwen38_p150x2.gdn_precision_staging]

## Phase 2: Implementation

- [ ] [serial] I1 Add the pure staging core: stage order, boundary list, closed validation. r[tenstorrent.qwen38_p150x2.gdn_precision_staging.invalid]
- [ ] [serial] I2 Stage the recurrence so each composite precision boundary is reproduced. r[tenstorrent.qwen38_p150x2.gdn_precision_staging]
- [ ] [serial] I3 Keep the existing fused route selected for every malformed plan and every disabled gate value. r[tenstorrent.qwen38_p150x2.gdn_precision_staging.invalid]
- [ ] [serial] I4 Bind the staged kernel, the staging core, and the gate into the package execution identity. r[tenstorrent.qwen38_p150x2.gdn_precision_staging]

## Phase 3: Verification

- [ ] [parallel] V1 Add positive device-free probes for valid stage orders and boundary lists. r[tenstorrent.qwen38_p150x2.gdn_precision_staging.valid]
- [ ] [parallel] V2 Add negative probes for every malformed plan shape. r[tenstorrent.qwen38_p150x2.gdn_precision_staging.invalid]
- [ ] [serial] V3 Run the package build and the device-free check. r[tenstorrent.qwen38_p150x2.gdn_precision_staging]
- [ ] [serial] V4 Run the bounded Blackhole kernel hazard audits over the staged reader and compute kernels. r[tenstorrent.qwen38_p150x2.gdn_precision_staging]
- [ ] [serial] V5 Run a fresh Blackhole simulator smoke before hardware. r[tenstorrent.qwen38_p150x2.gdn_precision_staging]
- [ ] [serial] V6 Run the matched physical screen on devices 0 and 1 and require exact greedy token and text parity before reading any timing. r[tenstorrent.qwen38_p150x2.gdn_precision_staging.valid]
- [ ] [serial] V7 Measure the GDN ratio and the full-step ratio against the composite reference. r[tenstorrent.qwen38_p150x2.gdn_precision_staging.valid]

## Phase 4: Completion

- [ ] [serial] D1 On a failed parity result, record an exhausted verdict with the discriminating evidence and keep the fused route at sampled-only. r[tenstorrent.qwen38_p150x2.gdn_precision_staging.exhausted]
- [ ] [serial] D2 Record the stage order, the boundary list, the parity table, and every non-claim. r[tenstorrent.qwen38_p150x2.gdn_precision_staging]
- [ ] [serial] D3 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.gdn_precision_staging]

## Verification Coverage

- `Scenario: The staged recurrence passes exact greedy parity and both speed bounds` -> V1, V3, V4, V5, V6, V7
- `Scenario: The staged design cannot reproduce parity within budget` -> V6, V7, D1
- `Scenario: A malformed staging plan is rejected` -> V1, V2, I3
