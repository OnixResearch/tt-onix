# Tasks: Adopt and screen the fused GDN decode and verify op

## Phase 1: Source and selection

- [ ] [serial] A1 Record the upstream source authority: PR 53482, PR 53587, issue 50475, branch `atupe/qwen36-mtp-v2`, the pinned commits, and the review state. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]
- [ ] [serial] A2 Inspect the pinned runtime for `ttnn.transformer.decode_gated_delta_rule` and `ttnn.transformer.fused_recurrent_gated_delta_rule`; record which op the pin can accept and the exact blocker for the other. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]
- [ ] [serial] A3 Add the pure, device-free gate and shape-selection core with a closed mode set and a fail-closed default. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]
- [ ] [serial] A4 Add a Nickel profile that owns the gate name, the mode set, the parity requirement, and the step-time bound. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]

## Phase 2: Implementation

- [ ] [serial] I1 Vendor the fused T=1 decode op into the package overlay against the pinned runtime. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]
- [ ] [serial] I2 Vendor the T=K+1 verify path and its per-prefix recurrent and convolution state handling. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]
- [ ] [serial] I3 Route the GDN decode step through the gate with a byte-identical composite fallback. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]
- [ ] [serial] I4 Keep the fused output in TILE layout so the traced graph holds on device. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]
- [ ] [serial] I5 Bind the op, the gate, the selected mode, and the profile into the package execution identity. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]

## Phase 3: Verification

- [ ] [parallel] V1 Add positive device-free probes for every closed mode and for the composite fallback. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op.valid]
- [ ] [parallel] V2 Add negative probes for absent, empty, non-string, and unknown gate values and for unsupported shapes. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op.invalid]
- [ ] [serial] V3 Run the package build and the device-free check. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]
- [ ] [serial] V4 Run the bounded Blackhole kernel hazard audits over the vendored op. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]
- [ ] [serial] V5 Run a fresh Blackhole simulator smoke before hardware. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]
- [ ] [serial] V6 Run the matched physical screen on devices 0 and 1: exact greedy token parity, recurrent-state parity, and the step-time ratio. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op.valid]
- [ ] [serial] V7 Verify the disabled state produces a byte-identical graph to the current admitted graph. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op.fallback]
- [ ] [serial] V8 Re-run the MTP drafter and rollback probes under the fused verify kernel; report economics and outcome parity separately. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op.economics]

## Phase 4: Completion

- [ ] [serial] D1 Record the screen result, both kernel measurements, and every non-claim; reject the candidate on any failed gate. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]
- [ ] [serial] D2 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]

## Verification Coverage

- `Scenario: The fused path passes the physical screen` -> V1, V3, V4, V5, V6, V7
- `Scenario: Unsupported shape or missing op selects the composite path` -> V1, V2, V7
- `Scenario: Screen evidence is invalid or incomplete` -> V2, V5, V6, D1
- `Scenario: MTP economics are re-evaluated under the fused verify kernel` -> V8, D1
