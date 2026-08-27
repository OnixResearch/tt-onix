# Tasks: Fuse the Qwen3.8 GDN decode state update

## Phase 1: Attribution and selection

- [x] [serial] A1 Reconcile component timing with model dimensions and weight-bandwidth floors. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]
- [x] [serial] A2 Compare full-kernel, convolution, layout, and bounded TTNN fusion mechanisms. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]
- [x] [serial] A3 Select the exact-equation TTNN fusion candidate for bounded testing. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]

## Phase 2: Implementation

- [x] [serial] I1 Add the pure closed-mode fusion policy. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]
- [x] [serial] I2 Fold fixed Q/K scales into optional RMSNorm weights. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]
- [x] [serial] I3 Add the optional ternary fp32 state update. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]
- [x] [serial] I4 Add the package-matched fp32 recurrence device program. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]
- [x] [serial] I5 Add trace-safe in-place state output and fail-closed model routing. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]
- [x] [serial] I6 Bind all execution changes to the Nix execution identity. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]

## Phase 3: Verification

- [x] [parallel] V1 Add positive checks for every closed candidate mode. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion.valid]
- [x] [parallel] V2 Add negative checks for missing, unknown, and invalid gate values. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion.invalid]
- [x] [serial] V3 Run package build and device-free checks. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]
- [x] [serial] V3a Run the bounded Blackhole kernel hazard audit. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]
- [ ] [serial] V4 Run matched physical component and full-step controls. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion.valid]
- [ ] [serial] V5 Verify exact tokens, text, state isolation, and clean reset. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion.valid]
- [ ] [serial] V6 Run fresh B=4 admission for any selected mode. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion.valid]

## Phase 4: Completion

- [ ] [serial] D1 Record the physical result and all non-claims. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]
- [ ] [serial] G1 Sync the accepted specification, archive the change, and run lifecycle gates. r[tenstorrent.qwen38_p150x2.gdn_decode_fusion]

## Verification Coverage

- `Scenario: A fusion mode passes matched physical gates` -> V1, V3, V4, V5, V6
- `Scenario: Fusion evidence is invalid or incomplete` -> V2, V3, V4, D1
