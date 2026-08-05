# Tasks: RWKV-7 LoRA bring-up on one P150A

## Phase 1: Focused compiler portfolio

- [x] [serial] B1 Validate current platform and CPU reference receipts before accelerator work. r[tenstorrent.rwkv7_p150a_finetuning.authority]
- [x] [serial] B2 Add named Nickel profiles for static shapes, dtypes, compile limits, memory limits, tolerances, run criteria, health limits, and timeouts. r[tenstorrent.rwkv7_p150a_finetuning.configuration]
- [x] [depends:tenstorrent.rwkv7_p150a_finetuning.configuration] B3 Compile and run the direct TT-XLA WKV7 forward probe with placement evidence. r[tenstorrent.rwkv7_p150a_finetuning.compiler]
- [x] [depends:tenstorrent.rwkv7_p150a_finetuning.compiler] B4 Compile and run the direct TT-XLA WKV7 backward probe with CPU gradient comparison. r[tenstorrent.rwkv7_p150a_finetuning.compiler] r[tenstorrent.rwkv7_p150a_finetuning.parity]
- [x] [parallel] B5 Probe LoRA projections, masked loss, gradient norm, clipping, and adapter optimizer update. r[tenstorrent.rwkv7_p150a_finetuning.compiler] r[tenstorrent.rwkv7_p150a_finetuning.optimizer]
- [ ] [depends:tenstorrent.rwkv7_p150a_finetuning.compiler] B6 If direct WKV7 lowering fails, emit the exact blocker receipt and stop without host fallback or custom-op implementation. r[tenstorrent.rwkv7_p150a_finetuning.blocker]

## Phase 2: Full single-device training graph

- [ ] [depends:tenstorrent.rwkv7_p150a_finetuning.parity] B7 Build the full pinned RWKV-7 LoRA graph from the shared CPU authority and keep base tensors frozen. r[tenstorrent.rwkv7_p150a_finetuning.graph]
- [ ] [depends:tenstorrent.rwkv7_p150a_finetuning.graph] B8 Add the thin shell that selects one logical `1x1` P150A submesh while passing all host devices. r[tenstorrent.rwkv7_p150a_finetuning.device]
- [ ] [depends:tenstorrent.rwkv7_p150a_finetuning.graph] B9 Capture and validate full placement evidence with no model, loss, backward, clipping, or optimizer fallback. r[tenstorrent.rwkv7_p150a_finetuning.placement]
- [ ] [depends:tenstorrent.rwkv7_p150a_finetuning.placement] B10 Run one full step and compare logits, loss, selected states, gradients, clipping, and adapter deltas with CPU. r[tenstorrent.rwkv7_p150a_finetuning.parity] r[tenstorrent.rwkv7_p150a_finetuning.optimizer]

## Phase 3: Bounded fine-tuning and artifacts

- [ ] [depends:tenstorrent.rwkv7_p150a_finetuning.optimizer] B11 Run the named multi-step acceptance profile and validate finite values, adapter-only mutation, and the configured loss criterion. r[tenstorrent.rwkv7_p150a_finetuning.run]
- [ ] [depends:tenstorrent.rwkv7_p150a_finetuning.run] B12 Save, load, and resume the shared adapter artifact with TT-XLA graph, placement, precision, and device evidence. r[tenstorrent.rwkv7_p150a_finetuning.checkpoint]
- [ ] [parallel] V1 Add positive device-free fixtures for compile receipts, placement, numerical comparisons, optimizer results, artifacts, health, and restoration. r[tenstorrent.rwkv7_p150a_finetuning.verification]
- [ ] [parallel] V2 Add negative fixtures for unsupported operations, fallback, stale graphs, bad shapes, nonfinite values, missing gradients, base mutation, tolerance failures, close failures, and stale receipts. r[tenstorrent.rwkv7_p150a_finetuning.verification]
- [ ] [depends:tenstorrent.rwkv7_p150a_finetuning.checkpoint] V3 Produce the bounded single-P150A fine-tuning receipt and adapter artifact. r[tenstorrent.rwkv7_p150a_finetuning.evidence]
- [ ] [serial] B13 Document prerequisites, exact commands, compile diagnosis, recovery, artifact use, and non-claims. r[tenstorrent.rwkv7_p150a_finetuning.documentation]
- [ ] [depends:tenstorrent.rwkv7_p150a_finetuning.verification] V4 Run formatting, focused tests, package checks, platform and CPU regressions, Cairn validation, and all lifecycle gates. r[tenstorrent.rwkv7_p150a_finetuning.verification]

## Verification Coverage

- `Scenario: Current prerequisite receipts are admitted` -> B1, V1
- `Scenario: Stale prerequisite evidence blocks compilation` -> B1, V2
- `Scenario: Direct WKV7 graph compiles for the named profile` -> B3, B4, V1
- `Scenario: Unsupported WKV7 lowering produces a blocker` -> B6, V2
- `Scenario: Full trainable graph stays on TT` -> B7, B9, V1
- `Scenario: Forbidden host fallback is rejected` -> B9, V2
- `Scenario: One P150A step matches CPU authority` -> B10, V3
- `Scenario: Wrong or nonfinite device values fail parity` -> B10, V2
- `Scenario: Bounded fine-tuning changes only adapters` -> B11, V3
- `Scenario: P150A checkpoint resumes safely` -> B12, V1
- `Scenario: Device cleanup failure blocks success` -> B8, V2
- `Scenario: Evidence keeps the single-device boundary` -> V3, B13
