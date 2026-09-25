## Phase 1: Attribution

- [x] [serial] A1 Measure per-op traced cost of the B=1 GDN, MLP, attention, and LM-head components on silicon and record the kernel-versus-glue split. r[tenstorrent.qwen38_p150x2.gdn_decode_step]  (evidence/attribution.md)
- [x] [serial] A2 Measure the collective alternatives on the same mesh and record that none is a lever at B=1. r[tenstorrent.qwen38_p150x2.gdn_decode_step]  (evidence/attribution.md)

## Phase 2: Implementation

- [x] [serial] I1 Add the `conv_prep` program: conv, SiLU, gate SiLU, a/b copy, and in-place conv-state shift. r[tenstorrent.qwen38_p150x2.gdn_decode_step]
- [x] [serial] I2 Add the `recurrence` program with exact fp32 carried state and the gated RMSNorm into a persistent bf16 buffer. r[tenstorrent.qwen38_p150x2.gdn_decode_step]
- [x] [serial] I3 Add mode `step`, the geometry bound, and the `forward_decode` branch that keeps the composite route for unsupported shapes. r[tenstorrent.qwen38_p150x2.gdn_decode_step.invalid]
- [x] [serial] I4 Keep greedy requests on the step route when the layers start in it. r[tenstorrent.qwen38_p150x2.gdn_decode_step.valid]
- [x] [serial] I5 Add the NixOS option, default on, with the MTP rejection. r[tenstorrent.qwen38_p150x2.gdn_decode_step.invalid]

## Phase 3: Verification

- [x] [parallel] V1 Compare both programs against a float64 reference on silicon for B=1, B=4, and B=3 of 8. r[tenstorrent.qwen38_p150x2.gdn_decode_step]  (evidence/kernel-numerics.md)
- [x] [parallel] V2 Add device-free positive and negative probes for the mode, geometry, and router. r[tenstorrent.qwen38_p150x2.gdn_decode_step.invalid]
- [x] [serial] V3 Measure step time, teacher-forced agreement, and passage NLL against the composite and fused-recurrence routes. r[tenstorrent.qwen38_p150x2.gdn_decode_step.valid]  (evidence/model-eval.md)
- [x] [serial] V4 Build the package with the device-free check and the NixOS service check. r[tenstorrent.qwen38_p150x2.gdn_decode_step]
- [x] [serial] V5 Run the packaged serialized service per mode and read its own per-token latency and serving route. r[tenstorrent.qwen38_p150x2.gdn_decode_step.valid]  (evidence/model-eval.md)
- [x] [serial] V6 Measure per-token host time and remove the redundant device sync after replay, with identical tokens. r[tenstorrent.qwen38_p150x2.gdn_decode_step.valid]  (evidence/host-and-width.md)
- [x] [serial] V7 Measure widths 1 to 16 on the fused step and the packaged service under concurrent load. r[tenstorrent.qwen38_p150x2.gdn_decode_step.valid]  (evidence/host-and-width.md)
- [x] [serial] V8 Measure prefill cost and gate the runtime's fused chunk prefill behind an operator opt-in with its quality result. r[tenstorrent.qwen38_p150x2.gdn_decode_step]  (evidence/prefill.md)
- [x] [serial] V9 Chunk untraced prompts above 1,024 tokens so that prompts of 1,025 to 2,048 tokens no longer fail with an L1 overflow. r[tenstorrent.qwen38_p150x2.gdn_decode_step]  (evidence/prefill.md)

## Verification Coverage

- `Scenario: The fused step passes the quality gate and serves greedy decode` -> V1, V3, V4, V5
- `Scenario: Unsupported geometry, mode, or serving combination` -> V2, V4
