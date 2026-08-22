## ADDED Requirements

### Requirement: Native short-width target verification preserves hybrid state

r[tenstorrent.qwen38_p150x2.native_mtp_verifier] The repository MAY provide an operator-enabled native target verifier for the pinned P150x2 runtime. The verifier MUST remain diagnostic-only. It MUST process widths 1, 2, and 4 layer by layer. It MUST batch weight-bearing projections, MLPs, and target-logit work across positions. It MUST preserve token-ordered high-precision GDN recurrence and convolution updates. It MUST serialize paged-KV writes when candidate rows alias one physical block. It MUST execute the complete width-specific hybrid forward through one captured trace replay. It MUST return target logits for every position. Physical evidence MUST use at least five complete samples per width and compare exact target tokens, bounded full logits, final GDN recurrent and convolution state, written full-attention KV positions, identities, durations, and replay stability against sequential traced decode. Full-logit host readback MUST remain separately visible and MUST NOT stand in for device-resident acceptance timing. A missing, mismatched, unstable, or slower verifier MUST select no optimization. A valid target-only result MUST NOT select or admit MTP without drafter timing, local acceptance, rejection rollback, end-to-end correctness, and admission evidence.

#### Scenario: Native target verification is valid

- **GIVEN** the exact pinned P150x2 runtime, package, source, checkpoint, tokenizer, mesh, policy, prompt state, and physical device identities
- **WHEN** the operator runs five complete native-verifier samples for each required width
- **THEN** every target token, full-logit metric, GDN state metric, paged-KV metric, duration, identity, and stability check satisfies policy, while MTP remains unselected and unadmitted

#### Scenario: Native target verification fails closed

- **GIVEN** a verifier that repeats full-model steps, loses prompt state, races paged-KV writes, omits a position logit, changes a target token, exceeds a state bound, has unstable timing, or fails to amortize target work
- **WHEN** the deterministic evaluator checks its evidence
- **THEN** it rejects the verifier, selects no optimization, and preserves eager production
