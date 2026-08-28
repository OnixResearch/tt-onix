## ADDED Requirements

### Requirement: MTP draft selection uses bounded causal evidence

r[tenstorrent.qwen38_p150x2.mtp_draft_selection] The repository MAY provide a diagnostic P150x2 control that compares full-logit MTP draft selection with shard-local top-one selection. The control MUST use the exact pinned package identities, physical devices `0` and `1`, batch one, exactly three drafts, and at least five fixed prompts. It MUST warm both routes and execute exact per-prompt order `full,bounded,bounded,full`. The bounded route MUST read only one local index and value per vocabulary shard and MUST use the same deterministic cross-shard tie policy as traced target decode. The evaluator MUST reject missing roots, route drift, invalid durations, duplicate-root instability, draft-token mismatch, target-token mismatch, acceptance mismatch, state mismatch, rollback failure, insufficient selector gain, or emitted-token aggregate speedup not strictly greater than `1.0`. A valid result MAY select bounded MTP draft selection for further diagnostic work. It MUST NOT admit MTP, traced decode, reduced precision, or production behavior.

#### Scenario: Bounded draft selection is selected

- **GIVEN** exact matched full-readback and bounded-selector roots for all fixed prompts
- **WHEN** duplicate roots are stable, all tokens and states match, bounded selection is materially faster, and aggregate economics exceed break-even
- **THEN** the evaluator SHALL select bounded draft selection for the next diagnostic MTP stage while production remains eager

#### Scenario: Selector evidence fails closed

- **GIVEN** missing roots, route drift, unstable samples, invalid timing, any output mismatch, rollback failure, insufficient gain, or aggregate slowdown
- **WHEN** the deterministic evaluator runs
- **THEN** it MUST reject selector optimization and MUST select no production change
