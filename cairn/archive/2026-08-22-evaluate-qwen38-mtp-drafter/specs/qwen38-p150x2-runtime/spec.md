## ADDED Requirements

### Requirement: MTP drafter evidence binds shifted inputs, acceptance, and rollback

r[tenstorrent.qwen38_p150x2.mtp_drafter_probe] The repository MAY provide an operator-enabled Qwen3.8 MTP drafter probe for the pinned P150x2 runtime. The probe MUST remain diagnostic-only. It MUST bind the exact checkpoint MTP tensors and their shapes. It MUST share the target embedding and LM head. It MUST separately normalize shifted next-token embeddings and post-final-norm target hidden states before the MTP FC projection. It MUST run one full-attention MTP decoder layer with its own position-ordered KV cache. It MUST generate exactly three greedy drafts. Physical evidence MUST compare those drafts with aligned target logits across at least five fixed prompts. It MUST record accepted-prefix length, draft time, target-verifier time, emitted-token economics, identities, stability, and cleanup. MTP MUST remain unselected when weight loading, parity, acceptance, rollback, target correctness, or economics is missing or invalid. Target GDN state MUST restore to the accepted prefix before any end-to-end MTP selection.

#### Scenario: Drafter evidence is complete

- **GIVEN** the exact pinned P150x2 package, runtime, checkpoint, tokenizer, MTP tensors, prompts, and physical device identities
- **WHEN** the operator runs the bounded three-draft probe
- **THEN** shifted-input alignment, local acceptance, timing, target correctness, rollback, stability, and non-admission are explicit in the receipt

#### Scenario: Drafter evidence fails closed

- **GIVEN** missing or drifted MTP weights, an uninitialized drafter cache, misaligned logits, unstable timing, insufficient acceptance, incorrect rollback, or slower emitted-token economics
- **WHEN** the deterministic evaluator checks the evidence
- **THEN** it rejects MTP selection and preserves eager production
