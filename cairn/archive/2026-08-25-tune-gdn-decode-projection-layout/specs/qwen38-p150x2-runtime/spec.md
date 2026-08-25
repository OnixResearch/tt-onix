## ADDED Requirements

### Requirement: GDN decode projection layouts tune only behind matched evidence

r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout]

The repository MAY provide an operator-enabled layout-tuning candidate for the pinned P150x2 GDN decode projections. The candidate MUST stay default-off, MUST NOT change prefill, and MUST leave the automatic layout path as production behavior. Its shape eligibility MUST live in a pure device-free core. Any speedup or selection claim MUST use matched eager-versus-candidate evidence under the existing decode-performance-admission requirement; a per-operation result alone MUST NOT imply a model speedup.

#### Scenario: Gated candidate is selected for diagnostic work

- GIVEN a matched physical eager-versus-candidate pair with stable medians
- WHEN the attribution comparator evaluates the gated candidate against the AUTO baseline
- THEN it MAY name the layout as a selected optimization candidate
- AND production keeps the automatic layouts until full admission evidence exists

#### Scenario: Candidate evidence is missing or unstable

- GIVEN a missing baseline root, identity drift, unstable medians, or an allocation failure under the gate
- WHEN the evaluator processes the candidate
- THEN it MUST reject the candidate and keep the automatic layout path selected
