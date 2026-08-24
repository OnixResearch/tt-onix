## ADDED Requirements

### Requirement: Sampled MTP acceptance is distribution-exact and measured

r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling]

The repository MAY provide a pure, device-free MTP acceptance core for the pinned P150x2 runtime. For an argmax (point-mass) draft, the core MUST accept the draft with probability equal to its target probability `p(x̂)` when the exact rule is selected, so the emitted stream equals the target sampling distribution. It MUST report the realized per-position total-variation divergence `|a − p(x̂)|` for every rule. A lenient rule MUST reduce to the exact rule at lenience `1.0`. The greedy exact-match rule MUST remain available and MUST remain the production default. The sampled rules MUST remain diagnostic-only and MUST NOT change eager production behavior.

#### Scenario: Exact sampled acceptance preserves the target distribution

- GIVEN a point-mass draft and one per-row target candidate distribution
- WHEN the pure acceptance core resolves one round
- THEN each position reports its acceptance probability and realized divergence
- AND any rejected draft draws from the renormalised target residual

#### Scenario: Sampled acceptance fails closed on invalid input

- GIVEN a zero-length candidate list, a non-finite acceptance probability, a bounded lenience, or a rule identifier outside the closed set
- WHEN the pure acceptance core validates the round
- THEN the core MUST reject the round with a bounded reason
- AND it MUST NOT emit a token from an invalid distribution
