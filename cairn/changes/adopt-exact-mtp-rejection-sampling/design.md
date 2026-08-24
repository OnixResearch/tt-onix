## Context

The Qwen3.8-27B P150x2 MTP path drafts greedily (`argmax`) and accepts by
exact token match. The reviewed Adartras Qwen3.6-A3B bundle ships a sampled
speculative-decoding core with a proven distribution-exactness argument. Its
key result: for an argmax draft, the acceptance probability collapses to
`p(x̂)`, the residual renormalises, and the emitted stream equals the target
distribution. The bundle reports the exact per-position total-variation
divergence `|a − p(x̂)|` for every rule, so the distributional cost of a
speedup is measured rather than assumed.

## Decisions

### Decision: Adopt the reviewed acceptance-rule core as a pure component

**Choice:** Port the exact, lenient, relaxed, and greedy acceptance rules and
the `realized_tv` divergence computation into a new device-free core in the
Qwen3.8 service, matched to the existing drafter, selection, and rollback
types.

**Rationale:** The reviewed core is deliberately pure torch with no ttnn, so
the distribution-exactness claim is unit-testable on a laptop with
deterministic uniform draws and chi-squared gates. Reusing a reviewed and
measured design avoids re-deriving a subtle probability argument.

### Decision: Keep greedy exact-match as the production default

**Choice:** Exact-match acceptance remains the default and only production
rule. The sampled rules are added behind the existing diagnostic-only MTP
gating.

**Rationale:** MTP is not yet admitted to production. The accepted
`qwen38-p150x2-runtime` spec keeps every MTP stage diagnostic-only until it
has complete matched physical evidence. A new rule must earn admission with
the same evidence discipline. Nothing changes for the current greedy MTP
evidence.

### Decision: Candidate truncation is proven exact, not approximated

**Choice:** Build the truncated target distribution from the top-NCAND
candidate logits the verify step returns, applying the presence penalty
before truncation.

**Rationale:** Penalising then truncating equals truncating then penalising
when at most `NCAND − top_k` candidates are penalised. The kept top-k set is
then identical to penalising the whole vocabulary. The restriction is exact
for the supported `top_k` range, not close enough.

## Risks / Trade-offs

- The sampled rule changes emitted tokens versus exact-match acceptance. The
  change is reported per position and gated by the existing acceptance and
  economics evidence.
- The rule set must stay pure: host-only, deterministic in the uniform draws,
  and free of device state. A device dependency would make the core untestable
  on a laptop and is rejected by design.
- The lenient rule buys acceptance at a bounded divergence. Its bound must be
  reported, not hidden, in the receipt.
