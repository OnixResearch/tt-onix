# Tasks: Gated MTP speculative serving

## Phase 1: Policy core

- [x] [serial] P1 Pure serving round policy core (`mtp_serving_core.py`): round schedule, exact greedy commit rule, closed validation. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.policy]
- [x] [serial] P2 Device-free check probe (`probe_mtp_serving_core`) with positive and negative cases, bound to the spec delta. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.valid]
- [x] [serial] P3 Bind the module into the package execution identity and pass the qwen36 device-free check. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]

## Phase 2: Serving integration

- [ ] [serial] I1 Serving adapter: request-scoped trace capture in hygiene order (verify, draft, decode), round replay, snapshot rollback per round, plain-step fallback, disengage accounting. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]
- [ ] [serial] I2 Server wiring: `QWEN36_MTP_SERVING` gate (default off), module option, readiness payload field, metrics for round outcomes. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]
- [ ] [serial] I3 Physical exactness screen: greedy requests through the gated path versus the plain path, 64 tokens, exact token parity required; draft and verify trace capture costs excluded from round economics. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.valid]
- [ ] [serial] I4 Owner review: k operating point, disengage thresholds, enablement. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]

## Phase 3: Invalid-input verification

- [ ] [parallel] V1 Extend the probe negatives to cover every rejected round input with its own diagnostic: mismatched lengths, draft width outside 2..6, negative counters, underflowing remaining-token counts, and non-integer token identifiers. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.invalid]

## Verification Coverage

- `Requirement: The serialized greedy route runs speculative MTP rounds behind a default-off gate` -> P3, I1, I2, I3, I4
- `Requirement: The round schedule and commit rule are owned by a pure device-free core` -> P1, P2
- `Requirement: A valid device-free probe demonstrates scheduling, commit, and bound behavior` -> P2, I3
- `Requirement: The probe rejects malformed round inputs with specific diagnostics` -> V1
