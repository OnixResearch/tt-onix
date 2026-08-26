## ADDED Requirements

### Requirement: B=8 batched throughput admission uses its own matched physical roots

r[tenstorrent.qwen38_p150x2.batch_b8_admission]

The repository MUST define B=8 batch admission in a closed typed profile separate from the admitted B=4 profile. The profile MUST compare the same traced production recapture path at widths B=1 and B=8. Physical execution MUST use devices 0 and 1 in exact `single,batch,batch,single` root order with an eight-prompt corpus. Each root MUST bind the same execution identity. The shared admission cores MUST accept only candidate widths 4 and 8 and MUST reject every other width before device work. The B=4 profile, binding, and receipt MUST remain unchanged. Admission MUST require exact per-request token parity, a complete active state prefix across eight slices, distinct state slices, clean reset, aggregate decode speedup at least `6.0`, per-user decode-rate retention at least `0.75`, TTFT ratio at most `9.0`, and tail-latency ratio at most `1.5`. A passing B=8 receipt MUST NOT admit B=16, slot recycling, mixed generation bounds, or live request insertion.

#### Scenario: B=8 passes every batch gate

- GIVEN complete current-identity B=1 and B=8 physical roots with matching requests
- WHEN the deterministic evaluator processes all four roots
- THEN every predeclared bound MUST pass
- AND the B=8 binding MAY carry the new receipt hash.

#### Scenario: B=8 evidence fails a gate

- GIVEN any parity, isolation, reset, identity, throughput, TTFT, or tail failure
- WHEN the deterministic evaluator processes the roots
- THEN it MUST withhold admission with bounded reasons
- AND the B=4 admission MUST remain the only admitted batch width.

#### Scenario: A width outside the admitted set is requested

- GIVEN a profile or request declaring a candidate width other than 4 or 8
- WHEN the pure admission core validates it
- THEN validation MUST fail before any device or filesystem effect.
