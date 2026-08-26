## ADDED Requirements

### Requirement: Same-width groups reuse at most one decode trace

r[tenstorrent.qwen38_p150x2.same_width_trace_reuse]

The service MAY reuse one resident decode trace across consecutive request groups of the same exact width. Reuse MUST require a complete prior reset and stable token, position, RoPE, page-table, KV, GDN, output, and trace bindings. The service MUST release the resident trace before prefill when the width changes, policy disables reuse, reset is incomplete, binding drifts, or more than one trace exists. The service MUST NOT keep two decode traces resident. Production reuse MUST remain disabled until matched physical evidence passes every profile-owned correctness, identity, stability, and latency bound. A passing reuse result MUST NOT admit slot recycling, mixed generation bounds, dynamic compaction, or live request insertion.

#### Scenario: A same-width group reuses one stable trace

- GIVEN one resident trace, the same requested width, a complete reset, and stable bindings
- WHEN the next group completes prefill and prepares decode
- THEN the service MUST reuse the existing trace without another capture
- AND the response evidence MUST report one resident trace and exact output parity.

#### Scenario: Reuse eligibility fails

- GIVEN a width change, disabled policy, incomplete reset, binding drift, or multiple resident traces
- WHEN the service plans the next prefill
- THEN it MUST release every resident trace before prefill
- AND it MUST capture at most one new trace after prefill.

#### Scenario: Physical reuse evidence fails

- GIVEN missing identity, token mismatch, reset failure, trace-count drift, unstable timing, or insufficient wall-time improvement
- WHEN the deterministic evaluator reads the matched roots
- THEN it MUST reject production reuse with bounded reasons
- AND the service MUST preserve the admitted recapture path.
