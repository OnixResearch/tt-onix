## ADDED Requirements

### Requirement: MTP draft tracing uses stable causal evidence

r[tenstorrent.qwen38_p150x2.mtp_draft_trace] The repository MAY provide a diagnostic P150x2 control that compares eager and persistent-trace execution of the fixed-shape MTP draft decode step. The control MUST use exact pinned identities, physical devices `0` and `1`, batch one, exactly three drafts, the bounded shard-local selector, and at least five fixed prompts. It MUST warm both routes and retain at least five complete samples per route and prompt in balanced `eager,traced,traced,eager` blocks. Trace capture time MUST be recorded and excluded from replay economics. The traced route MUST bind stable token, position, RoPE, feedback, MTP KV-cache, selector-output, and trace-output addresses. It MUST restore capture side effects before measurement or prove that repeated same-input writes at the captured position are exact. The evaluator MUST reject missing samples, unknown fields, route drift, invalid durations, address drift, duplicate-root spread above `1.25`, token mismatch, target mismatch, acceptance mismatch, MTP KV-prefix mismatch, target rollback failure, traced/eager draft ratio above the configured bound, or emitted-token aggregate speedup not strictly greater than `1.0`. A valid result MAY select traced MTP drafting for later adaptive-width investigation. It MUST NOT admit MTP, traced target decode, reduced precision, or production behavior.

#### Scenario: Traced drafting is selected

- **GIVEN** matched eager and traced roots for all fixed prompts
- **WHEN** every binding and output matches, retained roots are stable, traced drafting is materially faster, and aggregate economics exceed break-even
- **THEN** the evaluator SHALL select traced MTP drafting for the next diagnostic stage while production remains eager

#### Scenario: Trace evidence fails closed

- **GIVEN** missing evidence, route drift, address drift, unstable timing, invalid timing, any output or state mismatch, insufficient draft gain, or aggregate slowdown
- **WHEN** the deterministic evaluator runs
- **THEN** it MUST reject traced drafting and MUST select no production change
