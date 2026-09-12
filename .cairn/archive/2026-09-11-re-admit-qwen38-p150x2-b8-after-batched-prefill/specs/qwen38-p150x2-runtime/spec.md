## ADDED Requirements

### Requirement: B=8 admission is re-evaluated under batched prefill without weaker bounds

r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission]

The repository MUST re-evaluate the existing B=8 batch-admission profile against a package that carries the batched-prefill mechanism. The profile, its policy, its contracts, its corpus, its root count, its order, and every bound MUST remain unchanged. Admission MUST require all of: aggregate decode speedup at or above the bound, per-user decode retention at or above the bound, TTFT ratio at or below the bound, tail-latency ratio at or below the bound, exact token parity, exact text parity, distinct state slices for every active prefix, clean reset, and identity match. A pass on any subset MUST NOT admit B=8. The receipt MUST bind the new package identity, the runtime identity, the prefill mechanism identity, and the profile identity. The B=4 profile, binding, and receipt MUST remain untouched. B=32 MUST stay outside this requirement.

#### Scenario: Every bound passes and B=8 is admitted

r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission.valid]

- **GIVEN** the rebuilt package, the unchanged B=8 profile, and matched physical roots on devices 0 and 1
- **WHEN** the balanced `single,batch,batch,single` run completes
- **THEN** every correctness bound and every performance bound MUST pass
- **AND** the binding MAY set `physically_admitted` to true for exactly this execution identity
- **AND** the receipt MUST record the previous rejection and the new result together.

#### Scenario: Any bound fails and the rejection stands

r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission.rejection]

- **GIVEN** a run where aggregate speedup, per-user retention, TTFT ratio, tail ratio, parity, state isolation, reset, or identity fails
- **WHEN** the deterministic evaluator reads the matched roots
- **THEN** it MUST reject the candidate with the failing bound named
- **AND** the binding MUST keep `physically_admitted` false
- **AND** the B=4 admission MUST remain the only admitted batch width.

#### Scenario: A weaker or edited bound is rejected before device work

r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission.invalid]

- **GIVEN** an edited threshold, a shortened corpus, a reduced root or repetition count, a changed order, or a changed identity set
- **WHEN** the profile and receipt contracts are validated
- **THEN** validation MUST fail before any device work
- **AND** the receipt MUST NOT be admissible.
