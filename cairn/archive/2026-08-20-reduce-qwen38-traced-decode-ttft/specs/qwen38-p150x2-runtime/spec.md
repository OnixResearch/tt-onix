# Qwen3.8-27B P150x2 Traced-Decode TTFT Specification Delta

## ADDED Requirements

### Requirement: Traced decode uses request-bounded prefill capacity
r[tenstorrent.qwen38_p150x2.traced_decode_ttft] The traced-decode candidate MUST size each prefill page table from the bounded prompt and generation token counts. The prefill table MUST be a monotonic prefix of the stable decode table. The trace-bound decode table MUST retain its maximum profile capacity, shape, contents, and device address across requests. Invalid or capacity-exceeding requests MUST fail before prefill dispatch.

#### Scenario: Request uses a bounded prefill table
r[tenstorrent.qwen38_p150x2.traced_decode_ttft.valid]
- GIVEN positive prompt and generation counts within the trace capacity
- WHEN the pure block-count rule and adapter build the prefill table
- THEN its width MUST equal the rounded-up required block count
- AND its entries MUST equal the same prefix of the stable decode table.

#### Scenario: Request exceeds trace capacity
r[tenstorrent.qwen38_p150x2.traced_decode_ttft.capacity]
- GIVEN a prompt and generation bound that need more blocks than the trace capacity
- WHEN request block-count validation runs
- THEN it MUST reject the request before prefill dispatch
- AND it MUST NOT resize or replace the trace-bound decode table.

#### Scenario: Prefill width changes trace binding
r[tenstorrent.qwen38_p150x2.traced_decode_ttft.binding]
- GIVEN a request-bounded prefill table and an existing decode trace
- WHEN trace binding validation runs before replay
- THEN the stable decode page-table identity MUST remain unchanged
- AND any identity drift MUST select the fail-closed eager path.

### Requirement: TTFT admission uses phase-attributed matched evidence
r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence] Each measured eager and candidate response MUST record finite non-negative reset, prefill-device, first-token-readback, trace-preparation, and total TTFT durations. It MUST record prefill and decode page-table widths. The total TTFT boundaries MUST remain unchanged. Admission MUST use order-balanced matched physical evidence and MUST reject missing, non-finite, inconsistent, or identity-mismatched observations.

#### Scenario: Phase evidence is complete
r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence.valid]
- GIVEN matched physical responses with complete phase timings and table widths
- WHEN the deterministic receipt collector evaluates them
- THEN it MUST retain phase medians and total TTFT ratios
- AND admission MAY pass only when every existing correctness and performance gate passes.

#### Scenario: Phase evidence is incomplete
r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence.invalid]
- GIVEN a missing phase, non-finite duration, negative duration, invalid width, or mismatched identity
- WHEN receipt validation runs
- THEN it MUST reject the candidate evidence
- AND production MUST remain eager.

#### Scenario: Physical candidate passes
r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence.physical]
- GIVEN matched `eager, candidate, candidate, eager` physical roots on device IDs `0` and `1`
- WHEN exact parity, reset, decode, TTFT, tail, phase, and identity gates pass
- THEN traced decode MAY become physically admitted
- AND the admitted profile MUST bind the new receipt BLAKE3.
