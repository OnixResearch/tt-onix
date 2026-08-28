## ADDED Requirements

### Requirement: Traced TTFT control separates cache capacity and parked-trace effects

r[tenstorrent.qwen38_p150x2.traced_ttft_control] The repository MAY provide an operator-enabled same-process TTFT control for the pinned Qwen3.8 P150x2 runtime. The control MUST remain diagnostic-only. It MUST use one loaded model, one bounded prompt, stable request-sized and trace-capacity KV cache sets, and balanced root order. It MUST compare cache capacity without a parked trace and compare trace-capacity prefill before, during, and after a parked decode trace. Each root MUST contain at least five complete retained samples after warm-up. The deterministic evaluator MUST reject missing identities, wrong route order, token mismatch, invalid cache widths, malformed durations, unstable roots, unstable duplicate roots, or incomplete cleanup. The lifecycle MUST stop if eager-service restoration fails. A rejected result MUST select no optimization and MUST preserve eager production.

#### Scenario: Same-process TTFT control is valid

r[tenstorrent.qwen38_p150x2.traced_ttft_control.valid]
- **GIVEN** the exact pinned package, runtime, checkpoint, prompt, topology, cache sets, route order, and physical device identities
- **WHEN** every root completes repeated TTFT samples within the configured stability bounds
- **THEN** the receipt MUST report root medians, cache-capacity ratios, parked-trace ratios, combined ratios, and any selected bounded mechanism
- **AND** the result MUST remain diagnostic and ineligible for production admission.

#### Scenario: Same-process TTFT control is invalid

r[tenstorrent.qwen38_p150x2.traced_ttft_control.invalid]
- **GIVEN** identity drift, wrong route order, token mismatch, invalid widths, malformed timing, unstable roots, incomplete cleanup, or failed service restoration
- **WHEN** the deterministic evaluator and lifecycle gates process the evidence
- **THEN** they MUST reject the control and name each bounded reason
- **AND** it MUST select no optimization or production change.
