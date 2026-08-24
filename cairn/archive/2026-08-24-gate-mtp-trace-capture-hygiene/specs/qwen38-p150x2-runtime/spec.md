## ADDED Requirements

### Requirement: MTP trace capture satisfies explicit hygiene gates

r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene]

The repository MUST validate MTP trace evidence with explicit pure capture-hygiene gates before the evidence is accepted. A control MUST hold at most one resident MTP trace at a time and MUST release a resident trace before capturing another. Capture MUST prime from real warm execution rounds, including one one-row seed round before a multi-row capture. Trace-bound buffers MUST be mutated in place at stable device addresses. The validation core MUST reject evidence from a capture that kept a second trace resident, captured from zeroed priming, or changed a trace-bound buffer address. The gates MUST remain pure and device-free and MUST NOT change eager production behavior.

#### Scenario: Capture hygiene passes

- GIVEN a control that released all resident traces, primed a one-row seed round, and captured with stable buffer addresses
- WHEN the pure hygiene gate validates the capture evidence
- THEN the gate MUST accept the trace evidence
- AND the evidence remains subject to the existing token, state, and parity gates

#### Scenario: Capture hygiene fails closed

- GIVEN a second trace resident during capture, zeroed priming, or a reallocated trace-bound buffer
- WHEN the pure hygiene gate validates the capture evidence
- THEN the gate MUST reject the evidence with a bounded reason
- AND the control MUST NOT accept a trace claim without an admitted provenance
