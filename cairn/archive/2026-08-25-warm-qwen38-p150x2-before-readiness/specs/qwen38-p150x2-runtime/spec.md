## ADDED Requirements

### Requirement: P150x2 readiness follows bounded startup warm-up

r[tenstorrent.qwen38_p150x2.startup_readiness]

The Qwen3.8-27B P150x2 service MUST NOT report HTTP or systemd readiness before bounded startup warm-up succeeds. Warm-up MUST execute production prefill, exact-width trace capture, one trace replay, and clean reset for widths B=1 and B=4. Warm-up policy and lifecycle transitions MUST remain device-free. TTNN and systemd effects MUST remain in adapters or the service shell. Any warm-up, reset, or readiness-notification failure MUST abort startup. Health output MUST report the observed warm-up widths and elapsed time. Startup evidence MUST NOT imply physical admission or complete prompt-shape coverage.

#### Scenario: Warm-up completes before readiness

- GIVEN the physical P150x2 service starts with its pinned model and profile
- WHEN B=1 and B=4 warm-up complete with clean reset
- THEN the service MAY notify systemd and bind the ready HTTP endpoint
- AND `/healthz` and `/readyz` report the completed widths and elapsed time

#### Scenario: Warm-up fails closed

- GIVEN prefill, trace capture, trace replay, reset, or readiness notification fails
- WHEN startup evaluates the failure
- THEN the process MUST exit before it reports readiness
- AND the service MUST NOT accept an inference request
