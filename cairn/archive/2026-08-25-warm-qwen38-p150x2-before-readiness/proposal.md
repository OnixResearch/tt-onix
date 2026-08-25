# Warm Qwen3.8-27B before reporting readiness

## Why

The service reports healthy after model load, but before TTNN compiles the production prefill and decode paths. The first B=1 request then pays about 119 seconds for prefill and 45 seconds for trace preparation. A healthy response must mean that the bounded production path is ready for traffic.

## What Changes

- Validate a startup warm-up plan in a pure device-free core.
- Warm B=1 and B=4 prefill, trace capture, one trace replay, and clean reset before the HTTP server reports readiness.
- Use systemd notification readiness so the unit stays `activating` until warm-up succeeds.
- Fail startup if warm-up, trace replay, reset, or readiness notification fails.
- Expose the observed warm-up widths and duration in `/healthz` and `/readyz`.

## Impact

- Files: service readiness core, batch shell, HTTP server, NixOS service module, and packaged checks.
- Validation: positive and negative device-free tests, package checks, one physical restart, and bounded B=1/B=4/B=8/B=16 production requests.
- Non-claims: startup warm-up does not admit the profile or prove all prompt shapes are compiled.
