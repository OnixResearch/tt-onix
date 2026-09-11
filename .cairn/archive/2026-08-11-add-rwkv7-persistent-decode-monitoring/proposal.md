# Change: Add bounded persistent decode monitoring

## Why

Persistent decode worker receipts now expose stable telemetry, but no production tool aggregates a bounded receipt batch or emits fail-closed alert decisions. Operators cannot measure fallback rates or convert parity, cleanup, timeout, and malformed-event signals into one deterministic response.

## What Changes

- Add a pure Rust monitoring core that validates and aggregates an explicit bounded list of worker receipts.
- Add a strict Nickel monitoring policy for batch size, fallback-rate thresholds, and fixed alert actions.
- Add a production command that reads explicit receipt paths and emits one BLAKE3-bound monitoring receipt.
- Emit fixed critical alerts for parity, cleanup, timeout, terminal failure, unsupported admission, malformed telemetry, and duplicate events.
- Emit a fixed warning after the configured minimum sample count exceeds the configured fallback rate.
- Recommend fail-closed persistent admission disablement for every critical result without mutating runtime configuration.

## Scope

The change adds local collection and alert classification. It does not add a network metrics exporter, paging provider, long-running daemon, canary traffic controller, or automatic profile mutation.
