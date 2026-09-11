# Persistent decode monitoring acceptance

Date: 2026-08-11

## Result

The bounded collector and fixed production alert classifier passed package, source-boundary, and retained-receipt validation.

The installed policy has these limits:

- Maximum batch size: `256` events.
- Maximum event size: `1048576` bytes.
- Maximum bounded timing value: `60` seconds.
- Minimum fallback sample count: `8` accepted events.
- Maximum fallback rate: `0.25`.
- Admitted windows: `[2, 4]`.
- Critical action: `disable_persistent_admission`.
- Warning action: `investigate_fallback_rate`.

## Clean physical telemetry batch

`clean-monitoring-receipt.json` collects the retained physical persistent, fallback, and pre-replay cancellation receipts.

- Accepted events: `3`.
- Rejected events: `0`.
- Persistent events: `2`.
- Normal fallback events: `1`.
- Completed slots: `4`.
- Cancelled slots: `4`.
- Alerts: none.
- Verdict: `clean`.
- Admission recommendation: `retain`.
- Policy BLAKE3: `1f7d19e70df728919350b7b5f8242b2592fb2037366b2d88398b945698c0100e`.
- Batch BLAKE3: `95cd863b1af9f932670bd1db327b4015f0645b3e5c614a355adf57a7246deb70`.

The fallback rate is above the configured rate, but this batch is below the minimum sample count. The collector correctly emits no warning.

## Critical duplicate event

`critical-duplicate-monitoring-receipt.json` presents the same canonical physical persistent receipt twice.

- Accepted events: `1`.
- Rejected events: `1`.
- Alert code: `duplicate_event`.
- Severity: `critical`.
- Action: `disable_persistent_admission`.
- Verdict: `critical`.
- Admission recommendation: `disable`.
- Batch BLAKE3: `fb45856eace4ba08e8aafbb95779e2f3f7f5f8c43c1069f8646e8b41f854b5ae`.

## Fallback-rate warning

`warning-fallback-monitoring-receipt.json` uses eight canonical fallback fixtures with bounded ordinal differences.

- Accepted events: `8`.
- Fallback rate: `1.0`.
- Alert code: `fallback_rate_high`.
- Severity: `warning`.
- Action: `investigate_fallback_rate`.
- Verdict: `warning`.
- Admission recommendation: `retain`.

## Positive tests

The Rust and package tests cover:

- Deterministic replay of the same ordered batch.
- Clean persistent and normal fallback aggregation.
- Terminal cancellation aggregation.
- Fallback warning only after the minimum sample count.
- Stable clean, warning, and critical process statuses.
- Deterministic Nickel export.
- Path exclusion from monitoring receipts.

## Negative tests

The Rust, Nickel, and package tests reject or alert on:

- Empty and oversized batches.
- Oversized events.
- Invalid, non-finite, and unknown policy values.
- Malformed and high-cardinality telemetry.
- Duplicate canonical event bytes.
- Parity and cleanup failures.
- Replay timeout and terminal failure.
- Unsupported persistent admission.
- Missing command arguments.

## Non-claims

This evidence does not provide a network exporter, paging provider, long-running daemon, canary traffic controller, or automatic profile mutation.

A critical result recommends fail-closed admission disablement and returns status `3`. A production controller or operator must apply the recommendation.

A fatal worker error can stop receipt creation. Process supervision remains responsible for that path.
