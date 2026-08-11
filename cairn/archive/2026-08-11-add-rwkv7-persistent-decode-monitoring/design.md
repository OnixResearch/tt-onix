# Design: Bounded persistent decode monitoring

## Context

The fast worker writes one schema-versioned `persistent_decode_telemetry` object in each successful receipt. The existing evidence guard validates one receipt. Production monitoring needs a deterministic batch boundary, aggregate counters, fixed alerts, and a fail-closed admission recommendation.

The exact result is one command that accepts a typed policy and explicit receipt paths. It emits one bounded JSON receipt. The receipt contains no source paths, prompt data, token data, request identifiers, or model binding identities.

False completion includes unbounded directory watching, network transport, free-form alert labels, silent invalid-event drops, path disclosure, floating thresholds embedded in Rust, or an alert that does not recommend persistent admission disablement after a critical safety signal.

## Bounded review lenses

Subagent consent is absent, so the review uses serial lenses:

1. **Safety lens:** parity, cleanup, timeout, terminal failure, unsupported admission, malformed telemetry, and duplicate inputs.
2. **Determinism lens:** explicit input order, BLAKE3 source identities, fixed alert order, stable arithmetic, and strict schemas.
3. **Operations lens:** bounded batch size, fallback-rate warning, machine-readable severity, exit status, and non-claims.

## Decisions

### Pure monitoring core and thin command shell

The Rust core parses policy bytes and receipt bytes, validates each event, computes aggregates, classifies alerts, and returns a typed monitoring receipt. The command shell reads only explicit files, writes JSON, and maps the final severity to a stable process status.

This keeps policy and alert decisions testable without filesystem setup or mocks.

### Typed Nickel policy

Nickel owns all non-trivial limits and thresholds. The profile fixes the schema, profile ID, maximum batch size, minimum fallback sample count, maximum fallback rate, critical action, and warning action. Rust revalidates these values before collection.

### Fail-closed invalid events

The collector never silently drops a malformed or mismatched worker receipt. It binds the input bytes by BLAKE3, increments the rejected-event count, and emits `invalid_telemetry`. It also emits a more specific safety alert when bounded raw fields show failed parity, incomplete cleanup, or unsupported persistent admission.

### Deterministic alert order

The core evaluates a fixed alert registry. Output order does not depend on hash-map iteration, file names, or error text. Alert entries contain only code, severity, count, and fixed action.

### Critical admission recommendation

Any critical alert sets `persistent_admission_recommendation` to `disable`. A clean or warning-only batch sets it to `retain`. The command does not mutate the installed profile because profile mutation requires a separate controller contract.

### Explicit bounded collection

The command accepts at least one and at most the policy batch limit. It does not scan directories, follow links, watch files, retry reads, or fetch network data. Duplicate event bytes produce a critical alert instead of double counting.

## Receipt contract

The monitoring receipt contains:

- schema and event kind;
- policy ID and policy BLAKE3;
- source event count, accepted count, rejected count, ordered event BLAKE3 values, and one batch BLAKE3;
- bounded eligibility, dispatch, selected-window, fallback, terminal, parity, cleanup, and timing aggregates;
- fixed alerts;
- final severity and persistent admission recommendation;
- explicit non-claims.

## Verification

Positive tests cover clean persistent execution, bounded normal fallback, warning threshold behavior, and deterministic input replay. Negative tests cover empty and oversized batches, malformed policy, malformed telemetry, parity failure, cleanup failure, timeout, terminal failure, unsupported windows, duplicate events, and high-cardinality telemetry.

The package install check exports the Nickel policy twice, compares the results, runs clean and critical fixtures, and verifies stable process statuses.
