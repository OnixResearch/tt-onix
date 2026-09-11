## Context

The September research report supplies the baseline, source index, and rejected experiments. `authority.ncl` records release metadata, but it does not fully identify the converted backbone plus replaced MTP head.

## Decisions

A typed Nickel manifest describes model artifacts and the experiment cohort. New BLAKE3 identities include file content and conversion provenance. Existing SHA-256 Nix fetch identities remain unchanged because Nix requires them.

The pure evaluator consumes manifests, route facts, timing records, and expected policy. It returns accepted evidence, rejected evidence, or an exact blocker. It performs no filesystem, clock, network, or device work.

The shell obtains those facts. One device owner controls each approved window. The listener guard compares execution identity, not merely an HTTP success code. A terminal timeout prevents another request and concurrent device cleanup.

## Measurement contract

Profiles name prompt lengths, output lengths, arrivals, warmup counts, repeat counts, stability limits, timeouts, and quality thresholds. Reuse existing B4 and B8 bounds unchanged. New bounds require review before measurement.

The report distinguishes internal first-token readiness from client-visible first-byte latency. The current non-streaming service does not expose a token at internal readiness.

For MTP, complete round time includes draft, verifier, selection, commit, rollback, and synchronization. Request-wall time includes prefill and capture. Calibration binds the actual control route, not a fixed historical step constant.

## Validation

Positive fixtures cover a valid hybrid manifest and complete balanced roots. Negative fixtures cover swapped head bytes, missing conversion provenance, unknown routes, foreign listeners, partial roots, omitted rollback, and simulation-as-silicon claims.

The smallest hardware stage follows a representative simulator result or explicit simulator blocker. Hardware requires a separately approved outage and recovery path.

## Alternatives and rollback

A directory-name identity is rejected because the same path previously held different head precision. A monolithic benchmark script is rejected because policy and device effects need separate tests.

Existing receipt cores, Rust receipt tools, and execution-identity inputs remain the reuse boundary. No shared framework dependency is necessary. The old service stays selected until a new identity receives its own admission.
