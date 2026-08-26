# Search frame

## Goal

Reduce Qwen P150x2 request wall time without weakening exact token, state, reset, identity, TTFT, or tail-latency gates.

## Completion evidence

A valid result needs device-free positive and negative tests, one stable execution identity, matched physical roots, exact output parity, one resident trace, zero measured recaptures, clean resets, stable duplicate roots, and profile-owned wall-time gain.

## False completion cases

Simulator-only evidence, source comments, trace-count logs without output parity, unmatched package identities, warm-up savings, profiler-only timings, and two resident traces do not complete the goal.

## Audit risks

The primary risks are trace-address corruption, GDN state contamination, stale KV rows, reset failure, hidden package drift, and production restoration failure.

## Budget

The search uses four correlated serial lenses because subagent consent is absent. It uses the pinned source, the package-matched runtime, the latest local source reference, one candidate implementation, one device-free baseline, and one bounded physical control. Allowed outcomes are validated, blocked, exhausted, or user-decision-required.

## Approach registry

| Family | Mechanism | Evidence | State |
| --- | --- | --- | --- |
| kernel-layout | Change GDN or MLP decode layouts | GDN and MLP shares are close. The prior GDN layout candidate was slower. | falsified |
| speculative | Use MTP target verification and drafting | Verifier parity, rollback stability, and end-to-end economics remain rejected. | blocked |
| precision | Reduce Qwen weight or recurrent-state precision | No complete numerical and physical admission set exists. | blocked |
| concurrency | Reuse one same-width trace after complete reset | B4 scales by `3.5476x`. Warm trace preparation is about `0.64` seconds. The pinned source supports prewarmed slot prefill. | active |

## Adversarial check

The candidate must release the trace on any width, binding, reset, count, or policy mismatch. The physical control must try two consecutive reuse roots and compare them with two recapture roots.
