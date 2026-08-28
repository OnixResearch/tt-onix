# Design: Bounded window-16 evaluation

## Goal and evidence

The goal is to validate or reject persistent decode window `16` on the P150x2 host. Validation requires:

- one immutable profile identity that ends at window `16`;
- finite sixteen-step Metal Trace replay;
- exact generated-token and FP32 recurrent-state parity;
- completed acknowledgement, state zeroing, and slot release;
- simulator smoke without performance or determinism claims;
- complete profiler markers without dropped rows;
- profiler-free one-card and two-card A/B speedups of at least `1.0` for every pair;
- one bounded physical canary with clean monitoring and a `retain` recommendation;
- package, source-boundary, production-closure, repository-quality, and Cairn gates.

False completion includes simulator performance claims, partial receipts, window-`8` evidence, early admission, unbounded replay, missing cleanup, or candidate bypass.

Valid outcomes are `validated`, `blocked`, or `exhausted`. A failed evidence gate retains production windows `[2, 4, 8]`.

## Bounded portfolio review

Subagent consent is absent. The serial review passes are correlated.

| Family | Mechanism | Claim | State |
|---|---|---|---|
| Evidence-only profile | Use an immutable, non-admitted `[2, 4, 8, 16]` profile through explicit candidate controls. | Evaluation cannot expand production admission. | Selected |
| Immediate expansion | Change production before physical evidence. | This makes commands simpler. | Rejected because it admits an unproved window. |
| Runtime override | Bypass the admitted-window set with an environment value. | This avoids a second identity. | Rejected because it weakens fail-closed behavior. |
| Direct trace probe | Call trace code outside the production worker and receipt path. | This measures device behavior quickly. | Rejected because it does not prove dispatch or cleanup. |

The adversarial review targets profile confusion, stale hashes, fallback errors, trace capacity, parity, cleanup, profiler drops, order bias, timeout, and accidental admission.

## Evidence budget

- Source authority: package-matched runtime, accepted specification, TT-Metal trace API, and packaged `ttsim`.
- Devices: physical devices `[0, 1]` only.
- Simulator: one bounded single-card result for the final candidate.
- Profiler: two final runs. Retain setup errors without a success claim.
- A/B: five order-balanced samples for one-card and two-card modes.
- Canary: four samples, four requests per worker, two profiler runs, and a named timeout.
- Retry rule: fix orchestration or invalid input before a rerun. Do not retry a valid negative result.

## Decisions

### Keep production unchanged during evaluation

Use a separate window-`16` candidate runtime. The profile is not physically admitted.

### Use one finite trace

Capture all sixteen steps in one Metal Trace. Replay it once per bounded batch.

### Validate the production path

Reuse the production worker, benchmark, telemetry, monitoring, and canary paths with window-`16` contracts.

### Admit only after evidence passes

If all gates pass, update production to `[2, 4, 8, 16]`. Otherwise, keep `[2, 4, 8]`.

## Evaluation result

The result is `blocked` for admission. The final simulator passed, but the first required physical correctness sample failed exact CPU generated-token parity.

Normal and persistent device paths agreed with each other. Both diverged from the pinned CPU continuation at zero-based token index `10`. The evaluation stopped before final profiler acceptance, five complete A/B samples, and canary work.

Production remains `[2, 4, 8]`. The candidate remains non-admitted.

## Risks

- Trace memory and setup time increase.
- Host cancellation cannot interrupt an active trace replay.
- A longer window can preserve parity but lose its speed gain.
- Simulator success does not prove physical determinism or performance.
- Physical work uses explicit `/var/tmp` roots because `/tmp` is full.
