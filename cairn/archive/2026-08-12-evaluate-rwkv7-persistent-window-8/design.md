# Design: Bounded window-8 evaluation

## Goal and completion evidence

The exact goal is to validate or reject persistent decode window `8` on the P150x2 host. Validated completion requires:

- one immutable profile identity that ends at window `8`;
- finite eight-step Metal Trace replay;
- exact generated-token and FP32 recurrent-state parity;
- completed acknowledgement, state zeroing, and slot release;
- simulator smoke without performance or determinism claims;
- complete profiler markers without dropped rows;
- order-balanced profiler-free one-card and two-card A/B speedups of at least `1.0`;
- one bounded physical canary with clean monitoring and a `retain` recommendation;
- package, source-boundary, production-closure, repository-quality, and Cairn gates.

False completion includes simulator performance claims, partial receipts, historical window-`4` evidence, profile admission before all gates pass, unbounded replay, missing cleanup, candidate bypass, or a clean process status with invalid evidence.

Allowed outcomes are `validated`, `blocked`, or `exhausted`. A failed evidence gate retains production windows `[2, 4]`.

## Bounded portfolio review

Subagent consent is absent. The serial passes are correlated.

| Family | Mechanism | Claim | State |
|---|---|---|---|
| Evidence-only profile | Build an immutable, non-admitted `[2, 4, 8]` profile. Run window `8` only through explicit candidate controls. | Evaluation cannot silently expand production admission. | Selected |
| Immediate profile expansion | Change the production profile to `[2, 4, 8]` before physical evidence. | Simplifies command paths. | Rejected because it admits an unproved window. |
| Runtime step override | Let an environment value bypass the admitted-window set. | Avoids a second profile identity. | Rejected because it weakens fail-closed admission. |
| Direct trace probe | Call the trace code outside the production worker and receipt path. | Measures device behavior quickly. | Rejected because it does not prove production dispatch or cleanup. |

The adversarial audit targets profile confusion, stale hashes, unsupported-window fallback, interlock mismatch, trace capacity, missing parity, incomplete cleanup, profiler drops, order bias, timeout, duplicate telemetry, and accidental window admission.

## Evidence budget

- Source authority: package-matched runtime, evidence code, accepted Cairn specification, package-matched TT-Metal trace API, and packaged `ttsim`.
- Retrieval budget: six targeted source or documentation reads before each new mechanism. Retrieve more only for a missing owner, field, or command contract.
- Device budget: physical devices `[0, 1]` only.
- Simulator budget: one bounded single-card smoke result for the final candidate.
- Profiler budget: two final profiler runs, with any setup-only failure retained and no success claim.
- A/B budget: five order-balanced samples for one-card and two-card modes.
- Canary budget: four samples, four requests per worker, two profiler runs, at most `104` request executions, and `600` seconds.
- Retry rule: fix orchestration or invalid input before a rerun. Do not retry a valid negative runtime result to search for a pass.

## Decisions

### Decision: Keep production admission unchanged during evaluation

**Choice:** Parameterize the package profile input and add a separate window-`8` candidate runtime. Its profile is not physically admitted. Explicit candidate execution can use it, while the normal runtime stays at `[2, 4]`.

**Rationale:** This keeps the deployed production identity unchanged until evidence is complete. It also prevents an ordinary production command from selecting window `8`.

### Decision: Use one finite eight-step trace

**Choice:** Capture all eight steps in one Metal Trace and replay it once per bounded batch.

**Rationale:** This preserves zero per-token host submissions. It also keeps cancellation and recovery bounded by one finite window.

### Decision: Validate the production path

**Choice:** Reuse the production worker, throughput benchmark, telemetry, monitoring, and canary paths with profile-bound window-`8` contracts.

**Rationale:** A standalone microbenchmark cannot prove production dispatch, parity, cleanup, or monitoring behavior.

### Decision: Admit only after evidence passes

**Choice:** Keep the candidate profile separate if any required gate fails. If every gate passes, update the production profile to `[2, 4, 8]`, update all BLAKE3 bindings, and rerun deterministic gates before archive.

**Rationale:** Physical admission is the conclusion of the evidence, not an input to it.

## Risks and trade-offs

- Trace memory and setup time increase with the window.
- Host cancellation cannot interrupt an active trace replay.
- A longer window can preserve parity but lose its physical speedup.
- Simulator success does not prove physical determinism or performance.
- Host `/tmp` is near capacity. Physical work uses explicit `/var/tmp` roots and does not delete unrelated data.
