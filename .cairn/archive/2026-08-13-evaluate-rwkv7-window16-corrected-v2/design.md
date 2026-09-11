# Design: Corrected window-16 admission cycle

## Goal and completion evidence

The goal is to validate or reject a new immutable window-16 candidate. Completion requires one exact candidate identity and a complete evidence chain.

A validated result requires:

- a distinct v2 profile, runtime, evidence package, and closure;
- exact pinned CPU continuation and FP32 recurrent-state parity;
- one finite sixteen-step Metal Trace replay;
- completed acknowledgement, zeroing, and slot release;
- bounded simulator smoke with simulator-only claims;
- complete profiler markers without dropped rows;
- five order-balanced profiler-free A/B samples for one-card and two-card modes;
- a speedup of at least `1.0` in every pair;
- a bounded clean canary and monitoring `retain` decision;
- deployed and observed production evidence after admission; and
- package, source, closure, quality, and Cairn gates.

False completion includes reusing v1, accepting device-only agreement without CPU parity, simulator performance claims, missing receipts, early admission, or an unbounded retry.

## Candidate boundary

Candidate v1 remains immutable and rejected. Candidate v2 uses a new profile ID, profile BLAKE3, runtime name, evidence package name, and closure. It keeps `physically_admitted = false` during evaluation.

The candidate enables both diagnostic precision flags. This selects exact HiFi4 recurrence and the separate HiFi2 BF16 readout. The caller remains the sole correction owner.

## Gate order

1. Validate the typed profile and package closure.
2. Run the bounded single-card simulator smoke.
3. Require exact physical CPU parity before performance work.
4. Run terminal lifecycle and profiler gates.
5. Run five order-balanced A/B samples.
6. Run bounded canary and monitoring.
7. Update production only after all prior gates pass.
8. Deploy, observe, and preserve rollback evidence.

A valid failure stops all later gates and retains production `[2, 4, 8]`.

## Evidence budget

- Simulator: one bounded single-card result.
- Correctness: one bounded physical probe on devices `0` and `1`.
- Profiler: two accepted final runs.
- A/B: five balanced samples for one-card and two-card modes.
- Canary: four samples, four requests per worker, two profiler runs, and a named timeout.
- Observation: one bounded post-deployment interval with a rollback closure.

## Evaluation result

The result is rejected. The simulator passed, but the bounded physical request failed exact CPU generated-token parity.

Normal dispatch and persistent trace agreed. Both paths first differed from the pinned CPU continuation at zero-based index `10`.

The valid failure stopped all later gates. Production remains `[2, 4, 8]`.

## Claim boundary

Package-matched generic TT-NN is the exact fused-boundary device authority. The pinned CPU continuation remains the full-model admission authority. Both must pass.

Production stays `[2, 4, 8]`. The simulator and device-only parity results cannot change production admission.
