# Search frame

## Goal

Identify whether cold target compilation caused the two post-recovery timeouts. Preserve a complete diagnostic receipt or the exact last durable boundary.

## Completion evidence

- One loaded process prewarms and measures the same prompt and target-validation signature.
- The ledger records prewarm and measured boundaries with program-cache counts.
- A complete case preserves exact target output and rollback state.
- Eager production service is active and healthy after the bounded case.

## False completion

- A prewarm uses another prompt, acceptance width, target token sequence, or verifier token sequence.
- A timeout produces no durable boundary.
- A complete process lacks exact output or rollback evidence.
- The service restores without healthy devices.
- Prewarm time or a single root becomes performance or admission evidence.

## Approach registry

### Exact same-process prewarm

- **Family:** Program-cache control.
- **Mechanism:** Run the existing nonpersistent pilot for the same prompt before persistent recovery.
- **Claim:** A completed measured case after prewarm identifies cold target compilation as the prior blocker.
- **Artifact:** A prewarm-aware stage ledger and exact signature comparison.
- **Evidence:** Prompt 2 completed exact prewarm, persistent recovery, and measured target validation in one process. The measured target-validation cache count stayed at `408`.
- **Gap strength:** The control directly removed cold compilation without changing the measured recovery route.
- **Blocker:** One case does not establish repeated-root stability or admission.
- **Next check:** Use a new change and outage budget for matched repeated roots only if recovery research continues.
- **State:** Validated.

### Release the parked MTP trace before target validation

- **Family:** Trace-coexistence control.
- **Mechanism:** Release the persistent recovery trace after recovery and before target validation.
- **Claim:** Completion after release identifies parked-trace coexistence as the blocker.
- **Artifact:** None in this round.
- **Evidence:** The prewarmed measured case completed while the persistent MTP recovery trace remained valid and parked.
- **Gap strength:** It would change the tested persistence condition.
- **Blocker:** The exact prewarm control completed, so trace release cannot improve the current explanation.
- **Next check:** None in this change.
- **State:** Falsified as a necessary control.

### Increase the timeout

- **Family:** Unbounded observation.
- **Mechanism:** Repeat the same cold control with a larger timeout.
- **Claim:** Eventual completion can show that compilation is finite.
- **Artifact:** None.
- **Evidence:** The prior `1200`-second cases produced no receipts. The second forced stop required a device reset.
- **Gap strength:** Weaker.
- **Blocker:** It adds outage risk without separating mechanisms.
- **Next check:** None.
- **State:** Falsified.

### Add per-layer target progress only

- **Family:** Finer observation.
- **Mechanism:** Record each target layer boundary without changing execution order.
- **Claim:** The ledger can identify the last target layer.
- **Artifact:** None in this round.
- **Evidence:** Current boundaries already identify target validation as the blocked region.
- **Gap strength:** Weaker than removing cold compilation.
- **Blocker:** A layer index does not separate compilation from trace coexistence.
- **Next check:** None. Exact prewarm completed.
- **State:** Exhausted.

## Budgets

- Source authority: package-matched `tt-metal` `v0.74.0` and the pinned Qwen package.
- Retrieval: local source and archived physical evidence only.
- Implementation: one exact-prewarm mechanism round.
- Physical: the one authorized prompt-2 snapshot case is complete. No physical budget remains.
- Final outcomes: validated, blocked, exhausted, or user-decision-required.

## Outcome

The prompt-2 snapshot command returned status `0` and reached `complete` at `191.3243064180133` seconds. Exact prewarm and measured signatures matched.

The prewarm raised the program-cache count from `14` to `406`. Measured recovery raised it to `408`. The count stayed at `408` through measured target validation.

This bounded result identifies cold target-validation compilation as the observed blocker for this case. It does not establish stability, admission, or production performance.

Receipt BLAKE3: `04b8087eb04a8e609c0cb3bd83f419a21f1405360659bdb6aa24cf669da8c88a`.
