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
- **Evidence:** Both prior timeouts occurred in target validation while target programs compiled.
- **Gap strength:** Simpler than another full matched control.
- **Blocker:** Physical execution requires a new bounded outage authorization.
- **Next check:** Build and validate the package without devices.
- **State:** Active.

### Release the parked MTP trace before target validation

- **Family:** Trace-coexistence control.
- **Mechanism:** Release the persistent recovery trace after recovery and before target validation.
- **Claim:** Completion after release identifies parked-trace coexistence as the blocker.
- **Artifact:** None in this round.
- **Evidence:** The blocked target work ran while the persistent MTP trace remained parked.
- **Gap strength:** Equivalent diagnostic strength but changes the persistence condition.
- **Blocker:** It cannot distinguish cold compilation until the exact prewarm control runs.
- **Next check:** Use only if the prewarmed parked-trace case stalls.
- **State:** Blocked.

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
- **Next check:** Add only if exact prewarm also stalls.
- **State:** Audit.

## Budgets

- Source authority: package-matched `tt-metal` `v0.74.0` and the pinned Qwen package.
- Retrieval: local source and archived physical evidence only.
- Implementation: one exact-prewarm mechanism round.
- Physical: zero cases before explicit authorization. After authorization, at most one named bounded case.
- Final outcomes: validated, blocked, exhausted, or user-decision-required.
