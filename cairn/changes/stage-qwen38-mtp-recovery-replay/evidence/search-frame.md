# Search frame

## Goal

Find the last completed boundary in the persistent MTP recovery path without another unobserved long outage.

## Acceptance constraints

- Use the exact P150x2 model, target, MTP, trace, and recovery implementations.
- Preserve progress atomically before and after each major boundary.
- Keep progress writes outside measured timing windows.
- Stop each physical case at a short named bound.
- Restore eager service after success, failure, or timeout.
- Make no timing, MTP, traced-decode, or production claim from staged evidence.

## Candidate portfolio

### Existing-path stage ledger

Add an ordered progress callback to the unchanged full pilot path. Run one configured prompt and route per case.

**Strength:** Preserves the interaction that caused the prior noncompletion.

**Risk:** The last completed boundary can identify only a bounded region, not the exact instruction.

### Synthetic MTP-only replay

Run trace capture and controlled replay without target sequential decode or native verification.

**Strength:** Smaller and faster.

**Risk:** Removes trace coexistence and target-state interactions. A successful result could miss the actual blocker.

## Selection

Select the existing-path stage ledger. Reject the synthetic path for the first round.

## Adversarial audit

The ledger can perturb host scheduling. Therefore, no staged duration is admission-eligible. Only boundary completion and existing exact correctness fields can be interpreted.

A killed process might not run Python cleanup. The external owner-control trap remains responsible for eager restoration.

## Budget

- One implementation round.
- One positive and one negative device-free transition suite.
- At most two physical cases.
- A named bound for each case.
- Stop after an exact blocked boundary or two complete cases.
