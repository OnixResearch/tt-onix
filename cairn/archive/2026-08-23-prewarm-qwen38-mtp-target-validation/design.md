## Context

The first staged case reached `native-verifier-start` after re-prefill recovery. The second case reached `target-reference-state-start` after snapshot recovery. Each case reached its `1200`-second bound without a receipt. TT-Metal compiled target programs at both blocked boundaries.

The process program cache does not survive a process exit. A useful control must prewarm and measure in one loaded process. It must use the same prompt and exact target-validation shape.

## Decisions

### Decision: Prewarm the unchanged target-validation path before persistent trace capture

**Choice:** Run one same-prompt, nonpersistent `run_pilot` call before the staged persistent-recovery call. Compare exact target-validation signatures after both calls.

**Rationale:** This choice reuses the validated target path and program hashes. It changes no target operation, recovery operation, model state layout, or production route.

### Decision: Preserve program-cache observations in the stage ledger

**Choice:** Let the shell read the package-matched program-cache count. Pass that observation into the pure ordered-stage transition.

**Rationale:** A durable count distinguishes a completed prewarm from an unobserved assumption. The core rejects negative or regressing counts.

### Decision: Keep parked-trace release as a later discriminating control

**Choice:** Keep the persistent MTP trace parked during the first prewarmed case.

**Rationale:** If the prewarmed case completes, cold compilation caused the bounded stall. If it stalls, a later control can release the trace before validation.

## Risks / Trade-offs

- The prewarm can itself reach the outage bound. Its ledger must preserve the exact blocked stage.
- One completed case cannot establish repeated-root stability or MTP admission.
- A forced stop can leave a device in a reset-required state. The physical runbook must isolate, reset, and restore before health acceptance.
