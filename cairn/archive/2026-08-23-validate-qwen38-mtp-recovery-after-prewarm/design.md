## Context

The full recovery command already owns the required five prompts, balanced route schedule, persistent trace, state comparison, and deterministic evaluator. Its prior physical run entered cold target-validation compilation and produced no receipt. The exact prompt-2 prewarm control later completed the same blocked path.

## Decisions

### Decision: Reuse the full rollback policy and evaluator

**Choice:** Add a distinct `recovery-control` mode that uses the existing rollback profile, warmups, measured schedule, and evaluator.

**Rationale:** This changes only compilation control and evidence durability. It does not weaken sample count, route balance, state parity, stability, or recovery-ratio gates.

### Decision: Prewarm every fixed prompt

**Choice:** Run one discarded nonpersistent target-validation pilot for each configured prompt before persistent recovery warmups.

**Rationale:** Per-prompt prewarm avoids an unsupported assumption that every prompt selects identical target-validation programs or signatures.

### Decision: Freeze the measured program cache

**Choice:** Record program-cache counts and reject any count change after the measured phase starts.

**Rationale:** Measured recovery roots must not include hidden compilation after exact prewarm and route warmup.

### Decision: Preserve repeated-control progress in a pure ledger

**Choice:** Add a pure ordered progress core. The command shell atomically writes progress before and after each prewarm, warmup, measured root, evaluation, cleanup, and receipt boundary.

**Rationale:** A bounded stop must preserve the exact current prompt, route, root, and cache state without moving filesystem effects into the core.

## Risks / Trade-offs

- Per-prompt prewarm adds startup cost. The receipt must report it and cannot exclude it from deployment economics.
- Atomic progress writes add diagnostic host overhead. Recovery timing remains the existing synchronized device interval.
- A complete control still cannot admit MTP or traced target decode. It can only validate or reject persistent recovery stability.
- Any physical run needs a new exact outage bound and explicit authorization.
