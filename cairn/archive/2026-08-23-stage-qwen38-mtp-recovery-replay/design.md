## Context

The prior persistent-trace package fixed cache-address drift. Its first complete control then stopped making observable progress and reached the `7200`-second process bound without a receipt.

## Decisions

### Decision: Observe the unchanged production-sized path

**Choice:** Add a one-prompt, one-route staged mode. It records an atomic JSON progress ledger around model load, trace capture, replay, recovery, verification, cleanup, and receipt writing.

**Rationale:** The probe keeps the same MTP, target, cache, and trace interactions. A synthetic MTP-only path could hide the interaction that caused noncompletion.

### Decision: Keep progress outside timing economics

**Choice:** Write progress before and after each boundary, but never inside a measured device duration.

**Rationale:** The ledger locates a blocked stage. It is not performance or admission evidence.

### Decision: Fail closed on progress corruption

**Choice:** A pure ordered-stage core rejects unknown, repeated, route-incompatible, or time-regressing events.

**Rationale:** A durable ledger is useful only when its last stage has deterministic meaning.

## Risks / Trade-offs

- Atomic progress writes add host work outside measured intervals.
- A process kill can preserve only the last completed boundary.
- One prompt and one route isolate noncompletion but cannot establish timing stability or production admission.
