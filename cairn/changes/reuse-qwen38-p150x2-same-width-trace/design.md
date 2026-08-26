## Context

The startup warm-up ends with one resident B=4 trace. The first request releases that trace before prefill and captures another trace afterward. Consecutive groups repeat this work even when width and binding stay unchanged.

The pinned `d102e7f4512060e09e580e40cf26a870bfa88c55` source prewarms slot-prefill programs and keeps a persistent B=1 GDN scratch. It therefore supplies a source-backed reuse candidate. Earlier tests kept B=1 and B=4 traces resident together and corrupted one batch-sized GDN zero buffer.

## Decisions

### Decision: Preserve only one same-width trace

**Choice:** Preserve one resident trace only when the next group has the same width, the prior reset completed, and the binding stays stable.

**Rationale:** This removes repeated capture without repeating the known two-trace failure. A width change releases the trace before prefill.

### Decision: Keep transition policy in a pure core

**Choice:** A device-free core returns the preserve, release, reuse, or capture decision from explicit trace facts.

**Rationale:** The adapter owns TTNN effects. The core owns the safety policy and gives positive and negative tests deterministic inputs.

### Decision: Keep reuse operator-only before physical admission

**Choice:** A closed Nickel profile controls reuse. Normal production keeps recapture until a matched physical receipt passes.

**Rationale:** Source support and device-free tests cannot prove that prefill allocations preserve a resident trace on TT-Metal `0.74.0`.

### Decision: Use a matched same-process physical control

**Choice:** Run exact `recapture,reuse,reuse,recapture` roots at B=4. Require one resident trace, zero measured recaptures, exact tokens, clean reset, stable decode, and lower request wall time.

**Rationale:** The control isolates trace reuse from width scaling. A rejected result completes the investigation without weakening any gate.

## Risks / Trade-offs

- Prefill allocations can damage a parked trace or a trace-bound GDN buffer.
- A reset error can make same-width reuse unsafe.
- The code change creates a new execution identity and invalidates the prior admission binding.
- This change does not admit slot recycling, mixed generation bounds, dynamic compaction, or live request insertion.
- Physical validation needs one bounded service outage and mandatory production restoration.
