## Context

The local Qwen3.8 MTP line owns several trace controls: bounded draft
selection with persistent-trace execution, MTP recovery stability over one
trace, staged recovery replay, and a same-process TTFT control with a parked
decode trace. The reviewed Qwen3.6-A3B bundle documents trace-capture hygiene
as a hard rule. Keeping two verify traces resident corrupts the second. Trace
capture must run from real warm rounds: one one-row seed round, then a full
multi-row round. Trace-bound buffers are mutated in place at stable addresses
because a captured trace bakes the addresses it reads.

## Decisions

### Decision: Enforce release-before-capture

**Choice:** The hygiene gate rejects trace evidence collected while another
trace was resident for the same control.

**Rationale:** The reviewed bundle established the corruption by A/B
comparison. Greedy-then-sampling capture fails. Release-then-capture passes.
The gate turns unbounded output corruption into a bounded reject.

### Decision: Require real warm priming before capture

**Choice:** The hygiene gate requires capture from real warm execution
rounds, including one one-row seed round before a multi-row capture. It
rejects evidence captured from zeroed buffers.

**Rationale:** A multi-row capture primed from zeroed buffers writes state the
replay later slices out of. The result is wrong state that subsequent rounds
condition on.

### Decision: Require stable trace-bound addresses and in-place mutation

**Choice:** The hygiene gate rejects evidence when a trace-bound buffer was
reallocated at a different device address after capture.

**Rationale:** A captured trace bakes buffer addresses. Reallocation silently
invalidates the trace. In-place mutation of a stable-address buffer is the
supporting practice.

## Risks / Trade-offs

- A mandatory real warm seed round costs one extra warmup execution per
  capture. The bundle excludes warmup from measured economics; this change
  does the same.
- The gates validate evidence provenance, not token values. They complement,
  and do not replace, the existing token, state, and parity gates.
