## Context

The GDN chunk-prefill factory opens one program instance per `(batch, value
head)` pair. The instance count is `BH = B x Nv_tp`. The factory refuses any
launch where `BH` exceeds the compute-core count:

```
TT_FATAL(BH <= grid.x * grid.y)
```

At TP=2 the Qwen3.8 checkpoint gives 48 value heads, so `Nv_tp` is 24. The
B=8 case therefore requests 192 instances against 110 compute cores. The launch
fails before device work.

Two fallbacks hide the wall today. Prompts above 256 tokens prefill per user,
which serializes eight prefills inside one request. Short-context B=8 requires
`QWEN_BATCHED_GROUPED=0`. Both are correct and both cost time.

The rejected B=8 receipt measured the serialized-prefill cost directly: TTFT
ratio 9.0792 against a bound of 9.0, and per-user retention 0.7352 against a
bound of 0.75. Both misses sit within roughly two percentage points.

The wall is the only known mechanism that explains the short-context prefilling
behavior. This change removes the wall and measures whether the near miss was
caused by it.

## Decisions

### Decision: Vendor the grid fix alone, without the decode op

**Choice:** Take only the `BH > ncores` hardening from upstream PR 53482. Do not
take the fused decode op in the same change.

**Rationale:** The decode op and the grid fix have different failure modes and
different evidence. The decode op changes the decode numerics. The grid fix
changes prefill scheduling. Bundling them would make a parity failure
impossible to attribute.

### Decision: Keep the existing mapping for `BH` at or below the core count

**Choice:** When `BH <= ncores`, each core keeps exactly one instance and the
mapping is unchanged. Only the overflow case changes: each active core takes a
contiguous chunk of `ceil(BH / ncores)` instances, and the runtime arguments
carry `(bh_start, n_inst)`.

**Rationale:** The unchanged branch keeps every existing prefill receipt valid.
The contiguous chunk assignment keeps the per-core instance loop bounded and
avoids a strided gather.

### Decision: Require exact per-position parity against the serialized path

**Choice:** The screen compares batched GDN prefill against the serialized
per-user prefill reference at every position. It requires exact parity, not a
PCC bound.

**Rationale:** The serialized path is the current admitted behavior. A batched
prefill that only correlates with it would change every downstream state. This
repository already rejects drift that accumulates across 48 GDN layers.

### Decision: The mechanism change admits no batch width

**Choice:** This change produces a prefill mechanism and its parity evidence. It
does not move `physically_admitted` for any width and does not touch the B=8
profile or its bounds.

**Rationale:** The B=8 rejection stands until a fresh matched run passes the
unchanged bounds. Admission is a separate change with its own receipt.

### Decision: Keep the per-user fallback reachable and fail-closed

**Choice:** `QWEN_BATCHED_GROUPED=0` and the long-prompt per-user path stay
available. A malformed instance range, a missing runtime argument, or a shape
outside the closed set selects the per-user path before device work.

**Rationale:** The fallback is the only proven path for the failing cases. A
fail-closed fallback keeps the service correct while the new path is unproven.

## Risks / Trade-offs

- The grid fix may not explain the B=8 near miss. The admission change must then
  report a second rejection rather than a weaker bound.
- Removing a `TT_FATAL` widens the accepted input set. The instance-assignment
  core must therefore own the range validation the fatal used to provide.
- Contiguous chunking changes work distribution across cores. A parity pass does
  not prove that the new distribution is faster, so timing stays a separate
  measurement.
