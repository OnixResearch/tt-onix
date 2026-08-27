## Context

The pinned d102 source runs per-user prefill into a persistent B=1 GDN scratch, then snapshots each user's state to the host (`to_torch` per layer per tensor) and re-uploads it (`from_torch`) to write the decode slot. At 48 GDN layers times (recurrent plus K convolution taps) that is hundreds of device-host-device transfers per request group; the B=4 admission measured a `0.494`-second median state-install phase. Separately, `prepare_decode_inputs_host` recomputes `arange`/`outer`/`cos`/`sin` for RoPE on every decode step although the result depends only on position.

The upstream host round trip exists for one reason: under a parked bucket or chunk prefill trace, `ttnn.clone` allocations can alias the trace's baked-address intermediates and be silently overwritten by the next replay. The service path never parks a prefill trace, and it releases the decode trace before prefill.

## Decisions

### Decision: Gate device staging on no parked prefill trace

**Choice:** Clone the scratch state device-side inside the per-user loop only when `_chunked_trace_id` and `_bucket_trace_id` are both absent; keep the host round trip otherwise.

**Rationale:** The clone-aliasing hazard is real under a resident trace. The service satisfies the gate on every served request; the vLLM wrapper with a parked chunk trace falls back to proven behavior.

### Decision: Write through the proven write_slot path

**Choice:** After the loop, call `write_slot` per user per layer with the fresh clone allocations as sources.

**Rationale:** `write_slot` already accepts device tensors, handles dtype, preserves live rows, and consumes (deallocates) its sources — clones are fresh allocations, so consumption frees them safely. Semantics stay identical to the admitted host path.

### Decision: Keep the first-token evidence boundary

**Choice:** Defer all logits readbacks to one synchronized batch after the loop, fire `first_token_callback`, then write slots.

**Rationale:** TTFT keeps its exact meaning (first-token readiness, state install separate). One synchronization replaces per-user sync points.

### Decision: Bitwise-identical RoPE tables

**Choice:** Cache float32 cos/sin rows per position, computed by the same outer/cat/cos pipeline the per-step trig used; decode gathers rows and casts at use.

**Rationale:** Row p is bitwise identical to the recomputation it replaces, so decode outputs cannot change.

### Decision: Re-admit before any production claim

**Choice:** Reset the B=4 binding to not-measured and require the full matched admission control on the new execution identity.

**Rationale:** The patch changes execution inputs, so the prior receipt no longer binds. Receipt-only promotion without re-measurement would be a false claim.

## Risks / Trade-offs

- The clone aliasing hazard could resurface under an unforeseen trace state; the gate and fallback bound it, and the admission parity control would fail closed on any corruption.
- Per-user clones add transient device memory (bounded by one user's GDN state per active request); write_slot frees them.
- The deferred logits readback holds N small device tensors across the loop; at width 16 this is a few megabytes.
- If the matched control fails any gate, the change lands as a preserved rejection and production stays on the deployed admitted closure.
