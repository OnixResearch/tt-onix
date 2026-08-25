## Context

Served decode on P150x2 is single-stream. Measured: traced ~17 t/s (host-bound removed;
production wrapper already serves traced). The decode path is latency-bound: one GDN layer's
per-step time is flat across batch (4-6 ms at B=1..16), so aggregate tokens/s scales almost
linearly with active batch (layer-only ~207 to ~2505 device t/s from B=1 to B=16). The 0.74
service has one decode slot and one batch-one trace, so concurrency is unserved.

The full model exposes a batched decode primitive (`model.decode(token_ids [B,1], pos)` and the
trace-safe `_forward_decode` with staged page tables). Two integration gaps block a clean
full-model batch harness: `Qwen36Model.reset_state` forwards an argument into the argless TP GDN
`reset_state`, and production batched decode needs the adapter's staged page-table/position
tensors. Both are Phase-1 work, not dead ends.

## Decisions

### Decision: One full-model step per active batch, scheduler in a pure core

**Choice:** Each served step runs one full-model decode at the active batch width `B <= 16`; a
pure device-free core selects B and each user's state slice. The service shell assembles the
batch and distributes outputs.

**Rationale:** Matches the house core/shell split, keeps scheduling testable device-free, and
reuses the existing per-layer B<=16 decode that the layer batch curve already validated.

### Decision: State isolation is a hard invariant

**Choice:** Per-user recurrent (GDN) and KV state slices must not mix across users or steps.
The core rejects collisions; the shell preserves slice addresses between steps.

**Rationale:** The batched decode shares weight reads but must never share a user's hidden
state; isolation bugs produce silent cross-user contamination and must fail closed.

### Decision: No served-throughput claim without a matched pair

**Choice:** Adoption of batching as the served path requires a concurrent-versus-single-stream
physical pair in one window. Phase-1's full-model batch curve is the go/no-go gate: if step
growth is near-linear in B, re-scope the expected win.

**Rationale:** The existing admission discipline forbids per-op/per-layer claims; the service
must not flip to batched serving from extrapolation alone.

## Risks / Trade-offs

- Confirmed (harness v3, 2026-08-25): the plain eager `decode()` shortcut cannot
  run at any batch — `paged_update_cache` requires the staged prefill + page table, so the
  curve MUST go through the production `_forward_decode` staged path at width B.

- Full-model step growth may not follow the layer curve (full-attention KV, paged update,
  all-reduce at higher B). Phase-1 measures this before any service work.
- B=32 needs program-config geometry (`per_core_M` vs `out_subblock_h`) and fp32 recurrent
  state in L1 (50 MB); the operating range here is B <= 16.
- Trace capture/replay at batch width: the captured trace must be batch-width-stable; width
  changes may need re-capture or bucketed traces, verified in Phase 3.
