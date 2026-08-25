## Context

Served decode on P150x2 is single-stream. Measured: traced ~17 t/s (host-bound removed;
production wrapper already serves traced). The decode path is latency-bound: one GDN layer's
per-step time is flat across batch (4-6 ms at B=1..16), so aggregate tokens/s scales almost
linearly with active batch (layer-only ~207 to ~2505 device t/s from B=1 to B=16). The 0.74
service has one decode slot and one batch-one trace, so concurrency is unserved.

The full model exposes a batched decode primitive (`model.decode(token_ids [B,1], pos)` and the
trace-safe `_forward_decode` with staged page tables). The production primitive was measured on
P150x2 with a persistent B=16 model and unique page-table rows. Step time increased from 60.01 ms
at B=1 to 92.21 ms at B=16, while aggregate decode increased from 16.66 to 173.51 tokens/s.

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

### Decision: Keep one exact-width trace resident

**Choice:** Release the prior decode trace before prefill. After prefill, capture one exact-width
trace with the users' real first tokens and positions. Preserve GDN state through capture with
device snapshots.

**Rationale:** TT-Metal reported that allocations under a resident trace were unsafe. Retaining
B=1 and B=4 traces at the same time corrupted a batch-sized GDN zero buffer. Real capture inputs
make the repeated paged-KV writes idempotent.

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

- Full-model step growth did follow the layer curve closely. The measured B=16 step was 1.54x
  the B=1 step and delivered 10.41x the aggregate rate.
- B=32 needs program-config geometry (`per_core_M` vs `out_subblock_h`) and fp32 recurrent
  state in L1 (50 MB); the operating range here is B <= 16.
- Exact-width capture adds about 0.65 s at warmed B=1 and 1.25 s at warmed B=4. The adapter
  recaptures after each prefill to avoid allocations under a resident trace.
- Per-user prefill is serialized to preserve state isolation. In the warmed matched pair, B=4
  TTFT was about 8.99 s versus 1.94 s at B=1.
- The warmed service pair measured 16.44 aggregate tokens/s at B=1 and 57.12 at B=4, a 3.47x
  increase. The typed profile remains not admitted.
