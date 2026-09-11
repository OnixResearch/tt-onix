## Context

The batch coordinator collects up to 16 compatible requests per step, so an eight-request arrival already executes a width-8 batch in production without admission evidence. The full-model curve measured B=8 at `75.22` ms per step against `60.01` ms at B=1: aggregate `106.35` tokens/second, per-user `13.29`, retention `0.798`, tail ratio `1.25`. Those values sit above the same 75-percent-efficiency bounds that admitted B=4.

The trace-reuse rejection established that prefill under a parked trace is slower than per-group recapture, so B=8 evidence keeps the admitted recapture behavior.

## Decisions

### Decision: Predeclare B=8 bounds at the same efficiency floor

**Choice:** Require aggregate speedup at least `6.0` (eight users at 75 percent), per-user retention at least `0.75`, TTFT ratio at most `9.0` (eight serialized prefills plus 12.5 percent), and tail-latency ratio at most `1.5`.

**Rationale:** B=4 used the same 75-percent discipline. Projected B=8 service values are close to the aggregate bound, so the control is decisive rather than ceremonial: a rejection means the service-level B=8 efficiency is genuinely below the floor.

### Decision: Keep the B=4 evidence surface untouched

**Choice:** Add separate B=8 policy, contracts, binding, and profile files. Change no B=4 file. Parameterize the shared cores over the profile-declared width.

**Rationale:** The admitted B=4 receipt must remain valid without re-proof. Width acceptance in the shared cores stays closed at exactly {4, 8}; any other width fails before device work.

### Decision: Formalize existing serving behavior

**Choice:** The runtime coordinator is unchanged. Admission changes claims and evidence, not serving mechanics.

**Rationale:** Width-8 batches already execute by arrival. The change proves the existing behavior instead of deploying a new one.

## Risks / Trade-offs

- Service-level B=8 aggregate may land near the `6.0` bound; a marginal miss is an honest rejection.
- Eight serialized prefills raise TTFT to roughly seven times single-stream.
- The code change creates a new execution identity; the B=4 admission stays bound to its archived identity and receipt.
- This change does not admit B=16, whose curve efficiency (65 percent) is below the floor, nor slot recycling, mixed generation bounds, or live insertion.
