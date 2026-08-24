## Context

The archived decode-component attribution measures the P150x2 decode step at
roughly `54 ms`, with GDN mixers near half of it and MLP most of the rest.
The GDN TP projections (`qkvz`, `ba`, `out`) call `ttnn.linear` with a
compute-kernel config but rely on automatic layout selection. The reviewed
Qwen3.6-A3B bundle faced the same shape class at batch one and measured
`AUTO 34.4 us -> DRAM-shard 17.8 -> wide-1D multicast 12.7` for a small-N
decode projection, and `1.71x` to `1.73x` for DRAM-sharded input
projections.

## Decisions

### Decision: Explicit program configs behind a default-off gate

**Choice:** Select wide-1D or DRAM-sharded program configs for the decode
projections only when the operator sets the candidate gate; prefill and the
default path stay on AUTO.

**Rationale:** The accepted spec keeps tuning non-admitted until matched
physical evidence exists. A default-off gate gives a clean eager baseline for
the required A/B pair without touching production behavior.

### Decision: Layout selection lives in a pure core

**Choice:** A pure helper maps `(K, N, device grid)` to the candidate program
config or to None when the shape does not qualify; the patch consumes only
that helper.

**Rationale:** Shape eligibility and tie-breaking become device-free testable
policy, matching the house core/shell split, and the adapter stays a thin
consumer.

### Decision: Evidence before any speedup claim

**Choice:** The candidate is selected, never admitted, by this change. Any
end-to-end claim must go through the existing decode-performance-admission
requirement with order-balanced physical roots.

**Rationale:** Per-op microbenchmarks do not imply model speedup; the spec
forbids that inference explicitly.

## Risks / Trade-offs

- Wrong shard choices can regress individual ops; the pure core bounds shapes
  it accepts and the gate defaults off.
- L1 pressure from sharded activations can collide with the L1-resident
  recurrence state; the first physical pair watches for allocation failures
  and treats them as a failed candidate.
