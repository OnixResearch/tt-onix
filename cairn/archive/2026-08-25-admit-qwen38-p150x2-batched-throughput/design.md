## Context

The current batch service reports `admission_eligible = false` for every response. It has no batch-specific policy or receipt evaluator. It also sets batch TTFT after exact-width trace preparation. The existing single-stream admission contract stops TTFT after first-token readback and records trace preparation separately.

Physical service evidence already shows isolated B=4 execution and aggregate scaling. That evidence cannot admit the path because it lacks matched roots, a current identity, and a typed decision.

## Decisions

### Decision: Keep single-stream and batch admission separate

**Choice:** Keep the single-stream `1.05` TTFT regression gate unchanged. Add a batch profile that compares the same traced production path at B=1 and B=4.

**Rationale:** The single-stream gate asks whether tracing regresses eager first-token work. The batch gate asks whether four concurrent users gain enough aggregate throughput for a bounded latency trade-off. One result cannot replace the other.

### Decision: Measure first-token work before trace preparation

**Choice:** Start batch TTFT immediately before prefill. Stop it when prefill returns all first-token winners. Record exact-width trace preparation in its existing separate field.

**Rationale:** This matches the established phase boundary. It does not move prefill or first-token readback outside TTFT.

### Decision: Predeclare bounded B=4 gates

**Choice:** The typed profile requires at least `3.0` aggregate speedup, at least `0.75` per-user decode-rate retention, at most `4.5` TTFT ratio, and at most `1.5` tail-latency ratio.

**Rationale:** Four users have an ideal aggregate ratio of four. The aggregate and per-user bounds require at least 75 percent efficiency. Prefill remains serialized, so the TTFT bound permits four baseline prefill intervals plus 12.5 percent overhead. The tail bound limits decode-step inflation.

### Decision: Make admission evidence operator-only

**Choice:** Add an explicit batch-admission mode. Normal production responses remain ineligible. The mode requires the candidate decode path and emits the closed admission measurement name.

**Rationale:** Ordinary traffic and prior soak responses must not enter a formal receipt by accident.

### Decision: Evaluate receipts in a pure Rust core

**Choice:** A single-file Rust receipt command reads immutable response roots in exact `single,batch,batch,single` order. It validates identities, widths, request pairing, tokens, state slices, timing, cleanup, and profile bounds before writing JSON.

**Rationale:** The service owns device effects. The evaluator owns deterministic admission meaning.

### Decision: A rejected result completes the change

**Choice:** Set `physically_admitted = true` only after a passing receipt binds its BLAKE3 identity. Archive a failing receipt without changing admission state.

**Rationale:** Physical execution can disprove admission. The repository must preserve that result without weakening policy.

## Risks / Trade-offs

- B=4 prefill remains serialized, so TTFT grows with concurrent users.
- A B=4 admission does not admit B=8 or B=16.
- Non-streaming HTTP wall time is longer than the internal first-token phase.
- Four service roots require a bounded production outage. The lifecycle must restore the deployed unit and health on every exit.
