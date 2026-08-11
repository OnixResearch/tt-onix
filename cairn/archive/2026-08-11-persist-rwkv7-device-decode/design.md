## Context

The fast runtime keeps FP32 recurrent state on each card, but the host remains in the token loop. Program caching reuses compiled programs while preserving per-operation submissions. Earlier device-continuation experiments did not show a stable gain because they changed only token feedback, not the full dispatch boundary.

Persistent decode must therefore cover the complete repeated decode region. It must also preserve request isolation, bounded execution, exact token output, and clear host recovery.

## Goal and Success Contract

The change is complete when:

- A pure state machine defines request admission, active decode, completion, cancellation, timeout, and failure.
- Every persistent run has profile-owned maximum steps and maximum elapsed time.
- Dynamic token, state, slot, weight, output, and program identities are validated before start.
- The device carries the selected token and FP32 recurrent state between steps.
- The host performs no per-token submission inside the admitted persistent region.
- Completion and cancellation handshakes terminate without leaking or reusing live state.
- Exact generated tokens match normal dispatch and the pinned FP32 CPU reference.
- Simulator and physical evidence show the actual host-submission reduction.
- The persistent path becomes selectable only after stable physical throughput admission.

A trace that reuses stale addresses or a device loop without bounded exit is a failed result.

## Decisions

### Decision: Model persistence as a pure state machine

**Choice:** A deterministic core owns legal request states and transitions. The shell owns queues, device memory, clocks, cancellation input, and receipts.

**Rationale:** Lifecycle correctness can be tested without a device or timing assumptions.

### Decision: Bound every device run

**Choice:** The typed profile sets maximum decode steps, maximum elapsed time, queue depth, cancellation polling interval, and output capacity.

**Rationale:** Persistent execution must not become an uninterruptible device owner.

### Decision: Validate immutable binding identities

**Choice:** Bind checkpoint, runtime closure, program set, weights, state slots, token buffers, output buffers, and profile through BLAKE3 identities. Reject drift before execution.

**Rationale:** Replaying a valid program with stale token or state addresses can silently corrupt output.

### Decision: Keep token selection on device

**Choice:** Greedy selection writes the next token directly into the next decode input. The host reads token IDs at completion or bounded observation points outside the timed region.

**Rationale:** A persistent loop cannot depend on a host token round trip.

### Decision: Use explicit completion and cancellation handshakes

**Choice:** Device and host exchange typed status values through one declared channel. Every transition has one producer, one consumer, and a bounded wait.

**Rationale:** Hidden polling and unbalanced signals create deadlock and stale completion risk.

### Decision: Keep normal dispatch as fallback

**Choice:** Unsupported bindings, missing evidence, simulator failure, parity failure, or physical regression select normal dispatch. The receipt names the blocker.

**Rationale:** Persistence is an optimization, not a correctness requirement.

### Decision: Admit actual dispatch removal

**Choice:** Profiler evidence must show zero host submissions per token inside the persistent region. Setup and completion submissions remain outside that region.

**Rationale:** A wrapper that still submits each operation does not satisfy the goal.

## Validation Strategy

Positive tests cover normal completion, maximum-length completion, cancellation, two concurrent slots, slot recycling after acknowledgement, and deterministic receipt identities.

Negative tests cover stale token buffers, stale state buffers, changed checkpoint identity, changed runtime identity, output overflow, duplicate slots, invalid status transitions, cancellation timeout, device timeout, malformed completion, and parity mismatch.

Simulator smoke exercises at least two carried decode steps and every bounded exit type that the simulator supports. Physical tests compare normal and persistent dispatch with the same runtime, batch, workload, and machine facts.

## Risks and Trade-offs

- **Deadlock:** use bounded waits and balanced handshakes.
- **Stale bindings:** identity-check every dynamic and immutable binding.
- **Lost cancellation:** poll through a named bounded mechanism and test timeout behavior.
- **Output overflow:** derive output capacity from maximum steps before device start.
- **State leakage:** release a slot only after acknowledged completion and explicit zeroing.
- **Profiler blind spots:** retain host and device counters with dropped-marker diagnostics.
- **Nonstationary speed:** require alternating A/B evidence and strict admission.

## Non-Goals

- Endless generation.
- Host-free request admission.
- Network serving inside the device runtime.
- Speculative decoding.
- A single-stream latency claim without separate evidence.
