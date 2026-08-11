## Context

RWKV-7 uses twelve heads of width 64. Each head keeps a 64 by 64 FP32 recurrent matrix. Each token applies decay and rank updates, reads the state with the current read vector, applies the RWKV attention correction, and returns BF16 data to the remaining model.

The current implementation expresses this work through generic TT-NN matmul, multiply, add, normalization, reshape, and cast operations. Program caching reduces compilation cost but does not remove host dispatch or intermediate memory traffic.

The package-matched gated-delta operator cannot replace this path. Its contract requires 128-wide chunks and dimensions.

## Goal and Success Contract

The change is complete when:

- A pure reference core defines the exact fused equation and accepted tensor shapes.
- The kernel accepts BF16 operands, retains FP32 recurrent state, and emits BF16 output.
- Positive tensor tests match the reference state and output within profile-owned intermediate tolerances.
- End-to-end generated tokens exactly match the pinned FP32 CPU reference.
- Invalid shapes, dtypes, aliasing, and unsupported architecture fail before dispatch.
- The package-matched simulator runs one layer and carries state.
- All applicable MMIO, reconfiguration, config-word, semaphore, mailbox, dataflow, source-register, NoC, instruction-latency, and performance audits pass.
- Stable physical A/B evidence shows a gain before the runtime selects the fused path.

A lower launch count without stable throughput is a rejected candidate, not completion evidence.

## Decisions

### Decision: Build an RWKV-specific operator

**Choice:** Implement the exact token-at-a-time 64-wide RWKV recurrence instead of adapting the 128-wide gated-delta operator.

**Rationale:** Padding or chunk emulation would add work and weaken the shape contract.

### Decision: Keep a pure equation core

**Choice:** Define state update and output equations in a deterministic host reference with no device or file access. The device shell validates metadata and dispatches the kernel.

**Rationale:** Numerical behavior must be testable without hardware.

### Decision: Keep state FP32 and boundaries BF16

**Choice:** Read and write recurrent state as FP32. Consume model vectors in BF16 where the accepted runtime already does. Return BF16 attention output.

**Rationale:** This matches the accepted precision boundary and avoids unsupported quality claims.

### Decision: Fuse only the stable WKV boundary

**Choice:** Fuse recurrent decay, rank updates, state write, state readout, attention correction, and required layout handling. Keep unrelated projections outside the operator.

**Rationale:** Projection weights dominate a different matmul schedule. A bounded operator is easier to validate and audit.

### Decision: Shard state with explicit ownership

**Choice:** Assign heads and request tiles to fixed core groups. Each state tile has one writer. Data movement and completion signals use explicit counted handshakes.

**Rationale:** Single-writer ownership reduces cross-core race risk.

### Decision: Treat L1 state placement as measured work

**Choice:** Start with the simplest correct device placement. Add sharded L1 residency only after memory and profiler evidence show it fits and helps.

**Rationale:** L1 capacity, bank pressure, and request batch size can make a theoretical optimization slower.

### Decision: Keep fallback and selection fail closed

**Choice:** The generic TT-NN path remains available. A typed capability check selects the fused operator only for supported architecture, shapes, and evidence status.

**Rationale:** Unsupported input must not silently use a partially compatible kernel.

## Validation Strategy

Positive device-free tests compare the pure fused equation with the current decomposed equation across zero state, nonzero state, multiple requests, boundary BF16 values, and repeated state carry.

Negative tests reject wrong head width, wrong head count, wrong state rank, BF16 state, unsupported layout, overlapping state ownership, unsupported architecture, malformed runtime arguments, and non-finite reference inputs.

Simulator validation runs the production host and device binaries. Physical validation uses alternating A/B runs at the admitted batch capacities and strict P150x2 admission.

## Risks and Trade-offs

- **Numerical drift:** compare intermediate state and output, then require exact generated tokens.
- **Pipeline hazards:** run instruction-latency and performance audits together.
- **Synchronization races:** run every applicable LLK hazard audit before hardware.
- **L1 pressure:** derive capacity and keep a DRAM fallback.
- **Register pressure:** use bounded unrolling and inspect generated kernel resources.
- **False fusion claims:** verify program signatures and physical elapsed time.
- **Architecture lock-in:** fail closed outside the package-matched Blackhole target.

## Non-Goals

- A generic linear-attention API.
- Wormhole support.
- Tensor-parallel recurrence.
- Quantized recurrent state.
