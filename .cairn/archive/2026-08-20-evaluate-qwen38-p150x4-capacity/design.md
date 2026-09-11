## Context

The repository supports Qwen3.8-27B on P150x2. The proposed P150x4 evaluation needs an exact four-device physical mesh.

The available host exposes `/dev/tenstorrent/0` and `/dev/tenstorrent/1`. The operator confirmed that four devices are not available.

Hardware absence is a bounded deployment fact. It is not valid performance evidence.

## Goal and Success Contract

The goal is an explicit unsupported disposition for the unavailable P150x4 topology.

The disposition is complete when:

- the available device-node count is recorded;
- the operator's hardware statement is recorded;
- no P150x4 implementation or admission claim is added;
- P150x2 remains unchanged;
- the accepted specification requires a new change before future support;
- Cairn validation and lifecycle gates pass.

## Decisions

### Decision: Return P150x4 as unsupported

**Choice:** Do not implement or admit a P150x4 deployment on a two-device host.

**Rationale:** Physical topology, placement, fabric, power, and capacity evidence cannot be inferred from fewer devices.

### Decision: Keep the result time-bound

**Choice:** Bind this disposition to the hardware available on 2026-08-20.

**Rationale:** A future four-device host can support a new evaluation. This archive must not block valid future work.

### Decision: Require a new change for future support

**Choice:** A future P150x4 effort must create a new Cairn change with exact hardware authority and physical acceptance evidence.

**Rationale:** Archived absence evidence cannot admit a later topology.

### Decision: Preserve external results as non-claims

**Choice:** Upstream and simulator results remain external information only.

**Rationale:** They cannot prove local physical placement, performance, power, or capacity.

## Architecture

No product code, port, adapter, service, or configuration is added.

The accepted specification owns the support boundary. The evidence file owns the observed machine facts and operator decision.

## Validation Strategy

Cairn validates the proposal, design, completed tasks, and specification delta.

The positive scenario confirms rejection when the required physical topology is absent.

The negative scenario confirms that future hardware availability does not create support without a new validated change.

## Risks and Mitigations

- **Stale hardware result:** Record the observation date and require a new change for later hardware.
- **Unsupported performance claim:** Forbid local P150x4 claims without physical receipts.
- **Accidental P150x2 regression:** Make no product or configuration changes.

## Non-Goals

- Predicting P150x4 performance.
- Simulating a physical acceptance result.
- Adding a dormant P150x4 service.
- Changing P150x2 behavior.
