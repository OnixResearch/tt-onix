## Context

The Qwen3.8-27B P150x2 service serializes requests and uses eager decode.
The proposed batching design required one admitted traced replay for each batch shape.

The physical traced-decode candidate preserved parity and reset behavior.
It failed the TTFT admission bound and remained disabled.

## Goal and Success Contract

The goal is an explicit unsupported disposition for continuous batching on the current execution base.

The disposition is complete when:

- the rejected prerequisite receipt is recorded;
- no batch implementation or throughput claim is added;
- serialized eager execution remains the production path;
- future support requires a new validated change and complete physical evidence;
- Cairn validation and lifecycle gates pass.

## Decisions

### Decision: Do not build batching on an unadmitted trace

**Choice:** Return continuous batching as unsupported for the current service.

**Rationale:** The proposed runtime shell depends on traced replay.
Using an unadmitted path would bypass its fail-closed boundary.

### Decision: Keep serialized eager execution

**Choice:** Add no scheduler, queue, batched device state, mask, or trace path.

**Rationale:** Dormant batch code would add state and cleanup risk without an admissible execution base.

### Decision: Require a new change for future support

**Choice:** Future work must first name an admitted execution base.
It must then prove batch sizes separately with matched physical evidence.

**Rationale:** A later trace or runtime candidate can change state ownership, timing, and memory behavior.
This archive cannot admit it.

## Architecture

No product code, port, adapter, service, or configuration is added.

The accepted specification owns the support boundary.
The evidence file owns the prerequisite receipt and production health facts.

## Validation Strategy

Cairn validates the proposal, design, completed tasks, and specification delta.

The positive scenario rejects continuous batching when its execution base is unadmitted.
The negative scenario prevents a later trace admission from implying batch support automatically.

## Risks and Mitigations

- **State isolation risk:** Add no batch slots or mixed-lane state.
- **False throughput claim:** Forbid inference from single-request decode metrics.
- **Stale decision:** Require a new change after the prerequisite changes.
- **Service regression:** Keep production code and configuration unchanged.

## Non-Goals

- Predicting aggregate batch throughput.
- Implementing a scheduler for later use.
- Claiming that continuous batching cannot work.
- Changing the current eager service.
