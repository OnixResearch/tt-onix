## Context

Qwen3.8 runs with tensor parallel degree two on two P150 devices.
Some upstream program choices were tuned for larger tensor-parallel meshes.
The package also carries safe TP2 compatibility behavior.

The original tuning change did not capture an attributed TP2 operation profile.
It did not define or measure a complete legal candidate sweep.

## Goal and Success Contract

The goal is an explicit unselected disposition for this TP2 tuning sweep.

The disposition is complete when:

- the current fallback behavior and missing profile are recorded;
- no winner, generated configuration, or speedup result is invented;
- current package and service behavior remain unchanged;
- future tuning requires a new validated change with profile-first evidence;
- Cairn validation and lifecycle gates pass.

## Decisions

### Decision: Select no TP2 tuning winner

**Choice:** Return no operation or configuration winner.

**Rationale:** Candidate selection has no complete baseline profile or comparable sweep results.

### Decision: Keep current configurations

**Choice:** Preserve current decode configurations and safe TP2 prefill behavior.

**Rationale:** An unmeasured replacement has no admission authority.

### Decision: Do not label fallbacks as tuned

**Choice:** Describe current behavior as existing or fallback behavior only.

**Rationale:** Compatibility and successful execution do not prove optimality.

### Decision: Require profile-first future work

**Choice:** A future change must first attribute device time, host gaps, fabric time, launch counts, shapes, and memory.

**Rationale:** Candidate domains must follow measured bottlenecks, not assumptions.

## Architecture

No product code, port, adapter, service, sweep profile, or generated configuration is added.

The accepted specification owns the tuning-admission boundary.
The evidence file owns the current behavior facts and unselected verdict.

## Validation Strategy

Cairn validates the proposal, design, completed tasks, and specification delta.

The positive scenario selects no winner when the baseline profile or sweep is incomplete.
The negative scenario prevents faster isolated data from implying a model-level admission result.

## Risks and Mitigations

- **False winner:** Select no candidate from incomplete data.
- **Fallback regression:** Change no current configuration.
- **TP4 inference:** Forbid claims based on another mesh size.
- **False speed claim:** Require matched end-to-end physical evidence later.

## Non-Goals

- Proving that current configurations are optimal.
- Predicting a legal winner from tensor shapes.
- Running an unbounded hardware sweep.
- Changing production service behavior.
