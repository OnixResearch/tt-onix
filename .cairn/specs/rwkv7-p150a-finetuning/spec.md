# Rwkv7 P150a Finetuning Specification

## Purpose

Defines the `rwkv7-p150a-finetuning` capability.

## Requirements

### Requirement: Accelerator work requires current platform and CPU authority

r[tenstorrent.rwkv7_p150a_finetuning.authority] The P150A runner MUST require current passing platform and CPU reference receipts with matching runtime, model, tokenizer, data, LoRA, optimizer, and configuration identities.

#### Scenario: Current prerequisite receipts are admitted

r[tenstorrent.rwkv7_p150a_finetuning.authority.valid]
- GIVEN the platform and CPU receipts pass and bind the selected inputs
- WHEN accelerator admission runs
- THEN the runner MUST derive one deterministic prerequisite identity.

#### Scenario: Stale prerequisite evidence blocks compilation

r[tenstorrent.rwkv7_p150a_finetuning.authority.invalid]
- GIVEN a prerequisite receipt is missing, failing, stale, malformed, or bound to different inputs
- WHEN accelerator admission runs
- THEN compilation and device access MUST remain blocked.

### Requirement: P150A profiles are typed and named

r[tenstorrent.rwkv7_p150a_finetuning.configuration] Static shapes, dtypes, compile limits, memory limits, numerical tolerances, run criteria, health limits, timeouts, artifact policy, and evidence policy MUST come from typed Nickel profiles.

#### Scenario: Valid P150A profile exports deterministically

r[tenstorrent.rwkv7_p150a_finetuning.configuration.valid]
- GIVEN a named profile satisfies all model, device, static-shape, limit, and tolerance constraints
- WHEN profile export runs twice
- THEN both outputs and their BLAKE3 identities MUST match.

#### Scenario: Invalid P150A profile fails admission

r[tenstorrent.rwkv7_p150a_finetuning.configuration.invalid]
- GIVEN a profile has dynamic shapes, invalid limits, unknown precision, missing tolerances, or inconsistent artifact policy
- WHEN Nickel validation runs
- THEN export MUST fail before compilation.

### Requirement: Focused WKV7 compiler probes fail closed

r[tenstorrent.rwkv7_p150a_finetuning.compiler] The change MUST compile and run focused direct TT-XLA probes for WKV7 forward, WKV7 backward, LoRA projections, masked loss, gradient norm, clipping, and adapter update before full model compilation.

#### Scenario: Direct WKV7 graph compiles for the named profile

r[tenstorrent.rwkv7_p150a_finetuning.compiler.valid]
- GIVEN admitted fixed-shape WKV7 tensors and the selected runtime
- WHEN direct forward and backward compilation runs
- THEN compilation and execution MUST pass within named limits
- AND placement evidence MUST identify TT execution for every admitted tensor operation.

#### Scenario: Focused compiler probe fails

r[tenstorrent.rwkv7_p150a_finetuning.compiler.invalid]
- GIVEN compilation, placement, execution, memory, shape, or numerical validation fails
- WHEN the probe completes
- THEN the full model MUST remain blocked
- AND no passing support claim MUST be emitted.

### Requirement: Unsupported direct lowering produces an exact blocker

r[tenstorrent.rwkv7_p150a_finetuning.blocker] If direct WKV7 lowering fails, the change MUST emit a bounded blocker receipt with the unsupported operation, graph boundary, runtime identity, profile identity, diagnostic artifact identity, and next architecture decision.

#### Scenario: Unsupported WKV7 lowering produces a blocker

r[tenstorrent.rwkv7_p150a_finetuning.blocker.valid]
- GIVEN direct TT-XLA lowering cannot satisfy the compiler contract
- WHEN blocker generation runs
- THEN the receipt MUST identify the smallest discriminating failure
- AND it MUST NOT authorize host fallback, claim P150A support, or implement an unreviewed custom operation.

### Requirement: The full trainable graph uses one logical P150A

r[tenstorrent.rwkv7_p150a_finetuning.graph] The runner MUST compile the pinned full RWKV-7 model, masked loss, backward path, clipping, and adapter optimizer for one admitted static profile with frozen base tensors.

#### Scenario: Full graph compiles and runs

r[tenstorrent.rwkv7_p150a_finetuning.graph.valid]
- GIVEN all focused probes pass and the full model uses the admitted LoRA plan
- WHEN full graph compilation and execution run
- THEN outputs, loss, required gradients, clipping values, and adapter updates MUST be finite
- AND the graph MUST preserve the admitted trainable and frozen parameter sets.

#### Scenario: Full graph changes its admitted contract

r[tenstorrent.rwkv7_p150a_finetuning.graph.invalid]
- GIVEN the full graph adds an unknown operation, dynamic shape, trainable base tensor, missing adapter, or stale compile artifact
- WHEN graph admission runs
- THEN execution MUST fail before optimizer mutation.

### Requirement: One logical device is selected through supported runtime APIs

r[tenstorrent.rwkv7_p150a_finetuning.device] The launcher MUST pass all required physical devices and MUST select one logical `1x1` P150A submesh through the admitted TT-XLA API without container device isolation.

#### Scenario: Logical single-device selection passes

r[tenstorrent.rwkv7_p150a_finetuning.device.valid]
- GIVEN the dual-P150A platform is healthy and idle
- WHEN the single-device profile opens its logical submesh
- THEN exactly one logical device MUST execute model work
- AND both physical devices MUST have bounded before-and-after health evidence.

#### Scenario: Logical selection or cleanup fails

r[tenstorrent.rwkv7_p150a_finetuning.device.invalid]
- GIVEN logical selection is ambiguous, the unused device executes model work, ownership is busy, health fails, or close fails
- WHEN the runner exits
- THEN the run MUST fail
- AND restoration MUST remain unclaimed until separate evidence proves it.

### Requirement: Model and optimizer work cannot use silent host fallback

r[tenstorrent.rwkv7_p150a_finetuning.placement] The runner MUST save and validate placement evidence for model forward, masked loss, backward, gradient norm, clipping, and adapter optimizer operations.

#### Scenario: Full trainable graph stays on TT

r[tenstorrent.rwkv7_p150a_finetuning.placement.valid]
- GIVEN the full graph compiles under the admitted runtime
- WHEN placement validation runs
- THEN every required trainable operation MUST have current TT placement evidence.

#### Scenario: Forbidden host fallback is rejected

r[tenstorrent.rwkv7_p150a_finetuning.placement.invalid]
- GIVEN any required trainable operation is missing, stale, unknown, or placed on the host
- WHEN placement validation runs
- THEN the run MUST fail even if loss is finite.

### Requirement: P150A results satisfy CPU comparison profiles

r[tenstorrent.rwkv7_p150a_finetuning.parity] The P150A runner MUST compare logits, loss, selected WKV7 outputs and states, selected adapter gradients, gradient norm, clipping result, and selected adapter deltas with CPU authority under named tolerances.

#### Scenario: One P150A step matches CPU authority

r[tenstorrent.rwkv7_p150a_finetuning.parity.valid]
- GIVEN identical admitted model, adapters, data, labels, optimizer inputs, and seed
- WHEN CPU and P150A one-step runs complete
- THEN all required shapes and finite-value invariants MUST match
- AND all selected numerical comparisons MUST satisfy their named tolerances.

#### Scenario: Wrong or nonfinite device values fail parity

r[tenstorrent.rwkv7_p150a_finetuning.parity.invalid]
- GIVEN a required value is missing, stale, wrongly shaped, nonfinite, or outside tolerance
- WHEN parity validation runs
- THEN validation MUST fail with the exact tensor role and tolerance identity.

### Requirement: The optimizer updates adapters only

r[tenstorrent.rwkv7_p150a_finetuning.optimizer] The P150A optimizer MUST receive only admitted LoRA tensors and MUST preserve every frozen base tensor identity.

#### Scenario: P150A optimizer changes adapters only

r[tenstorrent.rwkv7_p150a_finetuning.optimizer.valid]
- GIVEN finite active adapter gradients and the admitted clipping result
- WHEN one optimizer step completes
- THEN at least one adapter tensor MUST change
- AND every base tensor MUST keep its pre-step identity.

#### Scenario: Optimizer contract is violated

r[tenstorrent.rwkv7_p150a_finetuning.optimizer.invalid]
- GIVEN no adapter changes, a base tensor changes, an unknown tensor enters the optimizer, or the clipping result violates policy
- WHEN optimizer validation runs
- THEN the run MUST fail without a passing adapter artifact.

### Requirement: Bounded P150A fine-tuning meets its run criterion

r[tenstorrent.rwkv7_p150a_finetuning.run] The runner MUST execute the named bounded multi-step profile and MUST satisfy its finite-value, adapter-only mutation, active-label, health, and loss-improvement criteria.

#### Scenario: Bounded fine-tuning run passes

r[tenstorrent.rwkv7_p150a_finetuning.run.valid]
- GIVEN the one-step parity gate passes
- WHEN the named fixed-corpus run completes
- THEN every required step MUST remain finite and valid
- AND the configured loss-improvement criterion MUST pass.

#### Scenario: Bounded fine-tuning run fails

r[tenstorrent.rwkv7_p150a_finetuning.run.invalid]
- GIVEN a step is missing, nonfinite, stale, unhealthy, label-empty, base-mutating, or outside the loss criterion
- WHEN run validation completes
- THEN the run MUST fail without a convergence claim.

### Requirement: P150A checkpoints use the portable adapter contract

r[tenstorrent.rwkv7_p150a_finetuning.checkpoint] The P150A runner MUST use the CPU adapter format and MUST add current graph, placement, runtime, precision, device, step, and health evidence.

#### Scenario: P150A checkpoint resumes safely

r[tenstorrent.rwkv7_p150a_finetuning.checkpoint.valid]
- GIVEN a passing adapter artifact and matching resume inputs
- WHEN save, load, and resume run
- THEN the resumed result MUST satisfy the named uninterrupted-run comparison
- AND the base checkpoint MUST remain unchanged.

#### Scenario: P150A resume evidence is incompatible

r[tenstorrent.rwkv7_p150a_finetuning.checkpoint.invalid]
- GIVEN any adapter, base, graph, placement, runtime, precision, device, optimizer, data, or configuration identity differs
- WHEN resume admission runs
- THEN resume MUST fail before device mutation.

### Requirement: Single-P150A evidence stays bounded

r[tenstorrent.rwkv7_p150a_finetuning.evidence] The runner MUST emit BLAKE3-bound compile, placement, parity, optimizer, run, artifact, health, restoration, and non-claim evidence.

#### Scenario: Evidence keeps the single-device boundary

r[tenstorrent.rwkv7_p150a_finetuning.evidence.valid]
- GIVEN all single-device acceptance gates pass
- WHEN evidence generation runs
- THEN the receipt MUST bind the exact runtime, profile, graph, inputs, comparisons, devices, and artifacts
- AND it MUST not claim dual-card scaling, useful generalization, speedup, or support for other models.

### Requirement: Single-P150A support has complete validation and documentation

r[tenstorrent.rwkv7_p150a_finetuning.verification] The change MUST include positive and negative compile, placement, parity, optimizer, artifact, device, health, restoration, and receipt fixtures plus bounded hardware evidence.

#### Scenario: Single-P150A validation closes the change

r[tenstorrent.rwkv7_p150a_finetuning.verification.complete]
- GIVEN maintainers intend to sync and archive single-P150A support
- WHEN required validation runs
- THEN all focused probes, fixtures, hardware gates, regressions, package checks, Cairn validation, and lifecycle gates MUST pass.

### Requirement: Single-P150A documentation gives exact recovery and limits

r[tenstorrent.rwkv7_p150a_finetuning.documentation] The documentation MUST give exact prerequisites, compile probes, launch commands, artifact paths, failure diagnosis, cleanup, recovery, and claim limits.

#### Scenario: Operator can run the accepted profile

r[tenstorrent.rwkv7_p150a_finetuning.documentation.complete]
- GIVEN the admitted platform, model inputs, and local fixture data
- WHEN an operator follows the documented procedure
- THEN the operator MUST reach either a passing bounded receipt or an exact blocker receipt.
