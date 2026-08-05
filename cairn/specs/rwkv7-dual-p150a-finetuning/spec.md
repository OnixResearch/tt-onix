# Rwkv7 Dual P150a Finetuning Specification

## Purpose

Defines the `rwkv7-dual-p150a-finetuning` capability.

## Requirements

### Requirement: Dual-card work requires the complete current authority chain

r[tenstorrent.rwkv7_dual_p150a.authority] The dual-card runner MUST require matching current platform, CPU reference, and single-P150A receipts for the selected runtime, model, data, LoRA, optimizer, graph, and configuration.

#### Scenario: Current prerequisite chain is admitted

r[tenstorrent.rwkv7_dual_p150a.authority.valid]
- GIVEN every prerequisite receipt passes and binds matching identities
- WHEN dual-card admission runs
- THEN the runner MUST derive one deterministic prerequisite-chain identity.

#### Scenario: Stale prerequisite chain blocks dual launch

r[tenstorrent.rwkv7_dual_p150a.authority.invalid]
- GIVEN any prerequisite is missing, failing, stale, malformed, or inconsistent
- WHEN dual-card admission runs
- THEN logical-mesh compilation and device access MUST remain blocked.

### Requirement: Dual-card profiles are typed and named

r[tenstorrent.rwkv7_dual_p150a.configuration] Physical mesh, logical data axis, shard policy, reduction policy, replica policy, run criteria, limits, timeouts, artifact policy, and evidence policy MUST come from typed Nickel profiles.

#### Scenario: Valid dual-card profile exports deterministically

r[tenstorrent.rwkv7_dual_p150a.configuration.valid]
- GIVEN a named profile satisfies all physical, logical, shard, collective, replica, and artifact constraints
- WHEN profile export runs twice
- THEN both outputs and their BLAKE3 identities MUST match.

#### Scenario: Invalid dual-card profile fails admission

r[tenstorrent.rwkv7_dual_p150a.configuration.invalid]
- GIVEN a profile has an invalid physical mesh, logical axis, shard rule, reduction rule, limit, timeout, or artifact policy
- WHEN Nickel validation runs
- THEN export MUST fail before compilation or device access.

### Requirement: Logical mesh and collectives have focused evidence

r[tenstorrent.rwkv7_dual_p150a.collectives] The runner MUST compile and run focused logical-mesh, synchronization, mean all-reduce, placement, timeout, and output probes over both P150A devices before model work.

#### Scenario: Logical two-replica mesh passes collective probes

r[tenstorrent.rwkv7_dual_p150a.collectives.valid]
- GIVEN the physical `p150_x2` `1x2` mesh is healthy and the logical data axis has two replicas
- WHEN focused collective probes run
- THEN both devices MUST contribute the admitted input
- AND the reduced outputs, synchronization, TT placement, and timeout results MUST satisfy the named profile.

#### Scenario: Collective or placement evidence fails

r[tenstorrent.rwkv7_dual_p150a.collectives.invalid]
- GIVEN a device, link, operation, placement, synchronization, output, or timeout result is missing, stale, host-placed, or wrong
- WHEN collective validation runs
- THEN full model work MUST remain blocked.

### Requirement: Global batches shard exactly once

r[tenstorrent.rwkv7_dual_p150a.sharding] The runner MUST derive a deterministic pure shard plan that assigns every admitted sample and active label to exactly one logical replica under the named even-shard policy.

#### Scenario: Global batch shards exactly once

r[tenstorrent.rwkv7_dual_p150a.sharding.valid]
- GIVEN an admitted global sample order, active-label map, accumulation step, and two-replica profile
- WHEN shard planning runs
- THEN every sample and active label MUST have exactly one owner
- AND shard order and identities MUST be deterministic.

#### Scenario: Invalid shard plan is rejected

r[tenstorrent.rwkv7_dual_p150a.sharding.invalid]
- GIVEN a batch cannot shard evenly or a plan duplicates, omits, reorders, or mislabels a required item
- WHEN shard admission runs
- THEN the step MUST fail before device execution.

### Requirement: Both replicas execute the admitted full graph

r[tenstorrent.rwkv7_dual_p150a.graph] The runner MUST replicate the admitted frozen model, LoRA adapters, optimizer state, and trainable graph on both logical devices.

#### Scenario: Both devices execute model work

r[tenstorrent.rwkv7_dual_p150a.graph.valid]
- GIVEN a passing shard plan and collective receipt
- WHEN the dual graph runs
- THEN each device MUST execute its complete local forward and backward graph
- AND per-device evidence MUST bind active tokens, operations, memory, and graph identity.

#### Scenario: One replica lacks admitted work

r[tenstorrent.rwkv7_dual_p150a.graph.invalid]
- GIVEN a replica has no active data, misses an operation, uses a stale graph, or moves required work to the host
- WHEN graph evidence validation runs
- THEN the dual step MUST fail.

### Requirement: Adapter gradients synchronize before clipping and update

r[tenstorrent.rwkv7_dual_p150a.optimizer] The runner MUST reduce adapter gradients with the admitted mean operation before global norm, clipping, and equal optimizer updates on both replicas.

#### Scenario: Adapter gradients synchronize before update

r[tenstorrent.rwkv7_dual_p150a.optimizer.valid]
- GIVEN both replicas produce finite local adapter gradients
- WHEN the optimizer boundary runs
- THEN reduced gradients, global norm, clipping result, and adapter updates MUST satisfy the named reduction policy
- AND base tensors MUST remain unchanged on both replicas.

#### Scenario: Optimizer order or reduction is wrong

r[tenstorrent.rwkv7_dual_p150a.optimizer.invalid]
- GIVEN clipping occurs before reduction, a gradient is missing, reduction uses the wrong operation, a value is nonfinite, or a base tensor changes
- WHEN optimizer validation runs
- THEN the step MUST fail without a passing artifact.

### Requirement: Replica state remains equal

r[tenstorrent.rwkv7_dual_p150a.replicas] The runner MUST compare adapter tensors and required optimizer state across replicas at every admitted validation boundary.

#### Scenario: Replica state stays equal

r[tenstorrent.rwkv7_dual_p150a.replicas.valid]
- GIVEN both replicas start from equal admitted state and complete a synchronized optimizer boundary
- WHEN replica validation runs
- THEN all required adapter and optimizer values MUST satisfy the named equality policy.

#### Scenario: Replica state divergence blocks progress

r[tenstorrent.rwkv7_dual_p150a.replicas.invalid]
- GIVEN any required adapter or optimizer value is missing, stale, nonfinite, or divergent
- WHEN replica validation runs
- THEN the next step, canonical save, and passing receipt MUST remain blocked.

### Requirement: Dual results match the equivalent global-batch reference

r[tenstorrent.rwkv7_dual_p150a.parity] The dual runner MUST compare global loss, selected reduced adapter gradients, gradient norm, clipping result, and selected adapter deltas with the single-device reference over the identical global batch.

#### Scenario: Dual step matches the global-batch reference

r[tenstorrent.rwkv7_dual_p150a.parity.valid]
- GIVEN equal initial state, sample order, active labels, loss reduction, optimizer inputs, and seed
- WHEN single-device and dual-replica steps complete
- THEN required invariants MUST match
- AND all selected numerical values MUST satisfy named tolerances.

#### Scenario: Dual result differs from the reference

r[tenstorrent.rwkv7_dual_p150a.parity.invalid]
- GIVEN a required value is missing, wrongly reduced, wrongly shaped, nonfinite, stale, or outside tolerance
- WHEN parity validation runs
- THEN validation MUST fail with the exact value role and profile identity.

### Requirement: Bounded dual-card fine-tuning meets run and health criteria

r[tenstorrent.rwkv7_dual_p150a.run] The runner MUST execute the named bounded profile and MUST satisfy finite-value, loss, per-device work, collective, replica, memory, health, cleanup, and restoration criteria.

#### Scenario: Bounded dual-card run passes

r[tenstorrent.rwkv7_dual_p150a.run.valid]
- GIVEN parity and replica gates pass
- WHEN the named fixed-corpus run completes
- THEN every required step and device MUST satisfy the profile
- AND the configured loss-improvement criterion MUST pass.

#### Scenario: Unsafe dual-card run fails

r[tenstorrent.rwkv7_dual_p150a.run.invalid]
- GIVEN a device stalls, times out, diverges, overheats beyond policy, becomes unhealthy, loses a link, fails cleanup, or fails restoration
- WHEN run validation completes
- THEN the run MUST fail
- AND restoration MUST remain unclaimed without separate passing evidence.

### Requirement: Canonical checkpoints require equal replicas

r[tenstorrent.rwkv7_dual_p150a.checkpoint] The runner MUST validate replica equality before it saves one canonical adapter and optimizer state with physical mesh, logical mesh, shard, collective, runtime, graph, and step identities.

#### Scenario: Canonical checkpoint resumes both replicas

r[tenstorrent.rwkv7_dual_p150a.checkpoint.valid]
- GIVEN equal passing replica state and matching resume inputs
- WHEN canonical save, load, replication, and resume run
- THEN both loaded replicas MUST satisfy equality before the next step
- AND the resumed result MUST match the uninterrupted path under the named profile.

#### Scenario: Divergent or incompatible checkpoint is rejected

r[tenstorrent.rwkv7_dual_p150a.checkpoint.invalid]
- GIVEN replicas diverge or any adapter, optimizer, mesh, shard, collective, runtime, graph, data, or configuration identity differs
- WHEN save or resume admission runs
- THEN canonical save or resume MUST fail before further mutation.

### Requirement: Dual-card evidence is bounded and does not imply speedup

r[tenstorrent.rwkv7_dual_p150a.evidence] The runner MUST emit BLAKE3-bound topology, logical-mesh, shard, collective, placement, parity, replica, run, artifact, health, cleanup, restoration, and non-claim evidence.

#### Scenario: Evidence avoids a speedup claim

r[tenstorrent.rwkv7_dual_p150a.evidence.valid]
- GIVEN all dual-card acceptance gates pass
- WHEN evidence generation runs
- THEN the receipt MUST bind per-device work and timing observations
- AND it MUST not claim speedup, scaling efficiency, long-run stability, useful generalization, or support for other models.

### Requirement: Dual-P150A support has complete validation and documentation

r[tenstorrent.rwkv7_dual_p150a.verification] The change MUST include positive and negative mesh, shard, collective, placement, replica, parity, run, artifact, health, cleanup, restoration, and receipt fixtures plus bounded hardware evidence.

#### Scenario: Dual-P150A validation closes the change

r[tenstorrent.rwkv7_dual_p150a.verification.complete]
- GIVEN maintainers intend to sync and archive dual-P150A support
- WHEN required validation runs
- THEN all fixtures, hardware gates, prerequisite regressions, package checks, Cairn validation, and lifecycle gates MUST pass.

### Requirement: Dual-card documentation gives one exact operator path

r[tenstorrent.rwkv7_dual_p150a.documentation] The documentation MUST give exact cable, mesh, runtime, data, configuration, launch, evidence, artifact, cleanup, recovery, and claim-boundary procedures.

#### Scenario: Operator can fine-tune with two P150A cards

r[tenstorrent.rwkv7_dual_p150a.documentation.complete]
- GIVEN two supported connected P150A cards and admitted local SFT data
- WHEN an operator follows the documented command path
- THEN the operator MUST reach either a passing bounded dual-card receipt or an exact blocker receipt.
