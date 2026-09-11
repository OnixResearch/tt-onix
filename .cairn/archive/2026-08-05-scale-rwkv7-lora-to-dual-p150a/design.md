# Design: Dual-P150A RWKV-7 LoRA fine-tuning

## Context

The physical P150-X2 descriptor defines a Blackhole `1x2` mesh. Each P150A provides 32 GB of GDDR6 and four fabric ports.

The single-device Cairn proves the complete LoRA training graph on one logical P150A. The 0.1B model does not need model sharding to fit.

This change scales the proven graph with data parallelism. It does not redesign WKV7 or divide model layers across devices.

## Completion Contract

A valid dual-card result has all of these facts:

- The physical `p150_x2` preflight passes with both cards idle and healthy.
- The logical mesh has two data replicas over the two physical devices.
- Each global batch partitions deterministically with no missing or duplicate sample.
- Both devices execute the complete model graph.
- Adapter gradients synchronize through current TT collective placement.
- Both replicas apply equal clipping and optimizer updates.
- Replica adapters and optimizer state match at every admitted validation boundary.
- The dual-card result matches the single-card global-batch reference within named tolerances.
- A bounded run satisfies its loss and health criteria.
- Canonical save, load, resume, cleanup, and restoration pass.

Two devices in `tt-smi` output are not completion evidence.

## Decisions

### Decision: Use data parallelism for the first two-card profile

**Choice:** Replicate the full frozen base model, LoRA adapters, and optimizer state. Shard the batch across a logical data axis of size two.

**Rationale:** The model fits one card. Data parallelism reuses the single-device graph and limits new numerical risk to sharding and collectives.

Model parallelism, optimizer sharding, and pipeline parallelism remain separate future decisions.

### Decision: Keep physical and logical mesh identities separate

**Choice:** Bind the physical `p150_x2` `1x2` descriptor and the logical two-replica data mesh as separate receipt fields.

**Rationale:** A compiler can map logical axes to physical dimensions. Separate identities prevent a logical configuration from becoming a physical cable claim.

The selected TT-XLA bundle must prove the admitted logical-axis orientation through a focused mesh probe.

### Decision: Partition batches with a pure plan

**Choice:** Compute shard ownership from the global sample order, active token counts, padding policy, replica count, and accumulation step.

**Rationale:** A pure plan can prove that each sample appears exactly once before device access.

The plan rejects a global batch that cannot satisfy the named even-shard policy. Later profiles can add explicit uneven-shard semantics.

### Decision: Synchronize gradients before clipping

**Choice:** Each replica computes local adapter gradients. The runtime reduces gradients with the configured mean operation before global norm, clipping, and optimizer update.

**Rationale:** This order matches the single-card mean-loss reference over the same global batch.

Loss reporting also uses the admitted weighted reduction over active labels. Padding cannot change the global mean.

### Decision: Validate collectives before the model run

**Choice:** Run focused all-reduce and synchronization probes on the selected logical mesh. Bind operation, topology, dtype, shape, placement, timeout, and output evidence.

**Rationale:** A model hang does not identify a collective fault. A focused probe gives the smallest failure boundary.

Missing, stale, host-placed, timed-out, or numerically wrong collective evidence blocks the full run.

### Decision: Require replica equality at optimizer boundaries

**Choice:** Compare adapter tensors and required optimizer state after each admitted validation boundary. Use exact equality where the runtime guarantees identical replicas.

Use named numerical tolerances only when the selected runtime documents nondeterministic reduction order.

**Rationale:** A falling aggregate loss can hide divergent replicas. Replica evidence must stand on its own.

### Decision: Compare equivalent global batches

**Choice:** Compare a dual-replica step with a single-device reference that uses the same sample order, active labels, mean-loss semantics, initial adapters, optimizer inputs, and seed.

**Rationale:** Different global batches cannot establish data-parallel equivalence.

The comparison covers global loss, selected reduced gradients, gradient norm, clipping result, and selected adapter deltas.

### Decision: Save one canonical adapter

**Choice:** Validate replica equality first. Then save one adapter and one optimizer state with physical mesh, logical mesh, shard plan, collective, runtime, and step identities.

**Rationale:** Duplicate adapter files add ambiguity without adding recovery value.

Resume reloads equivalent state on both replicas and validates equality before the next step.

### Decision: Measure work without requiring speedup

**Choice:** Record per-device operation counts, active tokens, step times, collective times, memory high-water marks, and health observations.

**Rationale:** Both devices must do model work. The first support slice does not know a defensible speedup threshold.

A later optimization Cairn can define performance budgets from measured baselines.

### Decision: Fail closed and restore ownership

**Choice:** A missing device, missing link, collective timeout, divergent replica, nonfinite value, health failure, close failure, or restoration failure blocks success.

**Rationale:** Partial two-card success cannot produce a trustworthy adapter.

The shell stops new work, records the failing stage, closes resources when safe, and validates restoration. It does not reset or flash devices automatically.

## Test Design

Positive fixtures cover physical and logical mesh admission, even shard plans, complete sample coverage, collective mean results, current placement, equal replicas, global-batch parity, bounded loss improvement, canonical save, load, resume, cleanup, and restoration.

Negative fixtures cover missing or extra devices, wrong descriptors, missing links, invalid logical axes, uneven batches, duplicate samples, omitted samples, wrong reductions, host fallback, collective timeout, replica divergence, nonfinite values, stale graphs, corrupt checkpoints, health faults, and failed restoration.

The hardware acceptance run first executes collective probes. It then runs one parity step and the named bounded fine-tuning profile.

## Adversarial Audit

The validation must try to falsify these false completions:

- One device did all model work.
- Both devices ran, but gradients never synchronized.
- Loss used an unweighted mean over padding.
- Shard plans duplicated or omitted records.
- Adapters matched, but optimizer states diverged.
- A stale single-card graph supplied dual-card placement evidence.
- Canonical save selected a divergent replica.
- A speedup claim used compile time or cache effects.
- Cleanup claimed restoration after a fabric or close failure.

## Risks and Trade-offs

- Data parallel overhead can exceed compute time for the 0.1B model. Support does not require speedup.
- TT-XLA can restrict logical mesh orientation. A focused mesh probe owns that decision.
- Collective reduction order can widen BF16 differences. Named profiles keep that difference explicit.
- Exact optimizer-state replication can cost validation time. Bounded acceptance profiles prioritize correctness.

## Claim Boundary

A passing receipt proves bounded two-replica RWKV-7 LoRA fine-tuning on one exact dual-P150A platform and runtime. It does not prove speedup, scaling efficiency, long-run stability, useful generalization, or other model support.
