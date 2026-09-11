# Tasks: Dual-P150A RWKV-7 LoRA fine-tuning

## Phase 1: Mesh and collective baseline

- [x] [serial] D1 Validate current platform, CPU, and single-P150A receipts with matching model and runtime identities. r[tenstorrent.rwkv7_dual_p150a.authority]
- [x] [serial] D2 Add typed Nickel profiles for physical mesh, logical data axis, shard policy, reduction policy, replica validation, run criteria, limits, timeouts, and artifacts. r[tenstorrent.rwkv7_dual_p150a.configuration]
- [x] [depends:tenstorrent.rwkv7_dual_p150a.configuration] D3 Add pure global-batch shard planning with exact sample and active-label coverage. r[tenstorrent.rwkv7_dual_p150a.sharding]
- [x] [depends:tenstorrent.rwkv7_dual_p150a.configuration] D4 Compile and run focused logical-mesh, all-reduce, barrier, placement, timeout, and output probes. r[tenstorrent.rwkv7_dual_p150a.collectives]

## Phase 2: Data-parallel training

- [x] [depends:tenstorrent.rwkv7_dual_p150a.collectives] D5 Replicate the admitted frozen model, LoRA adapters, optimizer state, and single-device graph over both logical replicas. r[tenstorrent.rwkv7_dual_p150a.graph]
- [x] [depends:tenstorrent.rwkv7_dual_p150a.sharding] D6 Shard admitted global batches across the logical data axis and reduce loss over active labels. r[tenstorrent.rwkv7_dual_p150a.sharding]
- [x] [depends:tenstorrent.rwkv7_dual_p150a.graph] D7 Synchronize adapter gradients with the admitted mean reduction before norm, clipping, and optimizer update. r[tenstorrent.rwkv7_dual_p150a.optimizer]
- [x] [depends:tenstorrent.rwkv7_dual_p150a.optimizer] D8 Validate replica adapter and optimizer equality at each admitted boundary. r[tenstorrent.rwkv7_dual_p150a.replicas]
- [x] [depends:tenstorrent.rwkv7_dual_p150a.replicas] D9 Compare one dual-replica step with the equivalent single-device global-batch reference. r[tenstorrent.rwkv7_dual_p150a.parity]

## Phase 3: Bounded run, artifacts, and evidence

- [x] [depends:tenstorrent.rwkv7_dual_p150a.parity] D10 Run the named bounded dual-card profile and validate loss, finite values, per-device work, collectives, memory, health, cleanup, and restoration. r[tenstorrent.rwkv7_dual_p150a.run]
- [x] [depends:tenstorrent.rwkv7_dual_p150a.run] D11 Save one canonical adapter after replica equality, then load and resume both replicas. r[tenstorrent.rwkv7_dual_p150a.checkpoint]
- [x] [parallel] V1 Add positive fixtures for mesh admission, shard coverage, collectives, placement, replica equality, parity, run criteria, artifacts, health, and restoration. r[tenstorrent.rwkv7_dual_p150a.verification]
- [x] [parallel] V2 Add negative fixtures for device mismatch, wrong mesh, missing links, invalid axes, bad shards, wrong reductions, fallback, timeout, divergence, nonfinite values, corrupt artifacts, health faults, and failed restoration. r[tenstorrent.rwkv7_dual_p150a.verification]
- [x] [depends:tenstorrent.rwkv7_dual_p150a.checkpoint] V3 Produce the bounded dual-P150A fine-tuning receipt and canonical adapter artifact. r[tenstorrent.rwkv7_dual_p150a.evidence]
- [x] [serial] D12 Document cable and mesh prerequisites, exact fine-tuning command, data contract, evidence, recovery, artifact use, and non-claims. r[tenstorrent.rwkv7_dual_p150a.documentation]
- [x] [depends:tenstorrent.rwkv7_dual_p150a.verification] V4 Run formatting, focused tests, package checks, all prerequisite regressions, Cairn validation, and all lifecycle gates. r[tenstorrent.rwkv7_dual_p150a.verification]

  V4 evidence: `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` pass on the dual core; `checks.x86_64-linux.rwkv7-lora-dual-p150a`, `rwkv7-lora-p150a`, `training-platform`, and `rwkv7-lora-reference` pass; `nix flake check` passes every attribute except the pre-existing `model-target-candidate-report` failure (missing `upstream_identity` reviewed fact), which reproduces on `origin/main` and belongs to the active `add-qwen3-6-a3b-candidate-target` change. Cairn validate and all three gates pass.

## Verification Coverage

- `Scenario: Current prerequisite chain is admitted` -> D1, V1
- `Scenario: Stale prerequisite chain blocks dual launch` -> D1, V2
- `Scenario: Logical two-replica mesh passes collective probes` -> D4, V1
- `Scenario: Collective or placement evidence fails` -> D4, V2
- `Scenario: Global batch shards exactly once` -> D3, D6, V1
- `Scenario: Invalid shard plan is rejected` -> D3, V2
- `Scenario: Adapter gradients synchronize before update` -> D7, V1
- `Scenario: Replica state divergence blocks progress` -> D8, V2
- `Scenario: Dual step matches the global-batch reference` -> D9, V3
- `Scenario: Bounded dual-card run passes` -> D10, V3
- `Scenario: Canonical checkpoint resumes both replicas` -> D11, V1
- `Scenario: Unsafe cleanup blocks success` -> D10, V2
- `Scenario: Evidence avoids a speedup claim` -> V3, D12
