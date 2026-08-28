## Why

Single-P150A support proves the model graph and numerical baseline. It does not prove that two connected P150A cards can train one consistent adapter.

The pinned 0.1B model fits one card. Data parallelism is the smallest useful dual-card strategy because it replicates the proven graph and synchronizes only training state.

Dual-card acceptance needs more than two visible devices. It needs deterministic batch sharding, current collective placement, equal adapter replicas, canonical checkpoints, and failure-safe cleanup.

## What Changes

- Add a logical two-replica data-parallel profile over the physical `p150_x2` `1x2` mesh.
- Replicate the frozen base model, LoRA adapters, and optimizer state on both devices.
- Shard each admitted global batch across the logical data axis.
- Synchronize adapter gradients with current TT collective operations before clipping and optimizer update.
- Compare dual-card loss, gradients, and adapter deltas with the admitted single-card global-batch reference.
- Add bounded multi-step, canonical checkpoint, save, load, and resume runs.
- Add per-device work, collective, health, cleanup, and restoration evidence.
- Add positive and negative device-free fixtures for shard, collective, replica, checkpoint, and receipt validation.

## Impact

- **Runtime**: both connected P150A cards under the `p150_x2` descriptor.
- **Parallelism**: replicated LoRA data parallelism with synchronized adapter gradients.
- **Artifacts**: one canonical adapter checkpoint after replica equality validation.
- **Evidence**: physical topology, logical mesh, shard, collective, parity, per-device work, health, and restoration receipts.
- **User path**: one documented dual-card fine-tuning command for local JSONL data.

## Out of Scope

- Model parallelism, pipeline parallelism, optimizer sharding, or full-parameter fine-tuning.
- Hosts with more or fewer than two selected P150A cards.
- Multi-host training, dynamic device membership, or automatic topology repair.
- A required speedup over one P150A.
- Support for other RWKV-7 sizes or checkpoints.
