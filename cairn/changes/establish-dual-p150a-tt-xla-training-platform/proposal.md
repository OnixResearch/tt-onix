## Why

The flake packages TT-Metal `v0.74.0-rc6`, device drivers, and operations tools. It does not package TT-XLA or TT-Blacksmith.

The upstream TT-Blacksmith quick start uses a mutable `latest` container image. That path cannot give this repository a reproducible training runtime.

Two P150A cards also need an explicit `p150_x2` mesh descriptor and a validated fabric. Model work cannot start safely before these platform facts are recorded.

## What Changes

- Add one immutable compatibility manifest for TT-XLA, TT-Forge, TT-MLIR, TT-Metal, PyTorch/XLA, PyTorch, TT-Blacksmith, drivers, and firmware.
- Add a Nix-owned launcher for an exact upstream runtime image digest and exact TT-Blacksmith source revision.
- Add typed Nickel configuration for runtime selection, device policy, hugepages, cache paths, mesh identity, and evidence output.
- Add a dual-P150A preflight that validates two Blackhole P150-class devices, firmware, driver compatibility, the `1x2` mesh, and fabric links.
- Add positive and negative device-free fixtures for manifest, configuration, launch-plan, and preflight receipt validation.
- Add a bounded hardware smoke run that opens and closes the `p150_x2` mesh without model execution.

## Impact

- **Packages**: new TT-XLA and TT-Blacksmith source artifacts, a runtime launcher, and validation tools.
- **NixOS**: documented `hardware.tenstorrent.meshName = "p150_x2"` training profile and hugepage requirements.
- **Configuration**: typed Nickel source with generated runtime data in the format that each upstream tool requires.
- **Hardware**: one bounded smoke run on two connected P150A cards.
- **Validation**: deterministic positive and negative fixtures plus a hardware receipt with explicit non-claims.

## Out of Scope

- RWKV-7 model code, gradients, LoRA adapters, datasets, or optimizer steps.
- A native Nix build of the complete TT-XLA compiler stack in the first slice.
- Firmware mutation, topology flashing, or cable installation.
- Performance, training correctness, or model convergence claims.
