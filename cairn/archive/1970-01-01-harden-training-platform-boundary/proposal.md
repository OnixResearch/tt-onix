## Why

The training tools claim one Nix-owned runtime and one fail-closed launch boundary. The current launcher accepts caller-controlled manifests and configuration files without matching their reviewed identities. The admission core also accepts nested values that violate the Nickel contracts.

The container changes its Python environment from package indexes during each hardware run. The portable receipt then binds raw logs and device identifiers that its evidence contract excludes. Some early and late failures exit without a terminal receipt.

The NixOS example also installs TT-KMD 2.7.0 while the admitted training manifest requires 2.10.1-pre. These gaps make a passing receipt weaker than its documented claim.

## What Changes

- Bind production launch commands to the Nix-owned manifest, configuration, descriptor, driver, and wheel identities.
- Make Rust admission enforce every generated Nickel contract, including closed nested records and exact schema versions.
- Require preflight in the same launch process before Docker receives any device.
- Replace runtime package-index access with a complete hash-locked offline wheel closure.
- Emit one validated terminal receipt for success and every failure stage.
- Sanitize portable evidence before hashing it. Keep raw logs and device identifiers outside the portable bundle.
- Align the NixOS module driver package, descriptor checks, hugepage policy, and public example export with the admitted platform.

## Impact

- **Packages**: `training-platform-core`, `training-platform-tools`, the single-card LoRA tools, and the dual-card LoRA tools.
- **NixOS**: `nixos/module.nix`, `nixos/examples/dual-p150a-training.nix`, and flake module exports.
- **Evidence**: the platform receipt schema and checked-in platform evidence need a sanitized replacement receipt.
- **Testing**: add device-free bypass fixtures, shell tests, NixOS assertions, offline-install checks, and one bounded dual-P150A smoke.

## Out of Scope

- Training correctness, convergence, speedup, thermal endurance, or broader model support.
- Firmware mutation or an automatic firmware update path.
- A general container runtime abstraction beyond the admitted Docker path.
