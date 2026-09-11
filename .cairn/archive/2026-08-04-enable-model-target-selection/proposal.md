## Why

The current model boundary is not replaceable. `rwkv-layer-harness` owns the checkpoint and tokenizer.

The runtime and classifier repeat RWKV-specific identities, dimensions, state rules, and operation names. Replacing only the checkpoint with a Qwen model will fail.

The repository needs a reproducible model-target boundary. A new model family must not weaken existing RWKV evidence or hardware safety rules.

## What Changes

- Add typed Nickel model profiles with canonical generated data and fixed artifact identities.
- Add a pure Nix target constructor that binds one profile, architecture adapter, evidence adapter, and topology.
- Keep model equations, tensor layouts, state rules, and reference logic inside architecture-specific adapters.
- Split common runtime evidence from model-family evidence without adding dynamic runtime plugins.
- Move the existing RWKV target through the new boundary and preserve its current flake outputs and command names.
- Add device-free positive and negative fixtures for a second architecture family and incompatible target combinations.
- Add a deterministic candidate report for proposed models such as Qwen3.6.
- Do not pin an unverified Qwen release or claim that it fits or runs on the P150x2 topology.

## Impact

- **Files**: model profiles, target constructors, RWKV runtime wiring, evidence packages, flake outputs, checks, documentation, and Cairn lifecycle files.
- **Testing**: profile export, Nix evaluation, adapter mismatch, evidence mismatch, RWKV compatibility, closure, formatting, Cairn gates, and flake checks.
- **Hardware**: This change does not open a device or claim new model execution, placement, capacity, or performance.
