## Why

Typed Nickel profiles already own canonical model and artifact identities. Runtime and evidence consumers still repeat model revisions, dimensions, topology values, tolerances, and hashes in Rust, Python, C++, Nix, fixtures, and documentation.

This duplication creates silent drift. One observed example accepts uppercase hexadecimal hashes in Rust while the Nickel contract accepts lowercase hashes only. A model update can therefore pass one boundary and fail another.

## What Changes

- Define the complete consumer identity projection in the typed model profile.
- Generate deterministic Rust, Python, C++, and Nix identity data from the canonical profile.
- Replace hand-written consumer identity constants with generated modules or data includes.
- Enforce one lowercase BLAKE3 and artifact-hash contract in every consumer.
- Add freshness checks and positive and negative cross-language fixtures.
- Keep runtime algorithms and architecture-specific equations in their existing adapters.

## Impact

- **Profiles**: `model-targets/profiles.ncl`, contracts, generated profile data, and freshness checks.
- **Consumers**: RWKV runtime, evidence, LoRA, harness, command, and model-target report packages.
- **Validation**: cross-language identity parity, stale generated output, uppercase hash, missing field, and wrong topology fixtures.
- **Documentation**: one update procedure for model identity changes.

## Out of Scope

- A new model target or model pin update.
- Generated algorithm implementations or generated numerical kernels.
- Changes to current RWKV tolerances without separate numerical evidence.
