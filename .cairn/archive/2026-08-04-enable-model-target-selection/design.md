## Context

The current flake builds one fixed RWKV-7 target. `rwkv-layer-harness` fetches its checkpoint and tokenizer assets.

`flake-modules/onix-packages.nix` passes those assets directly to `rwkv7-p150x2-runtime` and its evidence package. The C++ runtime repeats the model identity, dimensions, and checkpoint BLAKE3.

The Rust classifier repeats the same identity. It also embeds RWKV recurrent-state sizes, operation names, tokenizer limits, and parity rules.

This layout is safe for one pinned target, but it is not a model replacement interface. A Qwen-family model can use transformer attention, KV cache, RoPE, checkpoint shards, and different tokenizer assets.

A filename or checkpoint selector cannot represent those differences. The repository needs a small shared target contract and explicit architecture adapters.

## Decisions

### Decision: Author model profiles in typed Nickel

**Choice:** Add typed Nickel source for model profiles. Each profile binds a target name, model ID, revision, architecture family, precision, state kind, and resource facts.

Each profile also binds artifact roles, Nix hashes, BLAKE3 identities, and tokenizer assets. Nix will use SHA-256 where its fixed-output contract requires SHA-256.

A deterministic exporter will write checked-in canonical JSON. Freshness validation will reject generated data that differs from the Nickel source.

Nix evaluation and runtime packages will consume the canonical data. They will not run Nickel during model execution.

**Rationale:** One typed source removes repeated model identity data. Canonical output gives Nix, C++, and Rust one reviewed authority.

### Decision: Compose immutable model targets in pure Nix

**Choice:** Add a pure `mkModelTarget` constructor. It takes a model profile, fixed-output artifacts, an architecture adapter, an evidence adapter, and a topology profile.

The constructor returns exact runtime, command, evidence, source-boundary, and closure outputs. It rejects incompatible inputs during Nix evaluation.

The flake will expose each target under an exact attribute name. The repository will not have a mutable global current-model value.

Existing `rwkv7-p150x2-runtime` and `rwkv7-p150x2-evidence` attributes will remain compatibility aliases to the migrated target.

**Rationale:** Immutable package selection keeps builds reproducible. It also prevents an environment variable or local path from changing the model behind an evidence receipt.

### Decision: Keep architecture behavior inside explicit adapters

**Choice:** Each adapter declares its supported architecture families, state kinds, checkpoint layouts, required operations, and topology limits.

The adapter owns model equations, tensor names, tensor layouts, state lifecycle, reference generation, tolerances, and model-specific evidence rules.

The common target layer will not define a universal forward-pass interface. A second-family device-free adapter fixture will use transformer attention and KV-cache semantics.

**Rationale:** RWKV recurrent state and transformer KV cache are not interchangeable. A narrow adapter boundary prevents false portability while removing package-wiring duplication.

### Decision: Split common evidence from family evidence

**Choice:** Add a pure common evidence core for model identity, artifact identity, topology, process policy, profiler attribution, CPU fallback, and publication state.

Each architecture adapter will validate its typed family payload. The selected classifier must pass both parts before it publishes generated text.

The command shell can execute only the runtime in the selected target closure. It will reject a receipt from another profile or evidence adapter.

**Rationale:** Safety and attribution rules apply to all models. Numerical parity, state carry, operation inventory, and control cases remain model-specific.

### Decision: Migrate RWKV before onboarding a real second model

**Choice:** Move the current RWKV model and tokenizer pins into the first model profile. Keep its revision, artifacts, mesh, CLI, evidence semantics, and closure policy unchanged.

Use the current RWKV checks as compatibility tests. Use a synthetic transformer target only for device-free constructor and adapter tests.

**Rationale:** The migration proves that the boundary can represent the current authority. The synthetic target proves that shared code does not assume RWKV state.

### Decision: Add a fail-closed candidate report

**Choice:** Add a deterministic device-free report with a pure decision core and a thin file-reading shell.

The report covers exact upstream identity, revision, license, artifact pins, tokenizer support, state kind, required operators, memory bounds, reference authority, evidence adapter, and topology compatibility.

A complete report means that a candidate is eligible for a bounded simulator change. It does not mean that the model builds, fits, or runs.

Treat `Qwen3.6` as an illustrative candidate name until an exact reviewed upstream release exists. If identity or adapter facts are missing, report `unsupported` and expose no runnable target.

**Rationale:** Candidate research needs a useful negative result. The repository must not fabricate a pin or convert incomplete metadata into model support.

### Decision: Reject a universal runtime and package cloning

**Choice:** Do not add a dynamic graph loader, arbitrary checkpoint path, or runtime plugin system. Do not copy the complete RWKV package tree for each model.

**Rationale:** A universal runtime is larger than this change and can hide unsupported operations. Package cloning preserves the current coupling and duplicates safety logic.

## Validation Strategy

Device-free positive tests will cover the migrated RWKV target and a synthetic transformer target. Both targets must use the same constructor and common evidence envelope.

Negative tests will cover missing artifact hashes, generated-data drift, architecture mismatch, state mismatch, topology rejection, wrong evidence adapters, and arbitrary checkpoint overrides.

Existing RWKV package, source-boundary, closure, parity, profiler-policy, and retained-evidence checks must keep their current results.

No validation in this change can prove physical placement, hardware capacity, model performance, or successful Qwen execution.

## Risks / Trade-offs

- A broad common contract can hide architecture differences. The design keeps equations and state inside adapters.
- Generated profile data can drift. Required freshness validation fails when source and generated data differ.
- A factory can appear portable without a second path. The synthetic transformer target tests a different state kind and operation set.
- More target outputs can increase Nix evaluation work. The flake will construct only registered targets and keep artifacts lazy.
- RWKV migration can change old evidence by accident. Compatibility aliases and existing fixtures must fail on any identity or semantic drift.

## Non-Goals

- Implement Qwen equations or TT-NN operators.
- Assert that a model named Qwen3.6 exists or has a stable upstream release.
- Select arbitrary local checkpoints at runtime.
- Replace package-matched TT-Metal or topology safety rules.
- Claim simulator or silicon success for a new model.
