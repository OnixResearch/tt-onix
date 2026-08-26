## Context

The execution code and runtime are unchanged when the batch receipt or `physically_admitted` value changes. However, both values currently live inside the same Nix derivation as the executable payload. A metadata-only promotion therefore changes the reported package store path.

## Decisions

### Decision: Bind evidence to a stable execution artifact

**Choice:** Add a dedicated Nix artifact whose store identity depends on every execution-bearing input but excludes receipt-binding metadata.

**Rationale:** The artifact changes when code, patches, runtime, Python dependencies, or execution policy change. It stays fixed when only the receipt hash and admission decision change.

### Decision: Keep admission metadata in a separate binding

**Choice:** Split the closed Nickel profile into immutable execution policy and a small receipt binding. The exported profile remains the runtime contract.

**Rationale:** Operators can promote a measured execution artifact without rebuilding its identity. Nickel still validates the complete merged profile.

### Decision: Preserve fail-closed receipt evaluation

**Choice:** Continue to require matched identities across all roots. Require the final deployed closure to reference the exact measured execution artifact and receipt hash.

**Rationale:** This removes the promotion cycle without transferring evidence across executable identities.

## Architecture

The pure Nickel policy and receipt evaluator own identity semantics and admission rules. Nix constructs the immutable identity artifact and deployment envelope. The systemd service and physical runner are imperative shells that execute the approved artifact.

## Risks / Trade-offs

- The deployment envelope store path can change after receipt promotion. This is expected and is not an execution identity.
- An omitted execution input could create false stability. Positive and negative checks must prove the declared input boundary.
- Normal production evidence remains admission-ineligible.
