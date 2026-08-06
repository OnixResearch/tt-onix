## Context

The package exports a canonical Nickel manifest and configuration. Production commands still accept arbitrary files at runtime. The Rust core checks selected fields, but it does not enforce all nested Nickel fields.

A review fixture with schema version 999, board class `n300`, zero fabric channels, an empty exclusion list, and a relative cache path received an admitted result. A second fixture selected an unreviewed image digest and also produced a launch plan.

Hardware container scripts upgrade Pip and resolve JAX, Safetensors, and transitive dependencies from live indexes. Pip documents that `--extra-index-url` has no source priority and can cause dependency confusion.

The checked-in receipt binds preflight and smoke artifacts. Those artifacts contain raw runtime logs and stable board identifiers. This conflicts with the accepted portable-evidence requirement.

### Recorded device-free baseline

The 2026-08-06 baseline used the unmodified `origin/main` source in a dedicated
worktree. The existing training-platform Nix check and Rust package build
passed. Direct replay then showed both boundary failures:

- `validate-config` admitted schema `999`, board class `n300`, zero fabric
  channels, a relative cache path, and an unknown nested device field.
- `launch-plan` admitted a caller-owned manifest that selected
  `ghcr.io/unreviewed/image@sha256:0000000000000000000000000000000000000000000000000000000000000000`.
- ShellCheck reported missing shell declarations, unsafe optional argument
  expansion, and one unused `REPLICAS` assignment.

No Docker, simulator, or hardware workload was part of this baseline.

## Decisions

### Decision: Production launch uses package-owned authority

**Choice:** Embed the expected manifest, configuration, descriptor, and wheel-set BLAKE3 identities in the production closure. Reject caller input that does not match those identities. Keep arbitrary-input validation available only as a device-free diagnostic command.

**Rationale:** An immutable but unreviewed digest is not the selected compatibility bundle. The production launcher needs an allowlist, not only shape validation.

### Decision: Generate one admission policy from Nickel

**Choice:** Export a closed, versioned policy projection from Nickel. The pure Rust core consumes that checked-in projection and validates every root and nested field. It rejects unknown fields, wrong package names, wrong ABIs, invalid dates, unsafe paths, and unsupported values.

**Rationale:** Separate hand-written validators have already drifted. One generated policy keeps the functional core and typed source aligned.

### Decision: Preflight and launch form one operation

**Choice:** Run device, driver, firmware, hugepage, descriptor, mesh, fabric, health, and ownership checks in the launcher process immediately before Docker starts. Bind the preflight result to the launch plan and terminal receipt.

**Rationale:** A reusable receipt cannot prove current ownership. One process reduces the time-of-check and time-of-use gap.

### Decision: Hardware runs use an offline Python closure

**Choice:** Fetch every required wheel in Nix with an exact hash. Mount one read-only wheelhouse and install with `--no-index`, `--find-links`, and `--require-hashes`. Do not upgrade Pip during a run.

**Rationale:** Version-only network resolution can change the executed environment after review. An offline closure makes the environment reproducible and available without network access.

### Decision: Evidence is sanitized before publication

**Choice:** Keep private operational logs in a separate local directory. Build portable artifacts from typed aggregate facts only. Redact stable board identifiers before hashing or checking in an artifact.

**Rationale:** A receipt that hashes a private artifact still binds and publishes that private content.

### Decision: Every exit produces a validated terminal receipt

**Choice:** Use one shell trap and a pure stage-state classifier. Mark stages as `pass`, `fail`, or `not_observed`. Validate success and failure receipts before returning the workload status.

**Rationale:** Early preflight and late health failures need the same deterministic evidence boundary as workload failures.

### Decision: The NixOS module owns compatible host setup

**Choice:** Add a configurable driver package with a training-profile assertion. Verify the descriptor exists, configure named hugepages, export the documented example, and provide a group-restricted device-access mode.

**Rationale:** An example that evaluates but cannot satisfy runtime admission is not a complete host configuration.

## Validation Strategy

Device-free tests will replay the two observed bypass classes and require rejection. Additional negative fixtures will cover every nested contract field, duplicate board identities, stale authority identities, network-dependent installs, unsafe artifact content, and missing terminal stages.

Positive tests will cover canonical policy export, exact launch authority, offline wheel installation, sanitized evidence, success receipts, failure receipts, and NixOS evaluation.

The completed device-free run passed the focused platform check with 40 Rust
tests. It also passed the NixOS example, both LoRA checks, the TT-KMD build,
Clippy with warnings denied, full ShellCheck, formatting, and all-system flake
evaluation. Cairn validation and all three change gates passed.

A bounded hardware attempt on 2026-08-06 passed `health_before` and failed
closed at `preflight`. The host reported TT-KMD `2.10.1-pre`, but the reviewed
authority requires `2.10.0`. The terminal failure receipt validated, and Docker
did not start. The attempt also exposed an empty portable preflight file. The
launcher now keeps rejected preflight output private, validates each portable
artifact, and requires a fresh run directory under the authorized evidence
parent. The flake exports exact core and smoke apps plus a development shell.
The shell clears ambient runtime overrides but does not manage the kernel.

A successful bounded dual-P150A smoke must replace the stale platform receipt
after the host installs TT-KMD `2.10.0`. The receipt will keep all training and
performance non-claims.

## Risks / Trade-offs

- The offline wheel closure increases Nix store size.
- A driver update requires package-matched TT-Metal, TT-UMD, firmware, and hardware validation.
- Sanitized evidence removes some debugging detail from the portable bundle. Private logs remain local.
- Group-restricted access differs from the permissive upstream udev rule. The module will keep an explicit compatibility option.
