## Context

`nix flake check --all-systems --no-build` and the current focused checks evaluate successfully. Treefmt formats Nix only. It does not run Rust, shell, or Python quality tools.

Clippy with warnings denied reports two layer-harness functions with excessive argument counts. ShellCheck reports scripts without interpreter declarations and two runners that expand optional argument strings through word splitting.

Evidence parsers handle JSON, YAML, CSV, binary frames, vocabulary text, and receipts. Some enforce byte limits. Others rely on caller behavior or unbounded deserialization.

Large generated and evidence files mix with source-controlled data. The untracked generated directory is large, and no typed policy states which artifacts belong in Git.

## Decisions

### Decision: One device-free quality entry point

**Choice:** Add one flake check that composes formatting, lint, unit, property, freshness, lifecycle, and supported-system evaluation checks. Checked-in automation runs this entry point.

**Rationale:** Developers and automation need the same command and the same result.

### Decision: Language gates use package-owned tools

**Choice:** Run Rust formatting and Clippy through the pinned Nix toolchain. Run ShellCheck and Python format and lint tools through Nix. Do not depend on ambient developer installations.

**Rationale:** A quality gate is reproducible only when its tools are part of the closure.

### Decision: Fix warnings through structure

**Choice:** Replace excessive function arguments with typed context records. Replace optional shell argument strings with arrays. Add explicit interpreters through `writeShellApplication` or generated shebangs.

**Rationale:** Blanket lint allowances would preserve the design problems that the warnings found.

### Decision: Parser limits are shared policy

**Choice:** Author named parser byte, item, depth, and field limits in typed Nickel. Export checked-in generated data for pure Rust consumers. Reject oversized input before allocation or parsing.

**Rationale:** Evidence inputs cross trust boundaries. Consistent limits reduce denial-of-service and parser divergence risks.

### Decision: Property and fuzz tests stay device-free

**Choice:** Add deterministic property tests for round trips and rejection invariants. Keep a bounded corpus of malformed fuzz seeds for each parser family. Fuzz discovery runs separately, while checked-in seeds run in the normal gate.

**Rationale:** The normal gate must be deterministic and fast. Saved seeds preserve discovered failures without runtime entropy.

### Decision: Artifact policy is typed and enforced

**Choice:** Define artifact classes in Nickel: source, generated source, portable evidence, private operational log, and external large object. Each class has named size, location, identity, and Git rules.

**Rationale:** File size alone does not decide whether an artifact belongs in the repository. Type and evidence role do.

### Decision: Hardware remains explicit

**Choice:** The aggregate gate evaluates and tests device-free behavior only. Hardware and long simulator checks remain named, operator-run gates with saved receipts.

**Rationale:** General automation must not acquire devices or overstate hardware coverage.

## Validation Strategy

Positive checks will prove that all supported files are discovered, each language tool runs, parsers accept valid bounded data, generated policy is fresh, and allowed artifacts satisfy policy.

Negative checks will include a Clippy regression, missing shebang, unsafe shell expansion, malformed parser inputs, oversize inputs, stale generated data, private logs in portable evidence, and unapproved large files.

The aggregate check will run twice against one repository state and compare its deterministic outputs where applicable.

## Risks / Trade-offs

- The first gate can expose existing warnings outside the reviewed findings.
- Property tests increase device-free test time.
- Artifact limits need explicit updates when evidence formats grow.
- Full Python type checking is deferred until the repository defines one compatible type policy.
