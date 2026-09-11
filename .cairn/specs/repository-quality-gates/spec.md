# Repository Quality Gates Specification

## Purpose

Defines the `repository-quality-gates` capability.

## Requirements

### Requirement: One aggregate device-free gate

r[tenstorrent.quality.aggregate_gate] The repository MUST expose one Nix-owned aggregate gate for formatting, lint, tests, generated-data freshness, Cairn validation, and supported-system evaluation. Checked-in automation MUST run the same gate.

#### Scenario: Aggregate device-free gate passes

r[tenstorrent.quality.aggregate_gate.valid]
- GIVEN a repository state that satisfies every required device-free check
- WHEN a developer or automation runs the aggregate gate
- THEN every required check MUST pass through the pinned Nix tool closure
- AND the gate result MUST not depend on ambient developer tools.

#### Scenario: Gate tries to acquire hardware

r[tenstorrent.quality.aggregate_gate.invalid]
- GIVEN the aggregate device-free gate
- WHEN a check tries to open a Tenstorrent device, stop an owner, mutate firmware, or run an unbounded simulator workload
- THEN the gate MUST fail with the forbidden action.

### Requirement: Maintained language sources pass pinned gates

r[tenstorrent.quality.language_gates] Maintained Rust, shell, Python, Nix, and Nickel sources MUST pass their pinned format and lint gates. New suppressions MUST be narrow and justified at the diagnostic site.

#### Scenario: Maintained language sources pass

r[tenstorrent.quality.language_gates.valid]
- GIVEN maintained source files and the pinned quality tool closure
- WHEN language gates run
- THEN Rust formatting and Clippy, ShellCheck, Python formatting and lint, Nix formatting, and Nickel validation MUST pass.

#### Scenario: Language diagnostic fails the gate

r[tenstorrent.quality.language_gates.invalid]
- GIVEN a Clippy warning, unsafe shell expansion, missing interpreter, Python lint error, malformed Nix, or invalid Nickel contract
- WHEN the matching language gate runs
- THEN the aggregate gate MUST fail with the source path and diagnostic.

### Requirement: Parsers reject unsafe bounded input

r[tenstorrent.quality.parser_robustness] Every parser at an evidence or process boundary MUST have named byte, item, depth, and field limits where applicable. Pure parser cores MUST have positive, negative, property, and saved fuzz-seed tests.

#### Scenario: Bounded parser input is admitted

r[tenstorrent.quality.parser_robustness.valid]
- GIVEN valid input within every named limit
- WHEN the parser and admission core run
- THEN they MUST return the deterministic expected value
- AND round-trip or invariant properties MUST pass.

#### Scenario: Unsafe parser input is rejected

r[tenstorrent.quality.parser_robustness.invalid]
- GIVEN oversized, over-deep, over-count, malformed, truncated, duplicate, non-finite, or type-invalid input
- WHEN parsing starts
- THEN the parser MUST reject it with a stable diagnostic
- AND it MUST not panic or perform an unbounded allocation.

### Requirement: Repository artifacts follow typed policy

r[tenstorrent.quality.artifact_policy] The repository MUST author artifact classes and bounds in typed Nickel. Generated source, portable evidence, private logs, and external large objects MUST follow distinct location, identity, privacy, and Git rules.

#### Scenario: Approved artifact is admitted

r[tenstorrent.quality.artifact_policy.valid]
- GIVEN an artifact with an allowed class, root, size, BLAKE3 identity, privacy state, and freshness result
- WHEN artifact policy validation runs
- THEN validation MUST admit the artifact and report its class.

#### Scenario: Unapproved artifact is rejected

r[tenstorrent.quality.artifact_policy.invalid]
- GIVEN stale generated data, a private log in portable evidence, missing identity, unapproved large file, or generated output in an unapproved root
- WHEN artifact policy validation runs
- THEN validation MUST fail with the exact path and violated rule.

### Requirement: Quality-gate behavior has complete verification

r[tenstorrent.quality.verification] The quality-gate change MUST include positive and negative fixtures for every language gate, parser family, artifact class, aggregate result, and hardware-access prohibition.

#### Scenario: Quality-gate change is ready

r[tenstorrent.quality.verification.complete]
- GIVEN maintainers intend to sync and archive the quality-gate change
- WHEN all focused and aggregate validation runs
- THEN each required positive and negative fixture MUST pass
- AND the aggregate gate MUST produce the same result for the same repository state.
