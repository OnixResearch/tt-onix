# Tasks: Add repository quality gates

## Phase 1: Baseline and tool ownership

- [x] [serial] B1 Record current flake checks, focused tests, Clippy diagnostics, ShellCheck diagnostics, parser coverage, repository sizes, and Cairn validation. r[tenstorrent.quality.aggregate_gate]
- [x] [serial] Q1 Add pinned Rust, shell, and Python quality tools to Nix checks. r[tenstorrent.quality.language_gates]
- [x] [depends:tenstorrent.quality.language_gates] Q2 Add one aggregate device-free flake check and a checked-in automation runner. r[tenstorrent.quality.aggregate_gate]
- [x] [depends:tenstorrent.quality.aggregate_gate] Q3 Make the aggregate gate reject device access and document the separate hardware gates. r[tenstorrent.quality.aggregate_gate]

## Phase 2: Rust, shell, and Python cleanup

- [x] [depends:tenstorrent.quality.language_gates] R1 Replace the two excessive layer-harness argument lists with typed context records and add positive and negative core tests. r[tenstorrent.quality.language_gates]
- [x] [depends:tenstorrent.quality.language_gates] S1 Add explicit interpreters to maintained scripts and package them through `writeShellApplication` where applicable. r[tenstorrent.quality.language_gates]
- [x] [depends:tenstorrent.quality.language_gates] S2 Replace optional argument strings with arrays and resolve every actionable ShellCheck diagnostic without blanket suppression. r[tenstorrent.quality.language_gates]
- [x] [depends:tenstorrent.quality.language_gates] P1 Add deterministic Python formatting and lint configuration and fix the initial findings. r[tenstorrent.quality.language_gates]

## Phase 3: Parser robustness

- [x] [serial] C1 Inventory every JSON, YAML, CSV, binary-frame, vocabulary, and receipt parser with current input limits and trust boundary. r[tenstorrent.quality.parser_robustness]
- [x] [depends:tenstorrent.quality.parser_robustness] C2 Author named parser limits in Nickel and consume the checked-in generated projection in pure cores. r[tenstorrent.quality.parser_robustness]
- [x] [depends:tenstorrent.quality.parser_robustness] C3 Reject oversized, over-deep, over-count, malformed, truncated, duplicate, and non-finite inputs before unsafe allocation or admission. r[tenstorrent.quality.parser_robustness]
- [x] [depends:tenstorrent.quality.parser_robustness] C4 Add deterministic property tests and saved malformed fuzz seeds for each parser family. r[tenstorrent.quality.parser_robustness]

## Phase 4: Artifact policy

- [x] [serial] A1 Author typed artifact classes, named size bounds, allowed roots, privacy rules, and BLAKE3 identity rules in Nickel. r[tenstorrent.quality.artifact_policy]
- [x] [depends:tenstorrent.quality.artifact_policy] A2 Classify current generated, portable evidence, private log, and large fixture paths. r[tenstorrent.quality.artifact_policy]
- [x] [depends:tenstorrent.quality.artifact_policy] A3 Add positive and negative checks for approved artifacts, stale manifests, private content, untracked generated output, and unapproved large files. r[tenstorrent.quality.artifact_policy]

## Phase 5: Final validation

- [x] [parallel] V1 Add negative fixtures that prove each language gate, parser limit, and artifact rule fails with a stable diagnostic. r[tenstorrent.quality.verification]
- [x] [serial] D1 Document the local quality command, automation behavior, artifact workflow, fuzz-seed workflow, and hardware non-claims. r[tenstorrent.quality.verification]
- [x] [depends:tenstorrent.quality.aggregate_gate] V2 Run the aggregate gate twice, all supported-system evaluation, focused package builds, Cairn validation, and all change gates. r[tenstorrent.quality.verification]

## Verification Coverage

- `Scenario: Aggregate device-free gate passes` -> Q1, Q2, V2
- `Scenario: Gate tries to acquire hardware` -> Q3, V1
- `Scenario: Maintained language sources pass` -> R1, S1, S2, P1
- `Scenario: Language diagnostic fails the gate` -> R1, S2, V1
- `Scenario: Bounded parser input is admitted` -> C2, C4
- `Scenario: Unsafe parser input is rejected` -> C3, C4, V1
- `Scenario: Approved artifact is admitted` -> A1, A2, A3
- `Scenario: Unapproved artifact is rejected` -> A3, V1
