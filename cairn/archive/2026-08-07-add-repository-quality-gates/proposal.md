## Why

The flake evaluates and formats Nix, but the repository has no checked-in automation that runs the full device-free quality set. Rust Clippy currently fails in the layer harness. ShellCheck also reports missing shell declarations and unsafe optional argument expansion.

Several parsers protect evidence and process boundaries. Their test suites use focused examples, but they do not share input-size limits or property and fuzz coverage. The repository also has no enforced policy for large generated and hardware-evidence files.

## What Changes

- Add one repository-owned device-free quality command and run it in checked-in automation.
- Add Rust formatting and Clippy gates for every crate and target.
- Add ShellCheck and executable-shebang gates for every maintained shell script.
- Add Python formatting and lint gates for maintained Python sources.
- Add shared parser input limits, property tests, fuzz seeds, and malformed-input tests.
- Add a typed Nickel artifact policy for generated files, large fixtures, private logs, and portable evidence.
- Keep hardware and long simulator runs outside the default device-free gate.

## Impact

- **Flake**: quality checks and one aggregate device-free gate.
- **Rust**: layer-harness argument contexts, parser limits, Clippy cleanup, and property tests.
- **Shell**: declared interpreters, array-based optional arguments, and ShellCheck annotations.
- **Python**: repository-owned format and lint configuration.
- **Artifacts**: generated and evidence file classification, size policy, ignore rules, and BLAKE3 manifests.
- **Automation**: checked-in workflow or equivalent runner for pull and main-branch validation.

## Out of Scope

- Hardware execution in untrusted automation.
- Performance regression thresholds, which belong to the active harness regression change.
- Broad style-only rewrites of historical evidence or archived Cairns.
