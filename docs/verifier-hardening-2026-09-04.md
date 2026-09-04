# First verifier changes

## Scope

This work starts from research commit `72a3d94`, which descends from `origin/main` at `51e07b8`. The implementation branch is `work/verifier-hardening-20260904`.

The change adds strict Cairn verdict admission, a required package catalog, and byte-level seed tests. It also repairs the stale artifact-policy root from `cairn` to `.cairn`.

No model weights, inference-runtime source, package pins, service state, or hardware state changed. The branch remains local. The primary checkout remains at `51e07b8`. The research checkout remains at `72a3d94`.

## Implementation

- `repository-quality cairn-verdict` decodes bounded JSON and rejects duplicate keys at every depth. The pure core requires the top-level verdict, root resolution, and empty issue groups. The CLI owns file reads and exit status.
- `quality/required-checks.ncl` declares seven required package checks. These include the Qwen package, authority rejection, and NixOS service checks. Nix evaluation rejects a missing, duplicate, unregistered, or non-derivation check. Removal controls cover every required entry.
- Parser-owner tests consume actual JSON, YAML, CSV, binary-frame, vocabulary, and receipt bytes. Accepted counterparts prevent an always-reject implementation from passing. The synthetic vocabulary suffix reaches the existing row-count requirement without model downloads.
- Native `.cairn` evidence now satisfies artifact policy. The legacy root remains a negative fixture. The generated artifact manifest matches the covered files.

The seed index explains the corrected corpus labels. The old CSV input did not exceed the field limit. The old frame failed its magic check, not its truncation check. The new tests assert the actual parser diagnostics.

## Validation

| Boundary | Observed result |
| --- | --- |
| Original repository-quality tests | Ten passed before implementation |
| Original parser-owner library tests | Layer harness 41, evidence 211, persistent receipts nine passed |
| Changed Rust packages | All-target tests and Clippy with `-D warnings` passed for all four packages |
| Native-root repair | All-target, all-feature repository-quality tests and Clippy passed |
| Seed replacement control | Replacing duplicate JSON with valid JSON caused its rejection test to fail |
| Restored seed | Repository-quality tests passed again |
| Cairn receipts | Failed primary receipt rejected, valid research receipt accepted |
| Pinned Cairn Nix gate | Passed with the actual pinned binary and policy |
| Required package catalog | Real aggregate includes all seven entries, including all three Qwen checks |
| Focused Nix package checks | Qwen package, authority, NixOS service, and policy checks passed |
| Supported-system evaluation | `nix flake check --no-build --all-systems` passed |
| Native artifact policy | Actual tree failed before the root repair and passed after it |

The local logs are under `.pi/verifier-implementation/` in the implementation worktree. They include `parser-owner-baseline.log`, `parser-owner-tests.log`, `seed-replacement-control.log`, `final-nix-checks.log`, and `remaining-gates.log`.

## Remaining blocker and non-claims

The required Python gate reports 13 pre-existing lint findings in unchanged Qwen files. These include `F821` for `started` at `mtp_drafter_adapter.py:832` and `BLE001` at `mtp_serving_core.py:233`. Other findings concern imports and unused names. The formatter did not run after the lint failure.

The full aggregate is not green. A dry run listed 579 derivations. The review ran focused required checks instead of building the entire closure with a known failing dependency. No lint suppression or runtime patch masks the blocker.

The explicit evidence registry, adapter lifecycle harness, model-cohort receipts, tiered driver, and strict Octet integration remain separate work. The root repair does not add an explicit evidence registry.

The analysis used one assistant. Its passes are correlated. Tests and hashes do not prove model correctness, physical execution, performance, or deployment authority.
