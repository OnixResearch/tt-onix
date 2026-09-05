# First verifier changes

## First-phase scope

The first phase ends at `8f9e228`. The continuation section records later changes and current validation limits.

This work starts from research commit `72a3d94`, which descends from `origin/main` at `51e07b8`. The implementation branch is `work/verifier-hardening-20260904`.

The change adds strict Cairn verdict admission, a required package catalog, and byte-level seed tests. It also repairs the stale artifact-policy root from `cairn` to `.cairn`.

The first phase changed no model weights, inference-runtime source, package pins, service state, or hardware state. This phase ended before publication. The primary checkout stayed at `51e07b8`. The research checkout stayed at `72a3d94`.

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

## First-phase blocker and non-claims

The required Python gate reports 13 pre-existing lint findings in unchanged Qwen files. These include `F821` for `started` at `mtp_drafter_adapter.py:832` and `BLE001` at `mtp_serving_core.py:233`. Other findings concern imports and unused names. The formatter did not run after the lint failure.

The full aggregate is not green. A dry run listed 579 derivations. The review ran focused required checks instead of building the entire closure with a known failing dependency. No lint suppression or runtime patch masks the blocker.

The explicit evidence registry, adapter lifecycle harness, model-cohort receipts, tiered driver, and strict Octet integration remain separate work. The root repair does not add an explicit evidence registry.

The analysis used one assistant. Its passes are correlated. Tests and hashes do not prove model correctness, physical execution, performance, or deployment authority.

## Continuation

The owner authorized commits, branch publication, and further device-free work. The branch now includes published commit `2fa53a6`. Main remains unchanged.

The Qwen Python gate now passes. The continuation removes the 13 lint findings and the existing formatting errors. It adds an assertion for selected values instead of hiding the unused `cut_values` result.

The MTP completion timer starts before host preparation and ends after cleanup. Three fake-backend tests cover repeated tail-route requests, rejected prompts with no backend effects, and errors after cleanup. The original code fails with `NameError`. A separate mutation excludes page-table preparation and fails with `10.0 != 11.0` on both requests.

The watchdog now belongs to `mtp_watchdog_adapter.py`, not the pure round policy. Six host-thread tests cover values, errors, control signals, timeout, invalid input, and thread creation errors. The adapter rejects timeout values above `threading.TIMEOUT_MAX` before work starts. The old implementation raises `OverflowError` after thread creation.

The exception handler retains one narrow `BLE001` exemption to forward all worker errors unchanged. Tests include `KeyboardInterrupt`, `SystemExit`, and a synthetic `BaseException` subclass. An Exception-only handler does not satisfy that contract.

The package installs both test files. Its device-free checker requires each suite to contain tests and report success. The installed Qwen package, authority check, service check, and Python Nix gate pass. Supported-system flake evaluation also passes. The Nix format gate passes after a separate formatting repair in `pkgs/qwen36/default.nix`.

### Aggregate blocker

One bounded aggregate build returned exit 1. It exposed the Nix formatting error, which the follow-up repaired. The other blocker is the pinned training-platform dependency chain:

```text
nftables 1.1.6 -> moby 29.5.2 -> docker 29.5.2
  -> training-platform-launcher -> training-platform-check
  -> repository-quality
```

Three patches from `git.netfilter.org` fail their fixed-output hash checks:

- `0001-build-fix-.-configure-with-non-bash-shell.patch`
- `0002-build-simplify-the-instantation-of-nftversion.h.patch`
- `0003-build-generate-build-time-stamp-once-at-configure.patch`

The fourth patch, `0004-build-support-SOURCE_DATE_EPOCH-for-build-time-stamp.patch`, fails with connection-reset errors. The second patch provides this representative mismatch:

```text
specified: sha256-gzlMHaZTXCqR8IcccIsLaMZie3CaMEPa4lnGr/x7b/o=
got:       sha256-SOvwAQNpikDxeWCAxYS1DSN6qD1kxk3X4bgke7oUDK0=
```

The existing Nix contract requires those SHA-256 hashes. This work does not replace them with the returned hashes, change a package pin, substitute a source, or omit the required check. The next check requires the exact pinned patch bytes or an owner-approved dependency repair. The full aggregate remains unaccepted.

Local evidence is under `.pi/verifier-continue/`: `new-tests-baseline.log`, `behavior-final.log`, `late-timer-control.log`, `qwen-nix-final.log`, `nix-format-final.log`, and `aggregate-final.log`. The framework check reused its existing local RWKV checkpoint. No model download, simulator run, device access, service change, or deployment occurred.

The watchdog timeout test explicitly proves that timeout does not cancel work. Caller recovery, shared device ownership, speculative-round state, and physical MTP admission remain open. Import-time TTNN nanobind diagnostics also remain. The tests do not claim to repair these boundaries.
