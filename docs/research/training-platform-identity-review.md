# Training-platform identity review

## Result

The current core correctly rejects the retained training receipt. This review found no validator defect that permits a software-only repair under the existing contract.

The review starts at `61618d4` on `work/verifier-hardening-20260904`. It adds regression tests, not a replacement receipt or a change to admission policy. The full quality gate remains unaccepted.

## Reproduced mismatch

The pinned `training-platform-core` accepted the current package-owned authority and produced the same launch plan twice. It rejected the unchanged receipt at `pkgs/training-platform/evidence/2026-08-08-dual-p150a-smoke/receipt.json`:

```text
validate-terminal-receipt: stale_identity: receipt launch_plan_identity does not match the current admitted identity
```

| Identity | Retained evidence | Current authority or plan |
| --- | --- | --- |
| Bundle | `90edc4171f5f063d41af2fcdc5efa8253080f8befc43c0a8153a078ac7200a21` | Same |
| Configuration | `53e9eaf71f511ab9c6dacc75848f2baf5fb281ff59391b9e9a685de3176ae658` | Same |
| Runtime authority | `2491fb4f2c5a06c39c22521d4bdb1d1bdcb612976253d8f61ba3ba269f04bbb1` | `6fa6fa750d1934098f1745016fa4abdfda641f4302df56cc1b61b13f230cf631` |
| Launch plan | `2f181874c506cca6d9828966bb233b96181fbcbe32ed97b9d0b1d6db7b0ab823` | `1a4ff04b580a94a3a47a66799fba306401935f19ee1db064a303a6db3deff067` |

The retained preflight artifact supplies the old runtime-authority identity. The retained receipt supplies the other old identities. The current CLI plan supplies the current identities.

## Why the rejection matters

`admit_runtime_authority` hashes canonical JSON for the complete authority. Its inputs include the descriptor path, descriptor digest, wheelhouse path, requirements path, and requirements digest.

`build_authorized_launch_plan` binds that authority identity, the descriptor mount, the wheelhouse mount, and the requirements path. Equal bundle and configuration identities do not imply an equal launch plan.

The retained evidence files contain digests, not the full old authority and launch plan. This review did not recover those old inputs. It therefore cannot identify every changed field or prove equivalence between the two runtime closures.

Repository history confirms that the evidence bytes remain unchanged from `da06ef07cc620405895cc830f05e405030d38017`. Since that commit, the TT-Metal package gained SFPI metadata changes and the GDN patch. Those changes explain why a package path can change, but they do not establish the exact historical identity difference.

The accepted specification requires stale identities to fail receipt validation. The package check explicitly requires the retained receipt to satisfy current authority. Neither rule changed here.

## Regression tests

`pkgs/training-platform/core/src/tests/runtime_identity.rs` reuses the existing pure admission functions and synthetic fixture constructors. It adds four tests:

- A synthetic receipt with matching authority passes. JSON whitespace does not change the identity.
- Each stale bundle, configuration, or launch-plan digest fails separately.
- Each changed descriptor, wheelhouse, or requirements path invalidates the old receipt, even with unchanged content digests.
- Each changed descriptor or requirements digest invalidates the old receipt, even with unchanged paths.

These fixtures pass explicit synthetic hash observations into the core. They do not read model files, open devices, or prove filesystem content.

The baseline passed 40 Rust tests. The final all-target, all-feature run passed 44 tests. Formatting and Clippy with `-D warnings` passed before and after the change. The pinned Nix core build also passed all 44 tests.

An isolated mutation removed only the launch-plan comparison from receipt validation. The positive test still passed. All three negative tests failed because the mutated validator accepted stale plans. The production comparison remains unchanged.

## Remaining authority decision

Current physical admission needs evidence for the current runtime authority. A fresh bounded smoke requires explicit device-ownership, service-interruption, and recovery approval. This review does not grant that approval.

A separate change can review how the repository distinguishes device-free tests from current physical admission. Such a change must retain stale-receipt rejection and state the limits of each gate. It cannot present an old receipt or a synthetic fixture as a current physical result.

No aggregate retry was useful after this review because the known admission blocker remains. The review did not change pins, expected dependency hashes, hardware evidence, launcher behavior, service configuration, or devices. MTP physical admission remains a separate open issue.

## Evidence scope

Local evidence resides under `.pi/training-identity-review/` in the implementation worktree:

- `current-authority.json`, `current-plan.json`, and `current-plan-repeat.json` retain the current inputs and deterministic output.
- `historical-rejection.log` records the exact rejection from the pinned CLI.
- `receipt-origin.log` and `since-receipt-*.diff` record the focused history comparison.
- `rust-baseline.log`, `rust-final.log`, and `clippy-*.log` record the Rust checks.
- `no-plan-identity.log` records the isolated mutation result.
- `core-nix.log` records the pinned Nix build and its 44 passing tests.

One assistant performed three correlated analysis passes: identity drift, the package test contract, and validator behavior. The review found a valid rejection, not a route to full acceptance without new authority or evidence.

Source owners are `pkgs/training-platform/core/src/lib.rs`, `pkgs/training-platform/default.nix`, and `.cairn/specs/dual-p150a-training-platform/spec.md`. The test suite uses the existing core, with no new dependency or copied cross-repository source.
