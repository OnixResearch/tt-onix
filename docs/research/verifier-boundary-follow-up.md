# Verifier boundary follow-up

## Scope and status

This work follows `c27ebed` on `work/verifier-hardening-20260904`.

- `718d38c` adds the research-evidence registry and its rejection tests.
- `ffaceae` repairs the host MTP lifecycle and adds its history tests.

The four pinned patch inputs are available again. The artifact and MTP checks pass. The full quality gate remains unaccepted because an existing training receipt has a stale launch identity.

This work does not change main, package pins, expected source hashes, deployment configuration, services, or devices. It does not download model weights or run a simulator. No Cairn implementation task becomes complete from these checks alone.

## Pinned input recovery

A scoped `nix copy --from https://cache.nixos.org` recovered the original nftables patch outputs. Direct SHA-256 checks matched all four pinned digests:

| Patch | Required SHA-256 |
| --- | --- |
| `0001-build-fix-.-configure-with-non-bash-shell.patch` | `sha256-gJi6Q6mpURynzpnTFs1VJmZ+SpnNhyOrCzKarJmu/6w=` |
| `0002-build-simplify-the-instantation-of-nftversion.h.patch` | `sha256-gzlMHaZTXCqR8IcccIsLaMZie3CaMEPa4lnGr/x7b/o=` |
| `0003-build-generate-build-time-stamp-once-at-configure.patch` | `sha256-03nexKi/tH1pGNor2/q+Q5ps9+YQRGBbR/DR164Efpo=` |
| `0004-build-support-SOURCE_DATE_EPOCH-for-build-time-stamp.patch` | `sha256-vK/dsi9YXzEg2iShfbhnwlJJC27LTBA9fIjyJySoeFI=` |

The existing Nix fixed-output contract requires SHA-256. The recovered files use the original store paths. This recovery does not replace a source revision, change an expected hash, or accept different patch bytes. A separate bounded fetch from the original host did not complete.

The aggregate then reached `training-platform-check`. That check rejected the receipt at `pkgs/training-platform/evidence/2026-08-08-dual-p150a-smoke/receipt.json`:

```text
validate-terminal-receipt: stale_identity: receipt launch_plan_identity does not match the current admitted identity
```

The receipt and its admission check remain unchanged. This work does not relabel old physical evidence as current evidence. The remaining blocker needs a separate authority decision and matching evidence, not a replacement digest.

## Research-evidence coverage

`quality/policy.ncl` now registers `docs/research` through the required `evidence_roots` field. Every file in that scope requires a BLAKE3 manifest entry and private-marker checks.

The pure registry component rejects missing, empty, malformed, overlapping, reserved, and unowned roots. It also rejects an evidence rule that disables BLAKE3 or permits private content. The existing classifier owns the path-to-class decision. The filesystem adapter owns traversal and file reads.

The tests cover these cases:

- Valid research receipts and adjacent documentation outside the registered scope.
- Renamed receipts and nested directories named `target`, `.pi`, or `node_modules`.
- Modified, missing, relocated, or newly added receipts with an old manifest.
- Symbolic links inside the registered scope, without reads from their targets.
- A private marker followed by invalid UTF-8.

Eight new assertions failed against the old implementation before repair. The final Rust run passes 28 tests across the library and CLI suites. All-target, all-feature Clippy passes with `-D warnings`. The artifact, policy, and Cairn Nix checks pass.

The implementation reuses the existing classifier and manifest parser. Cap Root was reviewed as a reuse candidate. Its native-path normalization does not own this registry's canonical portable strings or evidence classes. No source was copied from it, and no sibling worktree became a product dependency.

This registry does not recognize every receipt by content. It does not prove mutable-filesystem confinement or detect every possible secret. Deliberate changes to evidence ownership still require review.

## MTP host lifecycle

The new suite uses a fake target that predicts each input token plus one. Feedback values identify host snapshots, not real tensors. The original adapter reproduced premature pool exhaustion, stale feedback, lost economic-loss counts, deadline fall-through, and incomplete cleanup.

The repairs keep policy in `mtp_serving_core.py` and effects in adapters:

- The core computes the maximum round count for rejection at every first draft.
- Each recorder refreshes its zero-prefix feedback from the current round input.
- The core returns the next economic-loss count and the disengagement decision.
- Round timing includes rollback and feedback selection.
- Entry checks reject invalid generation limits and expired deadlines before backend work.
- A deadline prevents further plain decode after speculation stops.
- Partial recorder allocation and capture errors release completed request resources.

The watchdog retains a pending worker and its eventual result after an incomplete start, wait, or liveness check. It propagates the same exception object. The MTP adapter retains request resources and rejects reuse or explicit close through that adapter. A worker that later finishes does not clear this quarantine.

The tests cover synthetic timeouts at prefill, head seeding, verifier capture, drafting, verification, rollback, and plain decode. A real host-thread test proves that the worker stays alive until released. It also proves that its late result remains retained. Interrupt tests cover `KeyboardInterrupt` and `SystemExit` without converting them into successful results.

A mutation that removed interrupted-wait quarantine failed both signal cases with this diagnostic:

```text
AssertionError: unsafe effect with live worker: verifier-close
```

An additional adversarial pass found unprotected startup and liveness-check windows. Their new tests failed before the final repair. The final package runs all 25 host tests. The Qwen package, authority-negative, NixOS service, and Python Nix checks pass.

## Remaining limits

These tests do not establish tensor parity, MTP cache alignment, trace compatibility across requests, or physical cleanup safety. They do not implement process restart, device recovery, or shared ownership across other adapters. MTP remains unapproved for physical serving.

The existing TTNN import still emits nanobind leak diagnostics. These checks do not repair that upstream behavior.

Older research digests describe their original source snapshots. The new host changes do not inherit their physical or performance admission.

## Evidence and bounds

Local logs remain under `.pi/verifier-next/` in the implementation worktree:

- `patch-derivations.json`, `patch-cache.log`, and `patch-digests.log` identify the recovered inputs.
- `artifact-baseline.log` and `registry-negative-baseline.log` record the pre-change controls.
- `registry-final.log`, `artifact-tree-final.log`, and `registry-nix.log` record artifact validation.
- `mtp-baseline.log`, `mtp-lifecycle-baseline.log`, and `watchdog-window-baseline.log` record host controls.
- `interrupt-mutation.log`, `mtp-lifecycle-final.log`, and `mtp-final-nix.log` record host validation.
- `aggregate-baseline.log` and `aggregate-final.log` record the unchanged training-receipt blocker. Both bounded aggregate attempts returned exit 1.
- `final-flake-evaluation-materialized.log` records successful evaluation of all supported systems.

The final aggregate also reports artifact, Cairn, and Qwen checks as passing. It still rejects the stale training receipt. The gate does not omit that dependency.

No-build evaluation initially encountered an unavailable filtered-source store path. An eval-cache-disabled retry did not resolve it. After a regular build materialized the source, the same no-build evaluation passed. A local GC root preserves that source for this validation session. This observation is not a package-pin change or a repair to the Nix evaluator.

The work uses one assistant and at most two concurrent commands. Analysis passes are correlated, not independent reviews. The contract permits two aggregate attempts, each capped at 20 minutes. No build result authorizes deployment or hardware access.
