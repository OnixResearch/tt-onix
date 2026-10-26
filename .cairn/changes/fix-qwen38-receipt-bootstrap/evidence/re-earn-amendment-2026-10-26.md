# Amendment 2026-10-26: the measurement profile is the unmeasured twin

This section amends `re-earn-procedure-2026-10-22.md`. It is dated, reviewable,
and narrower than either earlier formulation: it changes **which file the
measurement reads**, and nothing else. All six receipt gates and the promotion
gate are unchanged.

## What conflicted

- The runbook's preflight requires the profile passed to the server and the
  receipt builder to be a regular file **inside the served package**, equal to
  the package's installed v3 export.
- The 2026-10-22 amendment requires the campaign to run the **admitted** package,
  because an unmeasured package cannot build (its install check demands
  `physically_admitted: true`, `pkgs/qwen36/default.nix:1827`).
- `scripts/qwen36-batch-receipt.rs:439-441` refuses an input profile that is
  already admitted or measured (`input profile must be unadmitted and
  unmeasured`).

The admitted package ships only the admitted v3 export, so under the old
containment rule no profile could satisfy the builder and the package at once.
Observed on 2026-10-26 at `2ba2c19`:

```text
scripts/qwen36-batch-receipt.rs --profile <admitted package>/share/qwen36/batch-admission-b8-profile-v3.json ...
  -> exit 1: qwen36-batch-receipt.rs: input profile must be unadmitted and unmeasured
```

## Why the builder guard stays

The guard is anti-circularity: it prevents minting a receipt from a profile that
already claims admission. Removing it, or adding an admitted-input mode, would
let a stale admission be re-blessed by measurement. The stronger guard is kept
exactly as it is, with no new flag.

## What replaces the containment rule

The **measurement input profile** is the unmeasured twin:

- generated from the same Nickel source as the packaged admitted profile
  (`pkgs/qwen36/batch-admission-b8-profile-v3.ncl`) in a controlled tree whose
  only tracked difference is `pkgs/qwen36/batch-admission-b8-binding-v3.ncl`
  reverted to `not-measured`;
- proven byte-equivalent to the packaged profile except for exactly
  `physically_admitted` and `identities.receipt_blake3`, before any device run;
- passed to the server (`--batch-admission-profile`) and to the receipt builder
  (`--profile`), so both read the same bytes;
- recorded with its path, BLAKE3, source tree revision and the packaged
  counterpart's BLAKE3.

The **server binary, runtime and execution identity still come from the admitted
package**; only the profile file is the twin. The package containment requirement
therefore now applies to the *service code and identity*, and the twin's own
provenance is the hard precondition for the profile.

Twin provenance is a gate: missing, unreadable, non-equivalent, or differently
sourced twin aborts before any device run.

## Unchanged

- The six receipt gates: aggregate decode speedup >= 6.0, per-user decode
  retention >= 0.75, per-user TTFT ratio <= 9.0, tail latency ratio <= 1.5,
  exact token parity, exact text parity.
- Every structural gate: physical devices `[0,1]`, no simulation, active prefix,
  clean reset, distinct state slices, finite timings, one common
  package/runtime/model identity, exact v3 profile/width/route, the eight-prompt
  workload and 24 measured samples per root.
- The promotion gate: the promoted package's execution identity must equal the
  measured identity, and its profile must differ from the measured twin only in
  `physically_admitted` and `identities.receipt_blake3`, naming the fresh receipt.
