## Why

Every measured admission in this repository describes a runtime that is not
deployed. The running system is generation
`nixos-system-britton-desktop-26.11.20260908.e9b9cbe` (built 2026-09-08), whose
service runs `qwen36-0.1.0` against **tt-metal 0.74** with
`--decode-mode candidate`:

| | production now | B=8 admitted 2026-09-11 | B=4 admitted 2026-08-26 |
| --- | --- | --- | --- |
| runtime | 0.74 (`djchdwzk`) | 0.77 chunked variant (`6ccynlvv`) | 0.74 (`1l9nlvvv`) |
| package | `jsivpglnl` | execution identity `z31p1j9w` | `ixb9s8mb` |

A rebuild of this repository through the module default would deploy
`pkgs.qwen38` against the **shared 0.77** runtime (`dr9bj8qw`). That is a
different identity from every receipt, including the admitted B=8 receipt,
which was measured on the chunked variant where the width-8 fused decode can
launch at all.

So the gap is not a missing capability; it is an unverified deployment. The
served artifact must be brought onto a measured identity, and the switch must
be verified before and after rather than assumed.

## What Changes

- Run a readiness gate on the exact serving artifact a rebuild selects, and
  record its identities and first-request behavior.
- Switch the host configuration that imports this module onto the measured
  package, and re-run readiness, health, and the single-stream status after the
  switch.
- Keep the batch-admission route **off** in the service: the admitted B=8
  receipt's own non-claims exclude arbitrary continuous batching, so serving it
  would exceed the evidence. Enabling a batched serving route is its own change
  with its own admission.
- Repair the artifact manifest's merge damage so the artifact gate reports the
  pre-existing entry drift instead of a JSON syntax error.

## Impact

- **Files**: the host configuration that imports `nixos/module.nix` (not in this
  repository), this change directory, and
  `quality/generated/artifact-manifest.json` for the syntax repair.
- **Testing**: a readiness gate on the selected artifact, post-switch readiness
  and first-request checks, and the repository gates.
- **Production**: the service restarts onto a newer runtime. Decode mode stays
  `candidate`, exactly as it is today; no admission status changes, and no batch
  width becomes served.
