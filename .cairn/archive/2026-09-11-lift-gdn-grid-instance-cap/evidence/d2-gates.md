# D2: repository Cairn, artifact, and Qwen package gates

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task D2.

## Results

```
nix build --no-link \
  .#checks.x86_64-linux.repository-quality-cairn \
  .#checks.x86_64-linux.qwen38 \
  .#checks.x86_64-linux.qwen38-authority-negative \
  .#checks.x86_64-linux.qwen38-nixos-service \
  .#checks.x86_64-linux.qwen36 \
  .#checks.x86_64-linux.qwen36-nixos-service
rc=0
```

| Gate | Result |
| --- | --- |
| `repository-quality-cairn` | PASS |
| `qwen38` | PASS |
| `qwen38-authority-negative` | PASS |
| `qwen38-nixos-service` | PASS |
| `qwen36` | PASS |
| `qwen36-nixos-service` | PASS |
| `repository-quality-artifacts` | FAIL, pre-existing |

Plus, from this change's own verification:

- `nix build .#qwen38` PASS (the shared package now builds again; it was
  broken at HEAD by the V5 content assertions).
- `nix build .#qwen38-gdn-grid` PASS, device-free check `PASS` with 44 probes
  including the new in-range mapping probe.
- Both negative directions of the kernel-content binding fail closed with
  named diagnostics.

## The artifact-gate failure is pre-existing

```
repository-quality: artifact manifest is invalid JSON: expected `,` or `}` at line 1719 column 7
```

`quality/generated/artifact-manifest.json` is malformed at line 1719 (a
missing comma between two entries). The file is not touched by this change
(`git diff --name-only origin/main` does not list it), it is invalid at
`origin/main` and at `c095e3f` alike, and its last modifying commit is the
merge `e0035c3`. With the syntax error repaired the gate would compare
expected against actual entries and report the manifest drift that commit
`680afa7` already recorded as pre-existing, so this change leaves the file
alone rather than mixing an unrelated generated-file repair into it.

## Cairn CLI note

The repository Cairn gate above runs through the flake's pinned Cairn input
(`OnixResearch/cairn` rev `15f0087556…`) and passes. Running the live
`../cairn` checkout's CLI against the pinned policy is blocked by a pin skew
the CLI reports as
`cairn.workflow.profile.registry … initial workflow profile is missing:
outcome-machine`, the same class of skew recorded for other repositories on
2026-09-11. That skew is pre-existing and unrelated to this change; the flake
gate is the authoritative one.

## Also noted

Flake evaluation at `origin/main` (`c095e3f`) is broken: the `e0035c3` merge
resurrected `qwen36 = qwen36;` in the required-package-checks block after the
package binding was renamed to `qwen38`, so no flake output evaluates. This
change repairs it in its first commit (`42b2898`), which is why the gates
above can run at all.
