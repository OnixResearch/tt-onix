## Why

Steps 1-3 built the registry, the record, the checker, and the job runner. A
serve profile still has no way to become the owner: an operator stops the old
unit, resets the boards, starts the new one by hand, and checks readiness from
memory. Nothing proves that the process that came up is the one the record names,
and nothing restores the previous owner when the new one does not.

This change adds that sequence as one command, with the rollback the manual
procedure never had, plus the reviewable host fragment that would activate it.

## What Changes

- Add `tt-model switch <profile> [--package PATH] [--force] [--json]` for
  `mode = serve` only: lock, check, card holders, stop, reset, re-check, the
  package's own containment check, record and roots, start, readiness, identity
  readback, and the committed verdict. Any failure after the record is replaced
  restores the previous record and owner, or says plainly that it could not and
  withdraws the record rather than leaving an unverified owner recorded as active.
- Add an optional `containment` block to the runtime-profile contract (a
  repository-relative command, its arguments, and the verdict it must print) and
  record qwen38's existing containment checker in the registry. `--force` never
  bypasses it.
- Add an optional identity source to a profile's readiness block. qwen38 reads its
  identity from one bounded request's decode evidence, because `/healthz` does
  not carry identity today; a profile that declares nothing uses `/healthz`.
- Add `nixos/examples/qwen38-tt-model-switch.nix`, the reviewable fragment that
  would enable the owner unit and the helper grant on this host, and an
  evaluation check that proves it adds exactly one unit and one sudo rule.

## Impact

- **Files**: `pkgs/tt-model/{tt_model.py,default.nix}`,
  `model-targets/{runtime-contract,runtime-profiles,runtime-check}.ncl` and the
  export, `pkgs/tt-model-owner/**` (unchanged), `nixos/examples/qwen38-tt-model-switch.nix`,
  `checks/tt-model-host-config.nix`, `flake.nix`, `flake-modules/onix-packages.nix`,
  `quality/{required-checks.ncl,generated/required-checks.json}`, and this change.
- **Testing**: the `tt-model` install check with twelve switch scenarios, the
  helper check, the sudo and host-configuration evaluation checks, the registry
  check with an eighth fixture, and `nix run .#quality`.
- **Not performed**: no host change was applied. `nixos-rebuild` was not run, no
  unit was enabled or unmasked, no sudo rule was installed, and no hardware was
  touched. The live switch remains a deliberate, separately authorized action.
