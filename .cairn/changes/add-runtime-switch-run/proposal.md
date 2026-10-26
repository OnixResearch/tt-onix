## Why

The registry and the record answer "what would run" and "what is recorded", but
the job profiles still have no way to take the cards. `nanbeige43`,
`rwkv7-ttnn`, and `metalium-rs-rwkv7` are scripts an operator runs by hand: stop
the owner, reset the boards, export the right environment, remember the right
CLI. Every one of those steps is a place to start two runtimes on the same two
cards, and today nothing checks the order.

The device-free tool cannot do it alone. Step 1's install check asserts that
`tt_model.py` never names `/dev/tenstorrent`, `tt-smi`, or the owner helper, and
`quality/policy.ncl` forbids those strings in device-free gate actions. The
privileged operations therefore live in a separate helper, and this change adds
that helper, its single sudo grant, and the job runner that calls it.

## What Changes

- Add `pkgs/tt-model-owner`, the only privileged program in this repository. It
  has five subcommands: `cards`, `stop-owner`, `start-owner`, `reset`, `record`.
  It refuses any overridden state root, GC-root directory, unit name, or command
  path unless `TT_MODEL_OWNER_TEST=1`; it refuses unless the host's recorded
  helper pointer resolves to its own store path; it prints each acting command
  before running it; and it never guesses a unit name.
- Add `tt-model run <profile> [-- <args>]` for `mode = job` profiles only. It
  takes the run lock, checks the profile against the candidate package, asks the
  helper for the card holders, stops the owner if one is running, resets when the
  profile or the stopped owner requires it, runs the profile's argv with the
  profile's environment, and records the run event.
- Add `last-run.json` and `runs/<utc>-<profile>.json` under the state root. They
  are informational: `status` reports the most recent run and never treats it as
  the active owner.
- Add the `hardware.tenstorrent.ttModel` NixOS option. It renders exactly one
  sudo rule for `tt-model-owner`, one pointer rule that records which installed
  helper may act, and three evaluation-time assertions. It does not enable,
  unmask, or change any unit.

## Impact

- **Files**: `pkgs/tt-model-owner/{default.nix,check.nix}`,
  `pkgs/tt-model/{tt_model.py,default.nix}`, `checks/tt-model-nixos-sudo.nix`,
  `nixos/module.nix`, `overlay.nix`, `flake-modules/onix-packages.nix`,
  `quality/{required-checks.ncl,generated/required-checks.json}`, and this
  change directory.
- **Testing**: the helper check, the `tt-model` install check (all refusals, the
  run lifecycle, env isolation, and the lock), the sudo-rule evaluation check
  with three negative cases, and `nix run .#quality`.
- **Privilege**: the helper is the only program with a grant, and the grant is
  staged by the module, not installed on this host. No command in this change
  was run as root.
- **Not performed**: no hardware was touched, no real unit was started or
  stopped, no real `tt-smi` was called, the masked `qwen38-p150x2` unit is
  untouched, and the live bounded run is deferred to a separately authorized
  step.
