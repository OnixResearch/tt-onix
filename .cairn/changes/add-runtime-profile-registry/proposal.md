## Why

The repository ships four mutually exclusive Tenstorrent model runtimes on the
same two P150 cards, and nothing states in one place how each is launched. Today
the launch data is scattered across a NixOS module, a host drop-in, a host-local
profile service, two package wrappers, and a runner's `argparse` defaults:

- `nixos/module.nix:391-440` owns the Qwen environment and `ExecStart`.
- `scripts/qwen38-profile-service:44-63` re-states twelve environment variables
  and three required package files by hand, so a change to one copy makes the
  other copy's preflight reject the service.
- `pkgs/nanbeige43/runner/nanbeige43_runner/run.py:712-717` requires
  `--checkpoint`, `--cache-root`, and `--receipt` with no declared defaults, and
  the wrapper appends `--mesh p150x2` (`pkgs/nanbeige43/default.nix:145-150`).
- `pkgs/rwkv7-p150x2/default.nix:125-136` bakes its mesh descriptor and kernel
  root into the wrapper.
- `flake-modules/onix-packages.nix:43-67` names five native gate wrappers, each
  with its own pinned checkpoint argument.

An operator who wants to know "what is running, and what would running it take?"
has to read all of that. A *tool* cannot answer at all. Two concrete
consequences were measured on this host on 2026-10-26:

- the operator's served-version symlink
  `~/.local/state/qwen38-p150x2/current` points at
  `/nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0`, which no longer
  exists;
- the Nix profile generation `qwen38-p150x2-5-link` installs
  `/nix/store/inkfvsjqchqp7bki76asdib3b57nw6kj-qwen38-0.1.0`, which no admission
  receipt in this tree names.

Those are two different answers to "which Qwen package is active", neither is
usable, and the owner unit `qwen38-p150x2.service` is masked. That is the state
this change starts from; see `evidence/status-host.md`.

## What Changes

- Add `model-targets/runtime-contract.ncl`: a closed, typed Nickel contract for a
  runtime profile, with field-level and cross-field rules that each name their
  refusal.
- Add `model-targets/runtime-profiles.ncl`: four profiles (`qwen38`,
  `nanbeige43`, `rwkv7-ttnn`, `metalium-rs-rwkv7`). Every value is transcribed
  from a source in this tree or a recorded receipt, and every block cites that
  source. The registry is data: it evaluates no gate and re-serializes no
  receipt.
- Add `model-targets/generated/runtime-profiles.json`, the canonical export
  under the existing registered generated root.
- Add `model-targets/runtime-check.nix`: regenerates the export and compares it
  byte for byte, recomputes every authority and receipt digest, cross-checks the
  model-target identity pins, and rejects seven negative fixtures with their
  named messages.
- Add `pkgs/tt-model` with one subcommand, `status [--json]`. It is read-only,
  needs no root and no sudo, opens no device, writes nothing, and reports the
  root-owned record when one exists or the legacy layout otherwise. It exits
  non-zero, with the path and reason, when a reported source is dangling,
  unrooted, or contradicts another source.
- Register both checks in `quality/required-checks.ncl` and regenerate the
  closed catalog.

## Impact

- **Files**: `model-targets/{runtime-contract,runtime-profiles,runtime-check}.ncl`,
  its seven fixtures and one generated export, `pkgs/tt-model/{default.nix,tt_model.py}`,
  `flake-modules/onix-packages.nix`, `quality/required-checks.ncl`, and
  `quality/generated/{required-checks,artifact-manifest}.json`.
- **Not changed**: `model-targets/profiles.ncl`, `model-targets/generated/profiles.json`,
  `model-targets/check.nix` and its identity pins, every existing receipt, every
  existing package, and the NixOS module. `nix eval --raw .#qwen38.drvPath`
  remains `/nix/store/af68qaa6rimz65g49yhgkvv031znrsp4-qwen38-0.1.0.drv`;
  `evidence/identity-invariance.md` records the file digests.
- **Testing**: the new device-free registry check, the `tt-model` install check,
  a host `status` run, the closed required-check catalog, and `nix run .#quality`.
- **Production**: no service, unit, drop-in, or device is touched. This change
  adds no privileged path; the record reader is inert until a later change
  writes a record.
- **Non-claims**: this is step one of a phased switch design. It does not add a
  switch, does not reset or own the cards, and does not admit any package.
