## Why

The registry from `add-runtime-profile-registry` says what each runtime needs to
start, but nothing can answer two questions an operator actually asks: "would
this exact package satisfy this profile?" and "which package is active, and is
that record still true?".

Today the answers are spread across a host-local profile service, a Nix profile
generation, and an operator symlink that can be garbage-collected. This host
shows the consequence: the symlink points at a removed store path, the Nix
profile installs a package no admission receipt names, and the owner unit is
masked. A stale pointer is indistinguishable from a good one until something
fails at startup.

This change adds the two missing pieces without adding a privileged path: a
pure `check` that names the exact failing item, and a `record` writer that binds
the full identity tuple and roots every store path it names, so a later reader
can prove the record still means what it said.

## What Changes

- `tt-model check <profile> [--package PATH]`: validates the profile, the
  package, the entry binary, the runtime root, the admission references, the
  checkpoint, and the model-target identity. Cheap structural checks by default;
  `--deep` re-hashes the checkpoint behind a documented 1 GiB bound and prints
  progress.
- `tt-model record <profile> --package PATH`: writes `active.json`, an
  append-only `history/<utc>-<profile>.json`, and one GC root per store path the
  record names. The record carries the full identity tuple, including the
  registry digest, the argv digest, the receipt references with their admission
  verdicts, and the previous record.
- `tt-model status` consumes the record: `managed` requires the package to
  exist, to match the profile's store name, and to resolve from its GC root; a
  legacy source that disagrees is a contradiction.
- The registry gains the three fields `check` needs: a package-relative runtime
  root, gate artifacts as records, and `model_target_name` beside
  `model_target_blake3`.

## Impact

- **Files**: `pkgs/tt-model/{tt_model.py,default.nix}`,
  `model-targets/{runtime-contract,runtime-profiles,runtime-check}.ncl` and the
  committed export, this change directory.
- **Testing**: the `tt-model` install check (all refusals, the record lifecycle,
  and the deep bound), the registry check, real-profile verdicts on this host,
  and `nix run .#quality`.
- **Privilege**: none. No subcommand runs as root, uses sudo, opens a device,
  stops or starts a unit, or resets the cards. The privileged owner helper
  (stop/start, `lsof`, `tt-smi -r`) remains step 3-4 work.
- **Not changed**: `model-targets/profiles.ncl`, `generated/profiles.json`, the
  model-target factory and its identity pins, every existing receipt, the NixOS
  module, and every service. The qwen38 derivation path is unchanged.
- **Non-claims**: this is not a switch. It writes a record and proves it is
  rooted; it does not stop the current owner, reset the cards, or start another
  runtime.
