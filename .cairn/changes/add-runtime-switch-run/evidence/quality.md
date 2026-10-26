# Quality gate evidence

## Command

```text
nix run .#quality
```

Run from the repository root with `TMPDIR=/var/tmp` (the host's `/tmp` is full).
The launcher bootstraps `.#nix-patched-build-remote`, evaluates both supported
systems, and builds the aggregate through the matched hook.

## What this change adds to the gate

- `tt-model-owner` is a new required check: the helper behaviour suite
  (`evidence/helper-boundary.md`).
- `tt-model-nixos-sudo` is a new required check: the sudo-rule and unit-invariance
  evaluation (`evidence/sudo-grant.md`).
- `tt-model` keeps its install check, now covering the run lifecycle
  (`evidence/run-lifecycle.md`).
- The artifact manifest gains this change's evidence files.

## Order of operations

1. `nix build .#checks.x86_64-linux.tt-model-owner`,
   `nix build .#checks.x86_64-linux.tt-model-nixos-sudo`, and
   `nix build .#tt-model` for bounded local iterations.
2. The artifact manifest was regenerated from a clean copy of the staged tree
   (`git archive HEAD` plus the staged patch in `/var/tmp`) and compared byte for
   byte with the staged manifest.
3. `nix run .#quality` on the tree that includes that manifest.

## Non-claims

The gate is device-free. It does not open a device, start or stop a unit, reset a
board, or exercise a live systemd. The helper is exercised only against fake
tools, and the sudo rule is evaluated, not installed.
