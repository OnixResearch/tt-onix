# Quality gate evidence

## Command

```text
nix run .#quality
```

Run from the repository root with `TMPDIR=/var/tmp`. The launcher bootstraps
`.#nix-patched-build-remote`, evaluates both supported systems, and builds the
aggregate through the matched hook.

## What this change adds to the gate

- `tt-model` keeps its install check, now with twelve switch scenarios.
- `tt-model-host-config` is a new required check: the host fragment evaluation.
- The registry check gains the containment fixture and the containment-command
  existence proof.
- The artifact manifest gains this change's evidence files.

## Order of operations

1. Bounded local builds of `.#tt-model`,
   `.#checks.x86_64-linux.model-runtime-profiles`,
   `.#checks.x86_64-linux.tt-model-owner`, and
   `.#checks.x86_64-linux.tt-model-host-config`.
2. The artifact manifest was regenerated from a clean copy of the staged tree
   (`git archive HEAD` plus the staged patch in `/var/tmp`) and compared byte for
   byte with the staged manifest.
3. `nix run .#quality` on the tree that includes that manifest.

## Non-claims

The gate is device-free. It does not open a device, start or stop a unit, reset a
board, install a sudo rule, or run `nixos-rebuild`. The host fragment is evaluated,
not applied.
