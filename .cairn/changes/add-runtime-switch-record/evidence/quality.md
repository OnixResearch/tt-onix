# Quality gate evidence

## Command

```text
nix run .#quality
```

Run from the repository root with `TMPDIR=/var/tmp` (the host's `/tmp` is full).
The launcher bootstraps `.#nix-patched-build-remote`, evaluates both supported
systems, and builds the aggregate through the matched hook.

## What this change adds to the gate

- `tt-model` is a required package check. Its install check is the sandbox suite
  described in `check-verdicts.md` and `record-lifecycle.md`.
- `tt-model` runs `ruff check` and `ruff format --check` over `tt_model.py`, and
  `nixfmt --check` over its derivation and the extended registry check.
- The registry check now resolves every `model_target_name` through the target
  registry and refuses a gate artifact that is not package-relative.
- The artifact manifest gains this change's evidence files and the regenerated
  registry export.

## Order of operations

1. `nix build .#checks.x86_64-linux.model-runtime-profiles` and
   `nix build .#tt-model` for bounded local iterations, and
   `nix build .#checks.x86_64-linux.repository-quality-cairn` for the change
   directory.
2. The artifact manifest was regenerated from a clean copy of the staged tree
   (`git archive HEAD` plus the staged patch applied in `/var/tmp`), then
   compared byte for byte with the staged manifest.
3. `nix run .#quality` on the tree that includes that manifest.

## Non-claims

The gate is device-free. It proves formatting, lint, contract, freshness,
artifact, Cairn, and package-level behaviour. It does not open a device, start or
stop a service, reset a card, or exercise a live systemd.
