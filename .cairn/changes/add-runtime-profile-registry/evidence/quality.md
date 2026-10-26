# Quality gate evidence

## Command

```text
nix run .#quality
```

Run from the repository root with `TMPDIR=/var/tmp` (the host's `/tmp` is full).
Exit status 0. The launcher bootstrapped `.#nix-patched-build-remote`, evaluated
both supported systems, and built the aggregate; the log shows the aggregate
derivation being built on the configured remote builder
(`grpc://aspen1.local:50051`) and its output copied back, so the matched hook was
exercised rather than bypassed.

## What the run covers for this change

- `tenstorrent-nix-quality`: `nixfmt --check` over every maintained `.nix` file,
  including `model-targets/runtime-check.nix` and `pkgs/tt-model/default.nix`.
- `tenstorrent-python-quality`: `ruff check` and `ruff format --check` over
  `pkgs/tt-model/tt_model.py`.
- `tenstorrent-shell-quality`: ShellCheck and `bash -n`; the new package's
  install check is shell.
- `tenstorrent-quality-policy`: `quality/required-checks.ncl` exports exactly
  `quality/generated/required-checks.json`, now with `model-runtime-profiles` and
  `tt-model` in the closed catalog.
- `tenstorrent-artifact-quality`: the aggregate manifest matches the tree,
  including the five new portable-evidence files and the new generated export.
- `tenstorrent-cairn-quality`: `cairn validate` plus the verdict adapter, which
  requires a true verdict, valid root resolution, and no issues.
- The required package checks, including `model-runtime-profiles` (regenerate,
  compare, recompute every digest, reject every fixture), `tt-model` (its install
  check), `qwen36` (unchanged derivation), `qwen36-nixos-service`, and
  `metalium-rs-rwkv7-tokenizer`.

## Order of operations

1. `nix build .#checks.x86_64-linux.model-runtime-profiles` and
   `nix build .#tt-model` for the bounded local iterations
   (`--builders ''`, because the host's remote builder hits the documented
   `ENAMETOOLONG` slot-lock failure outside the quality launcher).
2. `nix run .#quality`: exit 0.
3. The artifact manifest was regenerated from a clean copy of the commit-ready
   tree (`git archive HEAD` plus the staged patch applied in `/var/tmp`), then
   `nix run .#quality` was repeated on the tree that includes that manifest.

## Non-claims

The gate is device-free. It proves formatting, lint, contract, freshness,
artifact, Cairn, and package-level behaviour. It does not open a device, start or
stop a service, or exercise a live systemd.
