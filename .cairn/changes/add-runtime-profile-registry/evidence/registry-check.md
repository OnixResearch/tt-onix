# Registry check evidence

Captured 2026-10-26 on britton-desktop, from the tree at the state described in
`../proposal.md`. The registry check copies the tracked tree, regenerates the
export, and verifies every reference; its verified content is the
`model-targets/` files and the receipts, so a later edit elsewhere changes the
derivation path but not the verified result.

## Command

```text
nix build --no-link --print-out-paths --builders '' .#checks.x86_64-linux.model-runtime-profiles
```

Exit status 0. The aggregate `nix run .#quality` run is recorded separately in
`quality.md`; `--builders ''` is used only for this bounded local run because the
host's configured remote builder hits the documented `ENAMETOOLONG` slot-lock
failure, which `nix run .#quality` repairs through its matched hook.

## Summary written by the check

```text
schema_version=1
profiles=qwen38,nanbeige43,rwkv7-ttnn,metalium-rs-rwkv7
authority_documents_verified=2
receipts_verified=2
fixtures_rejected=7
rwkv_model_target_identity=d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88
export_blake3=23b889c428323cd039f3d7b8b368afaeae252c7320fef0f23ddcdc8dcc1289e9
```

The two authority documents recomputed are
`pkgs/qwen36/generated/authority.json` and
`pkgs/nanbeige43/generated/authority.json`. The two receipts recomputed are the
B=8 continuous-slot and prefix-cache admissions the served Qwen profile names.

## Negative fixtures, each refused with its named rule

```text
invalid-runtime-unknown-field.ncl
       extra field `undocumented_launch_flag`
invalid-runtime-missing-field.ncl
error: missing definition for `mode`
invalid-runtime-mode.ncl
       runtime profile mode must be serve or job
invalid-runtime-authority-blake3.ncl
error: contract broken by the value of `authority_blake3`
invalid-runtime-serve-device-ids.ncl
       serve runtime profile mesh physical device ids must be exactly [0,1]
invalid-runtime-serve-without-port.ncl
       serve runtime profile must declare a port
invalid-runtime-duplicate-name.ncl
       runtime profile names must be unique
```

Each fixture also asserted that no JSON was emitted despite the failure
(`test ! -s invalid.json`), so a partially evaluated export cannot pass.

## Notes

- The `serve-device-ids` fixture reverses the order of the same two ids, so it
  exercises the order rule rather than the "ids are 0 or 1" rule.
- The `duplicate-name` fixture appends the first profile to the array, which is
  the realistic way a copy-paste edit would break the registry.
