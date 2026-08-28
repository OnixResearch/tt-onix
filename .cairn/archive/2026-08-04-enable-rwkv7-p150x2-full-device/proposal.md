## Why

The current repository proves one physical ttWKV7 operation, but it does not package a full RWKV-7 model for both Blackhole cards. A stale `onix-core` branch contains a replicated P150x2 TT-NN runtime and evidence classifier. The third hardware attempt opened devices 0 and 1, but its wrapper set `TT_METAL_WATCHER=0` while profiling. TT-Metal treats every present `TT_METAL_WATCHER` value as enabled, so Blackhole BRISC firmware exceeded its code region before a model operation ran.

`tenstorrent.nix` is now the sole authority for reusable Tenstorrent packages. The reusable runtime and evidence contract must move here before another machine-bound attempt.

## What Changes

- Add the package-matched full-model RWKV-7 TT-NN runtime for the exact replicated `1x2` P150 mesh.
- Add the reusable evidence classifier and bounded command package.
- Remove Watcher and kernel DPrint from profiler process environments instead of assigning disabled-looking values.
- Add pure environment-policy validation with positive and negative tests.
- Add device-free source, closure, package, and retained attempt-3 classification checks.
- Keep machine-specific owner isolation and fresh hardware execution outside this change.

## Impact

- **Files**: `pkgs/rwkv7-p150x2/`, `pkgs/rwkv7-p150x2-evidence/`, `checks/`, `flake-modules/onix-packages.nix`, `README.md`, and native Cairn lifecycle files.
- **Testing**: Cairn gates, package self-tests, profiler-policy negative tests, source and closure checks, retained attempt-3 classification, existing RWKV parity checks, formatting, and flake evaluation.
- **Hardware**: This change does not stop services, open devices, or claim a successful two-card generation.
