# Package check evidence

`pkgs/tt-model` runs its assertions as an install check, so
`nix build .#tt-model` (and `nix build .#checks.x86_64-linux.tt-model`) fails if
any of them regresses. This records what the check exercised on 2026-10-26.

## Command

```text
nix build --no-link --print-out-paths --builders '' .#tt-model
```

Exit status 0. The package installs `bin/tt-model`,
`share/tt-model/tt_model.py`, and `share/tt-model/runtime-profiles.json`.

## Assertions

1. **The shipped registry is the reviewed export.** `cmp` against
   `model-targets/generated/runtime-profiles.json`.
2. **The program text names no hardware tooling.** `grep -R -E
   '/dev/tenstorrent|tt-smi|ttwkv7-owner-control'` over `$out/bin` and
   `tt_model.py` returns nothing. The registry data is excluded, because it
   records the device paths a future launcher needs; the `cmp` above proves that
   data is exactly what the registry check reviewed.
3. **A skipped source degrades, it does not lie.** With
   `TT_MODEL_SYSTEMCTL=none` the verdict is `degraded`, `unavailable` is
   `["unit"]`, and there are no problems; exit 0.
4. **Every source readable and absent is `none`.** A stand-in `systemctl`
   printing `LoadState=not-found` yields `verdict: none`, `unit: not-found`;
   exit 0.
5. **A dangling operator symlink is broken.** `current -> /nix/store/does-not-exist-qwen38-0.1.0`
   yields `"state": "dangling"`, the exact target in the JSON, and
   `problem: operator symlink: <path>: dangling symlink to <target>`; exit 1.
6. **An unrooted record is broken.** A syntactically valid record whose package
   has no GC root yields `verdict: broken` and `record GC root is missing`;
   exit 1.
7. **A rooted record is managed, and a disagreeing legacy source is a
   contradiction.** With the GC root in place the verdict is `managed`; adding
   an operator symlink that resolves elsewhere makes it `broken` with
   `resolves to …`.
8. **An unknown profile in a record is refused.** `record names unknown profile
   'qwen38-not-in-the-registry'`; exit 1.
9. **A registry that does not match its contract is a usage error.** Schema
   version 2 and a duplicate profile name both exit 2 with
   `schema_version must be 1` and `duplicate profile name 'qwen38'`.

## Observed output excerpts

```text
verdict: degraded
  "state": "not-found",
verdict: none
  "state": "dangling",
  "reason": "dangling symlink to /nix/store/does-not-exist-qwen38-0.1.0",
problem: operator symlink: /build/home/.local/state/qwen38-p150x2/current: dangling symlink to /nix/store/does-not-exist-qwen38-0.1.0
  "verdict": "broken"
  "verdict": "managed"
  "reason": "record names unknown profile 'qwen38-not-in-the-registry'",
tt-model: /build/bad-schema.json: schema_version must be 1, got 2
tt-model: /build/duplicate.json: duplicate profile name 'qwen38'
```

## Non-claims

The check runs in a sandbox with no system bus and no device nodes. It proves the
read-only paths and the refusal logic; it does not prove behaviour against a live
systemd, and it does not exercise a record written by a future switch.
