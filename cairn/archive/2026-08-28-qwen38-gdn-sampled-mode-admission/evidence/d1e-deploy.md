# D1e: sampled-serving generation deployed (2026-08-28)

## Result

Production runs the sampled-serving package with the gate off.

- System: `43sm969d...` (onix-core `0f8210f9`, pinning tenstorrent.nix
  `a9e5b5f` on main)
- Package: `/nix/store/cdvkpnr3...-qwen36-0.1.0`
- Deployed execution identity: `/nix/store/llvy8p6f...-qwen36-execution-identity-0.1.0`
- Unit `qwen38-p150x2.service`: active, ready, greedy smoke passes with
  route `off`, no switch, the exact known off-path tokens
  `[11751, 13, 198, 760, 6511, 314]` at 16.28 tok/s.
- Sampled requests with the gate off still answer HTTP 400: today's
  greedy-only contract is preserved until the owner enables
  `QWEN36_SAMPLED_FUSION`.

## Deploy incidents

1. The first switch lost a deploy race: the pin update sat uncommitted in
   the onix-core working tree while a parallel session switched the host
   from main without the pin, reverting the unit to the old package. The
   fix was to commit and push the pin (`0f8210f9`) before switching again.
   Parallel deploys from origin/main now carry the new pin forward.
2. `switch-to-configuration` exits non-zero on unrelated service failures
   (`celld.service` from a parallel session's machine state); the qwen unit
   itself starts and reports ready.
3. Cold-start warmup for the new runtime took about six minutes on first
   boot (fresh JIT caches); the warm restart is immediate.

## Identity note

The deployed identity (`llvy8p6f`) differs from the locally screened
identity (`cpy7yayc`) for two reasons: onix-core overrides nixpkgs (python
3.14 environment, rebuilt tt-metal), and local `path:` builds carry the
service `__pycache__` into the identity while git flakes filter it. The
service source files are identical. The deployed-identity greedy smoke
above binds the deployment the same way the prior staged-prefill deploy
did. The `__pycache__` leakage into local identities is a hygiene debt for
a follow-up: local screens should build from committed trees.

## Remaining

- Owner act: enable `QWEN36_SAMPLED_FUSION=1` on the unit (module option
  or environment drop-in) when sampled traffic is wanted.
- D2: archive the change with the full evidence chain.
