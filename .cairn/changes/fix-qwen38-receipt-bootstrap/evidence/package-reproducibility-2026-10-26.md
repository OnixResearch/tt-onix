# Packaging observation: the qwen38 output is not bit-reproducible

Recorded 2026-10-26 at `2ba2c19`, while preparing the B=8 re-earn. This is a
separate honest note, not a blocker for the campaign and not a gate.

```text
nix build .#qwen38 --rebuild --no-link --builders ''
  -> exit 1
  error: derivation '/nix/store/af68qaa6rimz65g49yhgkvv031znrsp4-qwen38-0.1.0.drv'
         may not be deterministic:
         output 'out' ("/nix/store/wxnsxbhlx59vx4cfwi0881szxjfb32lm-qwen38-0.1.0") differs
```

The install check itself passes: the comparison runs only after a successful
build, and the existing output at that drv is from a passing build.

What is known:

- The output contains no `__pycache__`, `*.pyc` or post-epoch timestamps, and the
  execution-input manifest, the v3 profile and the wrappers are byte-stable
  across the two derivations compared in the bootstrap gate.
- The differing bytes were not isolated. `nix build --rebuild` reports only the
  output-level mismatch and discards its copy, and the old output cannot be
  renamed aside.
- The likely class is build-environment sensitivity in a phase that writes under
  `$out` (the install phase writes several JSON exports there), not model or
  receipt data. This is a hypothesis, not a finding.

Impact: identity-based binding (execution identity, input manifest, profile
bytes) is unaffected, because none of those moved. Re-deriving the exact package
store path from source to compare bytes is what does not currently work.

This note does not change the campaign, the six gates or the promotion gate.
