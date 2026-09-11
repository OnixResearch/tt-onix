# V5 status after the overlay fix

Recorded 2026-09-11. Supersedes the "Finding 2" section of `v5-two-findings.md`
and refines "Finding 1".

## Finding 1 is fixed and the fix is verified

`pkgs/qwen36/default.nix` now restores a list of patched JIT directories from
the installed runtime instead of one hard-coded path:

```sh
for patched_jit_path in \
  ttnn/operations/transformer/gated_delta_decode \
  ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels
do
  ...
done
```

The loop fails closed if a listed path is absent from the installed runtime, so
a future patch cannot be silently dropped by the full-tree copy.

**Verification:** the shared package `.#qwen38` builds and passes its device-free
check with this change in place. Log is 269 lines, ending
`qwen38 device-free check: PASS` and `installCheckPhase completed in 1 minutes
38 seconds`. Exit code `0`.

The mechanism is now understood more precisely than the first note stated. The
prefill **guard** is host-side C++ compiled into `libttnn.so`, so it comes from
the patched derivation and was never at risk. The scan **kernels** are JIT
sources read from `libexec`, so they are the ones the full-tree copy reverted.
The mismatch would have been a patched factory passing instance arguments to
reverted kernels.

## Finding 2: the variant package build still fails, and the cause is narrower

| Item | Value |
| --- | --- |
| Variant drv (before the fix) | `/nix/store/xlxcgxf2kn96s6jxjrrn3yfl7pn1l127-qwen38-0.1.0.drv` |
| Variant drv (after the fix) | `/nix/store/l3y1ga213gnp5j1cadflcm2ia3fmbj2w-qwen38-0.1.0.drv` |
| Result | builder exit `1` |
| Log length | 93 lines, stops at `Running phase: installPhase` |

Excluded by test:

- **Not the remote builder.** The same failure reproduces locally with
  `--builders ""`.
- **Not log truncation.** `nix log` shows the same content.
- **Not the Tracy tools.** Two different overrides produced one derivation hash.
- **Not the new restore loop.** The shared package passes with it, and the
  variant's own log predates the loop's commands.
- **Not the model-tree patches.** Every hunk applies.

**New and decisive observation.** The build was re-run with `set -x` at the top
of `installPhase`. The trace is 5767 lines, and a search for the phase body
commands — `mkdir -p`, `cp -r`, `cp -as`, `find`, `nickel`, `makeWrapper`, and
the new loop — returns **zero matches**. The trace contains only `runHook`
machinery and teardown. The phase therefore fails before its first body command
executes.

That points away from the body entirely and toward the phase entry itself, which
is `runHook preInstall`. The next step is to check whether the package or
stdenv defines an implicit `preInstall` that depends on an input the variant
changed, and to print the phase environment at entry.

`set -x` was removed and is not committed.

## Consequence

Task `V5` stays open. No physical run should be attempted: a package that cannot
complete its own device-free check cannot produce trustworthy physical evidence,
and the variant package is currently the only vehicle for that check.

The two C++ mechanisms remain compile-verified and unexecuted.
