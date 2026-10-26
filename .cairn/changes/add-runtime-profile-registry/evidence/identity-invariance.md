# Identity invariance evidence

The change is additive. The model-target registry, its factory check, the
model-target identity pins, the Qwen package, and every existing receipt are
untouched, so `profile_identity` and the pinned rwkv7 persistent-decode profile do
not move.

## Derivation path

```text
$ nix eval --raw .#qwen38.drvPath
/nix/store/af68qaa6rimz65g49yhgkvv031znrsp4-qwen38-0.1.0.drv
```

Unchanged from before the change (the required value).

## File digests, before and after

```text
$ b3sum model-targets/profiles.ncl model-targets/generated/profiles.json
3883a12a5daeb8c1578f356d7db2a31456783746d024d70ee13094da4df863de  model-targets/profiles.ncl
88ad86c4d1ca4b879686a78638ddb7ebdbc771717274072b4f9862fc59216aa6  model-targets/generated/profiles.json
```

```text
$ git status --porcelain model-targets/profiles.ncl model-targets/generated/profiles.json model-targets/check.nix
(no output)
```

`model-targets/check.nix:10` still pins
`expectedRwkvProfileIdentity = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88"`,
`model-targets/check.nix:180` still asserts the live identity equals it, and
`pkgs/rwkv7-p150x2/persistent-decode-profile.ncl:6` still admits that same value.
The new registry references it as `model_target_blake3` for both RWKV profiles
instead of re-deriving it, and the registry check fails if the reference and the
live identity disagree.

## What this proves and what it does not

It proves this change did not alter model identity, the model-target factory, or
the Qwen package. It does not prove any hardware behaviour, and it makes no
admission claim.
