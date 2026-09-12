# I6: the variant, the assignment core, and both factories are bound into the package execution identity

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task I6.

## What was broken

The V5 content assertions (commit `0c8c7bc`) ran unconditionally in the
qwen36 package installPhase. They require the chunked kernel markers
`instance < head_count` and `instance < wi_count`, which only exist on the
GDN grid variant runtime. The shared `qwen38` package therefore failed to
build at HEAD:

```
nix build .#qwen38 --no-link
error: Cannot build '/nix/store/imf3l0cwahh3skcmnl1qls4p5d3ck5y1-qwen38-0.1.0.drv'.
Reason: builder failed with exit code 1.
```

The V5 evidence had tested only the variant, so the shared regression went
unnoticed until this session. I6 is exactly this binding done correctly.

## The binding

`pkgs/qwen36/default.nix` gains a `gdnGridRuntime ? false` argument. The
variant call in `flake-modules/onix-packages.nix` passes
`gdnGridRuntime = true`. The install assertions then fail closed in BOTH
directions against the runtime the package actually carries:

- Declared variant (`gdnGridRuntime = true`):
  - decode kernel must carry `instance < head_count`;
  - all three scan kernels must carry `instance < wi_count`;
  - the runtime's `lib/_ttnncpp.so` must NOT contain the pre-change decode
    guard string `num_heads exceeds the available compute grid`.
- Declared shared (`gdnGridRuntime = false`, the default):
  - neither chunked marker may appear in any kernel;
  - the runtime's `lib/_ttnncpp.so` MUST still contain the guard string its
    admitted receipts rely on.

A reverted patch can no longer build quietly on either side.

The remaining identity inputs were already bound: the assignment core
(`service/qwen36_service/gdn_grid_core.py`) is part of the stable service
source in `executionIdentityLocalInputs`, the variant tt-metal runtime is an
explicit `executionIdentityInputs` entry through the overridden `tt-metal`
argument, and both factories' kernels are the asserted content above.

## Evidence

Positive, both packages build:

```
nix build .#qwen38 .#qwen38-gdn-grid --no-link --print-out-paths
/nix/store/4r2dr9nqi3i1czzl030mirfq600cc7ca-qwen38-0.1.0
/nix/store/igycad7spr0c2v0ln9552dysn18sc8bs-qwen38-0.1.0
```

Negative 1 — variant runtime declared as shared:

```
shared runtime carries a chunked scan kernel marker in dataflow/reader_chunk_gdn_scan.cpp
```

Negative 2 — shared runtime declared as variant:

```
GDN grid variant runtime lost the chunked scan kernel marker in dataflow/reader_chunk_gdn_scan.cpp
```

Both negative builds exited 1 in installPhase with the named diagnostic; the
test edits were reverted and both packages rebuilt green.

## Non-claims

This binds build-time content identity. It does not execute either factory,
does not move any admission binding, and does not claim the chunked kernels
are correct — that is V8 and V9.
