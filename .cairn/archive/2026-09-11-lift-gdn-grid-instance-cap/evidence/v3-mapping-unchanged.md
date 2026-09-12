# V3: the in-range program mapping is unchanged for both factories

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task V3.

## Result

The device-free gate now carries a permanent probe,
`probe_gdn_grid_mapping_unchanged` in `pkgs/qwen36/check.py`, that transcribes
both factory mappings line by line from the shipped sources and requires exact
equality over the entire in-range domain. Build
`/nix/store/5fbdlxzd…-qwen38-0.1.0.drv` (variant) reports:

```
PASS GDN grid instance cap assignment
PASS GDN grid in-range mapping unchanged
qwen36 device-free check: PASS
```

## What is compared

**Decode.** Pre-change `gdn-decode-fp32.patch` (at `8bf9f35^`) maps head `h`
to `CoreCoord{h / grid_y, h % grid_y}`, one head per core. The chunked factory
computes `heads_per_core = ceil(num_heads / ncores)`; for `num_heads <=
ncores` that is one, `active_cores = num_heads`, core `i` takes
`head_start = i`, and the core enumeration is the identical expression. The
probe compares the complete core-to-instance mapping as dicts.

**Prefill.** The pinned `distribute_scan` searches `NV` from `Vt` down with
the bound `BH * cand <= ncores` and maps instance `i` to
`CoreCoord{i % grid.x, i / grid.x}` with `(head, v-block) = (i / NV, i % NV)`.
The chunked variant keeps the NV search byte-for-byte (the patch comment
states it and the transcription asserts it) and adds
`per_core = ceil(BH*NV / ncores)`. In range, `BH*NV <= ncores` by the search
bound itself, so `per_core` is one and the mapping is identical; the kernel
derives `(h, vb)` from `wi` with the same expressions.

**Domain.** Grids 13x10 (measured sim) and 11x10 (planning constant), head
counts 1..129 across twelve values, `Vt` in {1,2,4,8}. Every in-range
combination must match exactly for both factories.

**Overflow behavior.** Both pre-change transcriptions refuse the B=8 launch
(they return the removed fatal's refusal); the chunked transcriptions must
accept it and partition all 192 instances with no gap or overlap.

## Negative teeth

The probe must distinguish real changes, so it fails when either mutation is
applied:

- floor instead of ceiling division in the decode chunk arithmetic;
- an unbounded NV search that accepts `BH * cand > ncores`.

Both mutants were written into the probe's own comparisons and are rejected
at multiple points of the domain, so the equality cannot pass vacuously.

## Non-claims

Source-level equivalence of the host-side mapping arithmetic. It does not
execute either factory on a device; V7 covers launch-level behavior on the
simulator and V8/V9 cover execution parity on hardware.
