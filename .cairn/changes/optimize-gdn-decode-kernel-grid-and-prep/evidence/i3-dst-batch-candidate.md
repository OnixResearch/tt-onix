# I3 candidate: batched destination registers in the fused GDN decode kernel

Recorded 2026-09-12. Candidate built as a variant runtime patch. Nothing here is a
measurement; the results section stays empty until the matched physical screen runs.

## Why this shape of change

`evidence/loop-cost-nature.md` bounds the loop's per-instance cost two to three
orders of magnitude above its arithmetic, so the cost is the dependency chain, not
the multiply-accumulates. The reader is not on that chain, which is why prefetching
it changed nothing. The chain inside the compute kernel is the per-tile sync: each
tile loop acquires the destination registers, runs one tile, commits, waits for the
math, packs, and releases -- then repeats for the next tile.

## The change

`pkgs/tt-metal/gdn-decode-dst-batch.patch`, applied after the grid-chunk patch so it
sees the instance loop that patch introduces. Seven tile loops in the compute kernel
batch the destination registers across the whole loop:

- `state_decay`, `outer_scaled`, and `state_new` (8 tiles each),
- `outer` (4 key by 2 value tiles = 8),
- `v_read` and `output` (2 value tiles each, with 4 key tiles accumulated per
  value tile),
- `delta` (2 tiles).

Each loop now acquires once, issues every tile into its own destination register,
commits once, waits once, then packs every register. The per-tile operations, their
operands, and their order are unchanged; only the synchronisation granularity
changes. The transpose loop and the single-tile `exp(g)` keep their original shape.

Guard rails:

- the patch touches exactly one file, the decode compute kernel;
- the variant package asserts the deployed kernel carries the batched markers and
  that exactly two `tile_regs_acquire()` calls remain (the single-tile `exp` and the
  transpose loop);
- the shared runtime asserts it carries **neither** the chunked-loop marker nor the
  batched marker, so a reverted patch cannot build quietly on either side.

## What would make this wrong

- **Parity.** The accumulation order per destination register is unchanged, but a
  mistake in a destination index would silently permute tiles. Exact token parity at
  width 8 is therefore read before any timing, per the plan's V6 rule.
- **Destination capacity.** `key_tiles = 4`, `val_tiles = 2`, `state_tiles = 8` at
  this geometry, so the widest batch uses 8 of the 16 destination registers. If a
  future geometry raised `state_tiles` above 16 the batched form would not fit and
  the guard's sync-count assertion would need revisiting rather than relaxing.
- **Instruction issue depth.** The math thread now issues eight operations between
  syncs instead of one. If the instruction queue cannot hold that, kernels would
  fault rather than run slowly, which the screen would show immediately.

## Acceptance

Screened with `scripts/qwen38-step-probe.sh` at width 8 against the reference
variant package in this order:

1. exact token parity for all eight prompts;
2. median step at or below **78.0 ms**, the width-8 retention bound, against the
   reference median of 78.9 ms.

The candidate is rejected on either failure, and the admitted kernel stays the
served route.
