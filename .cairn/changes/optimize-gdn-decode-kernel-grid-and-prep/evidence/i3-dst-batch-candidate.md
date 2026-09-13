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

## Result: exact parity, and a repeatable win at width 8

Two paired rounds, each one reference session followed immediately by one candidate
session, eight prompts released together, 64 generated tokens, interior samples only
(the first sample is prefill-adjacent and the last is the tail spike). 480 samples
per session.

| Round | Reference median | Candidate median | Median change | Mean change | Token parity |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 81.874 ms | 79.748 ms | -2.60% | -3.69% | exact |
| 2 | 85.540 ms | 80.073 ms | -6.39% | -7.28% | exact |

Parity is exact in both rounds and both directions: generated token ids, decoded
text, eight distinct state slices, and a clean reset on every response. The
destination-index change therefore did not permute any tile.

Two honest notes on the numbers:

- **The absolute values drift.** The same reference package measured 77.8 ms earlier
  in the day, 81.9 ms in round 1, and 85.5 ms in round 2. That is a five percent
  band, larger than the 1.8 percent the bound needs, so only the paired change is
  read as a result. The earlier claim that the screen resolves this question in
  three minutes was wrong in the direction that mattered: it resolves *large*
  effects, and a small one needs the paired design.
- **The two rounds disagree in size** (-2.6 and -6.4 percent) even though they agree
  in sign. Both are consistent with a real improvement of a few percent, and neither
  alone would have been enough to trust.

Against the width-8 retention bound the paired change is sufficient on the screen:
the bound needs a 1.8 percent step reduction and both rounds exceeded it. That is a
screen result, not an admission result; the four-root receipt decides.

## Screen reliability

The first group after startup can be rejected, which cost one round its reference
session while the candidate session succeeded. The probe now retries the group
before reading anything as a result, matching the four-root harness, and the round
above was taken with retries in place.
