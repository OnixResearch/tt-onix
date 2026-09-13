# Is the GDN loop latency-bound or arithmetic-bound?

Recorded 2026-09-12. Analysis of existing measurements. No new claim about the
kernel; the point is to choose the right next candidate instead of guessing.

## What the numbers already say

`x_1 = 0.549 ms` is the width-1 GDN cost per layer with one instance per core
(component probe: 26.34 ms over 48 layers). At `k = 2` the per-instance marginal is
at most 0.39 ms and at `k = 4` at most 0.33 ms, so the loop's cost per instance
*falls* as more instances share a core.

Take the second and third widths as a test of the two possible readings of that
fall:

- **Reading A, latency:** the per-instance marginal falls because more instances in
  flight hide each other's stalls. `step = C + 48k x_k`, with `C` nearly constant in
  the slot count. Width 8: `79.4 = C + 96 x_2`; width 16: `94.7 = C + 192 x_4`. With
  `C = 42 ms`, `x_2 = 0.385` and `x_4 = 0.275` -- a smooth, monotone 30 percent drop
  per doubling of `k`.
- **Reading B, non-GDN slots:** the fall is really the non-GDN path charging per
  active slot. Then width 16 gives `C(16) = 94.7 - 192 x` which, at a constant
  `x = 0.385`, is 20.8 ms against `C(8) = 42 ms`: the non-GDN cost would have to
  *halve* when the slot count doubles.

Reading B needs a quantity to fall when the work it measures doubles. Reading A
needs one smooth monotone curve. The data favours Reading A, which puts the loop at
about 37 ms of the 79.4 ms width-8 step, or 47 percent.

## An order-of-magnitude check on the loop's own cost

Rough arithmetic per instance per layer, counted in 32x32 tile operations: three
matmuls of a 1x4 tile activation against a 4x4 tile operand (about 192
multiply-accumulates of a 32x32x32 tile each), about 60 elementwise tile operations
(decay, delta, outer scale, state add, `exp`, `beta`), and 16 transpose tiles. That
is roughly 270 tile operations per instance per layer.

At the Tensix tile rate of one 32x32x32 operation per cycle or two, this is on the
order of a thousand cycles, or about one microsecond. The measured per-instance,
per-layer figure is 549 microseconds at `k = 1` and 385 at `k = 2`.

The estimate is an order-of-magnitude bound, not a measurement, and it ignores
unpack, pack, and L1 traffic. Even so, the gap is two to three orders of magnitude.
**The loop cannot be arithmetic-bound.** Its cost is a chain of dependent stages:
unpack, matmul wait, transpose, SFPU, pack, and the circular-buffer and semaphore
handshakes between five RISC processors. That is why feeding the reader faster
changed nothing.

## What this changes about the plan

- **I3's interleaving direction is confirmed as the right lever**, and it should be
  entered as a *two-stage hoist* rather than a full rewrite: move the next instance's
  independent opening stages (transpose and state decay) in front of the current
  instance's consuming stages. Same idea as the full pipeline, far smaller edit.
- **I4 (folding the prep chain) is de-prioritised, with a reason.** Its measured
  total is 0.470 ms per step against a 1.4 ms acceptance gap, and it reduces
  arithmetic in a loop that is not arithmetic-bound. It should not consume a rebuild
  before an interleaving candidate has been screened.
- The reader-prefetch rejection now has a mechanism, not just an observation.

## The discriminating experiment, if the estimate is not trusted

Split the state at fixed width: doubling the instances halves the per-instance
arithmetic without changing the depth of the dependency chain. If the step time is
unchanged, the chain dominates; if it falls with the arithmetic, the estimate above
is wrong. That is exactly the split the occupancy planner already computes -- so I2
now has a measurement purpose beyond occupancy, and its policy input is already
identity-bound and fail-closed.
