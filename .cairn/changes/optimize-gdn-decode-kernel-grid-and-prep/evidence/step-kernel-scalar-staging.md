# Step-kernel instance loop: destination capacity, rejected levers, and the staged scalar reads

Recorded 2026-10-21 on physical PCIe devices 0 and 1 (two P150a, firmware 19.11.0.0,
driver 2.10.0). This record applies the change's instance-loop question to the route the
service actually serves: the `step` recurrence kernel in
`models/demos/blackhole/qwen36/tt/gdn/kernels/`, not the TT-Metal fused op that the
change's proposal names. That op is still selected only by the older `full` route.

## Method

Three measurements, each a separate throwaway harness:

1. **Numerical parity.** `GdnDecodeStep.conv_prep` + `GdnDecodeStep.recurrence` against
   `reference_step` (float64) at `(batch, max_batch)` of `(1,1)`, `(4,4)`, `(3,8)`,
   `(8,8)`, and `(16,16)`. The widths 8 and 16 cases put two and four instances on each
   core, so the instance loop and chunked reader/writer are exercised.
2. **Kernel timing.** Trace capture of the recurrence alone, 200 replays per sample,
   seven samples, median reported. Two paired rounds, served package then candidate
   package, in one session each.
3. **End-to-end parity.** The production `BatchedTracedDecodeAdapter` at width 1, 16
   greedy tokens, compared token by token with the served package.

The served package is `qxwnq7213yavfq3skr21vanwwfj03bk9-qwen38-0.1.0` (execution
identity `9mm1yvljb8pj8jvgfd5r1prfgjvp0k21`). Every candidate below differs from it only
in the step kernel sources recorded here.

## Destination-register capacity (recorded because it constrains every batching idea)

With `fp32_dest_acc_en=True` and `dst_full_sync_en=False` (the recurrence's shipped
compute config), only **four** destination tiles are usable per acquire/pack section. A
probe that changed nothing but the existing state-decay loop from three tiles to seven
destinations produced `state rel err=3.47` against a reference of `1.4e-3`: the extra
destinations do not survive the pack section.

Setting `dst_full_sync_en=True` doubles the usable depth to **eight**. An eight-register
batch is bit-exact at every tested width. A sixteen-register batch still fails
(`state rel err=2.0`) under full sync, so the practical ceiling is eight.

## Rejected levers

### Batched destination registers with full sync

`step_compute.cpp` batches the four tile loops that dominate the per-instance sync chain:
state decay (6 rounds to 2), delta (4 to 2 at eight registers), the rank-1 state update
(4 to 2), and the gated norm (4 to 2). The operations, operands, and per-tile order are
unchanged, and the parity values are identical to the reference to four significant
figures.

Timing says it does not pay:

| Width | Served recurrence | Full-sync original | Full-sync + 8-register batches |
| ---: | ---: | ---: | ---: |
| 1 | 54.1 us | 55.7 us | 55.2 us |
| 8 | 113.7 us | 117.1 us | 115.6 us |
| 16 | 198.8 us | 203.7 us | 200.5 us |

Full sync costs about 2.5-3.0 percent (it removes the half-destination ping-pong between
math and pack), and the reduced sync count recovers most but not all of it. The net is a
one-to-two percent regression against the served kernel. Rejected.

### Double-buffered input and output circular buffers

Inputs 0-8 and the compute-to-writer buffers 16/22 were given two instances of depth so
the reader and writer could work one instance ahead. The state stays exact at every
width, but the gated output is wrong at every second instance per core (all odd heads at
widths 8 and 16: `gated rel err=0.98`). Single-buffering q, z, or the output buffer one
at a time did not change the error, so the failure is not a single stale page and its
root cause is not established here. Rejected on correctness.

### Scalar-read staging (kept)

The reader resolved the four per-instance scalars (`a`, `b`, `dt_bias`, `-exp(A)`) with a
64-byte NOC read and a full `noc_async_read_barrier()` each: four NOC round trips per
instance on top of the row and state reads. The change stages all four reads into the
existing scratch page and folds them behind the barrier that already covers the row and
state reads. Values, word offsets, and destination pages are unchanged.

## Results

Parity, candidate against the float64 reference (all cases pass):

| batch / max | state rel err | gated rel err |
| --- | ---: | ---: |
| 1 / 1 | 1.237e-3 | 1.887e-3 |
| 4 / 4 | 1.937e-3 | 1.972e-3 |
| 3 / 8 | 1.348e-3 | 1.902e-3 |
| 8 / 8 | 1.359e-3 | 1.935e-3 |
| 16 / 16 | 1.425e-3 | 1.964e-3 |

The served package reproduces these values to four significant figures at every case the
two harnesses share, so the change is numerically transparent.

Paired traced timing (two rounds; recurrence alone):

| Width | Served | Staged scalars | Change |
| ---: | ---: | ---: | ---: |
| 1 | 54.1 / 54.0 us | 51.2 / 51.5 us | -5.1 percent |
| 8 | 113.7 / 113.4 us | 107.8 / 108.3 us | -4.7 percent |
| 16 | 198.8 / 198.7 us | 186.5 / 186.7 us | -6.1 percent |

Including `conv_prep`, the pair moves 72.9-73.1 to 70.4-70.7 us at width 1, 129.5 to
123.8-124.0 us at width 8, and 214.4-214.5 to 201.3-201.5 us at width 16.

End-to-end at width 1, the first sixteen greedy tokens are identical to the served
package (`89454, 4384, 20635, 369, 264, 8751, 7059, 303, 3061, 14246, 11, 7695, 948,
14131, 440, 61911`). Median step time for that single 16-token session was 49.9 ms served
and 44.4 ms staged; that is one uncontrolled session per package and is **not** a timing
claim.

## What this establishes, and what it does not

- The staged scalar reads remove three NOC round trips per instance, are bit-exact, and
  cut the recurrence by 4.7-6.1 percent in a paired traced measurement. That is the
  kept change.
- The per-instance destination-register batching family is closed at this geometry: four
  destinations without full sync, eight with it, and full sync costs more than the
  batching saves.
- The double-buffered-prefetch family has an unresolved correctness defect and produced
  no admissible timing result.
- The recurrence is 51-187 us per layer at widths 1-16, or about 2.5-9.0 ms of a decode
  step across 48 GDN layers. The instance loop is therefore a small share of the step,
  unlike the composite path the change's proposal measured. Further step-level work
  belongs in the projection/MLP path, not here.
- No admission binding moves. No width-8 or width-16 performance claim is made. The
  single-session model probe is an end-to-end parity check, not a controlled timing
  comparison.
