# I3 candidate: reader prefetch across the instance loop — exact parity, no benefit, rejected

Recorded 2026-09-12 for change `optimize-gdn-decode-kernel-grid-and-prep`.

## The candidate

`evidence/rejected-instance-prefetch.patch` (kept here, removed from the variant's
patch list). It gave the reader-side circular buffers (q, k, v, beta, g, initial
state) a second instance's room when `heads_per_core > 1`, so the reader could
stream instance `i+1` while compute worked on instance `i`. With one instance per
core the depth stayed one, so the in-range layout and numerics were unchanged. No
kernel was touched, which is why exact parity was the expectation.

## V6: exact parity passed at both looping widths

Matched four-root protocol (`single,batch,batch,single`, one warmup plus three
measured runs per prompt) on the candidate package:

| Width | Token parity | Text parity | State slices | Clean reset | Active prefix |
| --- | --- | --- | --- | --- | --- |
| 8 | exact | exact | distinct | yes | yes |
| 16 | exact | exact | distinct | yes | yes |

So the scheduling change is numerically transparent, as designed.

## V7: the step time did not improve

| Width | Measurement | Candidate | No-prefetch reference |
| ---: | --- | ---: | ---: |
| 8 | batch step | 88.5 ms | 86.8 ms |
| 8 | per-user decode | 12.20 t/s | 12.60 t/s |
| 8 | retention ratio | 0.7440 | 0.7373 |
| 8 | aggregate | 97.59 t/s | ~97-101 t/s |
| 16 | batch step | 101.0 ms | 99.6 ms |
| 16 | per-user decode | 10.55 t/s | 10.56 t/s |
| 16 | retention ratio | 0.6379 | 0.6337 |
| 16 | aggregate | 168.80 t/s | 168.89 t/s |

At width 16 the aggregate and per-user figures are **identical to three
significant figures** (10.55 against 10.56 tokens/second per user, 168.80 against
168.89 aggregate). At width 8 both directions are inside the run-to-run spread
already recorded for this width (retention 0.7352, 0.7373, 0.7580 across earlier
runs; batch step 86.8 and 91.4 ms). The candidate is therefore neutral at best,
and 1.4-2.0 percent worse on the batch step at both widths.

## Why it did not help

The reader was not on the critical path. A4 already showed the state traffic is
about 2.7 GiB/s with the state L1-resident, so the per-instance read is small next
to the compute chain, and the circular-buffer flow control absorbs it without the
prefetch. The instance loop's cost is **compute-side**, which is what the
base-plus-marginal model in `evidence/retention-scaling.md` says: the marginal
per-user cost is arithmetic and latency inside the per-instance chain, not read
bandwidth.

## V8 and D2: the candidate is rejected and the admitted kernel stays

- removed from the variant patch list; the variant no longer carries it;
- the served route is unchanged: production runs the shared runtime, and the
  variant remains a candidate package that no service selects;
- the admitted kernel remains the reference everything is measured against.

## Identity note (recorded rather than papered over)

The revert restores the same tt-metal runtime the reference receipts used, but the
package execution identity moved from `nf2pd9d6` to `wjlvbcjk` because the
occupancy planner (`gdn_occupancy_core.py`) was added to the service source in
between. That module is **unreferenced** — no service module imports it, only the
device-free probes do — so the served path is behaviorally identical and the
tracked receipt remains valid in substance. It is nonetheless stale by identity,
and the next admission decision should refresh it on the current identity rather
than reuse the old hash.

## D1: non-claims

- No speedup claim: the candidate produced none.
- No admission movement and no bound change.
- The parity result applies to this candidate at widths 8 and 16 on this runtime.
- The B=8 and B=16 bounds still fail on the admitted kernel, and this candidate did
  not move them: speedup 5.9516 (B=8) and 10.2060 (B=16), retention 0.7440 and
  0.6379, TTFT 10.31 and 11.33.

## What the evidence leaves as the next lever

The loop is compute-side, so the remaining candidates are:

1. **Interleave instances inside the compute kernel** — two instances' independent
   stages in flight on one core — rather than feeding them faster;
2. **Reduce per-instance arithmetic** (I4's prep folding, or replacing the
   per-instance `k` transpose with one transpose per head group), which attacks the
   fixed per-instance cost the model in `evidence/a1-a4-attribution.md` identified.
