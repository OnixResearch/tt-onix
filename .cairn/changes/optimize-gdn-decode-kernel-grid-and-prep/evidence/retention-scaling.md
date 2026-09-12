# Retention scales with instances per core, not with width

Analysis of the four matched admission receipts this repository holds. No new
hardware run: these are their own numbers.

## The measurements

| Receipt | Width | Instances (24xW) | Instances per core | Single ms/step | Batch ms/step | Per-user retention | Extra ms per added user |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B=4 admitted 2026-08-26 | 4 | 96 | 1 | 75.2 | 91.5 | 0.8799 | 5.43 |
| B=8 rejected 2026-08-26 | 8 | 192 | 2 | 63.2 | 91.4 | 0.7352 | 4.03 |
| B=8 rejected 2026-09-12 | 8 | 192 | 2 | 63.8 | 86.8 | 0.7373 | 3.28 |
| B=16 rejected 2026-09-12 | 16 | 384 | 4 | 66.8 | 99.6 | 0.6337 | 2.18 |

(The 2026-08-26 rows were measured on tt-metal 0.74, the 2026-09-12 rows on the
0.77 chunked variant; the trend is what matters, not the absolute values.)

## The reading

A batched decode step costs a fixed base plus a **per-user marginal term**:

- base: 64-75 ms, the same band as a single-stream step;
- marginal: 2.2-5.4 ms for each extra user, and the marginal cost *falls* as the
  width grows (5.43 -> 4.03 -> 2.18), so part of the per-user work is already
  amortised.

`batch_ms/step ~ base + (width - 1) * marginal` reproduces every row to within a
few percent:

| Width | Predicted from its own base and marginal | Measured |
| ---: | ---: | ---: |
| 4 | 75.2 + 3 x 5.43 = 91.5 | 91.5 |
| 8 | 63.8 + 7 x 3.28 = 86.8 | 86.8 |
| 16 | 66.8 + 15 x 2.18 = 99.5 | 99.6 |

## Why this is the instances-per-core axis

The retained-per-user loss tracks the GDN instance multiplier, not the width:

| Width | Instances | Loop iterations per core | Retention |
| ---: | ---: | ---: | ---: |
| 4 | 96 <= 110 cores | 1 | 0.88 |
| 8 | 192 over 110 cores | 2 | 0.735 |
| 16 | 384 over 110 cores | 4 | 0.634 |

The chunked fusion that lifts the grid guard assigns each core
`ceil(instances / ncores)` instances and loops over them sequentially, so past
the core count the *step time grows with width* while a same-width parallel
mapping would keep it nearly flat. That is the occupancy limit this change's
proposal already names from the other side: at TP=2, once the fused decode has
more instances than cores, its cost per step becomes a serial loop.

Two consequences for the retention floor of `0.75`:

- B=8 sits at `0.735-0.737`, i.e. the floor is missed by the loop term alone. One
  fewer instance per core, or a ~10 percent cheaper instance, clears it.
- B=16 at `0.634` needs the per-instance cost cut to roughly a quarter, or the
  four instances per core processed concurrently rather than in a loop.

## The lever this names

Splitting the 128x128 recurrent state across cores per head — the plan's first
published item — is exactly the change that removes the loop: with the state
split, a head occupies several cores instead of one, so a core's per-step work
stops growing once instances exceed the grid. Pipelining heads achieves a
similar amortisation.

This also explains why the batched-prefill experiment at width 16 came out worse
(33.1 s against 23.6 s): it moved work *onto* the same instance loop rather than
off it, and it removed prefill pipelining besides.

## Non-claims

- The 2026-08-26 rows are a different runtime; only the direction is claimed.
- Per-user decode rates are the receipts' own protocol measurements, not serving
  benchmarks, and per-step milliseconds are derived from those medians.
- This is an attribution from measured aggregates, not a per-component profile.
  Confirming that the loop term is the GDN instance loop specifically needs a
  component-level split at width, which the component probe cannot yet do because
  it is bounded to width 1.
