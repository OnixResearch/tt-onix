# Where the width-dependent cost sits, and what it means for the retention bound

Recorded 2026-09-12. Measurement work only: no kernel was changed and no bound moved.

## The instrument that works

A width probe that starts one server session per width, releases `W` prompts
together, and reads the service's own `token_latency_seconds` list. The interior
median is the batch step time. It is calibrated against an admission receipt: at
width 8 it reports a 78.9 ms median step over 480 samples, and the width-8 receipt
reports 79.4 ms (1/12.5978). Script: `scripts/qwen38-step-probe.sh`.

This matters for cost. Screening a candidate at width 8 now takes about three
minutes of device time instead of the thirty minutes a four-root receipt needs, so
the full receipt can be reserved for the candidate that already looks promising.

## What the profile run established, and what it did not

A width-8 run with `TT_METAL_DEVICE_PROFILER=1` (device profiler output root
redirected to a writable directory, since `TT_METAL_HOME` is a store path) wrote a
683 MB `profile_log_device.csv`. It does **not** attribute time to operators: every
kernel zone is named `TRISC-KERNEL` or `BRISC-KERNEL`, and the kernel identity is
not present in a column, so the file cannot separate the GDN decode kernel from
anything else. The run also reported that the profiler DRAM buffers filled and
markers were dropped, so even a correct attribution would have covered only part of
the run.

Recorded as an instrument limit: the device CSV alone cannot answer this question.
A per-operator figure needs the host-side signposts, which the service does not
enable.

One correction to my own working notes, which I checked rather than assumed: the
earlier per-width probe's response files exist for widths 6 and 8, but they are
*error bodies*, and the session logs record `batch-failed` for both. The repository
record in `enable-qwen38-p150x2-b16/evidence/b16-result.md` already states this
correctly ("policy rejection", zero TT_FATALs). Only the absence of the
circular-buffer fatal was established there; nothing captured.

## The width-dependent cost, from measured step times

Step time from the receipts, where `k = ceil(24W/110)` is the GDN head-instances
per core on the 11x10 grid:

| Width | `k` | single step | batch step | retention |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 60.4 ms | 68.7 ms | 0.880 |
| 8 | 2 | 58.5 ms | 79.4 ms | 0.737 |
| 16 | 4 | 60.0 ms | 94.7 ms | 0.634 |

Reading these as `step = C(W) + 48 * k * x_k`:

- Width 1 -> 4 slots costs **+8.3 ms** at constant `k`, so the non-GDN path charges
  at most about 2.8 ms per added slot.
- Width 4 -> 8 slots costs **+10.7 ms** and doubles `k`.

If the non-GDN path charged the same 2.8 ms per slot across eight slots, it alone
would explain the whole width 4 -> 8 increase, leaving nothing for the loop. That
cannot be right, and it is the reason the boundary point matters: it is not possible
to separate the non-GDN per-slot cost from the GDN loop's marginal cost with points
that differ in both dimensions at once.

Under the mild assumption that the non-GDN cost does not *fall* when the slot count
doubles, the loop's per-instance marginal is bounded: `x_1 = 0.549 ms` (component
probe, width 1, `k = 1`), `x_2 <= 0.39 ms`, `x_4 <= 0.33 ms`. Written as a share of
the width-1 per-instance cost, the loop keeps at most 71 percent of it at `k = 2`
and at most 60 percent at `k = 4`.

That inference is bounded, not measured, and it is labelled as such. It does say
something useful: the fused kernel already overlaps part of the loop, so the reward
for interleaving work shrinks as `k` grows.

## What each bound actually needs

| Width | retention now | bound | step needed | shortfall | GDN loop must lose |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.737 | 0.75 | <= 78.0 ms | 1.4 ms (1.8%) | ~2% |
| 16 | 0.634 | 0.75 | <= 80.1 ms | 14.6 ms (15%) | ~25% |

So width 8 is a 1.8 percent change away, and the width-8 step probe can measure that
directly with its 480 samples. Width 16 needs a quarter of the loop.

## The boundary point, and the clean way to get it

The service serves only widths 8 and above: a group of 3, 4, or 5 requests fails
with `BatchDecodeAdapterError` under both the width-8 and the width-16 package,
with no TT_FATAL, and the warmup widths `[1, 4]` appear in the same log. So the
width-5 point that straddles the `k = 1 -> 2` boundary is not servable today.

Two ways to reach it, in order of preference:

1. **A diagnostic grid cap.** Cap the GDN decode grid so that four slots need two
   instances per core (a cap of 48 cores gives `ceil(96/48) = 2`), then run width 4
   twice, capped and uncapped. The slot count and the non-GDN cost are identical, so
   the difference is exactly `48 * (2*x_2 - x_1)`. This needs no new admission
   profile, only a diagnostic-only knob that defaults off and is proved off by a
   device-free probe.
2. A width-5 admission profile family, which invents a served configuration the
   admission policy does not want.

Option 1 is the next step.
