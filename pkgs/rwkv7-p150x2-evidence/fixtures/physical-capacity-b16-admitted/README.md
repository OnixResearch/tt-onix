# Physical card-local capacity B16 re-admission (2026-08-09)

This fixture re-admits B16 as the card-local decode capacity. It supercedes
the B4-retained decision in `physical-capacity-sweep-retained/`.

## Why the earlier block was wrong

The archived sweep rejected B16 because its per-request p99 wall latency
regressed against B4 under the 0.10 rule. That rule was designed for the
one-card-versus-two-card comparison, where the workload is the same and only
the number of cards changes. Applying it across different capacity workloads
measured request wall latency, which grows with concurrency, instead of the
real device cost per token.

The stall investigation (`investigate-rwkv7-b16-stall` in this repo) proved
the honest metric is per-token time:

- The decoder runs a fixed 35 steps; the variance sits entirely in the
  16-wide prompt phase.
- The stall is a device-side, intermittent, non-thermal event that also
  occurs at B4 (a B4 resample showed a 23.8 s two-card run from a single
  stalled wave).
- Per-token time normalizes the fixed stall across the tokens served, so B16
  is not penalized for amortizing one device event over 4x more tokens.

## The re-admission gate

`rwkv7-p150x2-throughput capacity-evaluate` now compares the candidate's
**p99 per-token time** against the current admitted capacity's p99 per-token
time, alongside the aggregate throughput gain. The pure core computes
per-token p99 as the p99 of (sample elapsed / tokens served in that sample),
which equals the p99 of 1/(tokens per second).

## Order-balanced evidence

Each run is the strict throughput benchmark with the token-balanced workload,
so the two-card scheduler no longer piles every long prompt onto one card.
The alternation was B4, B16, B16, B4.

| Run | Capacity | Two-card tps | p99 per-token (ms) |
| --- | --- | --- | --- |
| b4-r1 | B4 | 152.5 | 7.29 |
| b16-r1 | B16 | 408.4 | 5.53 |
| b16-r2 | B16 | 551.8 | 4.24 |
| b4-r2b | B4 | 116.6 | 57.05 |

The b4-r2b sample contains a stalled wave (57 ms per-token p99). B4 is not
immune to the stall; this is the same device event that previously inflated
B16's p99.

## Promotion decision

B16 is promoted in every order-balanced comparison against both B4 samples:

- B16-r1: gain +1.68x (vs b4-r1) / +2.50x (vs b4-r2b); per-token latency
  regression -24% / -90%.
- B16-r2: gain +2.62x (vs b4-r1) / +3.73x (vs b4-r2b); per-token latency
  regression -42% / -93%.

A negative latency regression means B16's per-token time is lower than B4's.
B16 is faster per token and faster overall, with exact FP32 token parity in
every run.

## What this does not claim

- B16 does not eliminate the intermittent device stall; the stall is a
  follow-up kernel investigation (see the investigation notes).
- This does not claim a fixed tokens-per-second rate.
- The residual 16-wide kernel stall remains a tracked risk and is not hidden
  by the per-token metric; it is amortized, not removed.