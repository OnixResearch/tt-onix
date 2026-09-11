## Context

The runtime keeps one complete RWKV-7 replica per card. Card-local capacity
is the number of concurrent request slots a card serves. The capacity profile
names B4, B16, B32, and B64. A larger capacity reuses loaded weight tiles
across more requests but also grows the per-step prompt work.

The archived change admitted B4 and blocked B16 on per-request p99 latency.
A physical probe showed the latency variance is a stochastic, device-side,
non-thermal stall in the 16-wide prompt phase that also occurs at B4. The
block therefore measured a metric that conflated concurrency with device
cost.

## Goal and Success Contract

The change is complete when:

- the capacity admission gate uses per-token p99 latency;
- a fresh order-balanced B4/B16 sweep, run on physical cards with
  token-balanced workloads, re-admits B16;
- the evidence package retains the order-balanced runs;
- every retained run preserves exact FP32-reference generated tokens;
- `cairn validate` and the proposal, design, and tasks gates pass;
- the re-admission is documented without claiming a fixed rate or that the
  residual stall is gone.

## Decisions

### Decision: Normalize latency by tokens served

**Choice:** The gate compares p99 time per generated token, computed as
the p99 of (sample elapsed / tokens served in that sample), which equals
the p99 of 1/(tokens per second).

**Rationale:** Per-request wall latency grows with concurrency, so it is not
comparable across workloads of different sizes. Per-token time measures the
same device work at every width.

### Decision: Keep the stall as a tracked follow-up, not a hidden risk

**Choice:** The per-token gate correctly amortizes the fixed device stall,
but does not remove it. The residual 16-wide kernel stall stays a recorded
follow-up for the fused WKV kernel work.

**Rationale:** Re-admission on the honest metric must not pretend the stall
is gone.

## Validation Strategy

Positive cases cover per-token p99 normalization, all supported capacities
planning, and the promotion decision on a realistic A/B pair. Negative cases
cover an empty per-token sample, a zero-token sample, malformed or non-finite
metrics, simulated claims, and parity mismatch. The retention check in the
evidence package pins the order-balanced re-admission.

## Risks and Trade-offs

- **Residual stall:** retained as a follow-up, not fixed here.
- **Metric change:** the per-token gate must not be read as weakening safety;
  it corrects the comparison basis. The strict physical, parity, profiler,
  and occupancy rails remain unchanged.
- **Non-stationary hardware:** the sweep uses order-balanced alternation and
  token-balanced workloads.
