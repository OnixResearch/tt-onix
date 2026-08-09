# Physical card-local capacity sweep (retained)

This fixture retains the bounded physical sweep of the card-local capacity
candidates. The sweep ran on physical P150A cards 0 and 1 through the
host's owner-isolation procedure. The owner services were inactive and no
process held either device node before the sweep.

The sweep measures each candidate with the same two-wave saturated shape as
the admitted B4 fixture. A candidate of capacity C runs 2*C requests: two
waves of C slots on one card, one wave of C slots per card on two cards.
The workload streams reuse the pinned FP32 CPU reference streams, so every
run is compared against the exact reference tokens.

## Order-balanced A/B pairs

The alternation was B4, B16, B16, B4, B16, so both orders appear. Each run
is a full strict benchmark: two profiler-attested samples plus five timed
one-card and five timed two-card samples. All streams matched the pinned
reference exactly in every run.

| Run | Candidate | Two-card tps | Two-card p99 (s) | Strict speedup | Strict p99 change |
| --- | --- | --- | --- | --- | --- |
| b4-baseline | B4 | 149.888 | 1.131 | 1.875 | -0.006 |
| b16-r1 | B16 | 493.850 | 2.648 | 2.601 | -0.290 |
| b16-r2 | B16 | 579.254 | 1.197 | 1.959 | -0.006 |
| b4-r2 | B4 | 155.537 | 1.090 | 1.995 | -0.120 |
| b16-r3 | B16 | 585.047 | 1.343 | 2.017 | -0.555 |

All five runs passed their own strict two-card admission: physical hardware,
profiler attribution, exact parity, and a dual-card speedup at or above
1.5. The B16 candidate therefore never hit an out-of-memory or parity stop.
The B4 absolute record stays at 181.820 tokens per second on two cards; the
B4 runs here are the current runtime's own baseline and are not a new claim.

## Blocker: repeated p99 latency-bound failure at B16

The pure capacity gate compares each candidate against the current admitted
runtime. The profile bound for p99 latency regression is 0.10 and the
throughput minimum is 0.00.

B16 beats the admitted B4 throughput in every round: the mean two-card rate
is 552.7 tokens per second against 152.7 for B4, a 2.90-3.62x gain
depending on the paired rounding. But B16's p99 latency is not stable:

- r1: 2.648 s (regression 1.34 against the B4 baseline p99);
- r2: 1.197 s (regression 0.06);
- r3: 1.343 s (regression 0.19).

The median B16 p99 exceeds the 0.10 regression bound, and two of three
rounds violate it. The change requires stable alternating A/B evidence and
a strict p99 admission. The B16 candidate does not satisfy the latency
bound, so the sweep stops and B4 remains the admitted card-local capacity.

The pure gate from `rwkv7-p150x2-throughput capacity-evaluate` reports the
same decision device-free:

```
{"promoted": false, "measured_gain_fraction": 2.903, "latency_regression_fraction": 0.187}
```

## What this evidence does not claim

The sweep does not claim that B16 fails on silicon. It claims that B16 does
not pass the retained stability and p99 rules that protect request
isolation. A later change may revisit the latency bound with evidence, but
this fixture does not relax it. Simulated runs never enter this fixture,
and no simulator result is used as throughput evidence.

## Machine facts

The benchmark captured `tt-smi -s --snapshot_no_tty` snapshots before and
after every run. The B4 baseline run's machine facts are retained in
`b4-baseline/machine-facts-*.json`; the B16 runs' facts are in the
`/var/tmp/scale-capacity-sweep-b16-r*` roots, whose receipts and raw
evidence are copied here.