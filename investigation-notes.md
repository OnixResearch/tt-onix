# B16 stall investigation — findings so far

Change: `investigate-rwkv7-b16-stall` (worktree `investigate-rwkv7-b16-stall`, repo `tt-onix`).

## What the archived change concluded

The physical card-local capacity sweep (`scale-rwkv7-card-local-batching`) kept B4 admitted and retained B16 as the blocker. B16 measured a stable ~3.6x two-card throughput gain over B4 but failed the retained p99 latency rule (regressions 1.34, 0.06, 0.19 against the 0.10 bound). The archived evidence lives in `pkgs/rwkv7-p150x2-evidence/fixtures/physical-capacity-sweep-retained/`.

## Question

Is B16's p99 regression a real hardware property, a workload artifact, or a measurement problem? If it is fixable or fair, B16 could be re-admitted; otherwise B4 stays.

## What we measured

### 1. The original two-card workload was token-imbalanced by construction

The round-robin scheduler assigns requests to cards by request index. The sweep workload alternates short (5-token) and long (35-token) prompts, so:

- card 0 received 16 short requests = 400 tokens total;
- card 1 received 16 long requests = 2800 tokens total.

That is a 7:1 token imbalance. Card 1 did seven times the work of card 0, so the two-card p99 and elapsed were dominated by card 1. The per-token rate was flat and equal across cards (~33 ms/token at B16), so the "stall" in the original runs was substantially card-1-doing-more-work, not a per-token regression.

### 2. A balanced workload still shows an intermittent stall

A balanced workload (each parity class gets 8 short + 8 long = 336 tokens) reproduces the same intermittent slow waves. Card symmetry became perfect (both cards 171.97 tps in one run) but p99 stayed high (2.22 s two-card; 7.83 s on one-card wave).

So the stall is real and independent of load balancing.

### 3. The stall is a device-side event, not host scheduling

Worker receipts show, for every run and both cards:

- 35 batch steps (constant);
- host gap ~80-160 us (negligible).

With a fixed step count and no host gap, the extra 2-9s must be spent inside a device decode step, not in host submission or scheduling.

### 4. The stall is not thermal

Telemetry (device 1) throughout the probe loop:

- aiclk pinned 800 MHz;
- asic temperature 60-63 C;
- power 37-39 W;
- voltage 0.74 V, current ~51 A;
- fan 45-55%.

All flat. No clock droop, no throttling signature. The slow waves are not thermal.

### 5. The stall is intermittent, bimodal, and non-monotonic

Ten consecutive single-worker runs on card 1 with the same request and the same 35 steps:

```
9.71  1.06  1.10  0.97  3.13  8.57  2.62  3.49  2.43  2.15   (seconds elapsed)
```

Fast runs are ~1.0-1.1 s; slow runs are 2.1-9.7 s. Fast and slow interleave: this is not a warm-up curve, a soak curve, or a monotonic mode shift. It is a stochastic bimodal device event.

### 6. The device profiler cannot see it

The raw `profile_log_device.csv` for a fast and a slow run are both exactly 2,358,962 lines. The TT-Metal device profiler buffer fills and drops markers before the slow step completes, so per-op CPU/cycle attribution of the specific stalled step is unavailable from the retained CSVs. The zone dwell totals also do not differ meaningfully between fast and slow runs for the same reason.

## Working hypothesis

A device-side serialization or synchronization event intermittently backs one decode step up 2-9x at 16-wide batch. Candidate mechanisms to probe next:

1. a specific TT-NN op (argmax/untilize readback of 16 rows, or a 16-wide matmul/lerp) waiting on a device barrier;
2. a NOC/Mailbox credit or semaphore stall in one kernel;
3. a device synchronization or completion-wait that intermittently sleeps longer;
4. host-deepening caused by a TT-NN op recompiling or recloning a program (though program-cache-stable holds, so this is unlikely).

## Per-step probe result (2026-08-09)

Added an env-gated per-step wall timer (`RWKV7_STALL_PROBE_STEPS=1`) to the fast worker and ran the exact B16 request repeatedly on card 1.

- Fast runs: all 35 steps ~27-29 ms each; prompt phase 836 ms; total ~0.92-0.97 s.
- Slow runs (intermittent): one or several prompt-phase steps take 100-306 ms (10x); prompt phase 1435-1916 ms; total up to 2.01 s in this batch (up to 9.7 s in an earlier batch).
- The generation phase (steps 32-34) is constant at ~80-96 ms in every run.

Attribution: the entire variance is in the 32-token prompt-decode phase of the 16-wide batch. Steps 9, 11, 16, and 28 each appeared as the slow step in different runs, so it is not a single op at a fixed index; it is a stochastic device-side slowdown of the 16-wide prompt matmuls/WKV, sometimes concentrated in one step, sometimes spread.

Ruled out: load imbalance (balanced workload still stalls; per-token rates flat across cards), host scheduling (host gap ~100 us, step count fixed), thermal (clock pinned 800 MHz, temp/power/voltage flat), profiler visibility (device profiler buffer drops markers).

Next probe: run the 16-wide prompt phase in isolation (one layer, no readback) with the device profiler to see which kernel surface absorbs the extra time, or test whether the stall correlates with L1/bank pressure at 16 rows.

## Decisive probe: all-long 16-wide workload (2026-08-09)

Ran the same 16-slot card with 16 requests that ALL have 32-token prompts (no mixed short/long), 20 iterations:

- Fast runs: all 35 steps 27-29 ms; prompt phase ~925 ms; total ~0.94-1.02 s.
- Intermittent slow runs (12, 15, 16 of 20): prompt phase 1100-1500 ms; steps 1-16 elevated 1.5-2x (45-98 ms); no single massive step needed, the slowdown spreads.
- Generation phase stays constant (~90-105 ms) in every run.

Conclusion: the stall is NOT specific to the mixed short/long batch. It is a stochastic 2x slowdown of the 16-wide prompt phase, sometimes concentrated in one step (0.3 s), sometimes spread across many steps. The mixed workload showed the same pattern (steps 9, 11, 16, 28 as spikes in different runs).

The stall is therefore a device-side property of the 16-wide prompt-decode kernels (matmul/WKV at 16 rows), intermittent and bimodal, not a workload shape, host, or thermal effect.

## B4 comparison: the stall exists at both widths, but B16 wins per token

Ran the same probe at B4 (4 all-long requests) on card 1, 12 iterations:

- B4 fast runs: prompt phase 875 ms; steps ~27 ms; total ~0.91-0.99 s.
- B4 mild stall (b4-5): steps 4-7 elevated 72/57/42/33 ms; prompt 1006 ms; total 1.10 s.

So B4 also has the intermittent prompt-phase step slowdown, just milder (4 rows serialize less than 16). The absolute step stall is a fixed device event present at both widths.

Per-token prompt-phase comparison (all-long workload):

| Width | Fast ms/token | Slow ms/token |
| --- | --- | --- |
| B4 | 6.84 | 7.86 |
| B16 | 1.81 | 2.92 |

B16's slowest run (2.92 ms/token) is still 2.3x faster per token than B4's fastest (6.84 ms/token). The intermittent step stall is a fixed absolute event (adds roughly 1-2 s to the prompt phase regardless of width), so it amortizes across 4x more tokens at B16.

## Why the archived p99 rejection was the wrong basis

The archived sweep compared B16's WALL-CLOCK p99 latency (per request) against B4's. Because B16 serves 4x the tokens per step and 4x the requests, its wall-clock per-request latency is naturally higher even though its per-token time is far lower. The p99 latency regression rule (0.10) was designed for the one-card-vs-two-card comparison (same workload, more cards). Applying it to a capacity comparison (different workload sizes) measured the wrong thing.

The honest capacity metric is per-token time (or tokens/sec), where B16 is decisively better in both fast and stalled runs. The B16 block should be revisited with a per-token or throughput-optimality gate, not a raw request p99 wall-clock regression.

Next probe: isolate the 16-wide prompt layer execution (one layer, minimal ops) to attribute the extra time to a kernel surface; or run the ttllk race/perturb audits on the 16-wide matmul and WKV kernels.

## Next probes

- Add a per-step host-side timer to the fast worker to identify WHICH decode step absorbs the extra seconds (device vs readback vs embedding).
- If it is the argmax readback step, count how often a 16-row readback takes 2-9s.
- If it is a matmul/lerp step, profile that op in isolation at 16-wide.
- Confirm whether B4 (4 slots) ever stalls; the archived B4 runs were tight (p99 1.13 s, max 1.13 s), suggesting the stall is specific to the 16-wide geometry or the larger state footprint.

## Non-claims

- No simulator result is used as physical evidence.
- No throughput or latency claim is made for B16 yet.
- This does not weaken the retained 0.10 p99 rule; it tests whether B16 can satisfy it.
