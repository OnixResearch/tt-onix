# t4: ABBA screen of the resident slot trace, and prefix-cache re-screen (2026-09-27)

**Verdicts: both admitted.**

- **Resident trace.** Receipt `qwen38-p150x2-resident-slot-trace-admission-v1` admits package
  `/nix/store/a32pfd9h7lkkidnpmni2lk7gs138dx0x-qwen38-0.1.0` to replace the deployed
  `/nix/store/vvi5i8wyxm3i4bjf1vb9fwxpxbvzcrs8-qwen38-0.1.0`.
  - Turns take a median 0.81 of the deployed wall time, with an unchanged TTFT.
  - All four roots decode identical tokens and make identical cache hits and saves.
  - The longest pause between tokens of a turn fell from 0.71–1.85 s to 0.05–0.11 s.
- **Prefix cache on the new package.** Policy v2 (`qwen38-p150x2-prefix-cache-admission-v2`,
  unchanged) admits cache profile `qwen38-p150x2-prefix-cache-v1` on `a32pfd9h`.
  - Hits take 0.42 of cold TTFT and 0.61 of cold wall.
  - Misses take a median 1.007 and at most 1.017 of cold wall.

## Protocol

- **Rules, fixed before measuring.** Change code, policy
  `pkgs/qwen36/resident-trace-admission-policy.ncl`, evaluator and driver were committed and
  pushed in `16c909a` before the first root started.
  - Resident-trace policy BLAKE3:
    `e006ff0b2343baf8a607ea288a5f354b0edad2f82bc85d116786eda6f6e4785e`.
  - Prefix-cache policy v2 BLAKE3:
    `3c862396f7028b45e83d78bfa5f2cf25aaa5092c23e8e64acd60b78341ca046c`.
- **Packages.**
  - Baseline: `vvi5i8wy`, the deployed package, which captures a decode trace for every slot
    group.
  - Candidate: `a32pfd9h`, built from `16c909a`. Its execution identity is
    `/nix/store/g5pdvpin2n221v8c95j1mxbw55zk98pj-qwen36-execution-identity-0.1.0`.
  - Runtime for both: `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.
- **Host.** britton-desktop, PCIe devices 0 and 1. Each root was a transient unit on port
  8011 with the production unit's environment and default CPU scheduling. Production
  `qwen38-p150x2.service` was stopped from 22:57:46 to 23:42:23 UTC.
- **Roots.** Six processes, in order:

  | Root | Package | Kind | Workloads | Measured (UTC) |
  | --- | --- | --- | --- | --- |
  | c1 | candidate | cold | w3, w5 | to 23:10 |
  | a1 | baseline | cached | w3, w5, w3s | 23:13:45–23:16:49 |
  | b1 | candidate | cached | w3, w5, w3s | 23:20:32–23:23:02 |
  | b2 | candidate | cached | w3, w5, w3s | 23:27:18–23:29:50 |
  | a2 | baseline | cached | w3, w5, w3s | 23:33:34–23:36:37 |
  | c2 | candidate | cold | w3, w5 | from 23:36:40 |

  - Resident-trace groups: a1, b1, b2, a2. Baseline and candidate mean positions are
    equal.
  - Prefix-cache groups: c1, b1, b2, c2. Cold and cached mean positions are equal.
  - Both designs cancel load that drifts linearly across the screen.
- **Workloads.** Driver `scripts/qwen38-prefix-cache-screen.py`:
  - w3 is 3 conversations × 4 turns, greedy (BLAKE3 `7bdff9db…c175b8`, as in p6 and p7).
  - w5 is 5 conversations × 2 turns, greedy (`38038f5d…6bbb6`).
  - w3s repeats w3's conversations at temperature 0.7, seeded per turn (seed 1000 +
    1000 × conversation + turn). It runs on the sampled route.
- **Early stops, declared in the runner.** The runner would have aborted if c1 or b1 had
  decoded any turn differently from the admitted package's p7 roots, or had captured on
  more than a quarter of their turns. Neither did: c1 and b1 matched every turn and
  captured on the first turn only.
- **Receipts.**
  - `qwen38-resident-trace-receipt --abba a1 b1 b2 a2`, for w3, w5 and w3s.
  - `qwen38-prefix-cache-receipt --abba c1 b1 b2 c2`, for w3 and w5.

Evidence under `pkgs/qwen36/evidence/2026-09-27/`:

- `qwen38-resident-trace-{a1,b1,b2,a2}-{w3,w5,w3s}.json.gz` and
  `qwen38-resident-trace-{c1,c2}-{w3,w5}.json.gz`: the roots. The receipts bind each by
  the BLAKE3 of its uncompressed JSON.
- `qwen38-resident-trace-admission-receipt.json` and
  `qwen38-resident-trace-prefix-cache-receipt.json`: the receipts.
- `qwen38-resident-trace-startup.json`: for each root, its readiness time, its startup
  events, and the host's CPU pressure when it started and ended. The service journals of
  a1, b1, b2 and a2 also hold the p7 roots of the same unit names; only the latest
  invocation's events are kept.

## Resident-trace gates

| Gate | Bound | Measured (ABBA means) | Pass |
| --- | --- | --- | --- |
| Matched identity, ABBA order, one package per role | required | yes | yes |
| Exact token parity across all four roots | required | 34 of 34 turns | yes |
| Identical prefix hits and saves across all four roots | required | 34 of 34 turns | yes |
| Largest share of a candidate root's turns that captured | ≤ 0.25 | 0.083 | yes |
| Median wall ratio | ≤ 0.95 | 0.812 | yes |
| Worst turn wall ratio | ≤ 1.10 | 1.041 | yes |
| Median TTFT ratio | ≤ 1.05 | 1.003 | yes |
| Startup DRAM drop | ≤ 64 MiB | 0 B | yes |

Adjacent-pair ratios (diagnostics only):

| Pair | Median wall | Worst turn | Median TTFT |
| --- | --- | --- | --- |
| a1 → b1 | 0.810 | 0.979 | 0.997 |
| b2 → a2 | 0.821 | 1.109 | 1.012 |

The worst turns in both pairs are the two turns on which both roles capture:

- the first turn of each process (w3 c0 t0, pooled 0.997);
- the first sampled turn after the greedy workloads, when the route changes (w3s c0 t0,
  pooled 1.041; 1.109 in the second pair alone).

## Per turn (client seconds, baseline mean → candidate mean)

Captures lists turns that captured a trace, in the order a1/b1/b2/a2. Longest pause is
the longest gap between two tokens of the turn.

| Workload | Conv. | Turn | Prompt tokens | Hit tokens | TTFT | Wall | Wall ratio | Captures | Longest pause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w3 | 0 | 0 | 3,959 | 0 | 2.31 → 2.30 | 4.91 → 4.90 | 1.00 | 1/1/1/1 | 0.94 → 0.94 |
| w3 | 1 | 0 | 3,201 | 0 | 1.96 → 1.97 | 4.40 → 3.54 | 0.80 | 1/0/0/1 | 0.92 → 0.05 |
| w3 | 2 | 0 | 3,250 | 0 | 1.95 → 1.96 | 4.86 → 4.08 | 0.84 | 1/0/0/1 | 0.87 → 0.08 |
| w3 | 0 | 1 | 5,756 | 3,072 | 1.76 → 1.76 | 4.74 → 3.88 | 0.82 | 1/0/0/1 | 0.92 → 0.05 |
| w3 | 1 | 1 | 5,045 | 3,072 | 1.18 → 1.20 | 3.58 → 2.78 | 0.78 | 1/0/0/1 | 0.87 → 0.05 |
| w3 | 2 | 1 | 5,078 | 3,072 | 1.21 → 1.21 | 3.78 → 2.84 | 0.75 | 1/0/0/1 | 1.01 → 0.05 |
| w3 | 0 | 2 | 7,504 | 5,120 | 1.58 → 1.64 | 3.90 → 3.09 | 0.79 | 1/0/0/1 | 0.93 → 0.05 |
| w3 | 1 | 2 | 6,736 | 4,096 | 1.76 → 1.78 | 4.15 → 3.37 | 0.81 | 1/0/0/1 | 0.86 → 0.05 |
| w3 | 2 | 2 | 6,452 | 4,096 | 1.56 → 1.58 | 4.53 → 3.69 | 0.81 | 1/0/0/1 | 0.91 → 0.05 |
| w3 | 0 | 3 | 9,267 | 7,168 | 1.45 → 1.45 | 4.31 → 3.58 | 0.83 | 1/0/0/1 | 0.79 → 0.05 |
| w3 | 1 | 3 | 8,385 | 6,144 | 1.51 → 1.50 | 4.38 → 3.63 | 0.83 | 1/0/0/1 | 0.80 → 0.05 |
| w3 | 2 | 3 | 7,953 | 6,144 | 1.22 → 1.25 | 3.64 → 2.88 | 0.79 | 1/0/0/1 | 0.84 → 0.05 |
| w5 | 0 | 0 | 3,959 | 0 | 2.27 → 2.28 | 4.86 → 4.00 | 0.82 | 1/0/0/1 | 0.94 → 0.05 |
| w5 | 1 | 0 | 3,201 | 0 | 1.95 → 1.96 | 4.33 → 3.52 | 0.81 | 1/0/0/1 | 0.86 → 0.06 |
| w5 | 2 | 0 | 3,250 | 0 | 1.96 → 1.97 | 5.15 → 4.07 | 0.79 | 1/0/0/1 | 1.14 → 0.07 |
| w5 | 3 | 0 | 3,550 | 0 | 2.04 → 2.11 | 5.03 → 4.23 | 0.84 | 1/0/0/1 | 0.93 → 0.05 |
| w5 | 4 | 0 | 3,820 | 0 | 2.27 → 2.29 | 5.32 → 4.42 | 0.83 | 1/0/0/1 | 0.99 → 0.07 |
| w5 | 0 | 1 | 5,756 | 0 | 3.38 → 3.42 | 6.25 → 5.54 | 0.89 | 1/0/0/1 | 0.80 → 0.06 |
| w5 | 1 | 1 | 5,045 | 0 | 2.82 → 2.87 | 5.22 → 4.46 | 0.85 | 1/0/0/1 | 0.87 → 0.07 |
| w5 | 2 | 1 | 5,078 | 0 | 2.82 → 2.83 | 5.20 → 4.45 | 0.86 | 1/0/0/1 | 0.80 → 0.05 |
| w5 | 3 | 1 | 5,346 | 0 | 3.08 → 3.11 | 5.93 → 5.04 | 0.85 | 1/0/0/1 | 0.96 → 0.05 |
| w5 | 4 | 1 | 6,011 | 0 | 3.39 → 3.39 | 6.46 → 5.50 | 0.85 | 1/0/0/1 | 1.01 → 0.05 |
| w3s | 0 | 0 | 3,959 | 0 | 2.25 → 2.26 | 6.14 → 6.39 | 1.04 | 1/1/1/1 | 1.44 → 1.61 |
| w3s | 1 | 0 | 3,201 | 0 | 1.97 → 1.95 | 6.65 → 5.32 | 0.80 | 1/0/0/1 | 1.46 → 0.10 |
| w3s | 2 | 0 | 3,250 | 0 | 1.96 → 1.97 | 7.46 → 6.10 | 0.82 | 1/0/0/1 | 1.49 → 0.10 |
| w3s | 0 | 1 | 5,748 | 3,072 | 1.77 → 1.77 | 6.93 → 5.38 | 0.78 | 1/0/0/1 | 1.63 → 0.10 |
| w3s | 1 | 1 | 5,049 | 3,072 | 1.19 → 1.20 | 5.88 → 4.45 | 0.76 | 1/0/0/1 | 1.53 → 0.09 |
| w3s | 2 | 1 | 5,078 | 3,072 | 1.20 → 1.18 | 6.65 → 5.34 | 0.80 | 1/0/0/1 | 1.42 → 0.10 |
| w3s | 0 | 2 | 7,490 | 5,120 | 1.58 → 1.59 | 5.64 → 4.33 | 0.77 | 1/0/0/1 | 1.43 → 0.09 |
| w3s | 1 | 2 | 6,743 | 4,096 | 1.77 → 1.77 | 6.66 → 5.27 | 0.79 | 1/0/0/1 | 1.48 → 0.09 |
| w3s | 2 | 2 | 6,465 | 4,096 | 1.60 → 1.56 | 7.23 → 5.59 | 0.77 | 1/0/0/1 | 1.71 → 0.09 |
| w3s | 0 | 3 | 9,253 | 7,168 | 1.46 → 1.46 | 6.09 → 4.72 | 0.78 | 1/0/0/1 | 1.49 → 0.09 |
| w3s | 1 | 3 | 8,398 | 6,144 | 1.50 → 1.50 | 7.00 → 5.67 | 0.81 | 1/0/0/1 | 1.49 → 0.09 |
| w3s | 2 | 3 | 7,964 | 6,144 | 1.23 → 1.23 | 6.25 → 4.65 | 0.74 | 1/0/0/1 | 1.68 → 0.09 |

- **Where the saving comes from.** The baseline captures on every turn. Its longest
  pause, 0.71–1.20 s greedy and 1.22–1.85 s sampled, sits between the first and second
  token. The candidate keeps its trace, and its longest pause is 0.05–0.11 s. A turn
  saves 0.7–1.6 s, with the sampled route saving the most.
- **Prefill is unchanged.** The candidate refills beside a resident trace, and every
  TTFT is within 0.07 s of the baseline. The TT-Metal 0.74 rejection came from slower
  prefill under a parked trace, and it does not recur with masked refills on 0.77.
- **The candidate captures only when a capture is due.** It captured on 1 of 12 turns in
  w3 (the process's first turn) and in w3s (the route change), and on none in w5. A
  route change releases the trace by design.

## Prefix-cache gates on the candidate package

| Gate | Bound | Measured (ABBA means) | Pass |
| --- | --- | --- | --- |
| Matched identity, ABBA order, cached roots agree | required | yes | yes |
| Exact token parity across all four roots | required | 22 of 22 turns | yes |
| Resumed hits in w3, both cached roots | required | 9 of 9 later turns | yes |
| Hit median TTFT ratio | ≤ 0.70 | 0.422 | yes |
| Hit median wall ratio | ≤ 0.85 | 0.608 | yes |
| Miss median wall ratio | ≤ 1.10 | 1.007 | yes |
| Maximum miss wall ratio | ≤ 1.25 | 1.017 | yes |
| Startup DRAM drop | ≤ 536,870,912 B | 364,904,448 B | yes |

- Both adjacent pairs would also pass alone. Their worst misses are 1.004 and 1.045.
- The cold roots keep their trace too, so the hit wall ratio (0.61) is lower than in p7
  (0.66): a cached turn now saves prefill on top of a shorter tail.

## Host conditions

- The median CPU stall fraction per root was 0.16–0.17, and no turn exceeded 0.224.
- The 1-minute load average when roots started and ended was 25–29 on 32 CPUs.
- Readiness took 218–252 s, except 626 s for c1, the first process after production
  stopped. The same happened to p7's a1. [INFERENCE] The first open of the devices after
  the production service stops is slow; readiness feeds no gate.

## Non-claims

- These are agent-shaped sequential workloads, greedy and seeded sampled, on two
  packages. They are not a general latency claim.
- Deployment must serve `a32pfd9h` exactly, or a later commit that builds the same path.
- The continuous-slot scheduler's own receipt names package `jxhspibs4ml8`. Later
  packages, including this one, were compared with a cold root of their own scheduler,
  not with the group coordinator under staggered load.
