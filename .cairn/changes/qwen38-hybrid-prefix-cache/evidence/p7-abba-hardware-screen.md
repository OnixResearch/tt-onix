# p7: ABBA re-screen of the prefix cache (2026-09-27)

**Verdict: admitted.** The receipt admits prefix-cache profile
`qwen38-p150x2-prefix-cache-v1` for package
`/nix/store/vvi5i8wyxm3i4bjf1vb9fwxpxbvzcrs8-qwen38-0.1.0` only. Every gate passes:

- Hits cut the median TTFT to 0.42 of cold and the median wall to 0.66.
- Misses cost a median of 1.02× cold and at most 1.08×.
- All four roots decode identical tokens.

The p6 screen failed under protocol v1, and that result stays recorded in
`p6-hardware-screen.md`. This screen used protocol v2, declared before it ran. Its
misses stayed within 8% of cold, whereas p6's worst miss took 1.95× cold. That supports
the p6 diagnosis that host contention caused the failed miss gates.

## Protocol

- Gates: `pkgs/qwen36/prefix-cache-admission-policy.ncl`, profile
  `qwen38-p150x2-prefix-cache-admission-v2`. It was committed and pushed in `e0f04a6`
  before the first root started. Exported policy BLAKE3:
  `3c862396f7028b45e83d78bfa5f2cf25aaa5092c23e8e64acd60b78341ca046c`.
- v2 keeps every v1 gate bound. It changes how the roots are measured and compared:
  - Each workload has four roots, each from a fresh process, measured in the order cold,
    cached, cached, cold. The evaluator checks that order against each root's
    `measured_at`.
  - Each gate compares, turn by turn, the mean of the two cached roots with the mean of
    the two cold roots. This cancels host load that drifts linearly across the four
    roots.
  - Parity must hold across all four roots, and both cached roots must resume the same
    prefix on every turn.
  - The receipt also reports each adjacent pair's ratios, as diagnostics only.
- Package: `/nix/store/vvi5i8wyxm3i4bjf1vb9fwxpxbvzcrs8-qwen38-0.1.0`, built from
  `e0f04a6`. Its serving code is unchanged from the p6 package; only the admission
  evaluator and policy differ.
  - Execution identity:
    `/nix/store/52cjnyvcv27imscmhp3b3y2pnayw1z2x-qwen36-execution-identity-0.1.0`.
  - Runtime: `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.
  - Cache policy BLAKE3:
    `f4567693570cb0e85213c83f6c08e91179b065c9beed1ea2479abdca97ab0ea6`.
- Host: britton-desktop, PCIe devices 0 and 1. Each root ran as a transient unit on port
  8011 with the production unit's environment and default CPU scheduling, as production
  has. Production `qwen38-p150x2.service` was stopped from 21:20:05 to 21:55:00 UTC.
- Workloads and driver: the same as p6. w3 is 3 conversations × 4 turns (BLAKE3
  `7bdff9db…c175b8`) and exercises hits. w5 is 5 conversations × 2 turns (BLAKE3
  `38038f5d…6bbb6`) and exercises misses and eviction. The driver,
  `scripts/qwen38-prefix-cache-screen.py`, now also records the host's CPU pressure over
  each request.
- Roots, measured requests in UTC:

  | Root | Kind | w3 | w5 |
  | --- | --- | --- | --- |
  | a1 | cold | 21:32:52–21:34:08 | 21:34:08–21:35:03 |
  | b1 | cached | 21:39:42–21:40:36 | 21:40:36–21:41:32 |
  | b2 | cached | 21:46:10–21:47:06 | 21:47:06–21:48:03 |
  | a2 | cold | 21:52:42–21:54:00 | 21:54:00–21:54:57 |

- Receipt: `qwen38-prefix-cache-receipt --abba a1-w3 b1-w3 b2-w3 a2-w3 --abba a1-w5
  b1-w5 b2-w5 a2-w5`.

Evidence under `pkgs/qwen36/evidence/2026-09-27/`:

- `qwen38-prefix-cache-abba-{a1,b1,b2,a2}-{w3,w5}.json.gz`: the eight roots. The
  receipt binds each by the BLAKE3 of its uncompressed JSON.
- `qwen38-prefix-cache-abba-admission-receipt.json`: the receipt.
- `qwen38-prefix-cache-abba-startup.json`: for each root, its readiness time, its
  startup warm-up and `device-dram` events, and the host's CPU pressure and load when
  the root started and ended.

## Gates

| Gate | Bound | Measured (ABBA means) | Pass |
| --- | --- | --- | --- |
| Matched identity, ABBA order, cached roots agree | required | yes | yes |
| Exact token parity across all four roots | required | 22 of 22 turns | yes |
| Resumed hits in w3, both cached roots | required | 9 of 9 later turns | yes |
| Hit median TTFT ratio | ≤ 0.70 | 0.417 | yes |
| Hit median wall ratio | ≤ 0.85 | 0.662 | yes |
| Miss median wall ratio | ≤ 1.10 | 1.021 | yes |
| Maximum miss wall ratio | ≤ 1.25 | 1.080 | yes |
| Startup DRAM drop | ≤ 536,870,912 B | 364,904,448 B | yes |

Adjacent-pair ratios (diagnostics only):

| Pair | Hit median TTFT | Hit median wall | Miss median wall | Maximum miss wall |
| --- | --- | --- | --- | --- |
| a1 → b1 | 0.416 | 0.665 | 1.021 | 1.060 |
| b2 → a2 | 0.419 | 0.633 | 1.033 | 1.134 |

Either pair alone would also pass every bound.

## Per turn (client seconds, mean of cold roots → mean of cached roots)

| Root | Conv. | Turn | Prompt tokens | Hit tokens | TTFT | Wall | Wall ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| w3 | 0 | 0 | 3,959 | 0 | 2.32 → 2.33 | 5.01 → 5.42 | 1.08 |
| w3 | 1 | 0 | 3,201 | 0 | 1.96 → 1.98 | 4.52 → 4.62 | 1.02 |
| w3 | 2 | 0 | 3,250 | 0 | 1.95 → 1.97 | 5.04 → 5.10 | 1.01 |
| w3 | 0 | 1 | 5,756 | 3,072 | 3.40 → 1.80 | 6.57 → 5.07 | 0.77 |
| w3 | 1 | 1 | 5,045 | 3,072 | 2.83 → 1.20 | 5.28 → 3.81 | 0.72 |
| w3 | 2 | 1 | 5,078 | 3,072 | 2.83 → 1.22 | 5.59 → 4.11 | 0.74 |
| w3 | 0 | 2 | 7,504 | 5,120 | 4.33 → 1.58 | 6.81 → 4.21 | 0.62 |
| w3 | 1 | 2 | 6,736 | 4,096 | 3.99 → 1.77 | 6.78 → 4.49 | 0.66 |
| w3 | 2 | 2 | 6,452 | 4,096 | 3.76 → 1.57 | 6.98 → 4.69 | 0.67 |
| w3 | 0 | 3 | 9,267 | 7,168 | 5.35 → 1.47 | 8.62 → 4.63 | 0.54 |
| w3 | 1 | 3 | 8,385 | 6,144 | 4.80 → 1.51 | 8.26 → 4.66 | 0.56 |
| w3 | 2 | 3 | 7,953 | 6,144 | 4.54 → 1.24 | 7.22 → 3.93 | 0.54 |
| w5 | 0 | 0 | 3,959 | 0 | 2.32 → 2.27 | 5.39 → 5.19 | 0.96 |
| w5 | 1 | 0 | 3,201 | 0 | 1.97 → 1.97 | 4.71 → 4.67 | 0.99 |
| w5 | 2 | 0 | 3,250 | 0 | 1.99 → 1.96 | 5.07 → 5.21 | 1.03 |
| w5 | 3 | 0 | 3,550 | 0 | 2.05 → 2.07 | 5.11 → 5.52 | 1.08 |
| w5 | 4 | 0 | 3,820 | 0 | 2.25 → 2.28 | 5.26 → 5.43 | 1.03 |
| w5 | 0 | 1 | 5,756 | 0 | 3.39 → 3.40 | 6.74 → 6.79 | 1.01 |
| w5 | 1 | 1 | 5,045 | 0 | 2.85 → 2.85 | 5.65 → 5.65 | 1.00 |
| w5 | 2 | 1 | 5,078 | 0 | 2.82 → 2.83 | 5.54 → 5.55 | 1.00 |
| w5 | 3 | 1 | 5,346 | 0 | 3.10 → 3.11 | 6.09 → 6.22 | 1.02 |
| w5 | 4 | 1 | 6,011 | 0 | 3.39 → 3.43 | 6.58 → 6.73 | 1.02 |

- Hit and miss patterns, and the hit sizes, match p6 turn for turn. Both cached roots
  resumed the same prefixes.
- A resumed turn's TTFT stays at 1.2–1.8 s from 5K to 9K prompt tokens, while cold TTFT
  grows from 2.8 to 5.4 s. The advantage grows with conversation length, and the wall
  ratio falls from about 0.75 at turn 1 to 0.54 at turn 3.
- The mean miss TTFT is within 0.05 s of cold on every turn. Each save runs inside
  prefill.

## Host conditions

- Each turn's CPU stall fraction is the share of the request during which runnable tasks
  waited for a CPU, from the `some` line of `/proc/pressure/cpu` read before and after
  the request.
  - Per-root medians: 0.19 (a1), 0.22 (b1), 0.22 (b2), 0.22 (a2).
  - Range: 0.16–0.28, except one b2 turn at 0.87 (w5 c4 t1). That turn took 6.94 s,
    against 6.53 s in b1.
- The 1-minute load average stayed between 26.6 and 36.6 on 32 CPUs. During p6 it was
  52, with CPU `some avg10=75`.
- Readiness took 273 s, 272 s and 273 s for b1, b2 and a2, but 766 s for a1:
  - a1's width-1 warm-up spent 39.3 s capturing the trace and 94 s in prefill, against
    about 1.5 s and 6 s in the other roots.
  - Its masked-trace capture took 413 s, against 140–156 s.
  - [INFERENCE] a1 was the first process to open the devices after production stopped.
    The CPU pressure sampled at its start and end matches the other roots, so those
    samples do not explain the slow startup.
  - Startup speed feeds no gate; the DRAM view after warm-up matches the other cold
    root.

## Non-claims

- This is one agent-shaped workload pair on one package. It is not a general latency
  claim.
- The admission names the package above. A deployment must serve that exact store
  path; a later commit that rebuilds the same path carries the admission, and any other
  package needs a new screen.
- ABBA means cancel linear drift, not bursts. Bursts stayed small in this window.
- Cache entries live in one process and are lost on restart.
