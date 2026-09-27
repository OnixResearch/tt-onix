# p6: cold-versus-cached hardware screen (2026-09-27)

**Verdict: not admitted.** The receipt fails `miss_median_wall_ratio` (1.132 against
≤ 1.10) and `miss_wall_ratio` (1.955 against ≤ 1.25). Every other gate passes. Cache
hits cut the median TTFT to 0.43 of cold and the median wall to 0.71 of cold, with
exact token parity.

The miss failure occurred during a period of measured host contention. The screen
cannot separate that contention from any cost the cache itself adds. The cache stays
off in deployment, and the profile stays unadmitted.

## Protocol

- Gates: `pkgs/qwen36/prefix-cache-admission-policy.ncl`, committed in `7a15fdd` before
  the measurement. Admission policy BLAKE3:
  `b5f87652c3710f0a321673aeaa1fef461d655faec48d2ec7da9b21d296e60586`. Cache policy
  BLAKE3: `f4567693570cb0e85213c83f6c08e91179b065c9beed1ea2479abdca97ab0ea6`.
- Package: `/nix/store/43ayflq56m4yhx3p5n6ds6mishwki1gd-qwen38-0.1.0`.
  - Execution identity:
    `/nix/store/mi1rjghx7g0d7039jzmzkd885c38dmlh-qwen36-execution-identity-0.1.0`.
  - Runtime: `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.
- Host: britton-desktop, PCIe devices 0 and 1. Each root ran as a transient unit on port
  8011 with the production unit's environment. Production `qwen38-p150x2.service` was
  stopped from 19:26 to 19:53 UTC.
- Roots, both with `--continuous-slot-policy`:
  - Cached, which adds `--prefix-cache-policy`: requests from 19:40:26 to 19:42:29 UTC.
  - Cold: requests from 19:50:38 to 19:52:47 UTC.
- Driver: `scripts/qwen38-prefix-cache-screen.py`. It builds conversations of filler
  text in the `qwen38-agent-conversations-v1` workload. Each turn appends the previous
  answer and a new 5,000-character passage (1,700–2,500 tokens), then streams one
  greedy request and times it at the client.
  - w3: 3 conversations × 4 turns (workload BLAKE3 `7bdff9db…c175b8`). This shape
    exercises hits: each conversation's entry fits the 4 state slots.
  - w5: 5 conversations × 2 turns (workload BLAKE3 `38038f5d…6bbb6`). This shape
    exercises misses and eviction: 5 conversations thrash 4 slots under LRU, so every
    turn misses and every save evicts.
- Receipt: `qwen38-prefix-cache-receipt --pair cold-w3 cached-w3 --pair cold-w5 cached-w5`.

Evidence under `pkgs/qwen36/evidence/2026-09-27/`:

- `qwen38-prefix-cache-{cold,cached}-{w3,w5}.json.gz`: the four roots. The receipt
  binds each by the BLAKE3 of its uncompressed JSON.
- `qwen38-prefix-cache-admission-receipt.json`: the receipt.

## Gates

| Gate | Bound | Measured | Pass |
| --- | --- | --- | --- |
| Matched identity (package, runtime, devices, workloads, not simulated) | required | yes | yes |
| Exact token parity, cached against cold, every turn | required | 22 of 22 | yes |
| Resumed hits in w3 | required | 9 of 9 later turns | yes |
| Hit median TTFT ratio | ≤ 0.70 | 0.430 | yes |
| Hit median wall ratio | ≤ 0.85 | 0.710 | yes |
| Miss median wall ratio | ≤ 1.10 | 1.132 | **no** |
| Maximum miss wall ratio | ≤ 1.25 | 1.955 | **no** |
| Startup DRAM drop | ≤ 536,870,912 B | 364,904,448 B | yes |

## Per turn (client seconds, cold → cached)

| Root | Conv. | Turn | Prompt tokens | Hit tokens | TTFT | Wall | Wall ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| w3 | 0 | 0 | 3,959 | 0 | 2.32 → 2.36 | 4.87 → 5.51 | 1.13 |
| w3 | 1 | 0 | 3,201 | 0 | 1.95 → 1.99 | 4.50 → 5.06 | 1.12 |
| w3 | 2 | 0 | 3,250 | 0 | 1.97 → 1.97 | 5.01 → 5.34 | 1.07 |
| w3 | 0 | 1 | 5,756 | 3,072 | 3.38 → 1.78 | 6.52 → 5.36 | 0.82 |
| w3 | 1 | 1 | 5,045 | 3,072 | 2.86 → 1.23 | 5.32 → 4.10 | 0.77 |
| w3 | 2 | 1 | 5,078 | 3,072 | 2.84 → 1.23 | 5.75 → 4.11 | 0.72 |
| w3 | 0 | 2 | 7,504 | 5,120 | 4.35 → 1.59 | 6.85 → 4.47 | 0.65 |
| w3 | 1 | 2 | 6,736 | 4,096 | 3.98 → 1.79 | 6.55 → 4.94 | 0.75 |
| w3 | 2 | 2 | 6,452 | 4,096 | 3.76 → 1.62 | 6.99 → 4.96 | 0.71 |
| w3 | 0 | 3 | 9,267 | 7,168 | 5.33 → 1.48 | 8.52 → 5.03 | 0.59 |
| w3 | 1 | 3 | 8,385 | 6,144 | 4.80 → 1.54 | 7.95 → 4.91 | 0.62 |
| w3 | 2 | 3 | 7,953 | 6,144 | 4.55 → 1.27 | 7.41 → 3.99 | 0.54 |
| w5 | 0 | 0 | 3,959 | 0 | 2.29 → 2.32 | 5.43 → 5.54 | 1.02 |
| w5 | 1 | 0 | 3,201 | 0 | 1.97 → 2.06 | 4.86 → 9.51 | 1.95 |
| w5 | 2 | 0 | 3,250 | 0 | 2.00 → 2.25 | 5.32 → 7.32 | 1.38 |
| w5 | 3 | 0 | 3,550 | 0 | 2.09 → 2.15 | 5.37 → 6.23 | 1.16 |
| w5 | 4 | 0 | 3,820 | 0 | 2.31 → 2.35 | 5.75 → 6.56 | 1.14 |
| w5 | 0 | 1 | 5,756 | 0 | 3.39 → 3.46 | 6.53 → 7.27 | 1.11 |
| w5 | 1 | 1 | 5,045 | 0 | 2.84 → 2.87 | 5.80 → 6.74 | 1.16 |
| w5 | 2 | 1 | 5,078 | 0 | 2.83 → 2.89 | 5.85 → 6.51 | 1.11 |
| w5 | 3 | 1 | 5,346 | 0 | 3.14 → 3.46 | 6.09 → 8.27 | 1.36 |
| w5 | 4 | 1 | 6,011 | 0 | 3.40 → 3.56 | 6.96 → 7.47 | 1.07 |

- Hits save the whole prefill of the resumed range. The TTFT of a hit barely depends on
  prompt length (1.2–1.8 s from 5K to 9K tokens), while cold TTFT grows from 2.8 to
  5.3 s. The advantage grows with conversation length.
- Cached misses, including the eviction-heavy w5 misses, have a TTFT within 0.1 s of
  cold, except w5 c2 t0 (+0.25 s), c3 t1 (+0.32 s) and c4 t1 (+0.16 s). Each save runs
  inside prefill, before the first token.
- The miss excess lies after the first token, in the decode phase:
  - In w3 misses, that phase took 2.55 → 3.15, 2.55 → 3.07 and 3.04 → 3.37 s.
  - In w5 c1 t0 it took 2.89 → 7.45 s. The server reported 4.6 decode tokens/s for
    that request, against 11–15 tokens/s elsewhere.

## Host contention

The Qwen service is host-dispatch bound. britton-desktop also ran long-lived work
unrelated to this screen, which the screen left running:

- 18-day-old `animus-steel` processes at about 95% CPU each;
- a QEMU guest at about 280%;
- `rustfs` and others.

At 19:55 UTC the host showed load average 52 on 32 CPUs and CPU PSI `some avg10=75`.

The service log gives a host-speed proxy that both roots share. Each new continuous-slot
group re-captures the decode trace, and every capture logs 516 identical
`moe_utils.cpp:160` warnings with millisecond timestamps. From those warnings:

| Measure (same code path in both roots) | Cold | Cached |
| --- | --- | --- |
| Startup width-1 decode trace capture | 2.88 s | 33.95 s |
| Startup width-1 warm-up prefill | 13.0 s | 81.1 s |
| Masked prefill trace capture (4 buckets) | 226 s | 501 s |
| Per-request capture span, w3 | 0.74–1.11 s | 0.97–1.36 s |
| Per-request capture span, w5 | 0.89–1.30 s | 1.48–4.46 s |
| Median gap between capture operations, w5 | 1 ms | 2–3 ms |

- Startup warm-up does not depend on the cache. It still ran 6–12× slower in the
  cached root, so that root started under heavy contention.
- The slowest miss (w5 c1 t0, 19:41:29 UTC) had a 4.46 s capture with a median 3 ms
  operation gap. The cached request just before it took 1.25 s with a 1 ms gap.
- [INFERENCE] The miss decode path runs the same code in both roots. The cache changes
  only admission bookkeeping and the prefill save, and both sit before the first token.
  The large w5 excess is therefore most likely host contention. A smaller systematic
  cache cost of up to 0.3–0.6 s per request, as the w3 turn-0 rows suggest, cannot be
  excluded: those rows also ran during the contended period.

The gates were fixed before the measurement. The failure stands as recorded. A new
admission attempt needs a new protocol, declared before its measurement, that controls
for host load. For example, it could interleave cold and cached roots in ABBA order and
record CPU pressure beside each root.

Follow-up: protocol v2 did this, and its screen admitted the profile. See
`p7-abba-hardware-screen.md`.

## Observations

- Every request starts a new continuous-slot group, and every group re-captures the
  decode trace (`trace_captures` 1 per request). A lone agent turn therefore pays about
  1 s of capture before its second token, even at 1 ms per operation. That cost is
  independent of the cache. Holding the trace between groups would remove it for every
  request, and is a separate optimisation.
- Readiness took 402 s for the cold root and 828 s for the cached root. Both include
  the masked-trace capture, which the cache requires and which also ran under the
  contention above.
