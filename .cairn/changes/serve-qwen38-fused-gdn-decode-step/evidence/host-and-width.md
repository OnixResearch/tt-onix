# Host overhead per token and batched widths on the fused step

Recorded 2026-09-24. This is a physical P150x2 run on package
`/nix/store/497albz9gzlrrjnbqc9vcdg9qba1vwx9-qwen38-0.1.0`, using the production
batched adapter with the model batch capacity at 16. It is diagnostic only. The
throwaway probes are `.pi/tmp/gdn-step/host_eval*.py`, `width_eval.py`, and
`concurrency_ab.sh`.

## Host time per token (width 1, `step`)

| Phase | ms |
| --- | ---: |
| trace-binding check | 0.62 |
| decode-input preparation | 0.29 |
| host-to-device input copy | 0.91 |
| trace replay (device) | 42.52 |
| `synchronize_device` after the replay | 0.92 |
| winner readback | 0.90 |

The replay is enqueued without blocking, and the winner readback runs on the
same command queue, so the readback already completes after the replay does.
The explicit `synchronize_device` added only fixed cost.

| Variant | ms/token (two runs) | Tokens |
| --- | --- | --- |
| sync after replay (previous) | 46.66, 46.11 | reference |
| no sync; readback on the trace queue | 44.96, 45.28 | identical |

The service adapters now omit that sync on every greedy and sampled replay and
name the trace queue on their readbacks. This saves about 1.2 ms per token,
2.6 percent.

## Width scaling on the fused step (adapter, 80 steps, median)

| Width | Step | Per-user tok/s | Aggregate tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 45.95 ms | 21.8 | 21.8 |
| 4 | 46.87 ms | 21.3 | 85.3 |
| 8 | 49.19 ms | 20.3 | 162.6 |
| 16 | 54.05 ms | 18.5 | 296.0 |

On the previous routes, the width-8 step was 79.4 ms and the width-16 step was
94.7 ms. That per-user retention (0.737 and 0.634) was below the 0.75 floor
blocking batch admission. On the fused step, retention is 0.934 at width 8 and
0.850 at width 16. User 0's continuation was identical across widths.

## Service under concurrent load (per-token decode latency, service evidence)

| Concurrent requests | `off` | `step` |
| ---: | ---: | ---: |
| 1 | 58.9–59.5 ms | 48.0–48.5 ms |
| 3 | 64.8–65.5 ms | 48.9–49.9 ms |
| 8 | 79.0–80.5 ms | 50.0–50.2 ms |
| 16 | 99.9–104.5 ms (split 14 + 2) | 56.4–57.9 ms (one group of 16) |

Every request succeeded. Time to first token dominates request wall time on
both routes. One short prompt takes 2.3 to 3.7 s to prefill. Grouped requests
prefill serially, so 16 concurrent users waited 49 to 100 s for their first
token.
