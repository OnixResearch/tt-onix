# Key-interpolation candidate

This candidate rewrote the key scaling equation as an exact interpolation:

`lerp(raw_k, raw_k * a_gate, k_a)`

The profiled B4 worker removed `936` launches. It removed scalar and row-broadcast binary work, then added one row-broadcast ternary launch per layer and step.

A six-pair order-balanced card-0 test preserved the expected generated-token streams. One candidate sample was slower than its pair by `20.342183411346326%`.

After that pair was excluded:

- Old aggregate throughput: `78.2086590235259` tokens/s.
- Candidate aggregate throughput: `81.85541723356575` tokens/s.
- Aggregate gain: `4.662857355657857%`.
- Faster candidate pairs: `3` of `5`.

Strict admission failed:

- One-card throughput: `88.48352791430418` tokens/s.
- Two-card throughput: `156.00165524423414` tokens/s.
- Dual-card speedup: `1.7630587174973507`.
- P99 latency regression: `0.5001155625063585`.
- Maximum allowed p99 regression: `0.10`.

The single-card gain did not survive concurrent execution. The source change is reverted.

The strict source remains at `/var/tmp/rwkv7-p150x2-key-lerp-saturated-20260809-2240/`.
The A/B source remains at `/var/tmp/rwkv7-p150x2-key-lerp-ab-20260809-2220/`.
The profiler source remains at `/var/tmp/rwkv7-p150x2-key-lerp-profiler-20260809-2210/`.
