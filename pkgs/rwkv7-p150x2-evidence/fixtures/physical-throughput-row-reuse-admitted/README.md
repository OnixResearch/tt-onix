# BF16 head-row reuse admission

This bundle retains the admitted physical result after BF16 head-row reuse.

The runtime keeps the BF16 `r`, `k`, and `v` head rows from the WKV step. Attention correction reuses those rows. The FP32 recurrent-state path uses separate cast views.

This change removes four reshape launches per layer and recurrent step. The profiled B4 worker reduced tiled reshape launches from `8,463` to `6,591`. This is a reduction of `1,872`, or `22.119815668202765%`.

A five-pair alternating card-0 test compared the previous runtime with the new runtime. Each sample processed 80 tokens and preserved the expected generated-token streams.

- Old aggregate throughput: `57.541243473495776` tokens/s.
- New aggregate throughput: `59.650820029299965` tokens/s.
- Aggregate change: `3.666199109471613%`.
- Mean throughput change: `3.6186551677841905%`.
- Median throughput change: `4.118086873583815%`.

The strict saturated benchmark also passed profiler attribution and exact FP32 CPU-reference token parity.

- One-card throughput: `60.59247374263283` tokens/s.
- Two-card throughput: `123.79287232979172` tokens/s.
- Dual-card speedup: `2.0430404088732743`.
- Scaling efficiency: `1.0215202044366372`.
- P99 latency change: `-0.4094103335089039`.
- Admission identity: `ee5e777f3bd81ca9c9452e846866c484fec53f63959a916203a7f53d84644c65`.

Machine latency remains nonstationary. Use the alternating A/B result for the optimization claim. Do not treat the difference between separate strict runs as an isolated optimization gain.

The strict source run remains at `/var/tmp/rwkv7-p150x2-row-reuse-saturated-20260809-0830/` on the measurement host.
