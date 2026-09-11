# WKV transpose-fusion admission

This bundle retains the admitted physical result after WKV transpose fusion.

The runtime uses the transpose flags of `ttnn::matmul`. This change removes separate transpose operations for the `a`, `v`, and `r` vectors.

The profiled B4 worker reduced transpose launches from `1,404` to zero. The other major launch counts did not increase.

A five-pair alternating card-0 test compared the head-row normalization runtime with the transpose-fusion runtime. Each sample processed 80 tokens and preserved the expected generated-token streams.

- Old aggregate throughput: `62.20151007588795` tokens/s.
- New aggregate throughput: `64.09375680893515` tokens/s.
- Aggregate change: `3.042123464106572%`.
- Mean throughput change: `3.0567486123711296%`.
- Median throughput change: `3.513791445136394%`.

The strict saturated benchmark passed profiler attribution and exact FP32 CPU-reference token parity.

- One-card throughput: `62.53832039352678` tokens/s.
- Two-card throughput: `125.00182683097606` tokens/s.
- Dual-card speedup: `1.998803710179507`.
- Scaling efficiency: `0.9994018550897535`.
- P99 latency change: `0.054866887031256466`.
- Admission identity: `4e9c6db426e09f1f8ed1e4e49d31d037764d32882b9754638145aee53e073bbd`.

Machine latency remains nonstationary. Use the alternating A/B result for the optimization claim. Do not compare separate strict runs as an isolated gain.

The strict source run remains at `/var/tmp/rwkv7-p150x2-transpose-fusion-saturated-20260809-1210/` on the measurement host.
