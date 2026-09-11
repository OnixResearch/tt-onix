# Saturated physical throughput admission

This bundle retains the admitted eight-request physical result.

One-card samples use two bounded B4 waves on physical card 0. Two-card samples run one B4 worker on each physical card at the same time.

Each worker uses `TT_VISIBLE_DEVICES`, the pinned single-card P150 descriptor, and filtered logical device 0. All profiler and timed token streams matched the pinned FP32 CPU reference.

- One-card throughput: `52.25896285041297` tokens/s.
- Two-card throughput: `109.66330232771391` tokens/s.
- Dual-card speedup: `2.0984592182132698`.
- Scaling efficiency: `1.0492296091066349`.
- P99 latency change: `-0.053870298117255126`.
- Admission identity: `b5e3d569a33d393d413b1a236f6c7b97705fd6b1bcd18c17e11d54a163966669`.

The result passes the `1.50` speedup bound and the `0.10` p99 regression bound. It does not prove long-duration stability or public cross-platform ranking.

The source run remains at `/var/tmp/rwkv7-p150x2-saturated-wave-benchmark-20260809-0550/` on the measurement host.
