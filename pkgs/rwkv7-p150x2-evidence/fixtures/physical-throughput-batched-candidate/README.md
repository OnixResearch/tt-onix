# Fixed-batch physical candidate

This candidate measured the first static card-local batch implementation on 2026-08-08. Five one-card samples and five two-card samples completed on physical P150 devices 0 and 1.

All worker token streams matched the pinned FP32 CPU reference. The candidate improved absolute throughput for both topologies:

- One card: `49.88481963465173` tokens/s.
- Two cards: `50.67726546622957` tokens/s.

The candidate is not admitted. Its speedup was `1.0158855106098728`, below the required `1.50`. Its p99 change was `0.25554955976220295`, above the allowed `0.10`.

The fixed four-request workload fills one four-row batch on one card. It gives each card only two rows in the two-card run. Use this candidate as runtime optimization evidence, not as saturated scaling evidence.

The receipt identity is `8b2f510c06c8556ef3d9fc23f61858ca814e2c6346df51f741eea05a2a8d2e35`.
