# Admitted physical throughput evidence

This directory retains the admitted aggregate evidence from 2026-08-08.

- One-card aggregate throughput: `21.289619409251138` tokens/s.
- Two-card aggregate throughput: `38.65481160499171` tokens/s.
- Speedup: `1.8156647548238811`.
- Scaling efficiency: `0.9078323774119406`.
- P99 latency change: `-0.2225636226581016`.

Physical devices 0 and 1 served distinct requests. Matching profiler preflight runs attributed both cards. The five timed repeats did not enable profiler instrumentation. Every short and long generated stream matched the pinned FP32 CPU reference.

The raw receipt binds before and after `tt-smi -s --snapshot_no_tty` snapshots by BLAKE3. The snapshots remain in the bounded live evidence root because they include machine-specific board data.
