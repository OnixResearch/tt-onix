# First physical throughput candidate

This fixture retains the aggregate receipt from the first complete five-repeat physical candidate on 2026-08-08.

The run used physical devices 0 and 1. Every generated stream matched the pinned FP32 CPU reference. The aggregate two-card speedup was `1.6663084339651104`, which passed the `1.50` bound.

The candidate failed admission because one profiled card stalled during `dual-1`. The p99 latency regression was `0.6458061323139735`, above the `0.10` bound. Device-profiler buffers also reported dropped markers.

The deterministic decision was **stop profiling timed repeats and rerun**. No fused operator was justified. The follow-up keeps physical profiler attestation in separate matching runs and leaves the five timed repeats free of profiler instrumentation.
