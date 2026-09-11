# Filtered-worker physical candidate

This rejected candidate resolves the dual-worker token corruption.

Each fast worker selects one physical card with `TT_VISIBLE_DEVICES`. It uses the pinned `p150_mesh_graph_descriptor.textproto` and opens filtered logical device 0.

All profiler and timed token streams matched the pinned FP32 CPU reference. The strict benchmark reached scaling admission without an attribution or parity failure.

The candidate is not admitted:

- One-card throughput: `51.61843281609733` tokens/s.
- Two-card throughput: `42.20120195168836` tokens/s.
- Dual-card speedup: `0.8175606977848389`.
- Scaling efficiency: `0.40878034889241943`.
- P99 latency change: `3.4918677732695094`.

The four-request workload gives one-card mode a B4 batch. It gives each two-card worker a B2 batch. This shape does not prove saturated two-card scaling.

The source run remains at `/var/tmp/rwkv7-p150x2-filtered-worker-benchmark-20260809-0330/` on the measurement host.
