# V7: fresh Blackhole simulator smoke on the variant runtime

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task V7.

## Result

The GDN grid variant runtime
(`/nix/store/6ccynlvv46p54pmp2lb7dbs65hkyj5w9-tt-metal-0.77.0`) opens the
packaged TTsim Blackhole `1x1` simulated device, compiles firmware, executes
representative TT-NN operations, and closes cleanly.

Probe: `/tmp/gdn-grid-sim/gdn-grid-sim-probe.py` under
`/tmp/gdn-grid-sim/run-gdn-grid-sim.sh` (mirrors `pkgs/ttsim/run-smoke.sh`
with the variant runtime substituted), `TT_METAL_DISABLE_SFPLOADMACRO=1`,
slow dispatch, fresh `HOME` and caches, `probe_exit=0`.

```json
{"event": "device_opened", "device_count": 1, "device_id": 0}
{"event": "compute_grid_measured", "grid_x": 13, "grid_y": 10, "compute_cores": 130}
{"event": "representative_operations_completed", "operations": ["ttnn.ones", "ttnn.add"], "result_shape": [1, 1, 32, 32]}
{"event": "clean_close"}
```

JIT telemetry shows 13 kernel compilations (0 cache hits), so firmware
compilation happened in this run.

## Launch-level guard behavior on the same simulator

Task A3's script extends this smoke to the change's own op. On the shared
(pre-change) runtime the 192-head fused decode launch fails host-side with
the exact guard fatal; on the variant runtime the same launch is accepted
and the program is created with the chunked assignment. Details and exact
diagnostics are in `attribution.md`.

## Simulator limitation that scopes this smoke

TTsim cannot execute the fused GDN decode op to completion at ANY head
count: its tensix matmul model rejects
`ALU_ACC_CTRL_Zero_Flag_disabled_src=1 (keep SrcB denormals) not modeled`,
which aborts the simulator process during deferred execution. The smoke
therefore proves the variant runtime's device open, firmware build, op
dispatch, and launch-level factory behavior on simulated Blackhole. It
cannot prove chunked kernel execution or numerics; per the
simulator-before-hardware workflow that requires hardware, recorded as the
reason V8 and V9 are physical tasks.

## Non-claims

Simulated evidence only. No physical placement, performance, or silicon
execution is claimed, and no admission binding moved.
