# Tasks: Optimize RWKV-7 P150x2 inference throughput

## Phase 1: Baseline and contracts

- [x] [serial] B1 Record the current package tests, Cairn validation, one-card runtime behavior, lockstep two-card behavior, and available profiler evidence before core changes. r[tenstorrent.rwkv7_p150x2.throughput_measurement]
- [x] [serial] B2 Add a typed Nickel throughput profile with named batch, long-context, repeat, speedup, and p99 latency bounds. r[tenstorrent.rwkv7_p150x2.throughput_measurement]
- [x] [depends:tenstorrent.rwkv7_p150x2.throughput_measurement] B3 Add a pure admission core for metric sanity, target equations, card attribution, and non-claim classification. r[tenstorrent.rwkv7_p150x2.throughput_measurement] r[tenstorrent.rwkv7_p150x2.physical_speedup_claim]

## Phase 2: Independent replica runtime

- [x] [serial] R1 Add a fast inference mode without changing the existing evidence mode. r[tenstorrent.rwkv7_p150x2.fast_inference_mode]
- [x] [depends:tenstorrent.rwkv7_p150x2.fast_inference_mode] R2 Implement a pure deterministic scheduler for request-to-card and request-to-state-slot assignment. r[tenstorrent.rwkv7_p150x2.independent_replicas]
- [x] [depends:tenstorrent.rwkv7_p150x2.independent_replicas] R3 Add one independent full model replica per physical card and reject lockstep request ownership in fast mode. r[tenstorrent.rwkv7_p150x2.independent_replicas]
- [x] [depends:tenstorrent.rwkv7_p150x2.independent_replicas] R4 Add deterministic continuous batching with card-local state slots and profile-owned capacity. r[tenstorrent.rwkv7_p150x2.independent_replicas]
- [x] [depends:tenstorrent.rwkv7_p150x2.fast_inference_mode] R5 Keep token-shift and WKV state on the device across decode steps and use FP32 recurrence. r[tenstorrent.rwkv7_p150x2.device_resident_state]
- [x] [depends:tenstorrent.rwkv7_p150x2.fast_inference_mode] R6 Remove state readback, profiler flush, control replay, and artifact writes from the fast timed interval. r[tenstorrent.rwkv7_p150x2.fast_inference_mode]

## Phase 3: Dispatch and measured optimization

- [x] [depends:tenstorrent.rwkv7_p150x2.fast_inference_mode] O1 Validate dynamic token and state bindings, then capture and replay the static decode graph or record the exact package-matched blocker. r[tenstorrent.rwkv7_p150x2.fast_inference_mode]
- [x] [depends:tenstorrent.rwkv7_p150x2.device_resident_state] O2 Run the packaged `libttsim_bh.so` `1x1` smoke for device open, firmware compilation, representative weight load, one RWKV layer, and recurrent-state carry. r[tenstorrent.rwkv7_p150x2.simulator_gate]
- [x] [depends:tenstorrent.rwkv7_p150x2.throughput_measurement] O3 Capture a physical profiler receipt for the fast one-card path and identify the largest steady-state bottleneck. r[tenstorrent.rwkv7_p150x2.throughput_measurement]
- [x] [depends:tenstorrent.rwkv7_p150x2.throughput_measurement] O4 Capture the first two-card candidate receipt and record a deterministic stop-or-fuse decision against the named throughput target. r[tenstorrent.rwkv7_p150x2.throughput_measurement]
- [x] [depends:tenstorrent.rwkv7_p150x2.device_resident_state] O5 If profiler evidence attributes a target miss to recurrent launches, implement the fused WKV operator with BF16 boundaries and FP32 state. r[tenstorrent.rwkv7_p150x2.device_resident_state]
- [x] [depends:tenstorrent.rwkv7_p150x2.device_resident_state] O6 For any custom kernel, run the applicable LLK hazard, instruction-latency, and performance audits before hardware execution. r[tenstorrent.rwkv7_p150x2.device_resident_state]

## Phase 4: Correctness and negative coverage

- [x] [parallel] V1 Add positive tests for deterministic scheduling, independent state carry, profile export, FP32 state metadata, and passing measurement fixtures. r[tenstorrent.rwkv7_p150x2.independent_replicas] r[tenstorrent.rwkv7_p150x2.device_resident_state] r[tenstorrent.rwkv7_p150x2.throughput_measurement]
- [x] [parallel] V2 Add negative tests for duplicate request ownership, live state-slot reuse, batch overflow, BF16 state, stale trace bindings, missing card attribution, and non-finite metrics. r[tenstorrent.rwkv7_p150x2.independent_replicas] r[tenstorrent.rwkv7_p150x2.device_resident_state] r[tenstorrent.rwkv7_p150x2.throughput_measurement]
- [x] [parallel] V3 Add a negative receipt fixture that reports speedup from simulator evidence or incomplete physical profiler data. r[tenstorrent.rwkv7_p150x2.physical_speedup_claim]
- [x] [serial] V4 Compare every independent stream with the existing CPU reference for the named short and long-context profiles. r[tenstorrent.rwkv7_p150x2.device_resident_state]

## Phase 5: Physical acceptance and documentation

- [x] [depends:tenstorrent.rwkv7_p150x2.simulator_gate] H1 Run the one-card baseline and two-card candidate on physical devices 0 and 1 with the same runtime, workload per replica, and machine facts. r[tenstorrent.rwkv7_p150x2.throughput_measurement]
- [x] [depends:tenstorrent.rwkv7_p150x2.throughput_measurement] H2 Admit the speedup claim only when the dual-card ratio and p99 latency bounds pass with complete per-card profiler attribution. r[tenstorrent.rwkv7_p150x2.physical_speedup_claim]
- [x] [serial] D1 Update the runtime README with fast-mode use, profile values, physical receipts, simulator labels, and single-stream non-claims. r[tenstorrent.rwkv7_p150x2.physical_speedup_claim]
- [x] [depends:tenstorrent.rwkv7_p150x2.physical_speedup_claim] G1 Run formatting, package tests, positive and negative fixtures, Cairn validation, lifecycle gates, and the smallest relevant Nix checks. r[tenstorrent.rwkv7_p150x2.fast_inference_mode]

## Verification Coverage

- `Scenario: Distinct requests use both replicas` -> R2, R3, R4, V1
- `Scenario: Lockstep ownership is rejected` -> R2, R3, V2
- `Scenario: Fast decode excludes evidence synchronization` -> R1, R6, V1
- `Scenario: Evidence work enters the timed interval` -> R6, V2
- `Scenario: FP32 state remains on device` -> R5, V1, V4
- `Scenario: BF16 recurrent state is rejected` -> R5, V2
- `Scenario: Complete physical comparison passes` -> B2, B3, H1, H2
- `Scenario: Incomplete or non-finite metrics fail` -> B3, V2
- `Scenario: Simulator smoke passes without a performance claim` -> O2, V3
- `Scenario: Simulator evidence claims speedup` -> V3
- `Scenario: Physical speedup target passes` -> O3, O4, H1, H2
- `Scenario: Target miss cannot claim speedup` -> B3, O4, H2, V3
