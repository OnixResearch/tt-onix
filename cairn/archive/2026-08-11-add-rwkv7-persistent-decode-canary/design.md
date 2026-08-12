# Design: Bounded canary and runtime interlock

## Goal and evidence

The exact goal is a one-machine, matched physical canary with a runtime-enforced interlock. Completion requires typed limits, deterministic canary and interlock receipts, matched A/B controls, clean candidate telemetry, exact parity, bounded cleanup, and one retained physical run.

False completion includes a recommendation that the runtime ignores, an unbounded worker loop, a simulator performance claim, unmatched controls, historical A/B evidence presented as a new canary, path-based identities, or a critical result that still permits persistent dispatch.

## Bounded portfolio review

Subagent consent is absent. The serial review uses four mechanism families.

| Family | Mechanism | Result |
|---|---|---|
| Runtime interlock | The runtime reads a BLAKE3-bound state file for each canary worker. | Selected. It enforces the decision at dispatch. |
| Supervisor-only environment override | A wrapper sets the existing candidate override after alerts. | Rejected. Later workers can omit the override. |
| Profile mutation | A controller rewrites `physically_admitted` and rebuilds the package. | Rejected. It is slow, global, and unsafe for a bounded canary. |
| Receipt-only gate | A controller emits a decision without a runtime input. | Rejected. It repeats the existing recommendation-only gap. |

The adversarial audit targets stale armed files, mismatched profile identities, missing environment hashes, warning versus critical behavior, controller timeout, duplicate telemetry, normal-control pollution, and controller failure before finalization.

## Decisions

### Strict Nickel policy

The policy fixes one machine, physical devices `0` and `1`, and profile `rwkv7-p150x2-persistent-decode-v2`. It admits windows `[2, 4]` and uses four matched samples.

Each worker gets at most four requests. The canary plans `72` executions under a hard maximum of `104`.

The wall limit is `600` seconds. The candidate share is `0.5` within the matched comparison.

The policy also fixes CPU affinities, FIFO priority, minimum paired gain, and clean-monitoring requirements. Rust revalidates all cross-field arithmetic before execution.

### Armed sidecar before execution

The controller writes a compact armed interlock beside the new output root before it starts the benchmark. It passes the exact path, file BLAKE3, and canary binding BLAKE3 through named environment variables.

The runtime permits persistent dispatch only for an armed or retained state whose file hash, canary binding, profile ID, profile BLAKE3, and bounded limits match. A present invalid, held, or disabled interlock forces normal dispatch with `canary_interlock_blocked`.

Normal production execution without the canary environment keeps the current profile-owned behavior.

### Bounded matched benchmark

The controller invokes the packaged `rwkv7-p150x2-throughput-benchmark` with persistent A/B mode. One-card control and candidate workers run concurrently on separate cards. Each sample swaps their card assignments.

Dual-card control and candidate samples keep alternating order and synchronized starts. Separate per-card TT-Metal caches prevent concurrent compiler conflicts and reduce repeated setup.

The controller does not scan directories. It derives candidate receipt paths from the typed sample count and fixed topology. It collects only persistent candidate telemetry. Normal controls remain in the strict A/B comparison and do not inflate the fallback metric.

### Named timeout and bounded request accounting

The shell polls the child process at a named interval, kills it at the policy duration, and writes a disabled final interlock. The pure core verifies the expected profiler, normal-control, and persistent-candidate request totals before execution.

### Final state

A clean monitoring result, valid physical A/B receipt, exact token and FP32 state parity, complete cleanup, zero terminal timeout or failure, stable positive paired gain, and bounded duration produce `retain`.

A warning produces `hold`. A critical signal, invalid evidence, timeout, benchmark failure, or budget mismatch produces `disable`.

The controller replaces the armed sidecar atomically with the final interlock. It also stores the final interlock and canary receipt in the output root.

## Functional core and shell

`persistent_decode_canary.rs` contains policy parsing, budget arithmetic, A/B validation, state transitions, receipt construction, and BLAKE3 identities. The canary binary owns file reads, atomic writes, child process control, timeout polling, and known-path receipt loading.

The existing C++ persistent decode core owns typed interlock validation and the dispatch decision. `main.cpp` owns environment and file reads.

## Verification

Positive tests cover armed admission, clean retain, exact budgets, deterministic identities, warning hold, and physical candidate aggregation. Negative tests cover invalid policy, request overflow, timeout, benchmark failure, invalid A/B evidence, critical monitoring, stale or mismatched hashes, malformed interlocks, and candidate override attempts under disabled state.

The physical canary must use the packaged runtime and controller. It must retain the A/B, monitoring, final interlock, and canary receipts.
