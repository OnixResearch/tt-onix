# RWKV-7 P150x2 Runtime Specification Delta

## ADDED Requirements

### Requirement: Fused RWKV-7 WKV kernel
r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel] The runtime MAY provide a package-matched Blackhole kernel for token-at-a-time 64-wide RWKV-7 heads. The kernel MUST preserve the accepted WKV equation, FP32 recurrent state, BF16 output boundary, unique state ownership, and exact generated-token parity. The runtime MUST use the fused path only after capability validation and stable physical admission pass.

#### Scenario: Supported fused execution passes
r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel.valid]
- GIVEN supported Blackhole hardware, 64-wide heads, BF16 model vectors, FP32 recurrent state, and unique state-tile ownership
- WHEN the fused WKV operator executes a decode step
- THEN its state and output MUST match the pure reference within profile-owned intermediate tolerances
- AND the complete workload MUST preserve exact FP32-reference generated tokens.

#### Scenario: Unsupported input fails closed
r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel.invalid_input]
- GIVEN an unsupported architecture, shape, dtype, layout, alias, or state-ownership plan
- WHEN fused-path capability validation runs
- THEN validation MUST reject the fused dispatch before device execution
- AND the runtime MUST retain the admitted generic path.

#### Scenario: Hazard evidence is incomplete
r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel.missing_audit]
- GIVEN a custom kernel without complete applicable LLK hazard, instruction-latency, performance, simulator, or parity evidence
- WHEN physical-run admission evaluates the candidate
- THEN admission MUST fail before silicon execution.

#### Scenario: Launches fall but throughput does not improve
r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel.no_gain]
- GIVEN a parity-correct candidate with fewer launches but no stable physical throughput gain
- WHEN runtime selection evaluates the candidate
- THEN the runtime MUST keep the generic path
- AND the evidence package MUST retain the measured counterexample without claiming an improvement.
