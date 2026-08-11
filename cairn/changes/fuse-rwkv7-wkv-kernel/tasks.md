# Tasks: Fuse the RWKV-7 WKV kernel

## Phase 1: Contract and reference

- [x] [serial] B1 Capture the accepted decomposed WKV equation, tensor shapes, precision boundaries, launch profile, and token streams. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [x] [depends:tenstorrent.rwkv7_p150x2.fused_wkv_kernel] B2 Add a pure fused-equation reference and typed capability contract. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [x] [depends:tenstorrent.rwkv7_p150x2.fused_wkv_kernel] B3 Add checked shape, dtype, layout, architecture, and state-ownership validation. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]

## Phase 2: Kernel implementation

- [x] [serial] K1 Implement bounded data movement for BF16 vectors and FP32 recurrent state. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [x] [depends:tenstorrent.rwkv7_p150x2.fused_wkv_kernel] K2 Implement decay, rank updates, state write, readout, attention correction, and BF16 output. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [x] [depends:tenstorrent.rwkv7_p150x2.fused_wkv_kernel] K3 Add deterministic head, request, core, buffer, and semaphore ownership. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [x] [depends:tenstorrent.rwkv7_p150x2.fused_wkv_kernel] K4 Add the capability-selected runtime shell and retain the generic TT-NN fallback. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]

## Phase 3: Correctness and negative coverage

- [x] [parallel] V1 Add positive tests for zero state, nonzero state, multiple requests, BF16 boundaries, and repeated state carry. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [x] [parallel] V2 Add negative tests for invalid width, head count, state rank, dtype, layout, architecture, aliasing, and ownership. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [x] [parallel] V3 Compare intermediate state and output with the pure reference under profile-owned tolerances. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [x] [serial] V4 Require exact end-to-end generated-token parity for every admitted workload. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]

## Phase 4: Hazard and simulator gates

- [x] [serial] A1 Run the complete LLK race audit set for config, semaphore, mailbox, dataflow, source-register, and NoC hazards. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [x] [serial] A2 Run instruction-latency and performance audits for every hand-written instruction sequence. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [x] [depends:tenstorrent.rwkv7_p150x2.fused_wkv_kernel] A3 Run package-matched Blackhole simulator smoke for one layer and recurrent-state carry. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]

## Phase 5: Physical acceptance

- [x] [depends:tenstorrent.rwkv7_p150x2.fused_wkv_kernel] H1 Capture before and after program signatures, launch counts, device time, and memory facts. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [x] [depends:tenstorrent.rwkv7_p150x2.fused_wkv_kernel] H2 Run stable alternating A/B measurements at each admitted batch capacity. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [ ] [depends:tenstorrent.rwkv7_p150x2.fused_wkv_kernel] H3 Run strict one-card and two-card admission. Select the fused path only after all gates pass. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]

  Blocker: synchronized launch raised fused two-card speedup to `1.6494731551938797`, above the required `1.5`. Its p99 latency regression is `0.24129836831923815`, above the permitted `0.10`. Physical captures located slow-wave time in on-CPU host layer submission. Fetch-queue waits covered less than `0.06%` of the excess, and device cycles fell. The receipt does not yet identify a smaller TT-NN or TT-Metal host call. Production admission remains closed.

## Phase 6: Documentation and gates

- [x] [serial] D1 Document supported shapes, precision, fallback, measured gain, receipt identities, and non-claims. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
- [x] [depends:tenstorrent.rwkv7_p150x2.fused_wkv_kernel] G1 Run formatting, package tests, negative fixtures, simulator smoke, Cairn gates, source-boundary checks, closure checks, and relevant Nix checks. r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
