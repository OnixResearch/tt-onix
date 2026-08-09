# Tasks: Scale RWKV-7 card-local batching

## Phase 1: Contracts and baseline

- [ ] [serial] B1 Record the current B4 runtime, profiler launch groups, memory use, exact token streams, and admitted physical baseline. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
- [ ] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching] B2 Add typed B4, B16, B32, and B64 capacity candidates with memory reserves and evidence bounds. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
- [ ] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching] B3 Add a pure capacity and memory planner with checked arithmetic. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]

## Phase 2: Runtime scheduling

- [ ] [serial] R1 Replace the fixed slot boundary with the admitted profile capacity. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
- [ ] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching] R2 Allocate card-local FP32 recurrent state and token-shift state from the checked plan. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
- [ ] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching] R3 Preserve deterministic slot ownership, sorted recycling, and state zeroing. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
- [ ] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching] R4 Record effective batch size, occupancy, host gaps, memory use, and per-request latency. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]

## Phase 3: Positive and negative tests

- [ ] [parallel] V1 Add positive tests for all candidate plans, deterministic assignment, slot recycling, and receipt calculation. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
- [ ] [parallel] V2 Add negative tests for duplicate ownership, stale state, unsupported capacity, checked-arithmetic overflow, and insufficient reserve. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
- [ ] [parallel] V3 Add negative evidence fixtures for parity mismatch, malformed metrics, non-finite metrics, and unsupported simulator claims. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]

## Phase 4: Simulator and physical sweep

- [ ] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching] H1 Run the package-matched simulator smoke for each physically attempted capacity. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
- [ ] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching] H2 Run bounded B4, B16, B32, and B64 physical sweeps until the stop rule fires. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
- [ ] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching] H3 Compare every stream with the pinned FP32 CPU reference. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
- [ ] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching] H4 Promote only a candidate with stable A/B gains and strict two-card admission. Otherwise retain the measured blocker. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]

## Phase 5: Documentation and gates

- [ ] [serial] D1 Document the selected capacity, throughput, latency, memory use, receipt identities, and all non-claims. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
- [ ] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching] G1 Run formatting, runtime tests, evidence tests, simulator smoke, Cairn validation, lifecycle gates, and relevant Nix checks. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
