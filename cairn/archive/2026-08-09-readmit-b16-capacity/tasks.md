# Tasks: Re-admit B16 card-local capacity

## Phase 1: Metric

- [x] [serial] M1 Add a pure per-token p99 latency helper. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission]
- [x] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission] M2 Change the capacity evaluation gate to compare p99 per-token time. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission]
- [x] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission] M3 Add positive per-token normalization and negative empty/zero-sample cases. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission]

## Phase 2: Evidence

- [x] [serial] E1 Run an order-balanced B4/B16 physical sweep with token-balanced workloads. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission]
- [x] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission] E2 Confirm exact FP32-reference token parity and physical/profiler attribution in every run. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission]
- [x] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission] E3 Retain the order-balanced B16-admitted evidence in the evidence package. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission]

## Phase 3: Admission

- [x] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission] A1 Evaluate B16 against both B4 baselines under the per-token gate. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission]
- [x] [depends:tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission] A2 Document the re-admission and the retained 16-wide kernel stall follow-up. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission]

## Phase 4: Gates

- [x] [serial] G1 Run formatting, evidence tests, package builds, and Cairn validation and gates. r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission]
