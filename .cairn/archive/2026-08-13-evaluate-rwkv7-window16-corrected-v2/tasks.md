# Tasks

## Phase 1: Candidate boundary

- [x] [serial] C1 Add a typed immutable v2 profile, runtime, evidence package, and closure while preserving v1. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] C2 Add positive and negative profile, runtime, precision, fallback, and identity tests. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] C3 Bind monitoring, throughput, canary, deployment, and observation evidence to v2. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]

## Phase 2: Device-free and simulator gates

- [x] [serial] V1 Pass package, source, closure, formatting, quality, and Cairn gates. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] V2 Run bounded Blackhole simulator smoke and retain simulator-only evidence. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]

## Phase 3: Physical correctness and performance

- [x] [serial] P1 Reject v2 after its bounded physical probe fails exact pinned CPU token parity. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] P2 Stop terminal lifecycle and profiler gates after the required correctness gate fails. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] P3 Do not run five order-balanced A/B samples after the correctness failure. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] P4 Do not run canary or monitoring after the correctness failure. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]

## Phase 4: Admission, deployment, and observation

- [x] [serial] A1 Retain `[2, 4, 8]` because v2 fails exact CPU token parity. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] A2 Do not deploy or observe the rejected candidate. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] A3 Sync, archive, run final gates, commit, push, and integrate. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
