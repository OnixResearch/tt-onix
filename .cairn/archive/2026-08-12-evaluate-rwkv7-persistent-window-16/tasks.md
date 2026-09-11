# Tasks

## Phase 1: Candidate boundary

- [x] [serial] C1 Add a typed, immutable, non-admitted window-`16` candidate profile and package. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_decode_window_16] C2 Add positive and negative runtime tests for window `16` and production fallback. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_decode_window_16] C3 Add profile-bound evidence, monitoring, throughput, and canary contracts. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]

## Phase 2: Device-free validation

- [x] [serial] V1 Pass package, source-boundary, production-closure, repository-quality, formatting, and Cairn gates. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] V2 Run bounded Blackhole simulator smoke and retain a simulated-only receipt. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]

## Phase 3: Physical evidence

- [x] [serial] P1 Stop competing services, require exact CPU token parity first, and stop after the bounded correctness probe rejects the candidate. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] P2 Do not accept partial profiler preflight after the required correctness gate fails. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] P3 Stop the five-sample A/B plan after its first sample fails exact CPU token attribution. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] P4 Do not run the canary or claim a monitoring recommendation after correctness failure. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]

## Phase 4: Admission

- [x] [serial] A1 Retain `[2, 4, 8]` because exact CPU generated-token parity failed. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
- [x] [serial] A2 Record the bounded rejection, run final gates, archive, commit, integrate, push, and remove the worktree. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]
