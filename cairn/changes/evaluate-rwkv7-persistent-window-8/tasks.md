# Tasks

## Phase 1: Candidate boundary

- [ ] [serial] C1 Add a typed, immutable, non-admitted window-`8` candidate profile and package without changing the production `[2, 4]` profile. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_8]
- [ ] [depends:tenstorrent.rwkv7_p150x2.persistent_decode_window_8] C2 Extend positive and negative runtime tests for window `8`, profile separation, malformed windows, and unsupported production dispatch. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_8]
- [ ] [depends:tenstorrent.rwkv7_p150x2.persistent_decode_window_8] C3 Add profile-bound evidence, monitoring, throughput, and canary contracts for window `8`. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_8]

## Phase 2: Device-free validation

- [ ] [serial] V1 Pass runtime, evidence, source-boundary, production-closure, repository-quality, formatting, and Cairn gates. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_8]
- [ ] [serial] V2 Run the bounded Blackhole simulator smoke and retain a simulated-only receipt. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_8]

## Phase 3: Physical evidence

- [ ] [serial] P1 Stop competing services, verify devices `[0, 1]` are unowned, and run bounded completion, cancellation, timeout, and failure probes. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_8]
- [ ] [serial] P2 Run final trace profiler validation and reject dropped or incomplete markers. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_8]
- [ ] [serial] P3 Run five order-balanced profiler-free one-card and two-card A/B samples. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_8]
- [ ] [serial] P4 Run one bounded interlocked canary and require clean aggregate monitoring. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_8]

## Phase 4: Admission

- [ ] [serial] A1 Admit window `8` only if every simulator, parity, cleanup, profiler, A/B, and canary gate passes. Otherwise retain `[2, 4]`. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_8]
- [ ] [serial] A2 Run final deterministic gates, sync the accepted specification, archive evidence, commit, integrate, push, and remove the worktree. r[tenstorrent.rwkv7_p150x2.persistent_decode_window_8]
