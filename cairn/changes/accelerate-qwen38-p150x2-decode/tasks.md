# Tasks: Accelerate Qwen3.8-27B P150x2 decode

## Phase 1: Baseline and contracts

- [x] [serial] B1 Reproduce package checks and retain the current batch-one physical baseline before core changes. r[tenstorrent.qwen38_p150x2.decode_performance_admission]
- [x] [serial] B2 Add a typed Nickel profile for trace, slot, timeout, workload, and admission bounds. r[tenstorrent.qwen38_p150x2.decode_performance_admission]
- [x] [depends:tenstorrent.qwen38_p150x2.decode_performance_admission] B3 Add pure lifecycle and evidence-admission types. r[tenstorrent.qwen38_p150x2.persistent_decode_buffers] r[tenstorrent.qwen38_p150x2.decode_performance_admission]

## Phase 2: Traced decode and stable buffers

- [x] [depends:tenstorrent.qwen38_p150x2.persistent_decode_buffers] T1 Add bounded startup allocation for request slots and stable decode tensors. r[tenstorrent.qwen38_p150x2.persistent_decode_buffers]
- [x] [depends:tenstorrent.qwen38_p150x2.traced_decode] T2 Snapshot GDN state, capture one decode step, and restore state before request execution. r[tenstorrent.qwen38_p150x2.traced_decode]
- [x] [depends:tenstorrent.qwen38_p150x2.traced_decode] T3 Refresh token, position, RoPE, and other mutable trace inputs before replay. r[tenstorrent.qwen38_p150x2.traced_decode]
- [x] [depends:tenstorrent.qwen38_p150x2.traced_decode] T4 Retain eager decode as a named fail-closed fallback. r[tenstorrent.qwen38_p150x2.traced_decode]
- [x] [depends:tenstorrent.qwen38_p150x2.persistent_decode_buffers] T5 Reset or quarantine every slot after success, rejection, cancellation, timeout, or failure. r[tenstorrent.qwen38_p150x2.persistent_decode_buffers]

## Phase 3: Bounded greedy selection

- [x] [depends:tenstorrent.qwen38_p150x2.shard_local_argmax] A1 Compute one maximum and global token index on each vocabulary shard. r[tenstorrent.qwen38_p150x2.shard_local_argmax]
- [x] [depends:tenstorrent.qwen38_p150x2.shard_local_argmax] A2 Add pure shard validation and lowest-global-index tie handling. r[tenstorrent.qwen38_p150x2.shard_local_argmax]
- [x] [depends:tenstorrent.qwen38_p150x2.shard_local_argmax] A3 Remove full-vocabulary host transfer from admitted traced greedy decode. r[tenstorrent.qwen38_p150x2.shard_local_argmax]

## Phase 4: Verification

- [ ] [parallel] V1 Add positive tests for lifecycle transitions, trace bindings, state restoration, shard reduction, and deterministic ties. r[tenstorrent.qwen38_p150x2.traced_decode] r[tenstorrent.qwen38_p150x2.persistent_decode_buffers] r[tenstorrent.qwen38_p150x2.shard_local_argmax]
- [ ] [parallel] V2 Add negative tests for stale bindings, state drift, invalid transitions, malformed shards, non-finite maxima, and cleanup failure. r[tenstorrent.qwen38_p150x2.traced_decode] r[tenstorrent.qwen38_p150x2.persistent_decode_buffers] r[tenstorrent.qwen38_p150x2.shard_local_argmax]
- [ ] [serial] V3 Compare eager and traced greedy continuations across the profile prompt corpus. r[tenstorrent.qwen38_p150x2.decode_performance_admission]
- [ ] [serial] V4 Verify sequential requests cannot observe prior KV, GDN, token, or output state. r[tenstorrent.qwen38_p150x2.persistent_decode_buffers]

## Phase 5: Physical acceptance

- [ ] [depends:tenstorrent.qwen38_p150x2.decode_performance_admission] H1 Run order-balanced eager and traced samples on physical devices `0` and `1`. r[tenstorrent.qwen38_p150x2.decode_performance_admission]
- [ ] [depends:tenstorrent.qwen38_p150x2.decode_performance_admission] H2 Admit traced decode only after parity, cleanup, throughput, TTFT, and tail-latency gates pass. r[tenstorrent.qwen38_p150x2.decode_performance_admission]
- [x] [serial] D1 Document the selected path, profile identity, receipt identity, fallbacks, and non-claims. r[tenstorrent.qwen38_p150x2.decode_performance_admission]
- [ ] [serial] G1 Run focused tests, positive and negative fixtures, package checks, Cairn validation, and all lifecycle gates. r[tenstorrent.qwen38_p150x2.traced_decode]

## Verification Coverage

- `Scenario: Eligible decode uses trace replay` -> T2, T3, V1
- `Scenario: Invalid trace binding falls back` -> T3, T4, V2
- `Scenario: Trace capture changes recurrent state` -> T2, V2
- `Scenario: Stable buffers serve one request` -> T1, T5, V1
- `Scenario: Slot cleanup fails` -> T5, V2, V4
- `Scenario: Valid shard winners select one token` -> A1, A2, V1
- `Scenario: Malformed shard result is rejected` -> A2, V2
- `Scenario: Physical candidate passes` -> B1, V3, H1, H2
- `Scenario: Candidate lacks complete evidence` -> B3, H2, V2
