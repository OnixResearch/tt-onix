# Tasks: Persist RWKV-7 device decode

## Phase 1: Lifecycle and binding contracts

- [x] [serial] B1 Record normal-dispatch token flow, host submissions, dynamic bindings, completion behavior, and token streams. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_device_decode] B2 Add a pure bounded request-lifecycle state machine. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_device_decode] B3 Add typed profile limits for steps, time, queue depth, cancellation polling, and output capacity. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_device_decode] B4 Add BLAKE3-bound token, state, slot, weight, program, runtime, checkpoint, and output identities. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]

## Phase 2: Persistent execution shell

- [x] [serial] R1 Add capability validation and retain normal dispatch as fallback. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_device_decode] R2 Keep selected token IDs and FP32 recurrent state on the device between bounded steps. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_device_decode] R3 Add explicit completion, cancellation, timeout, and failure handshakes. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_device_decode] R4 Prevent slot release until completion acknowledgement and state zeroing finish. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_device_decode] R5 Record setup submissions, persistent steps, exits, host submissions per token, and dropped-marker status. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]

## Phase 3: Positive and negative tests

- [x] [parallel] V1 Add positive tests for completion, maximum length, cancellation, concurrent slots, recycling, and deterministic receipts. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [parallel] V2 Add negative tests for stale bindings, identity drift, duplicate slots, invalid transitions, output overflow, and malformed completion. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [parallel] V3 Add timeout and lost-cancellation tests that prove bounded host recovery. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [serial] V4 Compare persistent tokens with normal dispatch and the pinned FP32 CPU reference. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]

## Phase 4: Synchronization and simulator gates

- [x] [serial] A1 Run applicable semaphore, mailbox, dataflow, NoC, MMIO, and source-register synchronization audits. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_device_decode] A2 Run simulator smoke for carried steps, completion, cancellation, and bounded timeout behavior. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]

## Phase 5: Physical acceptance

- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_device_decode] H1 Verify zero per-token host submissions inside the persistent region. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_device_decode] H2 Run stable normal-versus-persistent alternating A/B measurements at admitted capacities. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_device_decode] H3 Run strict one-card and two-card admission. Select persistence only after all gates pass. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]

## Phase 6: Documentation and gates

- [x] [serial] D1 Document limits, lifecycle states, fallback, cancellation, measured gain, receipt identities, and non-claims. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_device_decode] G1 Run formatting, package tests, negative fixtures, simulator smoke, Cairn gates, closure checks, and relevant Nix checks. r[tenstorrent.rwkv7_p150x2.persistent_device_decode]
