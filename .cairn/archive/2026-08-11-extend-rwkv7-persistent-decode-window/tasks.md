## Phase 1: Typed window contracts

- [x] [serial] W1 Add a closed, ordered admitted-window contract and pure request-window derivation. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_decode_windows] W2 Add positive tests for every admitted window and negative tests for duplicate, unordered, zero, excessive, and unlisted windows. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]

## Phase 2: Runtime integration

- [x] [serial] R1 Select the request window before device open and retain normal dispatch for unsupported windows. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_decode_windows] R2 Capture and replay one finite trace for the selected window with device-resident token and FP32 state carry. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_decode_windows] R3 Add bounded pre-replay cancellation, acknowledgement, state zeroing, and deterministic slot release. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]
- [x] [depends:tenstorrent.rwkv7_p150x2.persistent_decode_windows] R4 Record the selected window, replay count, trace bytes, fallback reason, and cleanup state in receipts. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]

## Phase 3: Correctness and evidence

- [x] [parallel] V1 Add the tokenizer-backed `Hello` prompt and pinned FP32 CPU-reference continuation. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]
- [x] [parallel] V2 Add exact token, FP32 recurrent-state, timeout, cancellation, overflow, and repeated recycling tests. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]
- [x] [serial] V3 Run Blackhole simulator smoke without performance or determinism claims. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]
- [x] [serial] V4 Run trace-only profiler validation and reject dropped markers. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]
- [x] [serial] V5 Run order-balanced one-card and two-card A/B measurements for the longer window. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]

## Phase 4: Admission and archive

- [x] [serial] A1 Admit only windows with complete simulator, parity, profiler, cleanup, and stable physical-gain evidence. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]
- [x] [serial] A2 Run formatting, package, repository-quality, Cairn, closure, and source-boundary checks. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]
- [x] [serial] A3 Sync the accepted spec, retain aggregate evidence, archive the change, and integrate it into `main`. r[tenstorrent.rwkv7_p150x2.persistent_decode_windows]
