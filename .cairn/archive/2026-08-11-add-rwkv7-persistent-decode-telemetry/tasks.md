## Phase 1: Telemetry contract

- [x] [serial] Add a pure typed telemetry summary and validation core with positive and negative lifecycle tests. r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.valid] r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.invalid]
- [x] [serial] Measure bounded setup, replay, and cleanup intervals in the worker shell and emit schema version `1`. r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.valid] r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.terminal]
- [x] [serial] Emit fail-closed fallback telemetry without trace or cleanup claims. r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.fallback]

## Phase 2: Evidence and documentation

- [x] [serial] Add strict Rust cross-field validation with positive and negative telemetry fixtures. r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.invalid] r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.aggregate_only]
- [x] [serial] Retain one validated runtime receipt and document the stable telemetry fields and non-claims. r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.valid]

## Phase 3: Completion

- [x] [serial] Run runtime, evidence, formatting, boundary, closure, repository-quality, and Cairn gates. r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.invalid]
- [x] [serial] Sync the accepted requirement, archive the change, regenerate the artifact manifest, and integrate the verified branch. r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.valid]
