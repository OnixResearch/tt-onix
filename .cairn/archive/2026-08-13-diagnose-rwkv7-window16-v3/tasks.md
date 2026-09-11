## Phase 1: Diagnostic contract

- [x] [serial] D1 Add the bounded full-model boundary contract and positive and negative tests. r[tenstorrent.rwkv7_p150x2.window16_full_model_diagnosis]
- [x] [serial] D2 Add a normal-dispatch diagnostic runtime and aggregate receipt path. r[tenstorrent.rwkv7_p150x2.window16_full_model_diagnosis]
- [x] [serial] D3 Pass focused package, source, closure, quality, and Cairn gates. r[tenstorrent.rwkv7_p150x2.window16_full_model_diagnosis]

## Phase 2: Bounded diagnosis

- [x] [serial] P1 Run at most one simulator smoke if device execution changed. Not required because the diagnostic reused package-matched TT-NN operations. r[tenstorrent.rwkv7_p150x2.window16_full_model_diagnosis]
- [x] [serial] P2 Run one bounded physical normal-dispatch diagnosis through generated-token index `10`. r[tenstorrent.rwkv7_p150x2.window16_full_model_diagnosis]
- [x] [serial] P3 Prove the first divergent boundary and retain aggregate-only evidence. r[tenstorrent.rwkv7_p150x2.window16_full_model_diagnosis]

## Phase 3: Correction and candidate decision

- [x] [serial] C1 Apply the smallest proven correction with positive and negative tests. The build-isolated HiFi4 control was sufficient for the bounded correction probe. r[tenstorrent.rwkv7_p150x2.window16_full_model_diagnosis]
- [x] [serial] C2 Require exact sixteen-token CPU and FP32 recurrent-state parity in one bounded probe. Tokens passed; the FP32 recurrent-matrix identity failed. r[tenstorrent.rwkv7_p150x2.window16_full_model_diagnosis]
- [x] [serial] C3 Create immutable candidate v3 only after exact parity passes. No candidate was created because recurrent-matrix parity failed. r[tenstorrent.rwkv7_p150x2.window16_full_model_diagnosis]
- [x] [serial] C4 Sync, archive, run final gates, commit, push, and integrate. r[tenstorrent.rwkv7_p150x2.window16_full_model_diagnosis]
