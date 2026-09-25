## Phase 1: Implementation

- [x] [serial] I1 Read stop ids from the generation config and tokenizer, and shape every response through `finish_generation`. r[tenstorrent.qwen38_p150x2.end_of_sequence]
- [x] [serial] I2 Stop the batch loop when every user has stopped, and stop the serialized loops at the first stop id. Keep admission measurements at the fixed length. r[tenstorrent.qwen38_p150x2.end_of_sequence.valid]

## Phase 2: Verification

- [x] [parallel] V1 Add device-free probes for truncation, finish reasons, and stop-id loading. r[tenstorrent.qwen38_p150x2.end_of_sequence.invalid]
- [x] [serial] V2 Compare chat requests on the packaged service before and after, alone and in a batch of four. r[tenstorrent.qwen38_p150x2.end_of_sequence.valid]  (evidence/chat-stop.md)

## Verification Coverage

- `Scenario: A chat answer ends before the bound` -> V2
- `Scenario: No stop id or an admission measurement` -> V1
