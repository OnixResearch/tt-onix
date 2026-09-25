## Phase 1: Implementation

- [x] [serial] I1 Combine the shard winners on the device and feed them into the next token input. r[tenstorrent.qwen38_p150x2.greedy_stream]
- [x] [serial] I2 Add `decode_greedy_stream` and use it for greedy batches in the worker. r[tenstorrent.qwen38_p150x2.greedy_stream.valid]
- [x] [serial] I3 Read greedy winners from one replica, and refuse mismatched tails. Remove the obsolete two-read probe. r[tenstorrent.qwen38_p150x2.greedy_stream.invalid]

## Phase 2: Verification

- [x] [parallel] V1 Compare the device combine with the host combine and torch argmax, including ties, on the mesh. r[tenstorrent.qwen38_p150x2.greedy_stream]  (evidence/stream.md)
- [x] [serial] V2 Compare streamed, step-by-step, and previous-package tokens and step times at widths 1, 8, and 16. r[tenstorrent.qwen38_p150x2.greedy_stream.valid]  (evidence/stream.md)
- [x] [parallel] V3 Update the device-free tail fixture for both mismatched readers. r[tenstorrent.qwen38_p150x2.greedy_stream.invalid]
- [x] [serial] V4 Run the packaged service before and after under concurrent greedy load. r[tenstorrent.qwen38_p150x2.greedy_stream.valid]  (evidence/stream.md)

## Verification Coverage

- `Scenario: A greedy batch streams its steps` -> V2, V4
- `Scenario: A reader meets the other mode's tail` -> V3
