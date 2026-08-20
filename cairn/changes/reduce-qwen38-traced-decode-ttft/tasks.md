# Tasks: Reduce Qwen3.8-27B traced-decode TTFT

## Phase 1: Baseline and diagnosis

- [x] [serial] B1 Retain the rejected physical receipt and current device-free baseline. r[tenstorrent.qwen38_p150x2.traced_decode_ttft]
- [x] [serial] B2 Record TTFT boundaries, candidate-only differences, hypotheses, false completions, and search budgets. r[tenstorrent.qwen38_p150x2.traced_decode_ttft]
- [x] [serial] B3 Add phase timing and width evidence without changing total TTFT boundaries. r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence]

## Phase 2: Bounded prefill table

- [x] [depends:tenstorrent.qwen38_p150x2.traced_decode_ttft] C1 Add pure request block-count validation. r[tenstorrent.qwen38_p150x2.traced_decode_ttft]
- [x] [depends:tenstorrent.qwen38_p150x2.traced_decode_ttft] C2 Slice a request-bounded prefill table from the monotonic host table. r[tenstorrent.qwen38_p150x2.traced_decode_ttft]
- [x] [depends:tenstorrent.qwen38_p150x2.traced_decode_ttft] C3 Keep the maximum-width trace-bound decode table and its identity unchanged. r[tenstorrent.qwen38_p150x2.traced_decode_ttft]

## Phase 3: Receipt and tests

- [x] [parallel] V1 Add positive tests for exact, rounded, maximum, and bounded-prefix cases. r[tenstorrent.qwen38_p150x2.traced_decode_ttft] r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence]
- [x] [parallel] V2 Add negative tests for invalid counts, capacity overflow, missing phases, non-finite timings, and inconsistent widths. r[tenstorrent.qwen38_p150x2.traced_decode_ttft] r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence]
- [x] [serial] V3 Extend the deterministic receipt with phase medians and page-table widths. r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence]
- [x] [serial] V4 Run Ruff, compilation, package, authority-negative, NixOS service, and receipt self-tests. r[tenstorrent.qwen38_p150x2.traced_decode_ttft]

## Phase 4: Physical decision

- [ ] [depends:tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence] H1 Run matched `eager, candidate, candidate, eager` samples on physical devices `0` and `1`. r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence]
- [ ] [depends:tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence] H2 Admit or reject the candidate from exact parity, clean reset, decode, TTFT, tail, phase, and identity gates. r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence]
- [ ] [serial] D1 Record the surviving mechanism, rejected hypotheses, receipt identity, and non-claims. r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence]
- [ ] [serial] G1 Run Cairn validation, lifecycle gates, sync, archive, and drain evidence checks. r[tenstorrent.qwen38_p150x2.traced_decode_ttft]

## Verification Coverage

- `Scenario: Request uses a bounded prefill table` -> C1, C2, V1
- `Scenario: Request exceeds trace capacity` -> C1, V2
- `Scenario: Prefill width changes trace binding` -> C3, V2
- `Scenario: Phase evidence is complete` -> B3, V1, V3
- `Scenario: Phase evidence is incomplete` -> V2, V3
- `Scenario: Physical candidate passes` -> H1, H2
