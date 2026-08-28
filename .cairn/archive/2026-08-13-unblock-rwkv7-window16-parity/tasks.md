# Tasks

- [x] [serial] D1 Record the corrected timeout diagnosis and package-binding defect from retained evidence. r[tenstorrent.rwkv7_p150x2.window16_fused_only_diagnostic]
- [x] [serial] D2 Replace generic device pre-work with a pure host reference and add positive and negative tests. r[tenstorrent.rwkv7_p150x2.window16_fused_only_diagnostic]
- [x] [serial] D3 Bind stage markers and receipts to the diagnostic runtime identity and precision setting. r[tenstorrent.rwkv7_p150x2.window16_fused_only_diagnostic]
- [x] [serial] V1 Run focused tests, Clippy, package isolation, source-boundary, closure, formatting, quality, and Cairn gates. r[tenstorrent.rwkv7_p150x2.window16_fused_only_diagnostic]
- [x] [serial] P1 Run one 30-minute fused-only physical diagnostic on device `0` and retain its exact verdict. The fused kernels compiled and were enqueued, but synchronized readback timed out. The authorized harness retry failed during Ethernet-dispatch firmware initialization before device open. Neither attempt wrote a correctness receipt. r[tenstorrent.rwkv7_p150x2.window16_fused_only_diagnostic]
- [x] [serial] E1 If exact parity passes, create a separate immutable window-16 evaluation and repeat all gates; otherwise stop and retain `[2, 4, 8]`. Exact physical parity was not evaluated because neither bounded attempt wrote a receipt. No later gate or candidate started. r[tenstorrent.rwkv7_p150x2.window16_reevaluation_after_diagnostic]
- [x] [serial] A1 Validate, sync, archive, commit, push, and integrate the bounded result. r[tenstorrent.rwkv7_p150x2.window16_fused_only_diagnostic]
