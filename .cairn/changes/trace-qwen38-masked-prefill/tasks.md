## Phase 1: Measurement

- [x] [serial] M1 Measure device replay time of a 128-token prefill trace against eager prefill. r[tenstorrent.qwen38_p150x2.masked_prefill_traces]  (evidence/traced-prefill.md)

## Phase 2: Implementation

- [x] [serial] I1 Pass the prompt length as device data to the GDN FIR conv and seq scan. r[tenstorrent.qwen38_p150x2.masked_prefill_traces]
- [x] [serial] I2 Capture and replay one masked prefill trace per bucket, and install each user's state before the next replay. r[tenstorrent.qwen38_p150x2.masked_prefill_traces.valid]
- [x] [serial] I3 Capture after every reachable program at startup, and release stale traces. r[tenstorrent.qwen38_p150x2.masked_prefill_traces.invalid]
- [x] [serial] I4 Prefill with full-width page-table rows. r[tenstorrent.qwen38_p150x2.masked_prefill_traces.valid]
- [x] [serial] I5 Shard width 11 as one grid row, and trim unshardable batch widths. r[tenstorrent.qwen38_p150x2.masked_prefill_traces.invalid]
- [x] [serial] I6 Add the NixOS option, default on. r[tenstorrent.qwen38_p150x2.masked_prefill_traces]
- [x] [serial] I7 Reset, compute logits, and install state and logits inside the trace from a staged slot index, and add the 512 and 1,024 buckets. r[tenstorrent.qwen38_p150x2.masked_prefill_traces.valid]

## Phase 3: Verification

- [x] [serial] V1 Compare traced and eager logits and GDN state at the model level for both buckets, including a full bucket. r[tenstorrent.qwen38_p150x2.masked_prefill_traces.valid]  (evidence/traced-prefill.md)
- [x] [serial] V2 Run a 14-batch adapter sequence traced against an eager reference from a separate process. r[tenstorrent.qwen38_p150x2.masked_prefill_traces.valid]  (evidence/traced-prefill.md)
- [x] [parallel] V3 Add device-free probes for widths, batch trimming, the gate, staleness, and the masks. r[tenstorrent.qwen38_p150x2.masked_prefill_traces.invalid]
- [x] [serial] V4 Run the packaged service with traced prefill off and on at widths 1, 4, 8, 11, and 16. r[tenstorrent.qwen38_p150x2.masked_prefill_traces.valid]  (evidence/traced-prefill.md)
- [x] [serial] V5 Check the install kernel against torch, eagerly and traced, then rerun the adapter sequence and the service with the in-trace install. r[tenstorrent.qwen38_p150x2.masked_prefill_traces.valid]  (evidence/traced-prefill.md)
- [x] [serial] V6 Serve mixed batches and prompts up to 2,047 tokens traced (eager users first, then continuation replays with a staged keep-state predicate), and check them against the eager reference and in the service. r[tenstorrent.qwen38_p150x2.masked_prefill_traces.valid]  (evidence/traced-prefill.md)

## Verification Coverage

- `Scenario: Traced prefill serves the batch exactly` -> V1, V2, V4
- `Scenario: A post-capture compile or an unshardable width` -> V2, V3
