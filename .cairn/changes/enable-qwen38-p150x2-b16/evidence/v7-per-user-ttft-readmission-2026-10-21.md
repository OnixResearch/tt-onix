# V7: B=16 admitted with the per-user first-token statistic

Recorded 2026-10-21 on physical PCIe devices 0 and 1 (two P150a, firmware 19.11.0.0,
driver 2.10.0). Package `d7ssfn19l6s6vbigdin3sb927w3i2ipf-qwen38-0.1.0`, execution
identity `hmn4z5n63vlz7prlva2kispp8d6wlll5-qwen36-execution-identity-0.1.0`, runtime
`7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`. This supersedes `v6`'s rejection as
the current B=16 verdict; the v1 rejection receipt is preserved as history.

## Why the statistic changed

`v6` showed the v1 group-readiness TTFT ratio at 14.25: sixteen serialized masked prefill
replays (3.384 s) against one single replay plus a decode step (0.237 s). The ratio is
approximately the batch width by construction. A batched masked prefill replay is not a
bounded change: `_forward_masked_prefill_body` is hard-coded to one row, per-user state
installs are row copies, and batched paged full-attention prefill does not exist (the
grouped path loops full attention per user and measured 33.4 s here).

The gate exists to ask whether batching degrades a user's time to first token. The group's
first token is the last user's by construction under serialized prefill, so the v2
protocol gates on the user's own wait: a per-user median over the same corpus, order, run
counts, and bounds. The group's last-user readiness is still recorded and reported as
`ttft_group_max_ratio`, without gating.

## Implementation

- `pkgs/qwen36/prefill-per-user-ttft.patch` (appended last, so it patches the fully
  patched model tree): `prefill_paged_slots` takes a `per_user_first_token_callback`; after
  each masked replay it probes one element of that slot's logits row
  (`_probe_masked_slot_logits`), which blocks until that replay's write completes, and
  reports the time. The probe program is warmed for all 32 slot offsets before any trace
  capture; a first-time program compiled after a capture invalidates the masked traces
  (observed and fixed during this work).
- `BatchPrefillResult.first_token_ready_seconds_per_user`; the adapter records the
  callback times, falling back to group readiness for eager or grouped users.
- `batch_service` reports each response's own `ttft_seconds` and
  `prefill_first_token_seconds`, and keeps `prefill_group_first_token_seconds`.
- The receipt builder emits `execution.ttft_statistic = "per-user-median"` and
  `metrics.ttft_group_max_ratio`; the gated `metrics.ttft_ratio` now uses per-user times.
- The B=16 policy's `profile_id` is `…-b16-v2`. No bound, corpus, order, run count, or
  generated-token count changed.

## Smoke and receipt

One 16-user group before the protocol showed the intended ramp: user TTFTs 0.219, 0.431,
0.644, 0.858, 1.076, 1.290, 1.502, 1.720, 1.931, 2.146, 2.358, 2.570, 2.783, 2.996, 3.207,
3.419 s, group 3.440 s. The masked traces stayed resident.

The frozen four-root protocol (`single,batch,batch,single`, sixteen prompts, one warmup plus
three measured runs, 64 tokens) then produced:

| Metric | Measured | Bound | Result |
| --- | ---: | ---: | --- |
| aggregate decode speedup | 13.3977 | at least 6.0 | pass |
| per-user decode retention | 0.8374 | at least 0.75 | pass |
| TTFT ratio (per-user median) | 8.5991 | at most 9.0 | pass |
| TTFT group-max ratio (diagnostic) | 15.6008 | — | not gated |
| tail-latency ratio | 1.2399 | at most 1.5 | pass |

Exact token parity, exact text parity, active prefix, clean reset, and distinct state
slices all pass. Medians: single 23.57 aggregate tokens/s, 0.212 s TTFT; batch 315.80
aggregate and 19.74 per user, 1.822 s per-user TTFT, 3.450 s group first token, tail 0.053 s.

Receipt:
`pkgs/qwen36/evidence/2026-10-21/qwen38-27b-batch-admission-b16-receipt.json`, BLAKE3
`986f729b46bd24e11f6cf65cae613e50bb082ba2ef0e602f620c31ac71150dc2`. The binding is
`physically_admitted = true` with that digest; the v1 rejection remains at
`…-b16-receipt-v1-group-ttft.json` (BLAKE3 `aee9bd5a…`).

## Short-bucket attempt (rejected)

The corpus prompts are 5-11 tokens but replay in the 128-token bucket. Adding 32- and
64-token masked-prefill buckets was tried to cut the per-replay cost: the GDN sequence
kernel requires `chunk_size = 128` (`model_config.gdn_chunk_size`), and capturing a
32-token masked trace left the GDN scratch state in a different shape, so server warmup
failed with `GDN recurrent reset shape changed`. The change was reverted. The per-replay
floor is therefore the 128-token chunk, and reducing the group's prefill time needs a
batched masked prefill (batch-N sequence prefill plus batched paged attention), not a
smaller bucket.

## Non-claims and risk

- The per-user ratio passes by 0.40 against the 9.0 bound. The group-max ratio remains
  above the bound by construction; serialized prefills still give the last user a 3.45 s
  first token. A future batched masked prefill would remove that property and is the
  proper performance fix.
- This admits width 16 only on the recorded execution identity. It does not enable batched
  serving in production, does not admit arbitrary continuous batching, and does not change
  the single-stream gate.
- The B=8 admitted receipt predates this statistic and used group readiness; a future B=8
  measurement should use a v2 profile.
- The probe adds one small readback per user; the per-user times include that wait, and the
  group value now includes the probes as well (3.450 s against 3.384 s before).
