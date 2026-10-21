# V6 step-route rescreen: B=16 passes everything except a structurally blocked TTFT ratio

Recorded 2026-10-21 on physical PCIe devices 0 and 1 (two P150a, firmware 19.11.0.0,
driver 2.10.0). Package `lxmlgkzvxfjijbmc6pc0yf22088v0002-qwen38-0.1.0`, execution
identity `hmn4z5n63vlz7prlva2kispp8d6wlll5-qwen36-execution-identity-0.1.0`, runtime
`7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`. The served GDN decode route is the
`step` kernel.

## Protocol

The frozen B=16 profile was re-run unchanged: roots in `single,batch,batch,single` order,
the sixteen-prompt corpus, one warmup plus three measured runs per prompt, 64 generated
tokens, and a batch root that releases all sixteen prompts as one barrier group. The
receipt is `pkgs/qwen36/evidence/2026-10-21/qwen38-27b-batch-admission-b16-receipt.json`,
BLAKE3 `aee9bd5a2617addb63932e9cee77587a86dd64c582f2f34381ea30b8aac97ab3`; the binding
stays unadmitted with that digest.

## Verdict

| Gate | Measured | Bound | Result |
| --- | ---: | ---: | --- |
| aggregate decode speedup | 13.4026 | at least 6.0 | pass |
| per-user decode retention | 0.8377 | at least 0.75 | pass |
| TTFT ratio | 14.2501 | at most 9.0 | **fail** |
| tail-latency ratio | 1.1228 | at most 1.5 | pass |

Exact token parity, exact text parity, active prefix, clean reset, and distinct state
slices all pass. Medians: single 23.57 aggregate tokens/s and 0.237 s TTFT; batch 315.88
aggregate and 19.74 per user, 3.384 s TTFT, tail 0.056 / 0.062 s.

## Why TTFT is structurally blocked on this route

Batch TTFT is the group's first token after sixteen serialized per-user masked prefill
replays: 3.384 s / 16 = 0.212 s per replay, against a single-root TTFT of 0.237 s, which
is one replay plus the first decode step. The ratio is therefore approximately the batch
width by construction: 16 replays divided by one, not a tuning defect.

Measured alternatives do not help. A direct adapter comparison on the same corpus and
width, with eager prefill and no masked traces, gave medians of 33.4 s for the grouped
route and 29.2 s for the per-user route across four runs each (first runs include
compilation: 115.6 s and 48.3 s). Both are an order of magnitude slower than the traced
replays, so switching the prefill route cannot clear the bound.

Clearing TTFT at width 16 needs one of:

1. **A batched masked prefill replay.** Run the sixteen users' prefills as one replay
   (per-row masks, per-row slot installs, per-row KV through each page-table row) instead
   of sixteen serial replays. This is a prefill-kernel and model change; it would also
   improve width-8 TTFT.
2. **A protocol-level per-user TTFT statistic.** The bound exists to ask whether batching
   degrades a user's time to first token; the group maximum is the last user by
   construction under serialized prefill. A per-user median would report the median
   user's wait. The service cannot expose per-user first-token completion today: the
   per-user route enqueues every replay back-to-back and reads all logits once, so
   per-user readiness is not observed. Exposing it means changing the enqueue/read
   structure, which changes what the measurement times.

Neither is in this change's scope. The binding stays unadmitted.

## Non-claims

- B=16 is not admitted; the receipt records an exact rejection on this identity.
- The B=8 admission is unaffected and remains admitted on this package.
- No bound, corpus, run count, order, or token count was edited.
- The prefill comparison is a diagnostic on the same corpus, not admission evidence.
