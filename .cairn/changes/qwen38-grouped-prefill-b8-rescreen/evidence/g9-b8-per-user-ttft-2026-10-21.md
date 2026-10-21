# g9: B=8 re-measured under the v2 per-user TTFT statistic

Recorded 2026-10-21. The g8 admission stands; this record moves its receipt and profile to
the statistic B=16 uses, so both widths report the same TTFT definition.

## What ran

The frozen B=8 protocol (`single,batch,batch,single`, eight prompts, one warmup plus three
measured runs, 64 tokens) was re-run on package
`psdg4iksyhhmxhpr7sqcd5f5gj8w6z3q-qwen38-0.1.0`, execution identity
`hmn4z5n63vlz7prlva2kispp8d6wlll5-qwen36-execution-identity-0.1.0`, with the `step` GDN
route and the v2 B=8 profile. The service reports each response's own time to first token
(observed after that user's masked prefill replay); the group's last-user value is kept as
`prefill_group_first_token_seconds`.

## Verdict

| Metric | v2 measured | Bound | v1 group receipt |
| --- | ---: | ---: | ---: |
| aggregate decode speedup | 7.4083 | at least 6.0 | 7.4077 |
| per-user decode retention | 0.9260 | at least 0.75 | 0.9260 |
| TTFT ratio (per-user median) | 4.5214 | at most 9.0 | 7.6697 (group) |
| TTFT group-max ratio | 7.7769 | — | 7.6697 |
| tail-latency ratio | 1.0797 | at most 1.5 | 1.0782 |

Exact token parity, exact text parity, active prefix, clean reset, and distinct state
slices all pass. Medians: single 23.57 aggregate tokens/s and 0.212 s TTFT; batch 174.61
aggregate and 21.83 per user, 0.957 s per-user TTFT, 1.713 s group first token, tail 0.049 s.

Receipt:
`pkgs/qwen36/evidence/2026-10-21/qwen38-27b-batch-admission-b8-receipt.json`, BLAKE3
`44960a77df0c7666624f4b0ae2e1f155f8b8861ad3de95b33886bcd358f398ab`. The binding remains
`physically_admitted = true` with that digest; the v1 group-readiness receipt remains at
`…-b8-receipt-v1-group-ttft.json` (BLAKE3 `f541b692…`).

## Non-claims

- This moves the receipt and profile statistic; it does not change any bound, corpus, order,
  run count, or generated-token count.
- Width 8 remains admitted on the recorded identity only; batched production serving stays
  off.
- The group-max ratio is a diagnostic, not a gate. Prefill is still serialized per user.
