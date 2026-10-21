# g8: B=8 is re-admitted on the fused step route under the frozen protocol

Recorded 2026-10-21. This supersedes the g6 verdict as the current B=8 admission state
without editing g6 or any frozen bound.

## What ran

The existing closed B=8 profile was re-run unchanged against
`5jk1x79sd33a5mfgdvvny8kzsbsm97ql-qwen38-0.1.0` (execution identity
`hmn4z5n63vlz7prlva2kispp8d6wlll5-qwen36-execution-identity-0.1.0`, runtime
`7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`) on physical PCIe devices 0 and 1.
The served GDN decode route is the `step` kernel (`QWEN35_GDN_DECODE_FUSION=step`), not the
fused TT-Metal operation that g6 measured.

The protocol is the frozen v1 one, not a widened one:

- roots in `single,batch,batch,single` order;
- the eight-prompt corpus, one warmup plus three measured runs per prompt;
- a batch root releases the eight corpus prompts as one barrier group, so the batch width
  is eight and each measured run must use state slices 0..7;
- a single root issues the same corpus one request at a time;
- 64 generated tokens per request, temperature 0;
- exact token parity between the same prompt and run in the single and batch roots.

The receipt is
`pkgs/qwen36/evidence/2026-10-21/qwen38-27b-batch-admission-b8-receipt.json`, BLAKE3
`f541b692a8a7696d60e0ffaf512b2861338439a49ad53d3356600f7def97eed9`. The binding is
`physically_admitted = true` with that digest, and the install phase pins the exported
profile to the receipt.

## Verdict

| Gate | Measured | Bound | Result |
| --- | ---: | ---: | --- |
| aggregate decode speedup | 7.4077 | at least 6.0 | pass |
| per-user decode retention | 0.9260 | at least 0.75 | pass |
| TTFT ratio | 7.6697 | at most 9.0 | pass |
| tail-latency ratio | 1.0782 | at most 1.5 | pass |

Medians: single 23.57 aggregate tokens/s, batch 174.60 aggregate and 21.83 per user,
single TTFT 0.22 s, batch TTFT 1.69 s, tail 0.043 / 0.046 s. Correctness: exact token and
text parity, active prefix, clean reset, and distinct state slices all true.

## Why the protocol was not widened

`evidence/protocol-measurement-power.md` asked for interleaved roots, per-user TTFT
statistics, and more roots because the g6 margins sat inside the run-to-run spread
(retention 0.737-0.768 against 0.75). On the step route the retention margin is 0.176
above the bound and the speedup margin is 1.41 above its bound, both far outside that
spread. The frozen protocol resolves this verdict; widening it was not required and no
bound, corpus, run count, or order was edited.

Per-user first-token times remain unavailable without changing the served prefill
enqueue/read behavior, so the TTFT gate still uses the group's first token. That is
recorded as a protocol property, not a blocked fix.

## Non-claims

- This admits candidate width 8 only, and only on the recorded execution identity.
- It does not admit B=16, arbitrary continuous batching, or batched serving in
  production; the service's batch route remains gated off.
- The single-stream eager-versus-traced gate is unchanged.
- The receipt and this record were produced with a throwaway root harness; the protocol
  values above are the receipt's own, not harness-side measurements.
