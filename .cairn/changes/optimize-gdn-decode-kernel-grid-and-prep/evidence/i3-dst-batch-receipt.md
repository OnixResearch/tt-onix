# The batched-destination kernel admits width 8 on one receipt

Recorded 2026-09-12. This is the candidate from
`evidence/i3-dst-batch-candidate.md` measured on the frozen four-root protocol.

## The receipt

Candidate package `/nix/store/yajnqp5cpbb65ky02nykj6m3yr0i8k3r-qwen38-0.1.0`,
variant runtime with the destination-register batching, four roots in the frozen
order `single,batch,batch,single` over eight prompts with 64 generated tokens each.

| Metric | Value | Bound | Verdict |
| --- | ---: | ---: | --- |
| aggregate decode speedup | 6.0338 | >= 6.0 | pass |
| per-user decode retention | 0.7542 | >= 0.75 | pass |
| TTFT ratio | 7.4895 | <= 9.0 | pass |
| tail latency ratio | 1.2990 | <= 1.5 | pass |

Admission verdict: **admitted, with no reasons**. Batch per-user decode 12.521
tokens/second, single 16.601. Receipt blake3
`9aefac99e6009f7b633a2207bbeb638bb887ed0345783fb28e1f98feb2224f2c`.

For comparison, the same protocol on the admitted kernel measured 5.898-6.049
speedup and 0.7373-0.7580 retention across three runs -- straddling both bounds.

## Why this receipt does not move the binding by itself

The margin is inside the measured spread. Speedup is 0.6 percent above its bound and
retention 0.6 percent above its bound, while the same protocol on the *unchanged*
kernel produced retention across a band of 0.021. A single passing run is therefore
consistent with both a real few-percent improvement and an unchanged kernel on a good
day, and the earlier B=8 decision was correctly reversed on exactly that reasoning:
three matched runs straddled the bounds and the binding went back to `false`.

The candidate's own screen supports the real-improvement reading: two paired rounds
showed the width-8 step 2.6 and 6.4 percent below the reference package with exact
parity. But a screen is not an admission measurement, and the decision rule recorded
for this work is that a passing receipt over a *single* run does not admit.

So the sequence from here is: repeat the frozen protocol for further matched runs,
and move `physically_admitted` to `true` only when the passes repeat outside the
band, exactly as the rejection was recorded.

## What is established either way

- Exact parity at width 8 for the batched destination registers, in both screens and
  in this receipt's roots (token ids, text, distinct state slices, clean reset).
- The width-8 step time falls by a few percent against the admitted kernel.
- The margin that separated B=8 from admission was about 1.5 ms of a 79 ms step, and
  batching the destination registers covers it on the screen and on one receipt.
- The served route is unchanged: this is a variant runtime package that no service
  selects, and the shared runtime still carries neither the chunked nor the batched
  kernel shape, which its package build asserts.
