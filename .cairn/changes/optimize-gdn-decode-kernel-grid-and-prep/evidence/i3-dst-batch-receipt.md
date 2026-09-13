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

## Two consecutive passes

| Round | speedup | bound | retention | bound | TTFT | tail | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 6.0338 | >= 6.0 | 0.7542 | >= 0.75 | 7.4895 | 1.2990 | admitted |
| 2 | 6.1401 | >= 6.0 | 0.7675 | >= 0.75 | 6.0018 | 1.2846 | admitted |

Receipts `9aefac99e6009f7b633a2207bbeb638bb887ed0345783fb28e1f98feb2224f2c` and
`edeebc8f9ad5c58129d62e027524f64543a183b7f2bb08c9e07d81e87a0c5f54`.

Against the admitted kernel's three matched runs, the candidate's *worst* result
beats the old kernel's worst on both bound metrics: speedup 6.0338 against 5.898,
retention 0.7542 against 0.7373, and the old kernel's whole band straddled both
bounds while two candidate rounds sit above them.

## Why two passes still do not move the binding

Because the old kernel's band straddled the bound, a uniform few-percent shift moves
most of the band above it but not necessarily its low tail: the old speedup floor
(5.898) times a two percent improvement is 6.02, which is above the bound by a
fraction of a percent. Two passes are consistent with a real improvement *and* with
a low tail that still dips below on a bad day, and the decision rule here is that
the admission follows the same three-run standard that reversed it.

A third matched run was taken, and it **failed**: speedup 5.5761, retention 0.6970,
TTFT ratio 20.4056, tail 1.6614. Every metric missed, and TTFT missed by more than
double, which is not the shape of a candidate regression -- it is the shape of a
session whose batch roots were re-issued.

That is what happened. The third session's logs carry eight `batch-failed` lines
against zero in both passing rounds, seven of them inside one batch root. The frozen
harness retries a group up to five times, and each failed attempt still spends
prefill, so the surviving measurements are not a clean reading of anything.

## Decision: the binding stays `false`, and the third round does not count either way

- Two clean passes are not the three-run standard that reversed the earlier
  admission, so the binding does not move.
- The failing round cannot be counted against the candidate either: it is a
  contaminated session, and its TTFT ratio is a signature of re-issued groups rather
  than of a slower kernel.
- What the failure *does* establish is that the group-rejection mode is frequent
  enough to poison a thirty-minute measurement session, which no earlier record
  captured. It appeared without the candidate as well (ten `batch-failed` lines in a
  reference screen, and a reference session that failed entirely during the screen
  rounds), so it is pre-existing rather than introduced here, but its rate is now on
  the critical path for every measurement and it deserves its own diagnosis.

The next measurement step is a clean third round rerun, and if clean sessions keep
rejecting groups, the rejection mode is the thing to fix first -- not the kernel.

## What the failure does not touch

The two clean rounds' parity and timing results stand. The candidate's parity was
exact in every run, including the contaminated one.

## What is established either way

- Exact parity at width 8 for the batched destination registers, in both screens and
  in this receipt's roots (token ids, text, distinct state slices, clean reset).
- The width-8 step time falls by a few percent against the admitted kernel.
- The margin that separated B=8 from admission was about 1.5 ms of a 79 ms step, and
  batching the destination registers covers it on the screen and on one receipt.
- The served route is unchanged: this is a variant runtime package that no service
  selects, and the shared runtime still carries neither the chunked nor the batched
  kernel shape, which its package build asserts.
