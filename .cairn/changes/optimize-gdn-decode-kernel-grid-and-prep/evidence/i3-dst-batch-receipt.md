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

## The fourth round is clean, and it fails

Round 4 fixed the contamination at its source. The shell harness released eight
separate `curl` processes, which under load arrive spread out; the service's
collection window is `BATCH_COLLECTION_SECONDS = 0.01`, so it dispatched partial
groups of width 3, 5, 6 and 7, and those widths have no warmed trace. A new driver
(`scripts/qwen38-batch-group.py`) opens every connection first and releases the
requests from a barrier.

That worked as intended: **zero group rejections** against eight in round 3, and all
four roots complete.

The receipt still fails:

| Round | Driver | Group rejections | speedup | retention | TTFT ratio | tail | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | curl | 0 | 6.0338 | 0.7542 | 7.4895 | 1.2990 | admitted |
| 2 | curl | 0 | 6.1401 | 0.7675 | 6.0018 | 1.2846 | admitted |
| 3 | curl | 8 | 5.5761 | 0.6970 | 20.4056 | 1.6614 | contaminated |
| 4 | barrier | 0 | 5.8395 | 0.7299 | 13.6542 | 1.4619 | rejected |

So the candidate's width-8 admission is **two passes in four**, and the clean
failing round is not explained by group rejections. Round 4's decode metrics
(5.8395 and 0.7299) are also slightly *worse* than the admitted kernel's median,
which no uniform improvement would produce.

## What this actually establishes

The kernel change is real: exact parity throughout, and paired screens show the step
2.6 and 6.4 percent below the reference. What is *not* established is that the
four-root protocol can see it. Across four rounds the TTFT ratio moved between 6.00
and 20.41, and retention between 0.6970 and 0.7675, while every bound sits within
about two percent of where the kernel actually performs.

The protocol's own spread therefore exceeds the margin being tested, and an
admission verdict at width 8 is decided by the weather. That is also the honest
reading of the earlier history: three matched runs straddled both bounds, and the
binding was reverted for exactly that reason.

This reframes the remaining work. More kernel work at width 8 cannot be evaluated
here until the measurement can resolve a two percent difference, so the next step is
not another candidate:

1. **Widen the protocol.** More roots, interleaved in the order
   `single,batch,single,batch,...` so the single and batch halves share the same
drift, and enough roots that the median is stable at the bound.
2. **Or reduce the dominant variance.** The TTFT ratio swings by more than a factor
   of two, and the batch TTFT is the slowest of eight serialized per-user prefills,
   which is where the variance concentrates.

Only when the protocol resolves two percent is a width-8 admission verdict
meaningful.

## Decision: the binding stays `false`

Two passes in four, with the clean failing round failing every metric, is not an
admission. The admitted kernel stays the served route, and the batch-destination
candidate stays available as a variant runtime for whoever widens the protocol.

The harness improvement stands on its own merit: it removed the group-rejection
contamination from measurement entirely, and it is now the only way this measurement
has produced four complete roots with no rejected group.

## What is established either way

- Exact parity at width 8 for the batched destination registers, in both screens and
  in this receipt's roots (token ids, text, distinct state slices, clean reset).
- The width-8 step time falls by a few percent against the admitted kernel.
- The margin that separated B=8 from admission was about 1.5 ms of a 79 ms step, and
  batching the destination registers covers it on the screen and on one receipt.
- The served route is unchanged: this is a variant runtime package that no service
  selects, and the shared runtime still carries neither the chunked nor the batched
  kernel shape, which its package build asserts.
