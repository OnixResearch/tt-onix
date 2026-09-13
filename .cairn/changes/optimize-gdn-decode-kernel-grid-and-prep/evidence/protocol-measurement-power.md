# The four-root protocol cannot decide the width-8 bound

Recorded 2026-09-12. This is a measurement-power finding, not a kernel result, and it
explains a decision that has now gone both ways twice.

## The observation

Four matched rounds of the frozen protocol, three on the same candidate kernel:

| Round | speedup | retention | TTFT ratio | tail | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 6.0338 | 0.7542 | 7.4895 | 1.2990 | admitted |
| 2 | 6.1401 | 0.7675 | 6.0018 | 1.2846 | admitted |
| 3 | 5.5761 | 0.6970 | 20.4056 | 1.6614 | contaminated (8 rejected groups) |
| 4 | 5.8395 | 0.7299 | 13.6542 | 1.4619 | rejected, zero rejected groups |

Two passes, one clean rejection, one contaminated run.

Round 4 is the informative one: it has **no rejected groups**, so nothing external
explains it, and it fails every metric. It is also the round where the decoder was
*not* re-issued, which rules out the obvious alternative explanation that the
previous failure was purely an artifact of retries.

## What the spread says

- Retention spans 0.6970 to 0.7675 across four rounds: a 0.070 band against a bound
  that sits 0.013 above where the kernel typically lands.
- TTFT ratio spans 6.00 to 20.41: more than a factor of two.
- The admitted kernel's own three-run history (speedup 5.898-6.106, retention
  0.7373-0.7580) straddled both bounds in the same way.

The protocol's run-to-run spread is therefore larger than the margin it is deciding,
which is why the same kernel has been "admitted" and "rejected" on different days,
and why the batched-destination candidate -- which is measurably faster on paired
screens -- can still fail a receipt.

## Why the TTFT term is the noisiest

The TTFT ratio is the batch root's time-to-first-token over the single root's. The
batch TTFT is set by the slowest of eight serialized per-user prefills, and the
single TTFT by one. Taking the maximum of eight samples of a variable quantity and
then dividing by a single sample of the same quantity amplifies every fluctuation
rather than averaging it out. That is the structural reason the ratio moved by 3.4x
while the decode metrics moved by 8 percent.

## What would fix it

Ordered by expected value per hour of device time:

1. **Interleave the roots.** `single,batch,single,batch,...` instead of
   `single,batch,batch,single`, so both halves of the ratio share the same drift and
   the ratio stops being a comparison of two sessions minutes apart.
2. **Reduce the maximum-of-eight effect** in the batch TTFT: record the per-user
   first-token times and compare distributions (median, or a trimmed mean) rather
   than the group's first token, if the service can expose them.
3. **More roots**, so the median of the roots is stable: the current cost is about
   eight minutes per root, and the decision needs a spread well under two percent.

Until one of these lands, a width-8 admission verdict is not reproducible, and
spending more device time on the kernel at that width cannot be evaluated. This is
why the batched-destination candidate stays unadmitted rather than rejected: its
paired-screen improvement is real, and the receipt simply cannot see it.

## What this does not change

- The candidate's parity result is exact in every run, including the contaminated
  one.
- The group-rejection contamination is fixed: pre-connected sockets released from a
  barrier produced zero rejected groups where eight `curl` processes produced eight.
  That defect also had a serving cost -- a rejected group spends a full prefill
  before failing -- and it is worth a service-side fix, since a real cluster can
  miss the service's ten-millisecond collection window the same way.
