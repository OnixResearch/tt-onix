## Context

The repository already vendored a fused GDN decode op. `pkgs/tt-metal/gdn-decode-fp32.patch`
adds `ttnn.transformer.gated_delta_decode`, the pinned 0.77 runtime exports it
(five occurrences in `_ttnn.so`), and `pkgs/qwen36/gdn-decode-fusion-candidate.patch`
wires it behind `QWEN35_GDN_DECODE_FUSION=full`.

That path is already admitted and already measured:

- distributional screen: ADMITTED for sampled mode, mean logit PCC `0.996948`,
  mean top-K total variation `0.018480`, stable-position agreement `1.0`;
- fresh B=4 admission under `full`: ADMITTED, aggregate decode speedup `3.5293`,
  per-user retention `0.8823`, TTFT ratio `4.0062`;
- component speed gates: GDN ratio `0.946456` against a bound of `0.95`,
  full-step ratio `0.939473` against `0.97`, speedup `1.0644x`.

So the fused decode speed exists. What it cannot do is scale past B=4, because
its program factory opens exactly one instance per head and refuses a head
count above the core count. The same shape of guard in the chunk-prefill
factory forces serialized per-user prefills and is the recorded root cause of
the B=8 TTFT rejection.

One mechanism removes both: contiguous per-core instance chunking.

## Decisions

### Decision: One instance-assignment core serves both sites

**Choice:** Put the chunk arithmetic and its range validation in one pure
device-free core. Use it for the decode factory and the prefill factory.

**Rationale:** Both factories need the same computation and the same rejection
rules. One core keeps the two sites from drifting, and one negative test set
covers both.

### Decision: Use uniform ceiling chunks, not the ragged-front split

**Choice:** Every active core takes `ceil(instances / ncores)` instances. Only
the final chunk may be short. Chunk `i` starts at `i * chunk_length`.

**Rationale:** The repository already fans work across the grid this way for the
PREP phase in the same file, `distribute_prep`, which has no core-count guard
because `P = min(total, max_cores)` and each core takes `base` or `base + 1`
items. That is precedent for the idea but not for the arithmetic: its ragged
split gives the first `rem` cores one extra item, so a chunk start is not
`index * chunk_length`. The uniform-ceiling form is chosen here because it needs
one forward pass, needs no running offset, and makes the kernel-side instance
loop bounds a plain multiply. The prefill and decode factories must both use the
uniform form so one assignment core describes both.

### Decision: Keep the in-range mapping byte-identical

**Choice:** When `BH <= ncores`, each core keeps exactly one instance and the
program mapping must equal the pre-change mapping. Only the overflow case
changes.

**Rationale:** Every admitted decode and prefill receipt was produced under the
current mapping. An unchanged in-range mapping keeps those receipts valid and
makes the change's blast radius exactly the overflow case.

### Decision: The assignment core takes over the removed fatal's validation

**Choice:** The core rejects a zero core count, a negative or zero instance
count, a chunk length below one, and a chunk range that does not partition
`[0, BH)`.

**Rationale:** Removing a `TT_FATAL` widens the accepted input set. The guard it
provided must move somewhere, and a pure core is testable without a device.

### Decision: Require exact parity at both sites, not a PCC bound

**Choice:** Decode compares output and recurrent state against the admitted
fused reference. Prefill compares every position against the serialized
per-user reference. Both require exact parity over a fixed corpus.

**Rationale:** This repository rejected an earlier GDN fusion whose state PCC
stayed near `0.99997` over two steps but drifted enough to flip late argmaxes
across 48 layers. An approximate bound does not describe that hazard.

### Decision: Build a patched tt-metal derivation and leave the shared pin alone

**Choice:** Add a tt-metal variant carrying the change. Do not modify the shared
`tt-metal` 0.77 output.

**Rationale:** RWKV-7 and the training platform consume the shared output and
carry their own admitted receipts. A separate derivation keeps them untouched
and costs one cached rebuild.

### Decision: The mechanism admits no width and no route

**Choice:** This change produces two mechanisms and their parity evidence. It
does not move `physically_admitted` for any width and does not touch the B=8
profile or its bounds.

**Rationale:** The B=8 rejection stands until a fresh matched run passes the
unchanged bounds. Admission is a separate change with its own receipt.

## Risks / Trade-offs

- The grid fix may not explain the B=8 near miss. The admission change must then
  report a second rejection rather than a weaker bound.
- Contiguous chunking changes work distribution. A parity pass does not prove the
  new distribution is faster, so timing stays a separate measurement.
- Chunking a per-head kernel means a core loops over several heads, which adds
  software loop overhead against the added parallelism. The step-time bound
  decides whether the trade is positive.
- The B=8 fused decode case needs 192 instances against about 110 cores, so each
  core takes two instances. That is a bounded and small chunk, not a general
  work-stealing scheme.
