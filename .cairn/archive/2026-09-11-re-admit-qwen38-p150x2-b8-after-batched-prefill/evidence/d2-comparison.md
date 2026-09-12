# D2: comparison with the prior rejection and every non-claim

Recorded 2026-09-11 for change `re-admit-qwen38-p150x2-b8-after-batched-prefill`.

## The two receipts, side by side

| | rejected 2026-08-26 | admitted 2026-09-11 |
| --- | ---: | ---: |
| runtime | tt-metal 0.74 (`mg0ha5ns…`) | tt-metal 0.77 chunked variant (`6ccynlvv…`) |
| package identity | `k1r12ppf…` | execution identity `z31p1j9w…` |
| aggregate decode speedup (>= 6.0) | 5.881509219624968 FAIL | 6.106294615434352 pass |
| per-user retention (>= 0.75) | 0.735188652453121 FAIL | 0.7632868269292941 pass |
| TTFT ratio (<= 9.0) | 9.079183271882933 FAIL | 5.421214324999622 pass |
| tail-latency ratio (<= 1.5) | 1.4461700431804296 pass | 1.288678052289753 pass |
| median single aggregate tokens/s | 16.88145527595542 | 16.468051354534474 |
| median B=8 aggregate tokens/s | 99.28843484621837 | 100.55877331289024 |
| correctness gates | all pass | all pass |
| receipt BLAKE3 | `e6641e7e…` | `2864c9b1…` |

The rejection receipt and its history are untouched; the new receipt is a new
artifact under a new identity in `pkgs/qwen36/evidence/2026-09-11/`.

## What decided the two failing margins

Both margins moved inside their bounds on the same measurement protocol:

- aggregate speedup 5.8815 -> 6.1063 (bound 6.0): the B=8 decode step now runs
  the fused chunked recurrence instead of the unfused path the one-instance-per-
  core guard forced on 0.74. Batch aggregate throughput rose slightly
  (99.29 -> 100.56) while single-stream throughput fell slightly
  (16.88 -> 16.47); the ratio, not the throughput, crossed the bound.
- per-user retention 0.7352 -> 0.7633 (bound 0.75): the same decode change.
- TTFT ratio 9.0792 -> 5.4212 (bound 9.0): the largest move, and it is not a
  prefill-batching claim. The served prefill is still per user, one
  `prefill_paged_slots` call per request (prefill state install 1.377 s for
  eight users versus 0.216 s for one). The ratio improved because the 0.77
  runtime plus the batched-decode path shortened the measured window
  differently for the two roots, not because eight prefills became one
  batched launch.

The change's design predicted "the expected result is another rejection". The
measurement did not agree. That is recorded rather than smoothed over, and the
three candidates for the improvement are named above so the attribution is
falsifiable: a fused B=8 decode, a 0.74 -> 0.77 runtime change, and a shorter
per-request measured window. This change does not separate them; it reports
that the frozen profile now passes.

## Non-claims, verbatim from the admitted receipt

- `this receipt admits only candidate width 8 on the recorded execution identity and does not admit any other width`
- `this receipt does not admit arbitrary continuous batching`
- `this receipt does not replace the single-stream eager-versus-traced gate`

The first non-claim is width-derived and wording-corrected in this change: the
builder previously emitted `this receipt does not admit B=8 or B=16`, which was
accurate for a rejected B=8 measurement and for the admitted B=4 receipt, but
becomes false on an admitted B=8 receipt.

Additional non-claims:

- **No B=16 or B=32 claim.** B=16 is out of scope; B=32 still fails
  `per_core_M % out_subblock_h == 0` at TP=2, a divisibility defect with a
  different owner.
- **No batched-prefill admission.** The chunked scan mechanism is proven on
  silicon at the op level by the sibling change; this change's served path
  prefills per user.
- **No timing claim beyond the frozen profile.** The medians above are the
  receipt's own numbers under its own protocol; no extrapolation to other
  widths, corpus, or context length is claimed.
- **No MTP interaction claim.** MTP stays gated off and is untouched here.
- **No claim that the prefill mechanism caused the pass.** See the three
  candidate causes above.
