# The admission protocol widened: an interleaved root order (2026-10-21)

Recorded for task V9 of `optimize-gdn-decode-kernel-grid-and-prep`. The four-root
protocol could not decide the width-8 bound: `protocol-measurement-power.md` records
the same kernel admitted twice and rejected twice, with retention spanning 0.070
against a bound 0.013 away and the TTFT ratio moving between 6.00 and 20.41. The
spread was larger than the margin being decided, so a width-8 verdict was not
reproducible, and the batched-destination candidate stayed unadmitted even though it
is measurably faster on paired screens.

## What the protocol is now

Two changes, neither of them a bound:

1. **The roots interleave.** The order is
   `single, batch, single, batch, single, batch, single, batch`: the two routes
   alternate, so a drift in the host's load reaches both halves of a ratio at the
   same point in time, instead of comparing two sessions minutes apart.
2. **There are four roots per route** instead of two, so the median over the roots
   settles.

The third fix named in the measurement-power evidence was already in place: the
receipt builder gates on the **per-user** time to first token (the median over the
users' own first-token times), and reports the group maximum as a diagnostic
(`ttft_group_max_ratio`). That landed with the B=16 admission
(`1737603`), and the structural reason it matters — the batch TTFT is the slowest of
eight serialized per-user prefills — is exactly what the per-user statistic removes.

## What changed in the repository

- `pkgs/qwen36/service/qwen36_service/batch_admission_core.py` accepts two root
  orders: the historical `single,batch,batch,single` and the interleaved one. The
  historical order stays valid, so the admitted receipts still validate.
- `scripts/qwen36-batch-receipt.rs` takes the number of roots from the order rather
  than from a constant, and accepts either order. Its self-test still covers the
  historical order and the reduced-root diagnostic.
- `pkgs/qwen36/batch-admission-b8-policy-v3.ncl`,
  `…-contracts-v3.ncl`, `…-policy-v3-validated.ncl`, `…-binding-v3.ncl`, and
  `…-profile-v3.ncl` are the v3 protocol: profile
  `qwen38-p150x2-batched-throughput-b8-v3`, the interleaved order, and **the same
  bounds** — speedup ≥ 6.0, retention ≥ 0.75, TTFT ≤ 9.0, tail ≤ 1.5, the same
  eight-prompt corpus, three measured runs, one warmup.
- The package exports the v3 profile and binds the v3 files into the execution
  identity.
- A device-free probe covers the accepted orders and the orders that must be
  refused.

The v1 and v2 policies, their contracts, and their receipts remain the historical
measurements; the v3 binding starts unmeasured and unadmitted, and only a receipt
from this protocol can fill it.

## The validation

The interleaved protocol was run end to end at width 8 on the admitted package
(`frf73q26…`, execution identity `l4arf461…`), eight fresh roots in the
interleaved order, each root one warmup round and three measured rounds of the
eight-prompt corpus, the single roots sequential and the batch roots released
through the collection window. The receipt admits the profile, and the spread across
the four pairs is 0.07 % on the two decode-throughput metrics. The receipt and the
per-root numbers are recorded in `protocol-v3-validation.md`.

## Non-claims

- Nothing about the kernel changed. This is the measurement instrument, not the
  thing being measured.
- The bounds did not move. A protocol that could not see a two percent difference
  would have been useless; loosening the bound to fit the noise would have made the
  admission meaningless.
- The historical receipts keep their own protocol and their own numbers. A receipt
  from the interleaved protocol covers the interleaved protocol.
- The single-stream gate, the decode route, and the serving configuration are
  untouched.
