# The interleaved protocol validated: a two percent difference is now visible (2026-10-22)

Recorded for task V9 of `optimize-gdn-decode-kernel-grid-and-prep`. The interleaved
root order was run end to end at width 8 on the admitted package, and the receipt
admits the profile. The point of the measurement is the *spread*: the four-root
protocol could not decide a two percent difference, and this one can, in the metrics
where the kernel candidates move.

## The run

Eight fresh roots in the predeclared order `single, batch, single, batch, single,
batch, single, batch`, each root one warmup round and three measured rounds of the
eight-prompt corpus, the single roots sequential and the batch roots released through
the collection window. Every root measured 24 responses: 24 samples per root, 192 in
all, the single roots at width 1 and the batch roots at width 8.

- Package `/nix/store/frf73q2612gczcj1qr43zz48i3frj2q4-qwen38-0.1.0`, execution
  identity `/nix/store/l4arf461kxbbns0nnlg2vd98anx943pc-qwen36-execution-identity-0.1.0`,
  runtime `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`, physical
  devices 0 and 1.
- Receipt `pkgs/qwen36/evidence/2026-10-22/qwen38-batch-admission-b8-v3-receipt.json`,
  BLAKE3 `af2365ffb09002bb5674e175374efda1a81ea1a665c1f199b16a069344ac785e`;
  the 192 roots ship beside it, gzipped.
- The v3 binding now names that receipt, so the package carries the admission.

## The receipt

| Gate | Bound | Measured | Pass |
| --- | ---: | ---: | --- |
| aggregate decode speedup | ≥ 6.0 | 7.4083 | yes |
| per-user decode retention | ≥ 0.75 | 0.9260 | yes |
| time to first token ratio | ≤ 9.0 | 4.7778 | yes |
| tail latency ratio | ≤ 1.5 | 1.1353 | yes |
| exact token parity | required | exact | yes |
| exact text parity | required | exact | yes |
| clean reset, active prefix, distinct state slices | required | yes | yes |

The group-maximum time-to-first-token ratio, which the protocol reports as a
diagnostic only, was 7.6404.

## The spread, which is the point

| Metric | Four pairs, interleaved | Historical four-root protocol |
| --- | --- | --- |
| aggregate decode speedup | 7.4082 – 7.4133, **0.07 %** | not reported per pair |
| per-user decode retention | 0.9260 – 0.9267, **0.07 %** | 0.6970 – 0.7675, a 0.070 band |
| time to first token ratio (per user) | 4.4902 – 4.7658, 5.99 % | 6.00 – 20.41, a factor of 3.4 |
| time to first token ratio (group max) | 7.5073 – 7.8814, 4.77 % | — |
| tail latency ratio | 0.9922 – 1.2035, 18.51 % | — |

The two decode-throughput metrics — the ones a kernel candidate moves — now hold to
**0.07 %** across the four pairs, against a bound that sits 0.013 away on retention in
the old protocol. A two percent difference in throughput is now thirty times the
noise. The time-to-first-token and tail ratios are still noisy, and their bounds are
wide (9.0 and 1.5) so they are not what decides; a candidate that moved them by a few
percent could not be read, and that should be said plainly rather than measured
repeatedly.

The per-pair ratios, with the roots they came from:

| Pair | Speedup | Retention | TTFT ratio | Group-max TTFT ratio | Tail ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-0 / batch-1 | 7.4085 | 0.9261 | 4.4902 | 7.5073 | 0.9922 |
| single-2 / batch-3 | 7.4086 | 0.9261 | 4.5835 | 7.8772 | 1.2035 |
| single-4 / batch-5 | 7.4133 | 0.9267 | 4.7658 | 7.8814 | 1.0819 |
| single-6 / batch-7 | 7.4082 | 0.9260 | 4.6216 | 7.8145 | 1.2002 |

## What this does and does not change

- The width-8 batch profile is admitted on the interleaved protocol, and the receipt
  is bound to the package and identity that served it.
- The protocol can now resolve a two percent difference in decode throughput, which
  is the difference the batched-destination candidate claims and the four-root
  protocol could not see. It is still unadmitted: its own receipt has to be earned on
  this protocol.
- The kernel is untouched. This is the instrument, measured against itself.

## Non-claims

- The spread is over four pairs on one host, one workload, and one day. It is a
  measurement of this protocol's noise, not a general statement about the host.
- The roots were measured across a session boundary: the sixth and seventh roots ran
  after a gap, because the campaign job hit its time limit. Every *pair* stayed
  adjacent in time, which is what the interleaving is for, and the gap fell between
  pairs.
- The admission covers the eight-prompt corpus at width 8 on one package. It says
  nothing about other widths, other corpora, or the single-stream route.

## Correction, 2026-10-22: which identity and package the receipt names

Added by `fix-qwen38-receipt-bootstrap`. This correction does not rewrite the
run record above; it fixes what the sentence "the receipt is bound to the
package and identity that served it" may be read to mean.

- The receipt names execution identity
  `/nix/store/l4arf461kxbbns0nnlg2vd98anx943pc-qwen36-execution-identity-0.1.0`,
  and no build in this repository reproduces that identity. It is not the
  identity of a package this tree can rebuild.
- All 192 saved roots report that same identity, with
  `profile_id=qwen38-p150x2-batched-throughput-b8-v3`. The served package
  `/nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0`, however, exports
  the generic B=8 profile as `qwen38-p150x2-batched-throughput-b8-v2`; its
  identity is `l4arf461…` only in the sense that its wrapper passes that
  identity.
- The campaign package named above,
  `/nix/store/frf73q2612gczcj1qr43zz48i3frj2q4-qwen38-0.1.0`, embeds identity
  `4115hgb2ri3168c9d0g4r4pjj8m8m06n-qwen36-execution-identity-0.1.0`, not
  `l4arf461…`.

So "bound to the package and identity that served it" names the identity the
served processes reported; it does not establish that the campaign package
embedded that identity, nor that any servable package carries this receipt.
The receipt still records its measurements, but its admission cannot be
carried by a build until it is re-earned on a buildable identity. The
procedure is
`.cairn/changes/fix-qwen38-receipt-bootstrap/evidence/re-earn-procedure-2026-10-22.md`,
and the investigation is
`.cairn/changes/fix-qwen38-receipt-bootstrap/evidence/identity-mismatch-2026-10-22.md`.
