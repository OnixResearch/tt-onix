# V1-V6 and D1: the fresh B=8 measurement, its gates, and the binding move

Recorded 2026-09-11 for change `re-admit-qwen38-p150x2-b8-after-batched-prefill`.

## Identities under test

| Item | Value |
| --- | --- |
| admission package | `/nix/store/mgi46qcz8mxs3sbmaq4ivyhs71y1jpac-qwen38-0.1.0` |
| runtime identity | `/nix/store/6ccynlvv46p54pmp2lb7dbs65hkyj5w9-tt-metal-0.77.0` |
| package execution identity (as measured) | `/nix/store/z31p1j9wa4af27vakffyzqp9cyypsjr1-qwen36-execution-identity-0.1.0` |
| exported profile blake3 | `679c8782ef0061a3c5ed98ee7e453d26c759b14f7598c15bbb02e62babd96388` (byte-identical to pre-change) |
| fused decode gate | `QWEN35_GDN_DECODE_FUSION=full` |
| devices | physical PCIe 0 and 1, `simulated: false` |

## V1: package build, profile export check, receipt self-tests

- `nix build .#qwen38 .#qwen38-gdn-grid` exits 0; the install phase re-exports
  the B=8 profile twice, requires byte equality between the two exports and
  against the installed artifact, and requires the exported admission state to
  match the tracked receipt (see D1).
- `./scripts/qwen36-batch-receipt.rs --self-test` exits 0 and now covers both
  candidate widths with the added negative cases from I2.

## V2: fresh Blackhole simulator smoke

Exit 0 on the packaged TTsim Blackhole `1x1` target with the variant runtime:
device opened, JIT/firmware compiled, representative TT-nn operations ran,
clean close.

**Ordering deviation, recorded.** This change's simulator smoke ran after the
physical roots rather than before them. The reason: the sibling change had run
the same smoke on the same runtime identity minutes earlier, and TTsim cannot
execute the fused GDN decode op at any width at all (`ALU_ACC_CTRL_Zero_Flag_
disabled_src` is not modeled), so the smoke can only ever prove launch-level
behavior on this path. The device-free gates and the physical parity evidence
carry the correctness weight.

## V3: matched physical roots in the frozen order

Four roots in exactly `single, batch, batch, single` order, eight-prompt corpus,
one warmup plus three measured runs per prompt, 64 generated tokens, on devices
0 and 1. All four roots produced 24 measured responses (32 requests each); the
transient HTTP 500s were absorbed by the runner's bounded group-retry loop, and
every retained response is a measured width-8 batch sample.

## V4: correctness gates

From the receipt's `correctness` block, all required gates pass:

```json
{"active_prefix": true, "clean_reset": true, "distinct_state_slices": true,
 "exact_text_parity": true, "exact_token_parity": true}
```

## V5: performance bounds, evaluated without editing a bound

Receipt: `pkgs/qwen36/evidence/2026-09-11/qwen38-27b-batch-admission-b8-receipt.json`,
BLAKE3 `2864c9b1e041af87b6bc3772bfe05b42df6f2055a3c06443ca8c48aadcdc1407`,
`admission.admitted = true`, `admission.reasons = []`.

| Bound (unchanged) | Measured | Result |
| --- | ---: | --- |
| aggregate decode speedup >= 6.0 | 6.106294615434352 | pass |
| per-user decode retention >= 0.75 | 0.7632868269292941 | pass |
| TTFT ratio <= 9.0 | 5.421214324999622 | pass |
| tail-latency ratio <= 1.5 | 1.288678052289753 | pass |

Median timings from the receipt:

| Metric | single | batch (width 8) |
| --- | ---: | ---: |
| aggregate decode tokens/s | 16.468051354534474 | 100.55877331289024 |
| per-user decode tokens/s | 16.468051354534474 | 12.569846664111282 |
| TTFT seconds | 2.365988109493628 | 12.82652863196563 |
| tail token latency seconds | 0.06650324247311801 | 0.08570126898121089 |
| prefill state install seconds | 0.21559453348163515 | 1.3768597880844027 |
| trace prepare seconds | 1.4250962694641203 | 1.1582073573954403 |
| cleanup seconds | 0.02597763598896563 | 0.023730888147838414 |

The B=8 batch path still installs prefill state once per user (1.377 s for
eight users against 0.216 s for one), so the served prefill remains per user;
the admitted mechanism is the width-8 batched decode.

## V6: production restored

`ttwkv7-owner-control restore` ran in the session trap; `qwen38-p150x2.service`
is active and `/healthz` returns `{"status": "ok", "ready": true}`.

## D1: the binding moved on a fully passing receipt

```
pkgs/qwen36/batch-admission-b8-binding.ncl
  physically_admitted = true,
  receipt_blake3 = "2864c9b1e041af87b6bc3772bfe05b42df6f2055a3c06443ca8c48aadcdc1407",
```

This is the digest of the tracked receipt file, verified by `b3sum` on the
tracked path. The install phase now binds the two together instead of asserting
the old rejection state: it requires the tracked receipt to contain
`"admitted": true`, requires the exported profile to say
`"physically_admitted": true`, and requires the exported `receipt_blake3`
to equal the tracked receipt's `b3sum`. A rejection receipt can no longer be
paired with an admitted binding, and the digest cannot drift from the artifact.

Rebuilt after the move: `qwen38` (shared runtime) =
`/nix/store/4jrx0g3rafcypgn3ndssw5s3dnblwhm7-qwen38-0.1.0` with runtime identity
`dr9bj8qw…` and execution identity
`/nix/store/3wnk77930rdz2nipxbzln47plnbkp2v6-qwen36-execution-identity-0.1.0`;
`qwen38-gdn-grid` (chunked runtime) =
`/nix/store/inmzbg5r1wk1pjivmisxwjb6gqicrpd5-qwen38-0.1.0` with runtime identity
`6ccynlvv…` and execution identity
`/nix/store/z31p1j9wa4af27vakffyzqp9cyypsjr1-qwen36-execution-identity-0.1.0`.
The admitted receipt records `z31p1j9w…`, which is still the variant package's
identity after the binding move: receipt bindings are deliberately excluded
from the execution identity (the package asserts that a receipt-only promotion
cannot change it), so the admission continues to describe the artifact that
exists now. The deployed B=8 profile reports `physically_admitted: true` with
the receipt digest, and the deployed B=4 profile is unchanged (`true`,
`489f2fbcb414ae04e536fc055857864c3255d727e4c12b79dc701a86ded3d028`).

One probe needed a correctness fix when the binding moved:
`probe_batch_admission` built its "admitted without a receipt" negative by
copying the shipped profile and setting `physically_admitted = true`, which
became a no-op once the shipped binding was admitted, so the negative stopped
being negative. It now sets `receipt_blake3 = "not-measured"` explicitly, the
same way the B=4 negative does, so it rejects the invalid state regardless of
the shipped binding.
