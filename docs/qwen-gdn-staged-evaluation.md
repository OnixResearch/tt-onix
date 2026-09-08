# Staged GDN candidate evaluation

## Candidates and boundaries

This round compares two distinct mechanisms. Neither is enabled by default.

- `full` combines norm fusion with the custom recurrent kernel.
- `norm-state` combines the existing TT-NN norm and state-update fusions. It retains the composite recurrence and does not call the custom kernel.

The `norm-state` selector appears in fusion profile version 2. Numerical tolerances and speed thresholds remain unchanged. The sampled-mode serving router still permits only its existing `off` and `full` routes.

The earlier B=1 attribution result did not select a dominant component. These are candidate checks, not claims that GDN dominates the workload.

## Full-fusion result

The physical B=1 component probe completed with `QWEN35_GDN_DECODE_FUSION=full`. The BF16 decode override was absent. Full trace replay measured 50.94 ms, against the earlier 54.34 ms baseline. This was not a balanced comparison and does not establish an accepted speedup.

The component evaluator rejected attribution because host input and readback timing varied too much and the dominance limit failed. It retained tokens 11751 and 13.

The subsequent B=4 serving smoke failed during the width-one startup warm-up:

```text
RuntimeError: shape '[2, 1]' is invalid for input of size 40
```

The model capacity was four. No long-output comparison completed. The driver closed normally.

A later device-free regression reproduced the exact error at the trace-output boundary. The `full` mode captures 20 candidates per device. The old startup warm-up passed those candidates to the greedy reader, which expected one winner per device.

The warm-up now selects the reader for the captured mode. Tests cover widths one and four, malformed output widths, read errors, cleanup, and denied readiness. This fix does not change the serving router or establish full-fusion numerical acceptance. A new physical B=4 full-fusion run remains outside this result.

The older full-fusion rejection remains relevant: `.cairn/archive/2026-08-27-fuse-qwen38-gdn-decode-state-update/evidence/parity-root-cause.md` records precision loss at mixed-kernel boundaries. The newer HiFi4 smoke does not erase that evidence.

## Staged candidate and serving boundary

The staged candidate passed the package checks and startup warm-up at widths one and four. The serving router then rejected `norm-state` with `RouterPolicyError`.

That rejection is intentional. An unadmitted candidate must not become a deterministic serving route. The router was not changed. The startup result alone does not prove long-output parity or a performance gain.

The diagnostic component command now accepts `--sequence-steps`. This option runs a bounded B=1 sequence without a network endpoint or serving router. It records the observed GDN mode, generated tokens, text, identities, and cleanup. Its receipt is explicitly ineligible for admission.

The diagnostic allocates page-table capacity for the full requested sequence. It rejects invalid lengths, missing or mixed GDN modes, and denied traces. Device-free tests cover token carry, positions, capacity requests, and those rejection paths.

## B=1 sequence result

The physical `norm-state` sequence completed with package `/nix/store/jbl3rh5j2gwaw05dgg2msj928gq1f9sw-qwen36-0.1.0`. All 64 tokens and the text matched the saved baseline for `The capital of France is`.

The [comparison](evidence/tt-metal-0.77/qwen-gdn-staged/sequence-comparison.json) passed positive, token-mutation, and empty-sequence controls. The [raw receipt](evidence/tt-metal-0.77/qwen-gdn-staged/norm-state-sequence.json) records the observed mode and successful cleanup.

This was one B=1 candidate sequence compared with a saved B=4 baseline response. It did not establish candidate B=4 correctness, exact recurrent-state equality, a controlled speedup, or production admission.

The later [balanced B=1 comparison](qwen38-staged-decode-comparison.md) preserved exact token/text parity but missed the existing full-step timing target. Broader parity and B=4 measurements remain open.

## Fusion-profile contract repair

The old negative fixture used `profile with`, which the packaged Nickel rejected as an unbound identifier. The package also copied that fixture into a directory that did not preserve its relative import. Any error counted as a passing negative test.

With valid syntax and imports, the old profile exported `diagnostic_only = false`. Constant defaults did not enforce the intended Boolean constraints. This does not prove that the serving router accepted such a profile.

The repaired contract requires true values for the diagnostic and parity/reset flags. The mode list is closed. Negative fixtures retain valid imports and must fail with `contract broken`. Tests cover production eligibility, an invalid mode spelling, and disabled exact-token parity.

## Execution limits

Each hardware probe used a finite timeout, private scratch space, read-only weights and caches, and a memory limit. The numerical simulator gap remains open. No simulator result proves these numerical paths.

Pueue reported a state-write error even though the first probe started. The active systemd unit prevented a duplicate launch. Later quota pressure caused Ruff to truncate `check.py`; the committed file was restored and compared byte-for-byte. Formatting then used tmpfs output followed by a new file and rename.

No serving gate was weakened. No deployment or push occurred.
