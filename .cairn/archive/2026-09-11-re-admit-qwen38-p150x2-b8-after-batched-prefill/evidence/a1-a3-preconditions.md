# A1-A3: preconditions for the B=8 re-admission

Recorded 2026-09-11 for change `re-admit-qwen38-p150x2-b8-after-batched-prefill`.

## A1: the sibling mechanism change passed and is archived

`lift-gdn-grid-instance-cap` completed all 25 tasks and is archived at
`.cairn/archive/2026-09-11-lift-gdn-grid-instance-cap` (completion commit
`6d713eb`). Its verified results, which this change consumes:

- Both GDN grid guards lifted in a separate runtime derivation
  (`/nix/store/6ccynlvv…-tt-metal-0.77.0`); the shared 0.77 output is
  unchanged.
- Decode parity: one 192-instance fused call is bitwise equal to eight
  serialized 24-instance calls on output and recurrent state, on both P150s
  (`evidence/v8-physical-b8-parity.md`).
- Prefill scan parity: 192 instances bitwise equal per position and state to
  the serialized per-user reference on both P150s
  (`evidence/v9-physical-prefill-parity.md`).
- Served B=8: every batch at width 8, 24/24 text-exact against the single
  roots, `fallback_reason: null`; the same group on the shared runtime fails
  with `BatchDecodeAdapterError`, which discriminates the mechanism.
- Measured physical grid: 11 x 10 = 110 compute cores on both dies.

## A2: the rejected receipt restated without editing a value

Source: `pkgs/qwen36/evidence/2026-08-26/qwen38-27b-batch-admission-b8-receipt.json`,
BLAKE3 `e6641e7efa02f384e550e0621fea9ea97f1eaffe5d258fb3634312cc183a3fa5`,
`admission.admitted = false`, measured on the 0.74 runtime.

| Bound (unchanged) | Measured then | Result |
| --- | ---: | --- |
| aggregate decode speedup >= 6.0 | 5.881509219624968 | FAIL |
| per-user decode retention >= 0.75 | 0.735188652453121 | FAIL |
| TTFT ratio <= 9.0 | 9.079183271882933 | FAIL |
| tail-latency ratio <= 1.5 | 1.4461700431804296 | pass |
| exact token parity, exact text parity, distinct state slices 0..7, clean reset, identity match | pass | pass |

Rejection reasons recorded: `aggregate-decode-speedup-below-bound`,
`per-user-decode-retention-below-bound`, `ttft-ratio-above-bound`. Single-stream
median 16.88145527595542 tokens/s, B=8 median 99.28843484621837 tokens/s.

Protocol restated from the frozen policy
(`pkgs/qwen36/batch-admission-b8-policy.ncl`), not from prose:

- widths: baseline 1, candidate 8;
- root order: `single, batch, batch, single` (four roots, two per route);
- corpus: the eight prompts listed in the policy;
- runs: one warmup plus three measured per prompt;
- generated tokens: 64 per request;
- gates: exact token parity, exact text parity, distinct state slices, clean
  reset, physical evidence, active prefix.

No value above is edited by this change; `git diff origin/main` shows no
`.ncl` policy, profile, binding, or receipt file modified on the B=4 side and
only the new pinning in the B=8 contracts file.

## A3: the profile export is deterministic and byte-identical

Nickel export of the B=8 profile, twice, byte-identical:

```
deterministic: identical bytes
baseline_blake3=679c8782ef0061a3c5ed98ee7e453d26c759b14f7598c15bbb02e62babd96388
```

Pre-change versus post-change export, re-exported from
`origin/main`'s `.ncl` files and from this change's files with the same
Nickel:

```
pre : 679c8782ef0061a3c5ed98ee
post: 679c8782ef0061a3c5ed98ee
EXPORT BYTE-IDENTICAL pre vs post
```

The deployed artifact inside the rebuilt package matches too:

```
679c8782ef0061a3c5ed98ee7e453d26c759b14f7598c15bbb02e62babd96388
/nix/store/mgi46qcz…-qwen38-0.1.0/share/qwen36/batch-admission-b8-profile.json
```

The new contract pinning therefore changes no exported value; it only refuses
to export edited ones (I2).
