# Qwen3.8 MTP drafter physical result

Date: 2026-08-22

## Decision

Reject MTP selection for this candidate. Keep production on eager decode.

The drafter and rollback paths are correct, but the measured economics fail. Aggregate emitted-token speedup was `0.9687222124830936`, below the strict `> 1.0` gate. Draft timing also exceeded the `1.25` spread bound.

## Bound candidate

- Package: `/nix/store/qy7l1pn3833rwbjar6xb7ysh3qzzjdpv-qwen36-0.1.0`
- Runtime: `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`
- Checkpoint and tokenizer: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Model source: `d102e7f4512060e09e580e40cf26a870bfa88c55`
- Authority BLAKE3: `c6146a5b5ea290afad2f5abbd99e73e987ed0d85e7d8df9ffb3b4ec0c56ee444`
- Physical PCIe devices: `0`, `1`
- Mesh: `(1, 2)`
- CPU affinity: `8-31`

The command loaded all 15 pinned `mtp.*` tensors. It shared the target embedding and LM head. It used separate BF16 normalization for shifted embeddings and post-final-norm target hidden rows.

## Smoke

The corrected one-prompt smoke accepted all three drafts. Draft tokens and target tokens were `[89454, 4384, 58377]`.

The smoke established weight loading and shifted-input alignment. Its cold compile timings were not used for economics.

## Final five-prompt pilot

The command warmed all five prompts before measurement. It then measured one complete speculative round for each prompt.

| Prompt | Accepted drafts | Total draft (s) | Target verify (s) | Readback (s) | Rollback + MTP re-prefill (s) | Sequential baseline (s) | Emitted-token speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 0.08102450499427505 | 0.10930749500403181 | 0.013962619006633759 | 0.0 | 0.282738859008532 | 1.383976045897197 |
| 2 | 0 | 0.07706965002580546 | 0.11477097601164132 | 0.012938837986439466 | 0.044844429969089106 | 0.286270285025239 | 0.2867016058099139 |
| 3 | 3 | 0.06635461599216796 | 0.10943353400216438 | 0.052020745002664626 | 0.0 | 0.2804581100062933 | 1.2311113225407215 |
| 4 | 3 | 0.13667992799310014 | 0.10961015601060353 | 0.01481975699425675 | 0.0 | 0.23244877500110306 | 0.890233681398928 |
| 5 | 3 | 0.07655773000442423 | 0.1094227520225104 | 0.011483792011858895 | 0.0 | 0.23742209901683964 | 1.2023547052880872 |

The mean accepted draft count was `2.4`. The drafter accepted 12 of 15 proposed tokens.

The evaluator used emitted-token-weighted totals:

- sequential baseline total: `1.1046354142890777` seconds;
- speculative total: `1.1403015230316669` seconds;
- aggregate speedup: `0.9687222124830936`.

Median per-prompt speedup was `1.202354705288087`. The evaluator did not use that median to hide the rejected prompt cost.

## Rollback result

Prompt 2 rejected its first draft. The verifier restored the accepted target prefix from trace-time GDN snapshots.

The restored recurrent state, convolution state, and committed paged KV had PCC `1.0` and NMSE `0.0`. The next target token also matched the sequential continuation.

Target rollback took `0.017195059976074845` seconds. MTP cache re-prefill took `0.02764936999301426` seconds. Thus, rollback correctness passed.

The receipt non-claim about rollback means that this diagnostic does not establish production rollback durability. The named `rollback_valid` and state fields are authoritative for this bounded pilot.

## Stability result

- Total draft spread: `2.059840539341421`.
- MTP prefill spread: `3.201045687765728`.
- Target verification spread: `1.0499826750892791`.
- Sequential spread: `1.231541379488365`.

The total draft spread exceeded the `1.25` bound. The evaluator therefore reported `sample-instability`.

## Evidence

- Receipt: `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-mtp-drafter-receipt.json`
- Receipt BLAKE3: `a102a170ce5d9fd1a2d2335bb6936ca1982e7a33df69b13df758b5073492141f`
- Measurement mode: `diagnostic-mtp-drafter-pilot`
- Final reasons: `sample-instability`, `emitted-token-economics`

The receipt is diagnostic-only and not admission eligible. The command restored device ownership. The eager service returned `{"model": "Qwen3.8-27B", "status": "ok"}` on port `8000` after the run.
