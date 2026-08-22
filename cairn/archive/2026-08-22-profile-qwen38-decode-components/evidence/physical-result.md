# Physical result

## Scope

- Physical PCIe devices: `0`, `1`
- Mesh: `(1, 2)`
- CPU affinity: `8-31`
- Runtime: `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`
- Final package: `/nix/store/16k19gpafpv6a9wadg742axf6qzdk2gd-qwen36-0.1.0`
- Checkpoint and tokenizer: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Authority BLAKE3: `c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd`
- Retained samples: five per group
- Unmeasured warmup replays: one per group

## Pilot

The first run exposed a cold bounded-readback path and one full-attention outlier. Its receipt was rejected for full-attention instability, readback instability, and insufficient GDN-to-MLP dominance. The pilot receipt BLAKE3 is `221e82f30fcef6c1137862ad833ab69b01e4f4dc96e76beab5be2bb95d17768c`.

The final package added one profile-owned unmeasured replay and readback before retained samples.

## Final result

The complete traced decode replay was stable:

- median device replay: `0.05421315398416482` seconds
- spread ratio: `1.004727385259557`
- repeated winner token: `13`

The additive component model totaled `0.05977224791422486` seconds. Its ratio to the measured full trace was `1.1025414225426509`, within the configured `0.8` to `1.25` closure range.

Weighted component results were:

| Component | Representative median | Multiplicity | Weighted seconds | Weighted share |
| --- | ---: | ---: | ---: | ---: |
| GDN mixer | `0.000580019986955449` | 48 | `0.02784095937386155` | `0.4657840443580814` |
| MLP | `0.0003762819978874177` | 64 | `0.02408204786479473` | `0.4028968075511107` |
| Full-attention mixer | `0.00035592998028732836` | 16 | `0.005694879684597254` | `0.09527631774481014` |
| LM head and argmax | `0.0020917720103170723` | 1 | `0.0020917720103170723` | `0.03499570592223391` |
| Embedding | `0.00006258898065425456` | 1 | `0.00006258898065425456` | `0.0010471244237638812` |

The direct weighted GDN-to-MLP ratio was `1.156087701933436`. It did not pass the configured `1.25` dominance gate. The result therefore does not select GDN or MLP work.

The final receipt also rejected overall attribution for these reasons:

- `embedding-instability`
- `host-input-update-instability`
- `host-winner-readback-instability`
- `minimum-dominance-ratio`

The median synchronized input update was `0.0006399830162990838` seconds. Its spread ratio was `1.611457421591108`. The median bounded readback was `0.0015717649948783219` seconds. One `0.0574714689864777`-second outlier raised its spread ratio to `38.500578404176125`.

## Decision

No kernel, precision, collective, or production change is selected. The direct replay result falsifies the earlier assumption that static MLP bytes alone identify a dominant runtime target. GDN and MLP costs are close under the exact traced P150x2 path.

Production returned to the eager service and passed health on port `8000` after both bounded runs.

## Evidence

- Final receipt: `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-decode-component-attribution-receipt.json`
- Pilot receipt: `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-decode-component-attribution-pilot-receipt.json`
- Final receipt BLAKE3: `58241201ed7eff7b1156b435e2af152eb7e10e359382af72e3b1ba7435df8025`
- Pilot receipt BLAKE3: `221e82f30fcef6c1137862ad833ab69b01e4f4dc96e76beab5be2bb95d17768c`

## Validation

The Qwen package, authority-negative, NixOS service, Python, Nix, policy, Cairn, and artifact gates passed. Ruff, Python compilation, Nix formatting, and Cairn lifecycle gates also passed.

Global Tracey coverage still fails on the existing repository-wide evidence-root policy gap. Its profile scans only `crates` and `tools`, so it lists this package-owned Python requirement with the other missing package requirements. Focused package and Cairn gates passed.

## Non-claims

- Independent component replay does not prove an optimization speedup.
- Rejected host timing does not establish a host bottleneck.
- The result does not apply to batching, concurrency, long context, TP4, thermal behavior, or another package.
- The result does not admit traced decode or change production from eager mode.
