# Traffic/retention CLI verification — 2026-10-23

## Scope

The initial focused tooling smokes below opened no hardware and sampled no traffic
counters. They separate analytic traffic, real local tokenizer freezing, CPU
synthetic-logit scoring and historical recovery. Later hardware-owner model
collection is recorded separately; none of this authorizes serving promotion.

## Focused package and actual CLI

`nix build .#tenstorrent-eval-tools --no-link --print-out-paths` built the package
with explicit Torch/Transformers dependencies. Its check phase ran **12 tests,
all passing**: exponent/padding arithmetic; A/B reread and grid correspondence;
intermediate/L1 buffers; invalid dimensions/buffers/dtypes; float32-before-float64
scoring and equal passage rather than token weighting; target-index off-by-one;
row count/nonfinite/truncated and extra-padded vocabulary failures; corpus hash
mismatch; baseline receipt binding; declared package-code treatment changes;
and refusal to change execution kind or undeclared runtime provenance.

Isolated offline-entrypoint package output used for those smokes:
`/nix/store/pxrpf7hg4lqf5k449ly83cdqd4k5gnd1-tenstorrent-eval-tools-0.1.0`.
Its wrapper removes ambient PYTHONPATH/PYTHONHOME and uses Python `-I`.
With both variables deliberately pointing at nonexistent hostile paths, installed
traffic, comparison and Torch scoring all succeeded. The final isolated scoring
smoke used the temporary small synthetic test vocabulary and was removed; the
full real-vocabulary CPU receipts below remain from the earlier installed build
with unchanged scoring logic. No user site or ambient Python dependency is
required by the installed offline commands.

The installed CLI actually ran:

```sh
tenstorrent-eval traffic --m 1 --k 5120 --n 8704 --b-dtype bfloat4_b \
  --bn 4 --bk 4 --cols 4 --multicast --out traffic-qwen-gate-b1.json
tenstorrent-eval traffic --m 8 --k 8704 --n 5120 --b-dtype bfloat8_b \
  --bn 4 --bk 4 --cols 4 --multicast --out traffic-qwen-down-b8.json
```

Raw modeled receipts are adjacent. Gate/up weight storage is **25,067,520 B**
including **2,785,280 exponent B**; down is **47,349,760 B**, same exponent B.
The illustrated gate stream totals **31,752,192 fused B** with **5,242,880 reread
B**; down totals **53,575,680 fused B** with **5,013,504 reread B**. These depend on
the explicitly supplied illustrative blocking/multicast assumptions, not deployed
kernel measurements. Logical M1 A is **10,240 payload B**, while its padded tile
payload is **327,680 B**. The receipts expose that distinction.

## Real frozen corpus, not inferred IDs

The model directory was discovered in existing `.pi/qwen38-step-probe.env` and
confirmed by service `ExecStart`/environment metadata; no credentials, downloads,
weights or home configuration were accessed/modified as part of freezing.
The existing packaged interpreter and then the **installed CLI** both ran:

```sh
tenstorrent-eval retention freeze --model "$model" \
  --checkpoint 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --out corpus.json
```

Both produced byte-identical corpus SHA-256
`07b95ff60e0a5f596c3751502c4819eab79233ab181aee5afad660a7cd53e77b`.
The checked-in fixture is
`pkgs/tenstorrent-eval-tools/data/qwen38-six-passages.json`.
It includes verbatim original passages, all IDs/target indices, tokenizer asset
hashes, config hash **74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc**,
checkpoint and exact **248320** model vocabulary columns. No absolute host model
path is serialized. Passage lengths **89,135,63,59,55,62** give scored lengths
**76,122,50,46,42,49**; every target index from 13 to end is retained.
The full asset SHA-256 map is in the fixture and CPU receipts, not inferred from a
checkpoint label. Transformers was **5.5.4**. No model assets are redistributed.

## Installed CPU scoring smoke

Actual installed `retention score` (baseline and treatment) and `retention compare
--original-diagnostic` consumed synthetic float32 logits of shape
`[scored_count,248320]` for the **actual frozen token IDs**. All non-target logits
were zero, target logit for passage i was `i+0.125`; the treatment increased only
the first passage's target logits by float32 `0.001`. This is a transparent
arithmetic smoke, **not a real model-retention measurement**. Temporary ~383 MB
logit files and generation script were removed; JSON receipts remain:

- `synthetic-cpu-baseline.json`: six-passage mean **9.79764756782609**.
- `synthetic-cpu-treatment.json`: mean **9.79748090158248**.
- `synthetic-cpu-comparison.json`: Δ **-0.00016666624360972548** nats.

Receipts explicitly label provenance `SYNTHETIC-LOGITS-NOT-MODEL-RETENTION` and
`synthetic-cpu-scoring-validation-NOT-model-inference`. CPU scoring used
Torch **2.12.0**. The optional diagnostic boolean is only the original six-passage
+0.010-nat diagnostic, not a quality claim about this synthetic smoke or a global
floor. Permanent tests separately demonstrate unequal passage lengths are not
pooled/token weighted and reject vocabulary padding and off-by-one alignment.

## Runtime export integration: bounded, no device access

The collector imports the actual packaged Qwen model, TTNN, component probe and
traced decode adapters. With resolved existing package/runtime/model paths, two
explicit **pre-device** error smokes ran through these imports: nonexistent cache
path produced exit 2; missing `QWEN35_GDN_DECODE_FUSION=step` produced exit 2 after
package/runtime import-identity checks. Neither reaches device enumeration or mesh opening.

The final installed package is
`/nix/store/l064ffcn30zr3r8s0z19pg5gm29j31w1-tenstorrent-eval-tools-0.1.0`.
Its installed `bin/tenstorrent-eval retention collect --runtime-python ...`
actually ran under hostile PYTHONPATH/PYTHONHOME, re-executed the existing
package-matched Python with `-I`, admitted only the explicitly supplied immutable
package/runtime module directories, imported the real adapters and reached the
intentional missing-step-route error (exit 2) before device enumeration/open.
This final focused rebuild also passed all 12 behavior tests.

The dedicated usage document supplies the real full-model collector command.
The recipe retains exact B1 protocol, not B8, and emits content-bound exports
rather than an HTTP scoring API. The pre-device import smokes above are not
inference proof; hardware-owner collection is separate follow-on evidence.

## Historical receipts recovered unchanged

`evidence/historical/` preserves the original script and raw JSON receipts,
including their originally missing provenance. SHA-256 matches were checked
against the ignored scratch originals and the existing 2026-10-22 corpus note:

| File | SHA-256 |
| --- | --- |
| bf4_eval.py.txt (byte-preserved original bf4_eval.py) | 6b71aabe18822b84ad3a06b09d99dc725727832913262630054d8ced6be0a58d |
| ref-bf8-step.json | 17b05e8fe78c0fd3b6282c8ed1593f081c68d4a6eb6bce84016f15ab884b7d44 |
| bf4-GDN.json | 6d5c152a488183a7dc193742e43cd1970e921bd7713f755f4596ccce32c332aa |
| bf4-MLP_DOWN.json | 9476b0db7706ee273fc6a6d88faa0da72874b7e6cbd134d6177e600db3ced32d |
| bf4-LM_HEAD.json | a936427c2ed13a798b073bf9ebafeb9bc4b927ef0d4ffd86af676caab50e6042 |

These historical files are evidence, not modern schema fixtures or newly
provenance-matched baseline/treatment results. No fields or identities were
retroactively invented. The historical rejected candidate remains unchanged.

## Checkpoint-content correction

The initial collector identified `model` only by config SHA and copied the
checkpoint revision from the profile. That did **not** bind loaded weights.
The corrected installed package is
`/nix/store/b2wy5v9fkqc2kafybwjyrz5wi2lmx7x1-tenstorrent-eval-tools-0.1.0`.
Its focused build passed **16 behavior checks**, including identical
config/tokenizer with a changed second weight shard refusing baseline matching,
missing hardware content manifests, missing shards and unsafe index paths.

`provenance.model` now hashes a canonical manifest of exact safetensors index
bytes and **every referenced shard**, with SHA-256 and size. The collector hashes
before loading and after model collection, refusing changed source content.
Export and score receipts retain the full manifest. Configuration stays in the
corpus, and the checkpoint revision remains a declaration, not a guessed origin
proof from a directory name. Old config-only hardware exports are rejected;
they were not silently relabeled or enriched as corrected-collector outputs.

The actual mutable checkpoint yielded **66 indexed shards, 31,239,065,128 bytes**;
the index is 137335 bytes with SHA-256
`f0838c766951bdfe76d6afbdb2771a8f67aaa2231dedb3d33cebd817729843a2`.
Content identity is
`weights-sha256:72437a18012f39c8abfe4be4889b8e33a2c3481b5910d5802ccf317df0977855`.
The portable complete manifest is
[`rounded-register-weight-content.json`](../../../../pkgs/qwen36/evidence/2026-10-23/rounded-register-weight-content.json).

## Actual installed hardware-owner collection and scoring

The hardware owner ran **fresh** baseline and treatment collections using the
corrected installed `b2wy5v9fkqc2kafybwjyrz5wi2lmx7x1` tool and its
`--runtime-python` interface on physical IDs **0,1**, TP2, B1, `step` recurrence.
Both used the same immutable Qwen runtime/package, resolved cache and exact
checkpoint-content manifest. Only the declared decode flag
`QWEN38_ROUNDED_REGISTER_SWIGLU=1` changed; `QWEN36_MLP_DOWN_BF4=0` stayed explicit.
All index/shard contents matched before/after each actual collection.

Installed `retention score` consumed all **385 × 248320** actual-model logit
values, following targets 13..end and equal six-passage weighting. Baseline and
treatment means were both **0.7451044766709086**. Installed `retention compare
--original-diagnostic` reported six zero passage deltas and ΔNLL **0.0**,
passing the unchanged +0.010-nat original diagnostic. Earlier config-only
outputs were **not** relabeled as produced by this corrected collector.

Exact package identities, commands/environments, model content identity,
tokenizer/config/corpus hashes, export/score-receipt hashes and raw comparison
are in the portable
[`rounded-register-retention.json`](../../../../pkgs/qwen36/evidence/2026-10-23/rounded-register-retention.json)
and adjacent comparison/weight-manifest records. Private resolved cache paths
remain only in scratch raw receipts, not the public fixture or source record.
This is actual B1 diagnostic retention, not B8 retention, global quality
acceptance, traffic-counter evidence, or performance/serving admission.

