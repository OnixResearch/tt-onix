# P1.2 retention screen — original six-passage bounded non-regression

## Decision and scope

Before any projection-layout implementation, the numerical screen is the
**original six-passage B1 diagnostic**: matched treatment mean NLL minus matched
baseline mean NLL must be **at most +0.010 natural-log nats**. Equivalently,
`exp(treatment_mean - baseline_mean) <= exp(0.010)`, approximately `1.010050167`.
The mean is the equal-weight mean of six passage NLLs, **not** a separate
+0.010 ceiling on every passage. Preserve all six deltas for inspection.

This is the existing diagnostic ceiling recorded in
[p1-retention-corpus-identity-2026-10-22.md](p1-retention-corpus-identity-2026-10-22.md),
not a new global quality floor. Historical baseline NLL
`0.7451044766709086` is an observation for one bound checkpoint/execution,
not a universal absolute floor. No B8 retention claim is established.
P1.2 is closed at this bounded scope; it neither admits serving nor authorizes
hardware. The bank-matching candidate remains rejected by the independent
[reader audit](p1-reader-transactions-2026-10-25.md).

## Corpus and scoring contract

The frozen executable corpus is
`pkgs/tenstorrent-eval-tools/data/qwen38-six-passages.json`:

| Field | Required identity or rule |
| --- | --- |
| Corpus SHA-256 | `07b95ff60e0a5f596c3751502c4819eab79233ab181aee5afad660a7cd53e77b` |
| Exact texts | Six historical passages, preserved in the frozen JSON and the prior corpus-identity note |
| Token IDs | Frozen JSON; `add_special_tokens=True`, no chat template |
| Full model vocabulary | 248320 logits per scored row, no truncation or extra padding |
| Scored target indices | `13..end` of each passage |
| Scored token counts | 76, 122, 50, 46, 42, 49 |
| Prefill/decode alignment | Prefill first 12 tokens; feed token 12 to predict target 13; target 12 is not scored |
| Precision | Convert logits to float32, then float64 before full-vocabulary `log_softmax` |
| Reduction | Negative mean target log-probability within each passage, then equal-weight mean of six passages |
| Delta and ratio | Treatment mean minus baseline mean; perplexity ratio is `exp(delta)` |

`pkgs/tenstorrent-eval-tools/retention.py` implements corpus validation,
`score()`, `matched()`, and `compare()`; `cli.py` implements the command contract.
`collector.py` records **all effective** `QWEN35_`, `QWEN36_`, and `QWEN38_`
environment flags, including explicit zero values, plus decode-only overrides.
A treatment declaration must therefore come from the actual exports, not a
hand-copied two-flag summary.

## Matched provenance and retained historical evidence

Future real baseline/treatment collections must retain immutable runtime and
package identities, checkpoint revision, model configuration, tokenizer asset
hashes and token IDs, complete effective precision flags, cache identity,
execution identity, and the exact checkpoint index/all-shard content manifest.
The collector hashes the index and 66 shards before loading and again before
export, rejecting changes. The scorer binds corpus and logits by digest;
comparison binds the treatment to the exact scored baseline receipt.
`matched()` allows only explicitly changed `package`, `cache`, or
`precision_flags`; it rejects undeclared differences and declarations of
unchanged values. `precision_flags` replaces the entire mapping, not a patch.

The existing rounded-register experiment is a **different mechanism**, retained
only as evidence that this original screen has a content-bound executable path:

- Summary: `pkgs/qwen36/evidence/2026-10-23/rounded-register-retention.json`.
- Comparison: `pkgs/qwen36/evidence/2026-10-23/rounded-register-retention-comparison.json`.
- Weight manifest: `pkgs/qwen36/evidence/2026-10-23/rounded-register-weight-content.json`.
- Checkpoint revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Model configuration SHA-256: `74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc`.
- Model content: `weights-sha256:72437a18012f39c8abfe4be4889b8e33a2c3481b5910d5802ccf317df0977855`.
- Runtime: `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.
- Historical collection package: `/nix/store/9p6q10s8cc4s17k1l2fxpy2d4rjl23p8-qwen38-0.1.0`.
- Tool: `/nix/store/b2wy5v9fkqc2kafybwjyrz5wi2lmx7x1-tenstorrent-eval-tools-0.1.0`.
- Historical execution: P150x2 physical devices 0,1, B1; same resolved cache
  in both collections, with private absolute path omitted from the public summary.
- Baseline summary flags: `QWEN35_GDN_DECODE_FUSION=step`,
  `QWEN36_MLP_DOWN_BF4=0`; treatment adds
  `QWEN38_ROUNDED_REGISTER_SWIGLU=1` during decode only.
- Both historical means: `0.7451044766709086`; mean delta `0.0`, all six
  passage deltas `0.0`, perplexity ratio `1.0`; original diagnostic passes.

Those historical results are not a bank-matching treatment result and were not
rerun here. The public experiment summary is **not** a scoring receipt or a
full effective-flags export; do not pass it as `--baseline` or reconstruct
`changes.json` from its abbreviated flag summaries.

## Offline export-to-score protocol

The following operates on already collected `baseline.pt` and `treatment.pt`
exports in a scratch directory; it opens no device. Use the exact common frozen
corpus and a scorer-compatible Python with PyTorch. Set `TOOLS` to the bound
installed tool above and `PYTHON` to its compatible offline interpreter, e.g.
`/nix/store/0vcarl8sm79bhld67c5qnl6gbvrslhar-python3-3.13.13-env/bin/python`.
Run only trusted locally produced tensor exports.

```bash
set -euo pipefail
CORPUS="$TOOLS/share/tenstorrent-eval-tools/data/qwen38-six-passages.json"

# Read actual exports. --changes consumes a JSON FILE, not inline JSON.
"$PYTHON" - baseline.pt treatment.pt changes.json <<'PY'
import json
import sys
import torch

base = torch.load(sys.argv[1], map_location="cpu", weights_only=True)["provenance"]
trial = torch.load(sys.argv[2], map_location="cpu", weights_only=True)["provenance"]
assert set(base) == set(trial), "provenance keys differ"
changes = {key: trial[key] for key in trial if trial[key] != base[key]}
assert set(changes) <= {"package", "cache", "precision_flags"}, "unmatched identities"
with open(sys.argv[3], "w") as stream:
    json.dump(changes, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

"$TOOLS/bin/tenstorrent-eval" retention score \
  --corpus "$CORPUS" --logits baseline.pt --out baseline.json
"$TOOLS/bin/tenstorrent-eval" retention score \
  --corpus "$CORPUS" --logits treatment.pt --baseline baseline.json \
  --changes changes.json --out treatment.json
"$TOOLS/bin/tenstorrent-eval" retention compare \
  --baseline baseline.json --treatment treatment.json \
  --original-diagnostic --out comparison.json

# compare exits 0 even when the diagnostic fails; require its Boolean result.
"$PYTHON" - comparison.json <<'PY'
import json
import math
import sys

with open(sys.argv[1]) as stream:
    result = json.load(stream)
print("delta_nll:", result["mean_nll_delta"])
print("perplexity_ratio:", math.exp(result["mean_nll_delta"]))
raise SystemExit(0 if result["original_diagnostic"]["passes"] is True else 1)
PY
```

This derives only actual differences. An unchanged package/cache is omitted;
a changed `precision_flags` contains the full effective treatment mapping.
Use fail-fast shell handling or require each command's success separately;
the final explicit Boolean check, not the comparison command's exit status,
decides the bounded diagnostic.

## Offline proof performed on 2026-10-25

A throwaway script exercised the installed tool above without model inference,
TTNN import, simulator, hardware, or checkpoint loading. It generated synthetic
full-vocabulary exports aligned to all six frozen target lists, derived a real
`changes.json` file from export provenance, scored baseline/treatment, compared,
and executed the explicit Boolean guard. Synthetic execution/model/runtime/cache
identities were labeled `synthetic:...`; no hardware identity was fabricated.

| Synthetic case | Mean delta | Perplexity ratio | `compare` exit | Diagnostic | Explicit guard exit |
| --- | --- | --- | --- | --- | --- |
| Identical uniform logits, declared flag change | 0.0 | 1.0 | 0 | true | 0 |
| Raise non-target vocabulary column 0 by 10 | 0.08498240708087579 | 1.088697913155579 | 0 | false | 1 |

The script asserted that token 0 was absent from every scored target and checked
the second delta against `log((exp(10) + 248320 - 1) / 248320)`.
Both changes files contained only a complete replacement `precision_flags`
mapping (`step`, MLP down BF4 `0`, and synthetic rounded-register flag `1`),
with no unchanged package declaration. This proves the offline protocol and
failure interpretation, **not** model retention or any new layout behavior.
The throwaway script/exports were removed after verification.

## Rejections and hardware boundary

- `QWEN36_MLP_DOWN_BF4=1` remains rejected by the historical original-screen
  delta of approximately `+0.0192` nats, above `+0.010`.
- `matmul_reduce_scatter_decode` remains rejected by the separate historical
  decode throughput/core-collapse finding in `../design.md`; it is not an NLL
  observation or a newly collected timing.
- Bank matching remains rejected independently: the exact selected reader
  already requests each weight tile once per projection, so placement alone
  eliminates no weight reread. No prototype or successor is implied.

Any future real collection first requires a separately declared owned hardware
window, stopped/released device ownership, precise stop conditions, matched
content-bound baseline/treatment identities, and an actual legal treatment.
There is no such treatment or hardware authorization in this P1 result.
The historical rounded-register flag is not a placeholder for a bank-matching
candidate. No global-floor, B8, bandwidth, serving, or promotion claim follows.
