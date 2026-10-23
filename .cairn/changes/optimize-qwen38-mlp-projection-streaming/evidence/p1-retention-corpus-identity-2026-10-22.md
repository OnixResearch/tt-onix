# Retention screen corpus identity — 2026-10-22

## Recovery and provenance

The scratch files still exist. Checked `.pi/tmp/gdn-step/bf4_eval.py`,
`.pi/tmp/gdn-step/bf4-*.json`, and recovered
`bf4-GDN.json`, `bf4-MLP_DOWN.json`, `bf4-LM_HEAD.json` plus the saved control
`.pi/tmp/gdn-step/ref-bf8-step.json`. No model, simulator, or hardware was run.

These five source files are untracked: `git ls-files --` with their exact paths
returned no entries. This screen was present only as untracked `.pi/tmp` scratch,
not a reproducible version-controlled experiment. This note preserves the exact
passages, scoring recipe, and reported NLLs; it does not preserve the executable
script, complete raw receipts, tokenizer assets, or runtime/package artifacts.

## SHA-256 identities

| Source path | SHA-256 |
| --- | --- |
| `.pi/tmp/gdn-step/bf4_eval.py` | `6b71aabe18822b84ad3a06b09d99dc725727832913262630054d8ced6be0a58d` |
| `.pi/tmp/gdn-step/ref-bf8-step.json` | `17b05e8fe78c0fd3b6282c8ed1593f081c68d4a6eb6bce84016f15ab884b7d44` |
| `.pi/tmp/gdn-step/bf4-GDN.json` | `6d5c152a488183a7dc193742e43cd1970e921bd7713f755f4596ccce32c332aa` |
| `.pi/tmp/gdn-step/bf4-MLP_DOWN.json` | `9476b0db7706ee273fc6a6d88faa0da72874b7e6cbd134d6177e600db3ced32d` |
| `.pi/tmp/gdn-step/bf4-LM_HEAD.json` | `a936427c2ed13a798b073bf9ebafeb9bc4b927ef0d4ffd86af676caab50e6042` |

## Exact six passages

The JSON strings below are extracted mechanically from the script's `PASSAGES`
literal, in order. Decode JSON escapes to obtain verbatim text, including the
Python/SQL trailing newlines; no prose reflow is part of the passages.

```json
[
  "The mitochondrion is an organelle found in the cells of most eukaryotes. It uses aerobic respiration to generate adenosine triphosphate, which is used throughout the cell as a source of chemical energy. Mitochondria have their own genome, which is inherited maternally in most species, and they are thought to have originated from an ancient symbiosis between a host cell and an alphaproteobacterium.",
  "def merge_sort(values):\n    if len(values) <= 1:\n        return values\n    middle = len(values) // 2\n    left = merge_sort(values[:middle])\n    right = merge_sort(values[middle:])\n    merged = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            merged.append(left[i])\n            i += 1\n        else:\n            merged.append(right[j])\n            j += 1\n    merged.extend(left[i:])\n    merged.extend(right[j:])\n    return merged\n",
  "In 1905, Albert Einstein published four papers that changed physics. The first explained the photoelectric effect by proposing that light is made of discrete quanta. The second gave a quantitative account of Brownian motion. The third introduced special relativity, and the fourth derived the equivalence of mass and energy.",
  "To make a simple tomato sauce, heat olive oil in a pan over medium heat and add chopped garlic. Cook until fragrant, then add crushed tomatoes, a pinch of salt, and a few basil leaves. Let the sauce simmer for about twenty minutes, stirring occasionally, until it thickens.",
  "The derivative of a function measures how its output changes as its input changes. For example, the derivative of x squared is two x, and the derivative of the sine function is the cosine function. Integration is the inverse operation, and the fundamental theorem of calculus connects the two.",
  "SELECT customers.name, COUNT(orders.id) AS order_count\nFROM customers\nLEFT JOIN orders ON orders.customer_id = customers.id\nWHERE customers.created_at >= '2024-01-01'\nGROUP BY customers.name\nORDER BY order_count DESC\nLIMIT 10;\n"
]
```

## Tokenization, scoring window, aggregation

- Source: `bf4_eval.py:24-60,94,103-114,116-117,143-159`.
- Tokenizer: `AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)`,
  with `model_path = Path(os.environ["QWEN38_MODEL"])`; each passage uses
  `tok(p, add_special_tokens=True).input_ids`. No chat template is applied.
- The script does not save the resolved tokenizer files/hash, token IDs, token
  counts, or `QWEN38_MODEL` path. The receipts contain no tokenizer identity.
- Let `ids` be one passage's tokens. Prefill `ids[:12]`; set `rest = ids[12:]`
  and `steps = len(rest) - 1`. At decode step `s`, feed `rest[s]` at position
  `12+s`, read logits, and score target `rest[s+1]`. Thus the scored zero-based
  original token indices are `13` through `len(ids)-1`, inclusive. The prefill
  prediction of token index 12 is not scored. All remaining positions are scored;
  the passage loop does not impose the prompt-loop `GEN=96` limit.
- `prefill` calls `probe._prefill(torch.tensor([ids], dtype=torch.int32),
  generation_tokens=GEN + 200)`; `GEN=96`, so the requested reserve is 296.
- Logits are read as float32, stacked, converted to float64, then normalized
  with `torch.log_softmax(..., -1)`. Passage NLL is the negative mean target
  log-probability over that passage's scored positions, in natural-log nats.
- `mean_nll = sum(passage_nll) / 6`: equal passage weighting, not token weighting.
  `mean_nll_delta` subtracts the saved reference's equal-passage mean.
- Control flag: `QWEN35_GDN_DECODE_FUSION=step`. Each treatment additionally
  sets exactly its recorded `QWEN36_GDN_BF4`, `QWEN36_MLP_DOWN_BF4`, or
  `QWEN36_LM_HEAD_BF4` flag to `1`.

## Recorded NLL values

Source: each raw receipt's `passage_nll` and `summary` fields. Passage columns
follow the literal order above; values below are the stored JSON numeric values.

| Passage | BFP8 control | GDN BF4 | MLP-down BF4 | LM-head BF4 |
| --- | ---: | ---: | ---: | ---: |
| prose | 0.8437592094805272 | 0.8939728056682931 | 0.8626758336308163 | 0.824223734184302 |
| Python | 0.12953256808947505 | 0.11267550908935713 | 0.13278670282327004 | 0.1385112903733144 |
| history | 0.9946418787190416 | 1.0228134293480329 | 1.0155421608883324 | 1.0195542801927813 |
| recipe | 0.8727227987705922 | 0.8875623699028902 | 0.8939042213118945 | 0.864648488052562 |
| calculus | 1.2131943407421244 | 1.2060897061154843 | 1.2587617820106478 | 1.2313729743782662 |
| SQL | 0.4167760642236913 | 0.3948095569282513 | 0.42193857090416553 | 0.42395948319961474 |
| Equal-passage mean | 0.7451044766709086 | 0.7529872295087182 | 0.7642682119281878 | 0.7503783750634735 |
| Stored Δ versus control | — | 0.007882752837809592 | 0.019163735257279213 | 0.005273898392564869 |

## Reproducibility verdict and remaining requirements

**Not reproducible from version-controlled in-repo state alone.** Exact corpus
text and the arithmetic recipe have now been recovered, but scratch receipts
do not record the tokenizer revision/assets, input IDs, resolved model/package/
runtime environment, or a content-bound link identifying the loaded reference.
The recovered reference has the BFP8 label/flag and its mean matches the reported
deltas; the script loads a reference through `EVAL_REF_LOAD` but does not save
that path in the receipts. The earlier screen note reports checkpoint
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, TT-Metal 0.77.0, P150x2 devices
0 and 1, B=1; these are contextual claims from
`.cairn/changes/enable-qwen38-bf4-weight-caches/evidence/screen-2026-09-24.md`
(Setup), not resolved identity metadata in these JSON files.

To make the existing ΔNLL ≤ +0.010-nat diagnostic ceiling usable reproducibly,
retain a version-controlled executable scoring fixture using these exact passages
and indexing/aggregation rules, pin tokenizer assets/revision and emitted token
IDs, and bind the model/checkpoint, runtime, package, cache/precision flags, and
reference receipt identities. Then run matched BFP8 control and MLP-layout
treatment measurements with full per-passage receipts. Do not replace these
passages with reconstructed text or treat the ceiling as corpus-independent.
