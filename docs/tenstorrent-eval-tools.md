# Traffic and six-passage retention CLI

Build/run the dependency-complete offline tool with `nix build .#tenstorrent-eval-tools`
and `nix run .#tenstorrent-eval -- --help`. Torch and Transformers belong to the
package; traffic itself imports neither. No model weights/tokenizer assets are
redistributed. The frozen fixture contains texts, token IDs and asset hashes only.

## Traffic: modeled, not measured

The equations follow [tensixfuse traffic.py at
6331e86b2169e7453f0a209b9d7e880dfa0aa38b](https://github.com/tanaymihani/tensixfuse/blob/6331e86b2169e7453f0a209b9d7e880dfa0aa38b/src/tensixfuse/traffic.py)
(MIT; notice shipped in the package). See also the existing
[Qwen streaming attribution](research/qwen38-mlp-streaming-attribution.md).

```sh
nix run .#tenstorrent-eval -- traffic --m 1 --k 5120 --n 8704 \
  --b-dtype bfloat4_b --bn 4 --bk 4 --cols 4 --multicast
nix run .#tenstorrent-eval -- traffic --m 8 --k 8704 --n 5120 \
  --b-dtype bfloat8_b --bn 4 --bk 4 --cols 4 --multicast
```

These are **illustrative block/grid assumptions at real TP2 Qwen projection
shapes**, not a reconstruction of the deployed kernel's traffic. The modeled
operation is `Y=act(A@B+C)`, not a whole gated MLP. Widths 1 and 8 pad to 32
rows; `logical_payload_bytes` excludes padding and exponent storage, whereas
all storage entries count padded tiles. Per 32×32 tile: BF16=2048 B,
FP32=4096 B, BF8=1024+64 B, BF4=512+64 B. Exponents are separate output fields.

For `Mt,Kt,Nt` padded tile counts and output-block counts `Mb,Nb`, A reads are
`Mt*Kt*Nb`, B reads `Kt*Nt*Mb`. Multicast divides A reads by grid columns and B
reads by grid rows, requiring evenly distributed output blocks. The tool exposes
compulsory one-read inputs plus one output write, additional rereads, multicast
savings, and two unfused intermediates each written and read once. Fusion removes
`4*Mt*Nt` intermediate tiles. L1 counts configurable buffer copies of A/B/C,
accumulator and Y per core with explicit mixed dtypes. It excludes firmware,
reserved L1, NoC, collectives, cache effects and allocator overhead. No output is a
hardware counter, performance prediction, or proof a candidate fits physical L1.

## Freeze actual tokenizer identities

```sh
# model points to an already-present, authorized local snapshot; offline only.
nix run .#tenstorrent-eval -- retention freeze --model "$model" \
  --checkpoint 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --out corpus.json
```

The shipped `pkgs/tenstorrent-eval-tools/data/qwen38-six-passages.json` freezes the
exact six original passages, trailing newlines included, token IDs, tokenizer
asset SHA-256s and authoritative `config.json` SHA-256/vocabulary width. No chat
template is applied; `add_special_tokens=True`, local files only, remote code
execution disabled. Host paths are not part of the fixture. It was generated from
the actual service-configured model assets, not reconstructed tokenizer output.
Token counts are **89,135,63,59,55,62**; scored counts **76,122,50,46,42,49**.
The full model scoring domain is exactly **248320** columns, from
`text_config.vocab_size`, not an assumed tokenizer length. Extra padding columns
and truncated vocabularies are rejected rather than silently renormalized.

## Score and compare exported logits

```sh
nix run .#tenstorrent-eval -- retention score --corpus corpus.json \
  --logits baseline.pt --out baseline.json
nix run .#tenstorrent-eval -- retention score --corpus corpus.json \
  --logits treatment.pt --baseline baseline.json --changes changes.json \
  --out treatment.json
nix run .#tenstorrent-eval -- retention compare --baseline baseline.json \
  --treatment treatment.json --original-diagnostic --out comparison.json
```

`torch.save` exports contain `corpus_sha256`, `provenance`, six `target_indices`
lists and six `rows` tensors. Each tensor is `[len(input_ids)-13,248320]`, row zero
predicting original token index **13**. Prefill the first 12 tokens, feed token
index 12 at position 12, score index 13, then continue through the final token.
The token-12 prefill prediction is deliberately not scored. Every tensor is cast
to float32 **before** float64 `log_softmax`; negative target log-probabilities are
averaged within each passage and then the six passage means receive equal weight,
not token-count weight. This is natural-log NLL, no generation-length cap.

Provenance requires nonempty `model`, `checkpoint`, `runtime`, `package`, `cache`,
`execution` strings and a `precision_flags` string mapping. Hardware `model`
identity is `weights-sha256:<canonical-manifest-digest>`, not the configuration
hash or a directory name. The retained `weight_manifest` binds the exact
safetensors index bytes and every unique referenced shard's SHA-256 and size.
The collector hashes before loading and again before export, refusing mutation.
Configuration/tokenizer identity remains separately bound by the frozen corpus.
`checkpoint` is the matching profile/corpus revision declaration; content hashes
identify the observed bytes, not an independent proof of their repository origin.
Hardware exports missing a matching content manifest are rejected, not enriched
or silently relabeled by the scorer. `execution` names measurement kind/geometry
(e.g. hardware P150x2 B1), **not code identity**; immutable `package` names code.
The scorer binds corpus/config/tokenizer, export SHA-256 and exact baseline
receipt SHA-256. Baseline and treatment must match except exact declared
`package`, `cache`, and/or complete `precision_flags` values in `changes.json`.
Runtime, model content, checkpoint and execution kind cannot be relaxed. For example:

```json
{"package":"/nix/store/<candidate-package>","precision_flags":{"QWEN35_GDN_DECODE_FUSION":"step","QWEN38_ROUNDED_REGISTER_SWIGLU":"1"}}
```

Use the **complete effective flag map** emitted by the collector, including
baseline flags and explicit zeroes, not just the abbreviated example above.
Comparison never manufactures missing provenance for historical receipts.
`--original-diagnostic` optionally reports ΔNLL ≤ +0.010 nats for these exact six
passages only; without it no threshold is applied. This is neither a global
quality floor nor serving admission. The comparison command reports a diagnostic
boolean in JSON; its success exit status means valid comparable receipts, not
that the optional diagnostic passed.

## Full-model collector: hardware owner only

`retention collect` integrates the existing `DecodeComponentProbeAdapter` and
`TracedDecodeAdapter`. It uses the original **B1** prefill/teacher-forced decode
recipe and exports every full-vocabulary row. It is not a B8 retention protocol.
The caller supplies `--runtime-python` pointing at the existing package-matched
interpreter and owns service isolation/devices. The **installed CLI wrapper**
re-executes that interpreter with Python `-I`, then admits only the explicit
package/runtime Nix-store module directories. It does not depend on ambient
PYTHONPATH; offline commands keep their isolated packaged dependencies.
The collector checks immutable package/runtime import locations, exact 1×2 mesh,
physical IDs 0,1, frozen checkpoint/config/tokenizer, and `step` route before
opening a mesh. No HTTP endpoint exposes these full logits.

```sh
# Existing local runtime interpreter; no pip installation and no credential access.
# package, runtime, model and cache are the owner's resolved deployment identities.
tools=$(nix build .#tenstorrent-eval-tools --no-link --print-out-paths)
unset PYTHONPATH PYTHONHOME
export TT_METAL_HOME="$package/libexec/tt-metalium"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export MESH_DEVICE=P150x2 HF_HUB_OFFLINE=1 QWEN35_GDN_DECODE_FUSION=step
# Keep ordinary runtime precision flags identical between separate processes.
"$tools/bin/tenstorrent-eval" retention collect --runtime-python "$runtime_python" \
  --corpus pkgs/tenstorrent-eval-tools/data/qwen38-six-passages.json \
  --model "$model" --package "$package" --runtime "$runtime" --cache "$cache" \
  --out baseline.pt
"$tools/bin/tenstorrent-eval" retention collect --runtime-python "$runtime_python" \
  --corpus pkgs/tenstorrent-eval-tools/data/qwen38-six-passages.json \
  --model "$model" --package "$candidate_package" --runtime "$runtime" --cache "$cache" \
  --decode-flags '{"QWEN38_ROUNDED_REGISTER_SWIGLU":"1"}' --out treatment.pt
```

For the second invocation, set `TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT` to the
candidate package too. Supply decode-only flags with `--decode-flags`, not inherited during
prefill/model setup; weight-loading precision flags belong in the process
environment, not this decode-only option. For the rounded-register experiment
this explicit scope remains valid even though the candidate also bypasses the
new route inside prefill. Run in an owner-controlled scratch HOME/cache when the
runtime needs writable user configuration. Do not use a simulator to produce a
receipt labeled hardware.

## Evidence and limits

[Dedicated evidence](../.cairn/changes/package-traffic-retention-cli/evidence/verification.md)
records installed CLI invocations and the final **16 passing behavior checks**.
The CPU synthetic-logit smoke validates scoring only; historical raw receipts
and original script remain byte-for-byte under `evidence/historical/`, without
retroactively invented provenance. The hardware owner separately exercised the
corrected installed collector and scorer on actual Qwen B1 baseline/candidate
model runs: exact 248320-way vocabulary, 385 scored rows, weight index/all66-shard
content bound before/after each collection, both means **0.7451044766709086** and
all six ΔNLL **0.0**. The unchanged original +0.010 diagnostic passes, not a global
quality floor. The complete portable record is
[`rounded-register-retention.json`](../pkgs/qwen36/evidence/2026-10-23/rounded-register-retention.json).
This is B1 retention only; the separate B8 model smoke establishes route/state
carry, not B8 retention. Traffic remains analytical, component timing is not
device profiling, and no performance/serving promotion or default change follows.
