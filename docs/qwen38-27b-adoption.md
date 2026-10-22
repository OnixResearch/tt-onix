# Qwen3.8-27B adoption (tt-metal PR 53543)

This document records the repository adoption of
[tenstorrent/tt-metal pull 53543](https://github.com/tenstorrent/tt-metal/pull/53543),
"Enable Qwen3.8-27B on the existing qwen36 Blackhole path".

## Current package names

The canonical package is `qwen38`. Its commands use the `qwen38` prefix.
The old `qwen36` names remain compatibility aliases. Upstream module paths,
cache paths, and historical receipts keep their original names.

The sections below retain the original adoption history. See the
[TT-Metal 0.77 port](tt-metal-0.77-port.md) for the current runtime candidate
and its acceptance limits.

## What the PR does

The PR registers `Qwen/Qwen3.8-27B` as an `HF_MODEL` weight-swap on the
main-line `models/demos/blackhole/qwen36` path. It:

- registers the Qwen3.8-27B checkpoint, prefill-chunk sizes, and tokenizer
  fallback in `models/tt_transformers/tt/model_config.py`;
- adds opt-in BF4 weight caches (`QWEN36_LM_HEAD_BF4`, `QWEN36_MLP_DOWN_BF4`,
  `QWEN36_GDN_BF4`) stored under distinct `.bfp4` cache names so default BFP8
  bins are never overwritten;
- keeps `QWEN_SDPA_BF8=1` as the long-context KV path;
- adds the `traced_150k` demo case and clips long prompts before tokenization;
- makes the vLLM adapter import-safe when Qwen3.5 multimodal helpers are
  absent, so text-only `vllm==0.10.1.1` can serve.

The PR targets P150x4 and P150x8 meshes only (4-way / 8-way tensor parallel).
It is **closed but not merged** upstream (`merged: false`).

## Adoption decision

The repository's packaged tt-metal runtime is pinned to `v0.74.0-rc6`, which
does **not** contain the `models/` tree at all. The qwen36 path — and every
change this PR builds on — exists only on tt-metal `main`. A full runtime pin
bump to main would rebuild the C++ stack and risk the validated RWKV-7 and
training deployments. This change therefore adopts the PR in the least
disruptive form:

1. **Pin the adopted source tree** at the PR head commit
   `d102e7f4512060e09e580e40cf26a870bfa88c55`. The tree already contains the
   PR on top of the qwen36 base, so no patch file is required.
2. **Layer the `models/` tree** from that pin over the packaged 0.74 runtime
   as a standalone deployable (`pkgs/qwen36`). No change to the cached tt-metal
   0.74 derivation, no RWKV/training rebuild.
3. **Gate the adoption on device-free evidence** (`qwen36-check`): the adopted
   modules must import against the packaged runtime, the opt-in BF4 gates must
   keep their documented BFP8 defaults, and the vLLM adapter must import with
   vLLM absent.

The runtime-compatibility question (0.74 vs main) is answered by the
device-free check, not by assumption. If the import gate ever fails against
0.74, the checked-in `qwen36` package records the exact blocker and the
bounded alternative (a main runtime pin) becomes the next change.

## Reviewed authority

The adoption facts live in `pkgs/qwen36/authority.ncl` with the contract in
`authority-contract.ncl` and the exported, committed copy in
`generated/authority.json`. The Nix validator (`authority.nix`) admits the
authority only when it matches:

- upstream repository `tenstorrent/tt-metal.git`, branch `qwen38-qb2`,
  commit `d102e7f...`, base `c71ea3ca...`, `merged_upstream = false`;
- PR 53543 facts (117 additions / 40 deletions / 9 files);
- release `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
  architecture `Qwen3_5ForConditionalGeneration`, and 18 checkpoint shards;
- 64 text layers: 48 linear-attention layers and 16 full-attention layers;
- one checkpoint MTP layer, with checkpoint weights excluded from authority hashing;
- hashed metadata artifacts (license, config, tokenizers, preprocessors,
  checkpoint index) with exact Nix and BLAKE3 identities.

Drift in any recorded fact fails the package evaluation. A separate
`qwen36-authority-negative` check proves a drifted commit is rejected.

The adopted `weight_mapping.py` declared eight full-attention indices for the older 32-layer model, but no code read that set. The remapper only changes tensor keys and layouts. Runtime layer construction reads the checkpoint `text_config.layer_types` list through `Qwen36ModelArgs.attention_type_list`. The accepted authority records interval `4` and all 16 full-attention indices. The local `layer-type-authority.patch` removes the unused set and validates the complete checkpoint list before layer construction. It permits the 32-layer fallback only for its exact layer count.

## Deployable surface

- `nix build .#qwen38` builds the package and its `models/` tree.
- `qwen38-check` runs the device-free adoption gate.
- `qwen38-p150x2-serve` runs the serialized P150x2 service.
- `qwen38-serve` runs the separate text-only vLLM 0.10 entry point. It rejects an absent vLLM dependency.
- `qwen38-p150x2-mtp-probe` compares target verification paths.
- `qwen38-p150x2-mtp-drafter-probe` runs the checkpoint drafter and rollback probe.
- `qwen38-p150x2-mtp-selection-probe` compares full-logit and bounded draft selection.
- `qwen38-p150x2-mtp-trace-probe` compares eager and persistent-trace MTP drafts.
- `qwen38-p150x2-mtp-rollback-probe` compares cache re-prefill and bounded row restoration.
- `qwen38-p150x2-mtp-recovery-prewarmed-control` runs the persistent recovery control after target prewarm.
- `qwen38-p150x2-decode-component-probe` measures component replay. Its `--sequence-steps` option records bounded B=1 sequences.
- `qwen38-p150x2-ttft-control` compares request-cache capacity and parked-trace effects.
- `nix build .#checks.x86_64-linux.qwen38` runs the package gate.
- `nix build .#checks.x86_64-linux.qwen38-authority-negative` runs the authority rejection gate.

The diagnostic commands do not admit candidates or change production policy.

## Deployment (hardware)

The 27B configurations require a P150x4 or P150x8 mesh for the PR's certified
path. This repository additionally carries a **P150x2 (`P300`)** build so the
two-P150 deployment can attempt Qwen3.8-27B at 2-way tensor parallel without
waiting for a larger box.

On a P150x4/x8 box:

```sh
HF_MODEL=Qwen/Qwen3.8-27B MESH_DEVICE=P150x4 nix run .#qwen36 -- \
  -m pytest models/demos/blackhole/qwen36/demo/text_demo.py -k 'traced_4k and not batched'
```

For long-context and the reported throughput, set the PR flags:

```sh
export QWEN_SDPA_BF8=1                # BF8 SDPA + paged KV (long context)
export QWEN36_LM_HEAD_BF4=1           # opt-in BF4, distinct .bfp4 caches
export QWEN36_MLP_DOWN_BF4=1
export QWEN36_GDN_BF4=1
```

Measured by the PR author on P150x4 (with all four flags): traced-4k
28.69 tok/s, traced-150k 25.54 tok/s, vLLM `max_model_len=4096` 22.63 e2e
tok/s. These are upstream-reported numbers, not repository evidence.

## Non-claims

Local silicon evidence proves a bounded TP=2 oracle, component PCC, recurrent
state carry, short-prompt paged serving, and B=4 active-prefix admission for one
stable execution identity. It does not prove full 64-layer logits PCC against a
CPU model. It also makes no long-context, streaming, B=8, B=16, arbitrary
continuous-batching, thermal, or general profiler claim. Each result applies
only to its recorded scope.

## Carried P150x2 (P300) build

`pkgs/qwen36/p150x2-tp.patch` adds a TP=2 path on top of the adopted PR. It:

- registers `P150x2 → (1, 2)` in the demo and test mesh maps;
- adds `P300` prefill-chunk entries (`2` in 1024-token units) for
  Qwen3.5/3.6/3.8-27B — `P300` is the unit-test built-in device name for
  two Blackhole devices (`determine_device_name`) and already appears in
  `get_trace_prefill_supported_seq_lens`;
- advertises **host sampling** at TP=2: `supports_sample_on_device` is made
  mesh-conditional because on-device sampling is certified only for 1x4/1x8
  (the Top-K 65536-logits/device shard limit; Qwen3.8-27B gives 124160
  logits/device at TP=2).

Why TP=2 divisibility is safe for this checkpoint: 24 heads, 4 KV heads,
16/48 linear-attention heads, hidden 5120, intermediate 17408, and vocab
248320 all divide by 2. No KV-head replication is needed. The bounded silicon
oracle now passes. Only bounded warm-service decode throughput is measured.

P150x2 validation status:

1. device-free gate — passed (`qwen36-check` TP=2/P300 probes);
2. simulator — pending for the experimental `libttsim_bh_x2.so` path;
3. hardware oracle — passed on a physical 1x2 mesh with eight greedy tokens;
4. component PCC and GDN state carry — passed against PyTorch references;
5. paged-contract logits for eight selected layers — passed against the bespoke path;
6. full 64-layer serialized paged service — passed for two sequential requests;
7. bounded warm-service TTFT and decode throughput — passed for three runs;
8. bounded B=4 active-prefix serving — admitted for execution identity
   `/nix/store/jilqzkn7gp3mxgnh41jgc6326z6n4rnj-qwen36-execution-identity-0.1.0`;
   long context, streaming, B=8, B=16, and arbitrary continuous batching remain pending.

## Hardware evidence (2026-08-19, britton-desktop)

The two-P150 box validated the adopted stack on real silicon with the packaged
0.74 runtime (no runtime bump):

- **Silicon op roundtrip** — device open on physical chip 1, host→device,
BF16 matmul 32x1024x1024, device→host; cosine 0.999968 against torch
  (`evidence/2026-08-19/hw-silicon-matmul.log`).
- **Qwen3.5-9B single-device generation — PASS**: the demo's own paged path
  (`_run_paged_generation`) produced `"The capital of France is Paris."` with
first generated token `Paris` and a non-degenerate continuation
  (`evidence/2026-08-19/hw-9b-paged-generate.log`). Physical chip 1,
`TT_VISIBLE_DEVICES=1`, `TT_MESH_GRAPH_DESC_PATH` unset (the ambient p150_x2
descriptor cannot map onto one device — the same hazard `rwkv7-chat`
documents).

Two blockers were found and fixed on the way, both recorded as carried patches
or package fixes:

1. `generate_tp` crashed on single-device models: `reset_tp()` calls
   `reset_state()` on every layer, but single-device full attention only
   exposes `reset_cache()` — carried as `generate-tp-reset.patch`. The PR's
   own oracle test box was unchecked upstream; the single-device oracle had
   never been run. (`generate_tp`/`prefill_tp` remain TP-only entry points —
   the 4D fractured-residual layout is wrong for single-device use; the 9B
   validation above uses the served paged path instead.)
2. The packaged 0.74 runtime include tree lacks
   `api/compute/experimental/matmul_custom.h` (and more), which the qwen36 SDPA
   kernels JIT-include. The package replaces the overlay's `hw/inc/api`
   subtree with files from the package-matched source.
3. The 0.74 bindings lack `ttnn.transformer.chunk_gated_delta_rule`. The
   `gdn-seq-fallback.patch` probes for that operator and uses
   `gated_delta_attn_seq` when the fused path is absent.
4. The installed runtime lacks required TP JIT kernel sources under
   `ttnn/cpp`. The package overlays the complete package-matched `ttnn/cpp`
   tree, including `all_gather_minimal_matmul_async`.
5. The TP=4-tuned GDN `matmul_reduce_scatter_prefill` waits indefinitely on
   P150x2. `tp2-runtime-compat.patch` disables that fusion below four devices
   and uses row projection plus all-reduce.
6. The 0.74 AGMM binding rejects the newer `fuse_swiglu` keyword.
   `tp2-runtime-compat.patch` probes the binding signature and uses the
   separate MLP matmuls when the keyword is absent.

The first TP=2 attempt started at 20:00 EDT. It opened the exact 1x2 mesh,
initialized `FABRIC_1D`, and loaded all 64 layers. A host reboot interrupted it
before token generation and removed its temporary log.

## Hardware evidence (2026-08-20, britton-desktop)

A later bounded run passed with both inference services stopped:

- physical PCIe device IDs 0 and 1 formed an exact `(1, 2)` mesh;
- `FABRIC_1D` initialized on both devices;
- all 64 Qwen3.8-27B layers loaded from revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- the bespoke TP state-carry oracle generated eight non-degenerate tokens;
- the first generated token was `Paris`;
- the full continuation was `The capital of France is Paris.\nThe capital of Germany is`;
- UMD closed both devices cleanly.

The run used tt-metal 0.74, TT-UMD 0.9.9, KMD 2.10.0, firmware 19.11.0,
and SFPI `7.70.0[847]`. Model loading took 77.221 seconds. Generation took
78.244 seconds, including first-run compilation. These timings are diagnostic,
not admitted performance results.

The physical PCC gate also passed:

- GDN component minimum PCC: `0.99999`;
- attention component minimum PCC: `0.99993`;
- MLP component PCC: `0.9992400660601553`;
- GDN four-step recurrent output/state minimum PCC: `0.999964`/`0.999995`;
- paged-contract prefill/decode minimum PCC: `0.9998629502385343`/
  `0.9999042691604841`;
- masked-bucket prefill PCC: `0.9998459531124145`.

The full 64-layer serialized paged path then served two requests from one
loaded model. It completed France with `" Paris.\nThe"` and Germany with
`" Berlin.\nThe"`. The packaged HTTP shell passed health, model, completion,
and chat requests. It also rejected streaming, wrong-model, and unknown-route
requests.

The full-model 2048-token chunk-trace warmup collided with L1 allocation on
runtime 0.74. The admitted service therefore uses masked-bucket paged prefill
and eager paged decode. Prompt plus generation is limited to 2048 tokens.

Evidence:

- `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-tp2-receipt.json`
- `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-tp2-summary.log`
- `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-tp2-pcc-receipt.json`
- `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-tp2-pcc-summary.log`
- `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-tp2-paged-service-receipt.json`
- `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-tp2-paged-service-summary.log`
- `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-tp2-service-perf-receipt.json`
- `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-tp2-service-perf-summary.log`

The warm-service benchmark used one warmup and three measured requests. Each
request had five prompt tokens and 64 generated tokens. Median TTFT was
`0.5750412510024034` seconds. Median decode rate was
`4.2306764265491275` tokens/second. These results do not extend to other
prompt lengths, batches, request concurrency, or meshes.

## Traced-decode candidate

The package includes `qwen36-p150x2-trace-candidate` for bounded physical evaluation.
The candidate uses stable decode buffers, GDN state restoration, and shard-local greedy reduction.

`pkgs/qwen36/decode-profile.ncl` owns the trace, buffer, workload, and admission bounds.
Its deterministic JSON export has `physically_admitted = false` until matched physical evidence passes.

The production `qwen36-p150x2-serve` wrapper remains eager.
Invalid trace bindings also select the named eager fallback before a decode replay.
Cleanup errors quarantine the only slot and block reuse.

The first 2026-08-20 attempt was blocked by the root-owned service.
The owner later authorized a bounded outage, and the physical candidate ran on device IDs `0` and `1`.

The repaired traced path preserved exact token IDs, exact text, matched identities, and clean resets across the profile corpus.
Its median decode rate was `15.336118315891143` tokens per second.
Its TTFT ratio against the matched eager run was `1.2164174383938555`, above the profile maximum of `1.05`.

The candidate was therefore rejected and remained physically unadmitted.

A second balanced physical run used `eager, candidate, candidate, eager` roots.
The candidate used a `2`-block request-bounded prefill table and retained its `32`-block trace table.
It preserved exact token IDs, exact text, clean reset, and matched identities.
Its median decode speedup ratio was `6.942337116061393`.
Its TTFT ratio was `1.1069467490614955`, which still exceeded the `1.05` bound.
The residual difference was concentrated in device prefill, not first-token readback.
The aggregate phase evidence does not identify the responsible prefill operation.

The candidate remains rejected and physically unadmitted.
The production wrapper remains eager.

### Prefill program diagnostic

Append `--prefill-profile` to either P150x2 service wrapper only during a bounded diagnostic run.
The service then collects package-matched TT-Metal real-time program records during prefill.
It validates and groups exact kernel-source signatures by physical chip.

Profiled responses use `measurement_mode = diagnostic-prefill-profile` and `admission_eligible = false`.
They include a receiver drain, so their TTFT values cannot enter admission.
The service fails startup when the real-time profiler is inactive.
The normal production wrapper does not register a callback or wait for receiver drain.

The 2026-08-21 physical diagnostic found `100` complete common signature-and-chip rows.
The candidate program-duration sum was only `1.0064446256760908` times the eager sum.
Its `2.306050567859769` millisecond delta explained `0.22521929455787217%` of the profiled wall-prefill delta.
The largest single difference was a `1.0996968146463241` millisecond MLP reduce-scatter increase on chip `0`.

No program difference was large enough to select an optimization.
The CCL settings must not change from this result alone.

A package-matched Tracy diagnostic then measured host enqueue and finish zones.
The qwen36 overlay carries `tracy-capture` and `tracy-csvexport` version `0.13.3`
from the same vendored fork as the packaged runtime.
Host-profiled responses use `measurement_mode = diagnostic-host-timeline` and remain ineligible for admission.

The bounded physical order was `eager,candidate,candidate,eager`.
Candidate duplicate ratios were at most `1.0760504091447658` across the checked wall, enqueue, and gap totals.
Eager duplicate ratios reached `2.900386615837245` for wall prefill and `9.206084094193452` for enqueue execution.
The comparator rejected the result as `duplicate-root-instability` against its `1.25` comparability bound.
No host attribution or optimization is selected from the rejected receipt.
A follow-up used Tracy signpost messages without Python call profiling. It still failed duplicate stability: eager enqueue ratio `3.174942201847243` and candidate enqueue ratio `1.4077723936685347`.

A second follow-up used five independent fixed-affinity samples per root. Candidate duplicates passed. Eager wall and gap ratios were `1.3928346091490837` and `1.4276727516605083`. This isolates the blocker to persistent eager cross-process drift. Both results were rejected, so no host attribution or optimization was selected.

Evidence:

- `pkgs/qwen36/evidence/2026-08-21/qwen38-27b-host-stability-attribution-receipt.json`
- `pkgs/qwen36/evidence/2026-08-21/qwen38-27b-host-signpost-attribution-receipt.json`
- `pkgs/qwen36/evidence/2026-08-21/qwen38-27b-host-timeline-attribution-receipt.json`
- `pkgs/qwen36/evidence/2026-08-21/qwen38-27b-prefill-profile-attribution-receipt.json`
- `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-traced-decode-candidate-receipt.json`
- `cairn/archive/2026-08-20-accelerate-qwen38-p150x2-decode/evidence/physical-result.md`
- `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-traced-decode-ttft-candidate-receipt.json`
- `cairn/archive/2026-08-20-reduce-qwen38-traced-decode-ttft/evidence/physical-result.md`

### MTP target-verification diagnostic

The package provides `qwen36-p150x2-mtp-probe` for an explicit physical diagnostic. The command compares traced sequential decode with teacher-forced widths `1`, `2`, and `4`. It uses five samples per width.

The first 128-token masked fallback failed token parity, GDN state equivalence, stability, and performance. The pinned runtime still lacks the fused `ttnn.transformer.chunk_gated_delta_rule` operation.

The package now uses a pinned-runtime hybrid packed-decode verifier instead. It batches projections, MLPs, and the LM head across candidate positions. It preserves ordered high-precision GDN recurrence. It also serializes paged-KV writes when candidate rows share a physical block. The full width-specific model forward runs through one captured trace replay.

The final 2026-08-22 P150x2 result validated this target-only base:

- target tokens matched exactly at widths `1`, `2`, and `4`;
- full logits, recurrent state, convolution state, and written paged KV had PCC `1.0` and NMSE `0.0` in every sample;
- width-four sequential median: `0.23198413103818893` seconds;
- width-four verifier replay median: `0.10929988999851048` seconds;
- width-four diagnostic readback median: `0.009195436985464767` seconds;
- width-four device amortization: `2.122455302025925`;
- width-four end-to-end amortization: `1.9469233481156285`;
- sequential and replay spread ratios stayed within `1.25` at every width.

Full-logit host readback remains separately visible and diagnostic. A speculative loop must keep target logits on device for acceptance. The untraced hybrid route was exact but slower than sequential execution, so tracing is load-bearing.

The valid target path did not select MTP by itself. At this stage, the checkpoint drafter, local acceptance, rejection rollback, and emitted-token latency were unmeasured.

At width four, drafter generation plus acceptance had an approximate `0.11282991201733239`-second budget before the iteration became slower than four traced steps.

Evidence:

- `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-native-mtp-verifier-receipt.json`
- final receipt BLAKE3: `2a4f171b36dcf31f4637d48706dfa3e037af100a5ca21bae91f47070b023ac19`
- `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-native-mtp-verifier-untraced-receipt.json`
- untraced receipt BLAKE3: `f87a06702f49047c921f78873d2eeceff1dadd8a478d7fab7f00328ec15ff473`
- prior masked-fallback receipt: `pkgs/qwen36/evidence/2026-08-21/qwen38-27b-mtp-target-verification-receipt.json`

### MTP drafter and rollback diagnostic

The package also provides `qwen36-p150x2-mtp-drafter-probe`. This explicit command loads the 15 pinned `mtp.*` tensors. Production wrappers do not load them.

The probe shares the target embedding and LM head. It separately normalizes shifted embeddings and post-final-norm target hidden rows. It runs one full-attention MTP layer with its own KV cache.

The final five-prompt P150x2 pilot produced these results:

- 12 of 15 drafts were accepted, for a mean prefix length of `2.4`;
- all target-verifier outputs matched the sequential target outputs;
- the rejected prompt restored recurrent state, convolution state, and committed paged KV exactly;
- rollback took `0.017195059976074845` seconds;
- MTP cache re-prefill took `0.02764936999301426` seconds;
- the emitted-token-weighted aggregate speedup was `0.9687222124830936`;
- total draft timing spread was `2.059840539341421`, above the `1.25` bound.

The deterministic evaluator rejected MTP for `sample-instability` and `emitted-token-economics`. A positive median did not override the negative aggregate result. Production remains eager.

Evidence:

- `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-mtp-drafter-receipt.json`
- receipt BLAKE3: `a102a170ce5d9fd1a2d2335bb6936ca1982e7a33df69b13df758b5073492141f`
- `cairn/archive/2026-08-22-evaluate-qwen38-mtp-drafter/evidence/physical-result.md`

### MTP draft-selector diagnostic

The package provides `qwen36-p150x2-mtp-selection-probe`. It compares historical full-vocabulary MTP selection with the target path's shard-local greedy policy.

The paired physical control produced identical draft, target, and continuation tokens. Prompt 2 restored hybrid state exactly after rejection. The observed selector reduction was `66.49554434507507%`, with bounded/full ratio `0.3350445565492493`.

The complete candidate remained invalid:

- aggregate emitted-token speedup was `0.8197972276186697`;
- one bounded MTP-prefill root took `0.4443297780235298` seconds;
- prompt 2 accepted zero drafts; and
- most duplicate prompt-route groups exceeded the `1.25` stability bound.

The implementation removes full-vocabulary draft readback. Unstable roots prevent a validated speedup claim. The next candidate traces the fixed-shape MTP draft step. Package source shows that later cache rows use explicit positions, but rejected-row overwrite still needs separate physical proof.

MTP, traced decode, and production behavior remain unselected.

Evidence:

- `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-mtp-draft-selection-receipt.json`
- receipt BLAKE3: `b0e0eb8df13fe8db1667cca7d82973cb069596675bdf50a5857b9f67e995d923`
- `cairn/archive/2026-08-22-optimize-qwen38-mtp-draft-selection/evidence/physical-result.md`

### MTP draft-trace diagnostic

The package provides `qwen36-p150x2-mtp-trace-probe`. It compares eager draft decode with a persistent trace that includes embedding, input fusion, the MTP layer, final normalization, shard-local LM-head projection, and bounded top-one reduction.

The trace owns stable token, position, RoPE, feedback, MTP KV-cache, and output bindings. Warmup and capture compare the same MTP KV row and feedback output exactly. Capture time is recorded and excluded from replay economics.

The profile retains six roots per route and prompt in three `eager,traced,traced,eager` blocks. The evaluator rejects address drift, capture mutation, token drift, target drift, rollback failure, timing spread above `1.25`, traced/eager draft ratio above `0.95`, or aggregate speedup at or below `1.0`.

The matched physical control preserved draft, target, and continuation tokens in all 60 roots. Every traced root passed address-binding, capture-idempotence, MTP KV-prefix, and target-rollback checks. Accepted prefix lengths remained `3,0,3,3,3`.

The traced draft-decode median was `0.01038352950126864` seconds, compared with `0.020045544501044787` seconds for eager draft decode. The traced/eager ratio was `0.507276627509049`, the total draft ratio was `0.7213822465863816`, and aggregate emitted-token speedup was `1.2921427224106086`.

The evaluator rejected traced drafting because every prompt-route group exceeded the `1.25` duplicate-root bound in at least one required timing group. The receipt's BLAKE3 is `c35ed4dacc2557086d811ee90f14c423f2a89b051bc91f716229dfb9320daddc`.

The observed gains are diagnostic opportunity bounds. The control does not prove rejected-row overwrite or multi-round MTP rollback. MTP and traced decode remain unadmitted, and production remains eager.

Evidence:

- `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-mtp-draft-trace-receipt.json`
- `cairn/archive/2026-08-22-stabilize-qwen38-mtp-drafting/evidence/physical-result.md`

### MTP K/V rollback diagnostic

The package provides `qwen36-p150x2-mtp-rollback-probe`. It compares full MTP cache re-prefill with device-resident restoration of the two rows that three-draft decode can overwrite.

The snapshot route allocates its K/V row and prefix-feedback buffers before trace capture. After target verification, it restores rejected rows, retains accepted rows, processes any accepted draft not represented in feedback, and advances one committed target token.

The evaluator compares the next draft token, feedback tensor, and committed MTP K/V prefix exactly. It requires six roots per route and prompt in three `reprefill,snapshot,snapshot,reprefill` blocks. It rejects recovery spread above `1.25` or snapshot/re-prefill ratio above `0.8`.

The matched physical control preserved the next draft token, feedback tensor, and committed MTP K/V prefix in all 60 roots. It covered natural accepted lengths zero and three. Target output and rollback also remained exact.

Re-prefill recovery had median `0.046100162508082576` seconds. Snapshot recovery had median `0.02023772850225214` seconds. The ratio was `0.43887204011198166`, for an observed reduction of `56.11279598880183%`.

The evaluator rejected snapshot rollback because six prompt-route groups exceeded the `1.25` recovery-stability bound. Exact state and a lower median do not override unstable roots.

Snapshot rollback, MTP, and traced target decode remain unselected. Production remains eager.

Evidence:

- `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-mtp-kv-rollback-receipt.json`
- receipt BLAKE3: `ea0b1d77b8c6be61a566778862a12e1faa620a597ec7cad1101a01efee929921`
- `cairn/archive/2026-08-22-reduce-qwen38-mtp-rollback-cost/evidence/physical-result.md`

### Persistent MTP recovery trace candidate

The version-two rollback control keeps one fixed-shape MTP trace and its buffers for the loaded process. It uses this trace for draft replay, accepted-prefix replay, and committed-token advance.

The control captures the trace during discarded warmup. Measured roots must report capture count one and one exact binding identity. Both recovery routes use explicit device synchronization before host-wall timing.

The control retains the five prompts and three `reprefill,snapshot,snapshot,reprefill` blocks. It does not remove outliers or weaken the `1.25` stability bound.

The first physical warmup rejected `MTP trace binding changed before-controlled-replay: kv_buffer_identity`. The package-matched attention reset allocated new K/V cache objects.

A second package reset the same K/V objects from fixed zero sources. This change removed the immediate binding error. The control still produced no receipt before its named `7200`-second timeout.

The watchdog restored eager service and health. No complete matched roots exist. Persistent recovery tracing, MTP, and traced target decode remain unselected.

Evidence:

- `cairn/archive/2026-08-22-isolate-qwen38-mtp-recovery-jitter/evidence/physical-result.md`

### Staged MTP recovery localization

The diagnostic package adds `qwen36-p150x2-mtp-recovery-stage`. It runs one prompt and one route through the unchanged full recovery path.

An atomic progress ledger records the last completed trace, replay, recovery, verification, cleanup, or receipt boundary. Diagnostic timings include ledger overhead.

The staged command is admission-ineligible. It cannot select MTP, traced target decode, or production behavior.

Two physical cases reached their `1200`-second bounds without receipts. Both completed MTP recovery first.

- Prompt 1 re-prefill ended after `native-verifier-start`.
- Prompt 2 snapshot ended after `target-reference-state-start`.

The logs were still compiling target validation kernels. The result localizes the bounded noncompletion after MTP recovery. It does not establish recovery parity or stability.

The second timeout required `tt-smi -r /dev/tenstorrent/0 /dev/tenstorrent/1`. Eager service and expected health were restored after the reset.

Evidence:

- `cairn/archive/2026-08-23-stage-qwen38-mtp-recovery-replay/evidence/physical-result.md`
- re-prefill ledger BLAKE3: `de1f81db3e9c9b279aacc8f28fcedb9fd6f00f71efbe7ada0e1bab2a79514d35`
- snapshot ledger BLAKE3: `64de84aa0cbe1a32ec1ed88b43a992a69500116b3b031cac92e3132a37e8c556`

The package also provides `qwen36-p150x2-mtp-recovery-prewarm-stage`. This command prewarms the same prompt through the nonpersistent target-validation path.

The same process then runs the persistent recovery route. The ledger records each prewarm boundary and the program-cache entry count.

The command requires exact prewarm and measured target-validation signatures. It rejects changed draft, target, verifier, acceptance, or rollback facts.

The bounded prompt-2 snapshot case returned status `0` and reached the complete boundary after `191.3243064180133` seconds. The prewarm and measured signatures matched exactly. The measured route retained acceptance length zero, exact target tokens, exact rollback state, one persistent trace capture, and valid trace bindings.

The prewarm increased the program-cache count from `14` to `406`. Measured setup raised it to `408`. Target reference-state construction and native verification then completed without increasing the count.

This bounded case identifies cold target-validation compilation as the observed blocker. It does not establish repeated-root stability, MTP selection, speedup, or production admission. Production remains eager.

Evidence:

- `pkgs/qwen36/evidence/2026-08-23/qwen38-27b-mtp-target-validation-prewarm-receipt.json`
- receipt BLAKE3: `04b8087eb04a8e609c0cb3bd83f419a21f1405360659bdb6aa24cf669da8c88a`
- `pkgs/qwen36/evidence/2026-08-23/qwen38-27b-mtp-target-validation-prewarm-progress.json`
- progress BLAKE3: `9b95efa146fb38de18fbc7da0a4fb304dd919be579714f2ba8adb752d7e6e3da`

The package also provides `qwen36-p150x2-mtp-recovery-prewarmed-control`. This diagnostic command prewarms all five fixed prompts before persistent recovery warmups.

It then runs both warmup routes for each prompt and the accepted sixty-root measured schedule. The pure progress ledger records every root boundary and program-cache count.

The authorized physical control completed all five prewarms, ten warmups, and sixty measured roots. All target-validation signatures matched their prompt prewarm signatures.

The program cache had `411` entries throughout measurement. One persistent trace capture served all measured roots.

The evaluator rejected the snapshot route. All six prompt-2 snapshot roots failed next-draft, feedback-state, and committed-prefix parity.

Eight prompt-route groups exceeded the `1.25` recovery-spread limit. The aggregate snapshot-to-re-prefill recovery ratio was `0.38005667764112233`, but correctness and stability gates failed.

The receipt has `rollback_valid=false`, `snapshot_rollback_selected=false`, and `admission_eligible=false`. Production remains eager.

Evidence:

- `pkgs/qwen36/evidence/2026-08-23/qwen38-27b-mtp-recovery-prewarmed-control-receipt.json`
- receipt BLAKE3: `9ab2ceef2cb3cbe710561ba0c56b61ad5155a3ce2159075deb3b37705ed0e24e`
- `pkgs/qwen36/evidence/2026-08-23/qwen38-27b-mtp-recovery-prewarmed-control-progress.json`
- progress BLAKE3: `d1d13010d5bd734a6124a6f3e4ddefb67c5783864ca777f8298b905d359cc36d`

### Decode component replay diagnostic

The package provides `qwen36-p150x2-decode-component-probe`. It captures coherent component traces around real package-selected layers. It measures three GDN mixers, three MLPs, three full-attention mixers, embedding, LM-head reduction, the full decode trace, synchronized input update, and bounded winner readback.

The final P150x2 run produced these results:

- full device replay median: `0.05421315398416482` seconds
- additive closure ratio: `1.1025414225426509`
- weighted GDN share: `0.4657840443580814`
- weighted MLP share: `0.4028968075511107`
- weighted full-attention share: `0.09527631774481014`
- weighted LM-head and argmax share: `0.03499570592223391`
- GDN-to-MLP weighted ratio: `1.156087701933436`

The additive model closed within its typed bounds. The full trace was stable. The evaluator still rejected attribution for embedding instability, host input-update instability, host readback instability, and insufficient GDN-to-MLP dominance.

No optimization is selected. Direct replay does not support the earlier assumption that static MLP bytes identify one dominant target. GDN and MLP costs are close on this exact traced path.

Evidence:

- `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-decode-component-attribution-receipt.json`
- `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-decode-component-attribution-pilot-receipt.json`
- final receipt BLAKE3: `58241201ed7eff7b1156b435e2af152eb7e10e359382af72e3b1ba7435df8025`
- `cairn/archive/2026-08-22-profile-qwen38-decode-components/evidence/physical-result.md`

### Same-process TTFT control diagnostic

The package provides `qwen36-p150x2-ttft-control`. It keeps one loaded model and compares stable request-sized and trace-capacity KV cache sets. It also compares full-capacity prefill before, during, and after a parked decode trace.

The final physical run used six balanced roots with five retained samples per root. Every sample returned token `11751`. The observed TTFT ratios were:

- trace-capacity cache over request cache: `1.0062416611969702`;
- parked trace over untraced capacity cache: `0.9951621585112468`; and
- parked capacity route over request route: `1.0013736235407196`.

All TTFT ratios stayed below `1.05`. All TTFT root and duplicate spreads stayed within `1.25`. The strict control still failed because readback spread exceeded `1.25` in roots `3`, `4`, and `6`.

TT-Metal also warned about buffer allocation while the decode trace was active. The control did not execute that trace. The result therefore makes no post-prefill trace-correctness claim.

No mechanism or optimization was selected. The result does not admit traced decode, and production remains eager.

A later device-free pass hardened invalid-profile and allocation-failure paths. The outage budget was exhausted, so that hardened package has no new physical claim.

Evidence:

- `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-traced-ttft-control-receipt.json`
- receipt BLAKE3: `fd127f0cf88d8393487dc95949b072f334037ab2cf3459db914db7e93660f759`
- `cairn/archive/2026-08-22-isolate-qwen38-traced-ttft/evidence/physical-result.md`

## Serialized P150x2 service

The package provides `qwen36-p150x2-serve`. The NixOS module owns a matching
`qwen38-p150x2.service` unit:

```nix
hardware.tenstorrent = {
  enable = true;
  qwen38 = {
    enable = true;
    modelPath = "/path/to/the/pinned/Qwen3.8-27B/snapshot";
    conflictingUnits = [
      "llamacpp-server-vibethinker-britton-desktop.service"
      "docker-tt-inference-server-llama-3-1-8b-instruct-p150.service"
    ];
  };
};
```

The service listens on `127.0.0.1:8000` by default. It provides `/healthz`,
`/readyz`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`. It can
assemble one active-prefix decode batch through B=16. Formal admission covers
only B=4. Requests with a temperature above zero need `sampledFusion`; without it
the service is greedy-only.

The updated package and module allow 131,072 combined prompt and generation
tokens per request, with at most 8,192 generated tokens. On britton-desktop
the enabled native NixOS unit still contains the older 65,536-token command.
The managed root profile deployment below replaces only its effective
`ExecStart` via a persistent `/usr/local/lib/systemd/system` drop-in, selecting
the admitted han2 package with a 131,072-token limit. The historical `/run`
overrides remain in place but no longer determine the effective command.
A future NixOS switch can still replace or mask the native unit.

OMP keeps `contextWindow: 65536` for ordinary `mesh/Qwen3.8-27B` sessions and
uses `maxContextWindow: 131072` in `~/.omp/agent/models.yml` for optional
`/extended-context on` sessions. `maxTokens` remains 8,192. Check the running
service's `ExecStart` and `/healthz` before using extended context, especially
after a reboot or NixOS switch; the OMP override alone cannot enlarge the
server's limit.

The HTTP endpoint also caps JSON request bodies at 1,048,576 bytes,
independently of token count; byte-heavy prompts can reach that limit first.

The running han2 package allocates one pool of 4,113 blocks of 64 tokens.
Blocks 0–15 reserve the structured-read prefix, blocks 16–4,095 hold
generation rows, blocks 4,096–4,111 provide four disjoint batched-read
rows, and block 4,112 is scratch. Each batch group gives every generation
row a contiguous run sized to its request: whole 1,024-token prefill
buckets plus every decoded position. A full 131,072-token row takes
2,048 blocks of the 4,080-block generation range, leaving 2,032 blocks
for another request before prefix-cache residency; two full-length rows
do not fit. The generation prefix-cache budget is 4,064 blocks. The worker
adds requests only while the pool holds them all. Startup logs the
allocator's DRAM totals as the `device-dram` stage. The earlier gfvk package
instead allocated 4,096 blocks with a 2,048-block prefix-cache budget.

On britton-desktop the pooled build became ready in 700 s with 9.2 GiB of DRAM
free per chip. A needle placed at the start of the prompt was recalled at 6K,
16K, 32K and 60K tokens. TTFT was 3.3, 9.2, 19.2 and 39.2 s, and decode was
23.2, 22.7, 22.0 and 21.1 tokens/s. A group of one 30K row and three short rows
kept the short outputs equal to their solo runs. Prefill still runs one user
after another within a group, and a group runs until its last request finishes,
so a long prompt or generation delays the requests queued behind it.

On 2026-09-29 the rebuilt `/nix/store/clm5m3463xsyw1q93ivqpf13i2gbya1w-qwen38-0.1.0`
started without `--max-sequence-length`, captured its traces, and completed a
physical `P150x2` warm-up. TT-SMI identified the serving process on device
IDs 0 and 1 (firmware 19.11.0.0, TT-KMD 2.10.0). Through the local OMP gateway
at `127.0.0.1:9337`, an uncached 130,544-token prompt recalled the two-word
passphrase from its start and generated four tokens; the 2,048-block row,
103.56 s TTFT, and zero prefix-cache hits are recorded in the
[physical receipt](../pkgs/qwen36/evidence/2026-09-29/qwen38-long-context-130k.json).
The fixture placed `violet ibis` at the start, repeated ` filler.\n` 43,500
times, then asked for the passphrase (`temperature: 0`,
`enable_thinking: false`, `max_tokens: 16`).
A 131,144-token prompt plus 16 generated tokens returned a 400 naming the
131,072-token cap; see the
[overflow receipt](../pkgs/qwen36/evidence/2026-09-29/qwen38-long-context-overflow.json).
`/healthz` still reported ready afterward. The rebuilt package's device-free
checks and the NixOS service check passed. This is one bounded recall and
admission check, not a quality survey, long-request concurrency or latency
admission, 262K support, or proof that the old systemd generation was upgraded.

After the candidate test, `ttwkv7-owner-control restore` restarted the old
65,536-token service. A subsequent root SSH maintenance session used
`ttwkv7-owner-control isolate`, pinned the tested package at
`/nix/var/nix/gcroots/qwen38-context-128k`, and loaded only
`/run/systemd/system/qwen38-p150x2.service.d/90-context-128k.conf` to replace
its `ExecStart`. `ttwkv7-owner-control restore` started the new package; systemd
reported notify-ready after 1,002.94 s of warm-up. The active service ran as PID
1889532 with `--max-sequence-length 131072` and remained healthy.

The drop-in leaves the base unit's environment, user, and notify settings
unchanged. Its `ExecStart` changes only the package path, sequence limit, and
continuous-slot/prefix-cache policy paths to files in the tested package.

Through the OMP gateway, the restarted service recalled `cobalt otter` from an
uncached 130,545-token prompt and generated four tokens on physical devices 0
and 1. Its 2,048-block row took 103.68 s to first token; a 131,145-token
prompt plus 16 requested generation tokens returned HTTP 400 naming the
131,072-token limit. `/healthz` was ready both before and after. See the
[live-service receipt](../pkgs/qwen36/evidence/2026-09-29/qwen38-live-128k-service.json).
This is one bounded physical recall, not a concurrency, quality, or latency
admission.

A separate non-interactive OMP run selected `mesh/Qwen3.8-27B` with a temporary
`extendedContext: true` config overlay and no tools. A 22,500-repeat prompt
returned `copper wren` with 69,459 input and 53 output tokens through the mesh
provider; see the [OMP receipt](../pkgs/qwen36/evidence/2026-09-29/qwen38-live-omp-extended.json).
The interactive `/extended-context on` command also reported enabled, while
ordinary OMP sessions remain in the 65,536-token mode. This proves a request
above the old limit reached the live model through OMP; it does not establish
general 128K quality or compaction behavior.

The first 128K activation capped cached prefixes at 1,024 KV blocks of 64
tokens, or 65,536 tokens. Before budget-aware donation, the scheduler tried to
save the last 1,024-token boundary even when it could not fit. The fix saves
the longest chunk-aligned conversation prefix within the configured budget.
The opt-in shared system/tool root is still saved only at its exact verified
boundary, never truncated into a different digest or state.

A matched physical two-turn baseline on the previous service used 130,549 and
130,559 prompt tokens with 130,540 identical leading token IDs. Both turns
answered `amber thrush`, but neither saved or hit a prefix; TTFT was 103.65
and 103.43 s. See the [baseline receipt](../pkgs/qwen36/evidence/2026-09-29/qwen38-long-prefix-baseline.json).
On the first bounded-prefix package, the same two requests returned the same
answer on physical device IDs 0 and 1. The cold turn saved 65,536 tokens at
103.69 s TTFT; the follow-up hit those 65,536 tokens and reached its first
token in 61.00 s, about 41% below the old follow-up. The saved carried GDN
state and cached KV remained coupled; masked-prefill traces stayed available.
See the [physical trial](../pkgs/qwen36/evidence/2026-09-30/qwen38-long-prefix-trial.json).
Immediately after that restart, the gateway temporarily answered 503 despite
direct `/healthz` reporting ready; a later short gateway request succeeded.
The quarter-pool package
`/nix/store/fgz9698w8g8c1qnz9hi3x3c59qanlkhw-qwen38-0.1.0` was rebuilt
after correcting the policy-source comment. Its scheduler module and exported
cache-policy JSON are byte-identical to the physical trial's, and its packaged
device-free checks pass. That live service repeated the exact two request
hashes on physical device IDs 0 and 1: the cold turn saved 65,536 tokens at
103.61 s TTFT; the follow-up hit them at 60.28 s TTFT, **41.7% less** than the
previous service's matched 103.43 s follow-up. Both produced `amber thrush`
with token IDs `[14778, 8541, 1080]`. A short request through the gateway
returned `READY`; an over-limit prompt of 131,149 tokens plus 16 generation
tokens returned HTTP 400, and `/healthz` stayed ready. See the
[deployed-package receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-long-prefix-final.json).
This is a bounded sequential 128K-prefix reuse and recall check, not an
admission of throughput, competing cache residents, arbitrary long-form
quality, or long-request concurrency.

TTsim's 0.77 fused GDN numerical gate stops on an unsupported FP32 matmul
control flag (see [the port blocker](tt-metal-0.77-port.md)); it cannot prove
the carried-state restore needed here. These receipts are physical, not
simulator recurrence evidence.

On 2026-09-30 the v1 cache budget was raised to 2,048 of the existing
4,096 KV blocks: the pool allocation and four carried-state buffers did not
grow. A 130,549-token cold prompt now saves 130,048 tokens (2,032 blocks) at
its last complete chunk; a 130,559-token continuation needs only 16 own
blocks. The opt-in shared-root v2 policy still holds a separate 1,024-block
quota and is not enabled on the serving unit.

On the same physical P150x2 and exact cold/follow-up request hashes, the
larger budget returned the same `amber thrush` token IDs
`[14778, 8541, 1080]`. Cold TTFT stayed at 103.58 s against 103.61 s
previously; after an unrelated short `READY` request, the 130,048-token
prefix hit reached its first token in **0.848 s versus 60.28 s** under the
quarter-pool quota (98.6% less for this turn). The short request itself took
0.224 s to first token. A different 130,549-token prompt returned
`copper wren` cold with zero prefix hits and a 103.40 s TTFT. The gateway
also served `READY`, rejected a 131,149-token prompt plus 16 generated
tokens with HTTP 400, and `/healthz` stayed ready; see the
[half-pool physical receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-long-prefix-half-pool.json).
The unchanged 4,096-block pool leaves only 2,047 blocks free while this
2,032-block prefix is cached (16 read-prefix blocks and one scratch block
are reserved). An unrelated 2,048-block row therefore evicts an unused
prefix; a pinned reader instead delays that full row until retirement.
The device-free checks exercise both transitions and the smaller quota.
Separate long conversations can evict each other's prefixes. The earlier
ABBA cache admission binds another package and cache budget; it is not reused
to admit this larger quota. These runs do not admit concurrent long requests,
broad model quality, or general latency/throughput across workloads.

At the earlier half-pool activation, package
`/nix/store/gfvkac4zchl11g5n3xh9zbkvc4451vbl-qwen38-0.1.0` was pinned at
`/nix/var/nix/gcroots/qwen38-prefix-128k-budget-candidate`. After
`ttwkv7-owner-control isolate` and `restore`, systemd reported PID 2446749
with `--max-sequence-length 131072`. The runtime-only
`/run/systemd/system/qwen38-p150x2.service.d/96-prefix-128k-budget.conf`
then superseded `95-prefix-65k.conf` and `90-context-128k.conf`; both
policy paths resolved inside the then-active package. The former 65K-prefix package
remains pinned at `/nix/var/nix/gcroots/qwen38-prefix-65k-final` for rollback,
and the original 128K package remains pinned separately.

At this earlier `/run`-only activation, NixOS was **not** switched.
`/etc/systemd/system` resolved into the read-only Nix store and `/run` was
tmpfs; a reboot at that point would have lost all three runtime drop-ins and
fallen back to the 65,536-token boot command. The later managed profile
deployment below addresses that specific loss **only while the current
enabled, unmasked native unit remains selected**; an actual reboot has not
been exercised. The unrelated, heavily edited `onix-core` checkout masks
Qwen and sets a 2,048-token limit, so switching it still would not preserve
this owner. Keep the old package GC root.

The historical `/run`-only rollback removed `96-prefix-128k-budget.conf`
to return to the earlier 65K-prefix package, or removed all three `/run`
drop-ins and lowered OMP's extended limit to return to the 65,536-token boot
unit. **Do not apply that old procedure to the managed profile owner:**
the persistent `99-profile.conf` takes precedence. The managed `restore`
command described below can return to the native `/run`96 old-package
command only while all three original `/run` drop-ins still exist; after a
reboot, first verify profile generation 1 still exists and contains only the
pinned old package; otherwise use the retained old-package GC root for reviewed
single-package profile recovery, not a vanished `/run` command.

On 2026-09-30 a separate **two-long-context candidate** increased the pool
from 4,096 to 4,097 blocks and the cache budget from 2,048 to 4,064 blocks.
The 16 structured-read blocks and one scratch block leave exactly 4,080
generation blocks. One 2,032-block cached 130,048-token prefix can coexist
with an unrelated cold 2,048-block row; after that row donates its own
2,032-block prefix, both distinct histories fit with 16 blocks free for one
continuation or short request. Device DRAM allocation increased by exactly
2,097,152 bytes (2 MiB per chip) versus the previous startup receipt.
The 131,072-token row limit, GDN state count and model computations did not
change. A third unrelated cold full row still reclaims unused entries; the
contiguous-run allocator may evict both prior prefixes, so this is not
admission for three independent long conversations or concurrent long rows.

The physical P150x2 AB-A baseline on the **active** half-pool package saved
each of two distinct 130,048-token prefixes, but the A revisit missed after
B's cold turn and took 103.446 s to first token. On the candidate, A and B
cold turns took 103.546 and 103.469 s, then the A and B revisits hit their
respective 130,048-token KV and carried GDN states at **0.864 and 0.857 s**
to first token. A's revisit was 99.16% faster than the measured baseline,
with the same answer and generated token IDs. Both hits survived an unrelated
`READY` request through the gateway; a third unrelated 130K-token prompt ran
cold in 103.558 s and saved its own prefix. The 131,072-token overflow
still returned HTTP 400. See the
[bounded physical receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-two-long-prefix-candidate.json).
The package, device-free checks (including ABAB state/KV ownership, pinned
readers, third-row reclaim and the old one-block-short geometry), and NixOS
service check passed. TTsim still cannot check the fused GDN recurrence;
this result is a bounded physical parity/latency check, not a broad
throughput, concurrency or quality admission.

**This candidate is not deployed.** The root-owned production unit cannot
accept an additional `/run/systemd/system` override through the available
passwordless owner helper. `sudo -n` could not create a root GC symlink
(`a password is required`); the listed passwordless operations cover only
stopping and starting the owner and inspecting device ownership. The direct
candidate on port 8000 was run only while the owner was isolated, with the
pinned TT-Metal source added to `PYTHONPATH` for its unshipped `infra` import.
It was stopped after the physical check; the original half-pool systemd unit
was restored. The candidate package is retained by
`.pi/qwen38-prefix-two-long-candidate`, a registered user-writable Nix GC
root. A privileged, reviewed service update would be needed to activate its
4,097-block pool; the existing reboot and extended-context cautions above
still apply to the unchanged production unit.

On 2026-09-30 a further **contiguous-victim candidate** changed only the
cache's host-side KV placement; its 4,097-block pool, 4,064-block budget,
model arithmetic and 131,072-token row limit are unchanged. With A's
2,032-block prefix at blocks 16–2,047 and B's at 2,048–4,079, a third
unrelated full row needs blocks 2,048–4,095. Evicting B alone makes that
run; the prior LRU evicted the older A first and then B. Placement now tries
each unpinned victim of the same priority for a sufficient contiguous run
before falling back to LRU. A reader's pin and the shared-root-last priority
still constrain eviction. This retains A and the new C prefix; B must miss
on its next turn. It does not hold three near-full prefixes simultaneously
or admit multiple concurrent full rows.

An exact-fixture, paired physical P150x2 ABCA comparison reproduced the
prior candidate's A miss after C: **103.601 s** to first token and zero
prefix-hit tokens. On the improved candidate, A hit 130,048 tokens and
reached its first token in **0.898 s** (99.13% less; 115.42× faster for that
turn). C subsequently hit its own prefix in 0.895 s; B's subsequent revisit
missed and took 103.597 s, while A still hit in 0.888 s. A, B and C cold
turns took 103.791, 103.581 and 103.586 s versus 103.756, 103.589 and
103.534 s on the prior candidate; all three passphrase answers and generated
token IDs matched. Both candidates reported physical device IDs 0 and 1 and
23,569,736,704 allocated DRAM bytes per device view. See the
[paired physical receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-contiguous-victim-physical.json).
The new device-free regression failed against the old eviction loop and
passed after the fix; package, device-free and NixOS service builds passed.
The GDN simulator limitation and bounded-concurrency/quality caveats above
still apply.

**This improvement is not deployed.** Its physical trial used a direct
unprivileged process after isolating the root-owned service; that process
was stopped. The package
`/nix/store/1yd8km80j1sajh4ig71da7jbkns9ynni-qwen38-0.1.0` is pinned
through the user-writable `.pi/qwen38-contiguous-victim-candidate` GC root.
The original root-owned half-pool unit still targets the older package;
activation of the 4,097-block candidate needs a privileged service
override, as described above.

`ttwkv7-owner-control restore` restarted that unchanged unit after the paired
trial. Systemd reported `ActiveState=active`, PID 4142000, and the old
`/nix/store/gfvkac4zchl11g5n3xh9zbkvc4451vbl-qwen38-0.1.0` executable;
`/healthz` returned `ready: true` after startup warmup.

On 2026-09-30 a further **masked-prefill host-transfer candidate** stopped
restaging the GDN valid mask when a captured trace's bucket and valid length
are unchanged. A changed length still rewrites the mask, including a partial
tail, the next full bucket, and a structured read. Mask buffers and trace
arithmetic are otherwise unchanged. This cuts 126 redundant mask copies
from a cold 130,549-token, 128-replay prefill (128 to 2).

On physical P150x2 IDs 0 and 1, the identical cold fixture returned
`[14778, 8541, 1080]` / `amber thrush` before and after. Timed host staging
fell from 62.920 s to 11.090 s and prefill enqueue from 65.661 s to
15.479 s, but **client TTFT did not measurably improve**: 103.642 s versus
103.629 s, with completion wall times of 104.626 s and 104.803 s.
The prior mask-copy call absorbed asynchronous replay backpressure;
those call timings are not a 62-second device-transfer cost. No end-to-end
speedup or concurrency benefit is claimed.

The candidate also returned the same tokens with a 130,048-token prefix
hit, and across independent 832, 892, 832, 1,024 and 1,024-token prompts
in the same trace bucket. The mask was refreshed at the length changes
and omitted for the second 1,024-token prompt. One 800-token
`/v1/systemone` read succeeded, followed by an unchanged 1,024-token
generation answer. Package, device-free Qwen and NixOS service builds passed.
See the [physical receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-mask-reuse-physical.json).

**This candidate is not deployed.** The direct trial process was stopped.
`ttwkv7-owner-control restore` restarted the unchanged root-owned service;
systemd reported `ActiveState=active`, PID 1526612 and the old
`/nix/store/gfvkac4zchl11g5n3xh9zbkvc4451vbl-qwen38-0.1.0` executable.
`/healthz` returned `ready: true` after the required startup warmup.
The candidate package
`/nix/store/90x2sz3nqbpmc1n5pdj5d98ws05gwi40-qwen38-0.1.0`
is pinned by `.pi/qwen38-mask-reuse-candidate`. A privileged service
override remains required to activate this candidate and the preceding
4,097-block cache candidates; this mask change alone has no measured
end-to-end latency benefit.

On 2026-09-30 a **selective structured-read invalidation candidate** stopped
discarding cached generation prefixes when a read writes disjoint KV blocks.
The adapter reports the physical blocks covered by the single-read padded
bucket or every row of a batched trace; the slot coordinator removes only
prefix entries intersecting those blocks. Eager reads, lost masked traces,
and read failures still invalidate the cache conservatively.

On physical P150x2 IDs 0 and 1, an unrelated 800-token structured read
previously made a 130,564-token continuation miss its 130,048-token prefix:
TTFT 103.696 s. The candidate returned the same structured answer and
generation tokens `[14778, 8541, 1080]`, retained all 130,048 prefix tokens,
and reached its first token in 1.084 s (95.6 times faster for this turn).
A 1,400-token read overlapping the cached KV correctly evicted the prefix;
the next identical continuation missed, took 103.953 s, and returned the
same tokens. The cold 130,549-token request remained 103.697 s before and
103.649 s after. Package, device-free Qwen, and NixOS service builds passed.
See the [paired physical receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-structured-read-prefix-retention.json).

**This candidate is not deployed.** The direct physical trial was stopped;
the package `/nix/store/5j41s231rcc6p35dpr9gxjlr70r0c3d4-qwen38-0.1.0`
is pinned by `.pi/qwen38-selective-read-candidate`. The original root-owned
service remains on its older package; activation requires a privileged
override. This receipt measures one cache-hit continuation, not concurrency
or average request latency.

The restore helper's caller timed out after 120 s while the unit was
activating. Subsequent systemd inspection reported `ActiveState=active`,
PID 1797706, and the unchanged
`/nix/store/gfvkac4zchl11g5n3xh9zbkvc4451vbl-qwen38-0.1.0` executable;
`/healthz` returned `ready: true`.

On 2026-09-30 a further **cache-aware read-batching candidate** protected
generation prefixes from the captured 128- and 256-token batch traces. Each
batch bucket's KV write footprint includes all four captured rows, even unused
lanes. If any retained generation prefix occupies those blocks, queued reads
use the single-read trace in the reserved KV range instead; disjoint batches
still run together. This changes routing, not model arithmetic or KV layout.

On physical P150x2 IDs 0 and 1, A held a 130,048-token prefix at blocks
16–2,047 and B held 16,384 tokens at blocks 2,048–2,303. Four concurrent
84-token decisions previously took three batched and one single route,
overwrote B's KV, and made B's next turn miss (TTFT 9.598 s). The candidate
ran four singles; B still hit all 16,384 tokens (TTFT 0.414 s, 23.2× faster
for that turn), and A still hit 130,048. Both answers, probabilities, and
generated token IDs matched the baseline. When only A was cached, the
candidate continued batching. Cold A/B prefill latency was unchanged at
measurement precision. Three read replies in the protected burst took
about 0.90 s rather than 0.64 s; this is a latency tradeoff, not a general
throughput gain. Package, device-free Qwen, and NixOS service builds passed.
See the [paired physical receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-cache-aware-batched-read-physical.json).

**This candidate is not deployed.** The direct trial was stopped; its
`/nix/store/yx1zm7rgqknqya1bgpccx5q8i9291dar-qwen38-0.1.0` package is
pinned by `.pi/qwen38-cache-aware-read-batch-candidate`. Activating it on
the root-owned unit requires a privileged service override.

`ttwkv7-owner-control restore` returned; systemd reported the unchanged
`/nix/store/gfvkac4zchl11g5n3xh9zbkvc4451vbl-qwen38-0.1.0` service
`ActiveState=active`, PID 2149125, and `/healthz` returned `ready: true`.

On 2026-09-30 the **dedicated batched-read KV-lane candidate** reserved a
4,113-block pool: prefix blocks 0–15, generation blocks 16–4,095, four
four-block read rows at 4,096–4,111, and scratch block 4,112. Both captured
128- and 256-token batch traces use the read rows rather than generation KV.
The package is
`/nix/store/y58n6n12g680m4g9bqmd5n68smhs0ha9-qwen38-0.1.0`, pinned by
`.pi/qwen38-read-batch-lanes-candidate`; its package, device-free Qwen checks,
and NixOS service check passed the offline build.

Two initial direct launchers omitted production traced/grouped-prefill flags:
the [first attempt](../pkgs/qwen36/evidence/2026-09-30/qwen38-read-batch-lanes-first-attempt.json)
timed out before a read and required an ethernet-core board reset during
original-owner restoration; the second found no B cache hit before reads.
With the production flags verified in the candidate process, a preflight
exposed a test-fixture error (82 actual read tokens, not the asserted 84).
The next trial protected A and B through three 198-token *single* reads, but
did not exercise the 256-token batched route. Its final fixture submitted
four concurrent medium reads so three could batch after an early single.

The final direct P150x2 run on physical IDs 0 and 1 saved A's 130,048-token
and B's 16,384-token generation prefixes. B hit all 16,384 before the reads.
Four concurrent 82-token reads in the 128 bucket then took one masked and
three batched traces; four 198-token reads in the 256 bucket also took one
masked and three batched traces. B still hit 16,384 and A still hit 130,048
after **each** burst, with unchanged generated token IDs and equal answers
within each read burst. The B cold request hash differs from the historical
comparison, so these are bounded physical routing and cache-retention
measurements, **not a paired performance claim**. See the
[physical lane receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-dedicated-read-batch-lanes-physical.json).

**The candidate is not deployed.** Its direct process stopped and
`ttwkv7-owner-control restore` returned. The unchanged root-owned
`/nix/store/gfvkac4zchl11g5n3xh9zbkvc4451vbl-qwen38-0.1.0` service was
active/running at PID 3738578 with zero restarts, and `/healthz` returned
HTTP 200 with `ready: true`.

This is **not** just a read-lane change. The candidate's packaged prefix
policy permits up to 4,064 cached generation blocks, versus the running
package's 2,048, and the KV pool grows from 4,096 to 4,113 blocks. The
separate two-long-prefix budget receipt does not substitute for a paired
read-lane admission. The direct candidate launcher also omitted
`--require-systemd-notify`, so it did not exercise systemd readiness,
root-owned unit startup, or reboot persistence.

The later [paired root-systemd physical receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-read-batch-lanes-systemd-paired.json)
exercised the built
`/nix/store/hd0r2drbsacdz14vdbw3ssv8sy8j6nxf-unit-qwen38-p150x2.service`
with the same six exact request SHA-256 values on the old and candidate
packages. Root-owned `Type=notify` started candidate PID 516860, became ready
after 762 s and returned `/healthz` 200. Four concurrent 82-token reads
(128 bucket) and four 198-token reads (256 bucket) each took three batched
routes and one masked trace, with numerically identical answers between
old and new services. A and B returned identical generated token IDs across
both packages. The candidate retained B's 16,384-token and A's 130,048-token
prefix hits before and after **each** burst. The old 2,048-cached-block
package cannot retain both prefixes at once and hit zero tokens after
the bursts; its warm B continuation fell from 16,384 hits before reads
to zero after the short burst. The candidate raises that budget to 4,064,
so the observed continuation TTFT difference is **not** attributable to
the dedicated read lanes alone. This is bounded paired correctness and
cache-retention evidence, not broad throughput, long-read or model-quality
admission.

The temporary `/run` `97-read-batch-lanes.conf` was removed. The old
`/nix/store/gfvkac4zchl11g5n3xh9zbkvc4451vbl-qwen38-0.1.0` service
restored at PID 658860 with zero restarts, `/healthz` 200 and `ready: true`
after a 111.55 s warm-up; the existing 90/95/96 drop-ins were preserved.
The candidate remains undeployed. A root GC link at
`/nix/var/nix/gcroots/qwen38-read-batch-lanes-systemd-trial` retains its
package for review; it does not persist the service configuration.

The old and y58 packages in that historical comparison link the TT-Metal
0.77 candidate; the later han2 package links the same runtime. A bounded
HiFi4 GDN recurrence pass does not resolve its outstanding production
numerical gate or make 0.77 a production release.

The one-line pool-geometry comment correction in source commit
`e3cfded523e873f51da87f9ff0ba46831884f636` changed the ordinary
Git-flake Qwen output to
`/nix/store/han2q1h90mcp8ad1k9qq7bz7sl8r4rf9-qwen38-0.1.0`.
The separate [han2 physical requalification](../pkgs/qwen36/evidence/2026-09-30/qwen38-han2-committed-read-lanes-physical.json)
ran that exact output as a temporary root-owned P150x2 `Type=notify` service:
candidate PID 835106 was ready after 622 s within its 1,250 s startup bound,
and the identical six-SHA-256 fixture completed in 125 s within its 1,100 s
bound. The old `gfvkac4` and historical y58 answers came from preserved
exact-byte response files, **not** from another baseline service run in this
trial. A/B answers and generated token IDs, plus the numeric answers to all
four 82-token and four 198-token reads, matched those preserved responses.
Each four-way read burst took three batched traces and one masked trace; the
han2 service retained A's 130,048-token and B's 16,384-token prefix hits
before and after both bursts. Its 4,064-block cache budget differs from the
old package's 2,048 blocks, so continuation TTFT changes are not an isolated
read-lane speedup.

The temporary candidate was stopped, only its `/run` 97 drop-in removed, and
the old root-owned `gfvkac4` service restored at PID 964848: `Type=notify`
active/running, zero restarts, `/healthz` 200 and ready after a 104 s warm-up.
Drop-ins 90/95/96 remained byte-identical; old, y58 and han2 package GC pins
were retained. No host generation was switched. This gate qualifies only the
exact six-request physical and root-systemd path for han2, not longer reads,
broad throughput, model quality, reboot persistence or TT-Metal 0.77
production numerical admission.

### Reviewed read-lane NixOS deployment (host generation not switched)

`nixosModules.qwen38-p150x2-read-lanes` is a guarded example to import
**alongside** `nixosModules.default` in a reviewed host generation. It selects
`pkgs.qwen38`, 131,072 combined tokens, grouped and traced prefill, sampled
fusion, continuous slots, and prefix cache. The host must separately supply
its own absolute `modelPath` and `conflictingUnits`; the example does not
publish either machine-specific value. The module's service still requires
device nodes 0 and 1, starts the serialized P150x2 executable, and reads both
scheduler policies from that same package.
Importing the example alone does not switch the host generation or replace
the running service. That host configuration masks Qwen and
sets 2,048/64 tokens; its lock pins `tenstorrent-nix` at `cbfcab9`, not a
revision containing both the reviewed read-lane source and the qualified
han2 profile guard (`e3cfded` alone retains the stale y58 guard). The source
and guard are committed here, but no reviewed host input or generation has
selected **han2**. Deploying that candidate through NixOS still requires a
pinned host-generation source selecting its exact requalified package while
preserving other host settings. Review owner conflicts, model path, driver,
firmware and OMP separately. The earlier 65,536-token boot command and
`/run/systemd` overrides alone do **not** provide durable 131,072-token
service after reboot; the separately managed han2 package profile below now
provides a persistent effective command under the *current* native unit.

For a host NixOS configuration, the smallest reviewable integration is:

1. Pin the final reviewed `tenstorrent-nix` flake revision containing this
   requalified package **and** updated profile guard; update the host lock
   from `cbfcab9` only after comparing the entire host closure. Pinning
   `e3cfded` alone would retain the old y58 guard, and a local store path
   alone is not a reproducible host-generation pin.
2. Import `inputs.tenstorrent-nix.nixosModules.qwen38-p150x2-read-lanes`
   beside the existing default module. Replace the host's explicit
   `hardware.tenstorrent.qwen38` attrset (including its
   `package = tenstorrentPackages.qwen36`, 2,048/64 limits) with that
   profile while setting `modelPath` to the host's reviewed snapshot and
   `conflictingUnits` to the existing competing owner services in the host
   configuration. The profile fails closed if either the model path or
   competing-owner list is absent. Retain the host's unrelated Tenstorrent
   driver, descriptor, mesh and access settings. Update the host's
   maximum-sequence and maximum-generation documentation to 131072 and 8192.
3. Remove `systemd.services.${qwenServiceName}.enable = false` and the
   stale host README text saying Qwen is masked or greedy-only. The
   profile now asserts against a masked unit or effective 2,048/64
   limits, including stronger host overrides; it does not unmask by
   silently overriding an operator's setting.
4. Preserve OMP's ordinary `contextWindow: 65536`, optional
   `maxContextWindow: 131072`, `maxTokens: 8192` and
   `extendedContext: false`. Raising the OMP budget is neither necessary
   nor evidence that the server can serve 131,072 tokens after reboot.

These are instructions for a **future reviewed host edit**, not edits made
to `onix-core`. A profile imported into its current host config is not
deployable: same-priority limits/package definitions conflict, while a
forced 2,048/64 override or the current mask fails the profile assertions.
The profile selects the host's overlaid `pkgs.qwen38` derivation and fails
closed unless its output is the requalified
`/nix/store/han2q1h90mcp8ad1k9qq7bz7sl8r4rf9-qwen38-0.1.0`.
This is an admission invariant, not a store-path package fallback. A host
with different `nixpkgs` or extra overlays can produce a different output;
the local flake result does not qualify that host package. Compare the
*rendered host unit* and package path against han2, and retain the verified
old-package rollback generation.

Before the source commit, the untracked example was missing from the
Git-filtered flake, while a `path:` checkout included extra local files and
resolved an unqualified `q291n7j5q72bdh8fllkkd2l6pisp31c0` package. Those
were pre-commit diagnostics, not the current source identity. Commit
`739abbf` added the reviewed example and y58-matched source; the one-line
cache-policy geometry comment correction in `e3cfded` changed the package
output. From `e3cfded`, both the ordinary Git flake and a revision-pinned
`git+file:` evaluation select the exact `han2q1h90mcp8ad1k9qq7bz7sl8r4rf9`
package. Offline package, device-free Qwen and NixOS service checks built.
The separate NixOS profile asserts the exact han2 output; a different
package fails closed. A flake export and local evaluation are not a pinned
host input, a rendered host generation or an activation. Future source
changes that alter the output require their own paired physical gate before
the guard can change again.

In the current publication-prep worktree, formatting and import-order fixes
change the Qwen package's source identity. With a host model path and competing
units supplied, evaluation still rejects the package at the exact-output
assertion. That is intentional: the historical han2 receipt does not qualify
the modified output. Requalify the new package physically before changing the
guard or selecting it in a host generation.

Before a privileged activation, evaluate/build the **reviewed host generation**
and inspect its rendered `qwen38-p150x2.service`, not just this example. For
`e3cfded`, `nix eval --offline --raw
.#packages.x86_64-linux.qwen38.outPath` resolved to
`/nix/store/han2q1h90mcp8ad1k9qq7bz7sl8r4rf9-qwen38-0.1.0`; reject a
different output unless separately requalified. Verify the generated
`ExecStart`, `Environment`, device-node conditions and owner conflicts:
`--max-sequence-length 131072`, `--require-systemd-notify`, both policy
arguments from this package, `MESH_DEVICE=P150x2`,
`QWEN35_GROUPED_PREFILL=1`, `QWEN36_TRACED_PREFILL=1`,
`QWEN36_SAMPLED_FUSION=1`; ensure the effective host generation does not
mask the unit. The existing device-free package and service checks are build
gates, not physical admission. Physical IDs 0 and 1 must still be checked
against the actual serving process after activation.

The **historical y58 paired** systemd gate measured the old `gfvkac4` package
on the same exact six-input fixture and exercised root-owned notify readiness.
The later **han2 same-fixture requalification** compared its newly captured
answers to preserved exact-byte old and y58 responses; it did not rerun the
old baseline. Both are bounded correctness and cache-retention gates. The
changed prefix budget is part of the comparison, so do not credit continuation
TTFT differences to read lanes alone. Neither qualifies longer reads, general
throughput, 128K structured-read quality, a reviewed host generation or
TT-Metal 0.77 production admission.

For a future **han2 host-generation deployment** (not performed), preserve
the old package
`/nix/store/gfvkac4zchl11g5n3xh9zbkvc4451vbl-qwen38-0.1.0` under
the existing `/nix/var/nix/gcroots/qwen38-prefix-128k-budget-candidate`
root. Profile generation 1 was verified at the cutover but is not a durable
GC pin: weekly `fast-nix-gc --delete-older-than 14d` can remove its link.
A separately reviewed host generation still needs to select the exact
qualified han2 output, keep this owner's 131,072/8,192 settings, remain
enabled and unmasked, and preserve device/model settings. The old
65,536-token native command is not a suitable 128K rollback if the managed
profile is absent.
Retain all three package GC pins: the old gfvk root, historical y58
`/nix/var/nix/gcroots/qwen38-read-batch-lanes-systemd-trial`, and han2
`/nix/var/nix/gcroots/qwen38-han2-physical-requal`.

`ttwkv7-owner-control` isolates/restores device owners; the managed script
below installs or rolls back a **package profile and its Qwen-only unit
drop-in**, not a NixOS generation. While the original `/run`90/95/96
drop-ins exist, `restore` can temporarily recover their old gfvk command.
After reboot, `install` can select the old pinned output only from an
already healthy managed owner; a failed-start profile needs the manual
verified-generation rollback described below. A failed candidate admission
during `install` automatically restores its previous generation and owner.
If a future NixOS switch masks Qwen or changes the native owner contract,
fix or roll back that reviewed host generation first; the profile drop-in
cannot override a mask. Do not substitute an unmanaged persistent override.
OMP's 131,072 `maxContextWindow` remains unsafe whenever the effective
service has fallen back to 65,536 or is not ready. The actual reboot has
not been exercised.

A continuous-slot scheduler is available behind a flag. With
`--continuous-slot-policy $out/share/qwen36/continuous-slot-policy.json`, in
candidate decode mode, it replaces the group coordinator at construction. It keeps
one B4 decode trace resident and refills a slot at the next window boundary once
its request ends. Queue, refill, fairness and stall bounds come from
`continuous-slot-policy.ncl`. Drift or a stalled device effect quarantines the
engine, and `/healthz` then answers 503.

On 2026-09-27 it was compared with the group coordinator on one package, using a
staggered workload of two 1,024-token requests and six short ones. The receipt
`pkgs/qwen36/evidence/2026-09-27/qwen38-continuous-slot-admission-receipt.json`
admits the profile against its predeclared gates:

- Short requests finished in 3.5–3.8 s at the median instead of about 52 s.
- Delivered tokens/s rose 12 %.
- The long requests took 20 % longer, because each refill pauses the live streams
  for its prefill. The longest pause was 5.5 s, for a 9.6K-token prompt.
- Every request decoded the same tokens as when it ran alone.

At that earlier trial, the service was not switched: serving the profile by
default traded pauses for the queue-time gain. The current root profile
does select the continuous-slot policy, but that trial is not a new live
throughput measurement. The change
`.cairn/changes/qwen38-continuous-slot-scheduler` records the evidence.

The scheduler also has a B=8 profile, exported as
`continuous-slot-policy-b8.json` and selected with
`--continuous-slot-policy $out/share/qwen36/continuous-slot-policy-b8.json`. Its
closed capacity set is {4, 8}. On 2026-10-21 it was compared with the group
coordinator on the same staggered workload, with configuration-matched roots. The
receipt
`pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json`
admits the profile against its predeclared gates:

- Short requests finished in 3.3 s at the median instead of 51–55 s, and their p95
  fell from about 61–65 s to 12.6 s.
- Delivered tokens/s rose 11 %.
- The long requests took 19 % longer (p95 ratio 1.187 against a 1.3 bound), and the
  longest pause was 5.542 s against a 6 s bound.
- Every request decoded the same tokens as when it ran alone, and the width-8 trace
  stayed resident: no decode trace was captured during the measured passes.
- Both roots left the same device DRAM free after startup.

A first attempt at that comparison was rejected by the startup-DRAM gate because
the candidate root alone ran with the prefix-cache policy, whose saved state slots
cost 365 MB; the gate was not changed and the matched rerun is the receipt above.
The B=8 profile is admitted for the package and execution identity named in the
receipt. The root profile still selects the B4 policy, and the comparison covered
one workload without the prefix cache. The change
`.cairn/changes/widen-qwen38-continuous-slots-b8` records the evidence.

The NixOS module now names the profile explicitly:
`hardware.tenstorrent.qwen38.continuousSlotProfile` takes `b4` or `b8` and refuses
every other value, and the service passes the matching packaged policy. On
2026-10-21 the admitted package was served with the B=8 profile at the served
limits (131,072 context, the prefix cache enabled) on devices 0 and 1: eight
concurrent requests were answered, the service reported the admitted execution
identity and the `qwen38_p150x2-continuous-slot-b8-v1` profile, one slot was
refilled inside the same run, and the client-observed walls were 0.77 s to 2.73 s.
The managed profile and the systemd unit were moved to the B=8 policy with the
profile service's `switch-policy` operation: the named profile holds the admitted
package, the managed drop-in names `continuous-slot-policy-b8.json`, and the live
unit on `127.0.0.1:8000` reported the admitted execution identity and the
`qwen38_p150x2-continuous-slot-b8-v1` profile under eight concurrent requests. The
unit is stopped after that check and starts on the B=8 profile when it is started.
The change `.cairn/changes/widen-qwen38-continuous-slots-b8` records the admission
evidence and `.cairn/changes/select-qwen38-continuous-slot-b8` records the
deployment. `docs/qwen38-served-deployment.md` is the standing procedure for moving
the served service between packages, for the privilege the move needs, and for
measuring while the service runs.

The deployed configuration also keeps the hybrid prefix cache, whose own receipt
covered neither that package nor the eight-slot profile. A schema-5 prefix-cache
admission policy now names the scheduler profile every root must have served, and
the root identity records the profile the service reported. On 2026-10-21 the
four-root ABBA comparison ran again on the current package with the eight-slot
profile at the served limits, and the receipt
`pkgs/qwen36/evidence/2026-10-21/qwen38-prefix-cache-abba-admission-receipt.json`
admits the cache profile `qwen38-p150x2-prefix-cache-v1`:

- A resumed turn's first token took 0.41 of a cold turn's, and the whole request
  took 0.57 of it.
- A miss cost 1.006 of a cold turn at the median and 1.07 at the worst.
- Every turn decoded the same tokens in all four roots, and both cached roots
  resumed the same prefix on every turn.
- The four saved states cost 364,904,448 B of device DRAM, inside the 512 MiB bound.

The eight-slot scheduler admission was measured again on that same package, so one
package carries both receipts. The change
`.cairn/changes/readmit-qwen38-prefix-cache-b8` records the evidence.

A request without `temperature` decodes greedily, including thinking. On
2026-09-27, eight reasoning prompts with known answers were run at the
template's default effort (`xhigh`) with `max_tokens` 8,192. Greedy (one
batch) and temperature 0.6 (sampled, two batches of four) each answered all
eight correctly, none reached the token cap, and neither repeated itself. The
median length was 181 and 179 tokens. Sampled decode ran at 11.5 tokens/s per
user against 21.8 greedy, so greedy stays the default. Clients that want
sampling send a temperature.

Both completion routes accept `"stream": true` and answer with server-sent events.
The first chat chunk carries the assistant role. Deltas follow as the batch worker
decodes each token; the detokenizer decodes only the tokens since its last output,
and a character split across tokens is held back until it is complete. One chunk
then carries the finish reason and the decode evidence. With
`"stream_options": {"include_usage": true}` a usage chunk follows, and the stream
ends with `data: [DONE]`. A request rejected before generation, for example one
over the sequence limit or one the chat template refuses, still gets a plain JSON
400. The MTP and eager routes send their whole text when generation ends.
`max_completion_tokens` is an alias of `max_tokens` (both must agree). Without
either, a request may generate up to `--max-generation-tokens` or the room its
prompt leaves in `--max-sequence-length`, whichever is smaller.

Chat requests follow the OpenAI tool-calling contract. `tools` are rendered by the
model's chat template. `tool_choice` `auto` (default) and `required` offer every
tool, a named function offers only that tool, and `none` offers none and turns
off call parsing; nothing constrains decoding, so `required` and a named choice
are best effort. `parallel_tool_calls: false` keeps only the first call. History
may hold `system` (first only), `user`, `assistant` and `tool` messages: assistant
`content` may be empty or null, `reasoning_content` is replayed, `tool_calls`
arguments may be a JSON string or an object, and tool results follow an assistant
or tool message. Content may be a list of text parts, joined as the template joins
them. `enable_thinking`, `preserve_thinking` and `reasoning_effort` are read from
the top level first, then from `chat_template_kwargs`; no other template argument
is forwarded. Efforts map `minimal`/`low` to `low`, `medium` to `medium` and
`high`/`xhigh`/`max` to `xhigh`. Thinking stays on unless disabled. Output before
`</think>` becomes `message.reasoning_content`, the text after it `content`, and
each `<tool_call>` becomes a `tool_calls` entry whose `arguments` is a JSON object
string typed from the tool's parameter schema; text around calls stays content,
and `content` is null when only calls remain. The stream carries the same result
as `reasoning_content`, `content` and indexed `tool_calls` deltas (id and name
first, then argument fragments). `finish_reason` is `tool_calls` only when
generation ended on the stop token with every call closed, `length` whenever the
token budget ran out (a truncated call is never reported as `tool_calls`), and
`stop` otherwise. `/v1/completions` returns the raw text.

The B=4 admission receipt used exact `single,batch,batch,single` order on the
Python 3.14 deployment runtime. It preserved every generated token, text,
state slice, and reset. Aggregate decode increased by `3.5475913484735178`.
Per-user retention was `0.8868978371183794`, TTFT ratio was
`3.1320421887440983`, and tail ratio was `1.066116770546012`. The receipt
binds the stable execution identity and BLAKE3
`2bf7d74e784d2c5aecfc60a4252f7141955faf1c735d25b94dbc3d832b84b136`.
The separate single-stream eager-versus-traced profile remains unadmitted.

At the time of that older deployment, this host used a lingering per-user
`qwen38-p150x2.service` on `127.0.0.1:8000`; its previous inference services
were stopped while Qwen owned the devices, and a live 64-token request passed.
The current owner is the root native NixOS service with a managed package
profile, documented next.

## B=8 batch admission on the fused step route (2026-10-21)

The frozen four-root B=8 protocol was re-run unchanged on the current package
(`5jk1x79sd33a5mfgdvvny8kzsbsm97ql-qwen38-0.1.0`, execution identity
`hmn4z5n63vlz7prlva2kispp8d6wlll5-qwen36-execution-identity-0.1.0`) with the
`step` GDN decode route. Every correctness gate passed and all four performance
bounds passed:

| Gate | Measured | Bound |
| --- | ---: | ---: |
| Aggregate decode speedup | 7.408 | at least 6.0 |
| Per-user decode retention | 0.9260 | at least 0.75 |
| TTFT ratio | 7.670 | at most 9.0 |
| Tail-latency ratio | 1.078 | at most 1.5 |

The protocol was the existing frozen one: roots in `single,batch,batch,single`
order, the eight-prompt corpus, one warmup plus three measured runs per prompt,
64 generated tokens, and physical PCIe devices 0 and 1. Exact token parity,
exact text parity, distinct state slices, active prefix, and clean reset all
report true. The receipt is
`pkgs/qwen36/evidence/2026-10-21/qwen38-27b-batch-admission-b8-receipt.json`
(BLAKE3 `f541b692a8a7696d60e0ffaf512b2861338439a49ad53d3356600f7def97eed9`);
the binding is admitted and the install phase pins the exported profile to that
receipt.

The route is what changed. The 2026-09-12 rejection measured retention 0.737 on
the fused TT-Metal operation path; the step route's retention is 0.926, far
outside the protocol's run-to-run spread against the 0.75 bound. The
protocol-widening task was therefore not needed for this verdict, and the frozen
bounds were not edited.

The width was later re-measured under the v2 per-user TTFT statistic (the same
statistic B=16 uses): speedup 7.408, retention 0.926, per-user TTFT ratio 4.521
against 9.0, group-max ratio 7.777 as a diagnostic, tail 1.080, with exact parity
throughout. The admitted receipt is now
`pkgs/qwen36/evidence/2026-10-21/qwen38-27b-batch-admission-b8-receipt.json`
(BLAKE3 `44960a77df0c7666624f4b0ae2e1f155f8b8861ad3de95b33886bcd358f398ab`); the
v1 group-readiness receipt remains at `…-b8-receipt-v1-group-ttft.json`.

This admits candidate width 8 on the recorded execution identity only. It does
not enable batched serving in production and does not admit arbitrary
continuous batching.

### B=16 admission on the per-user TTFT statistic (2026-10-21)

A first re-screen under the v1 group-readiness statistic failed TTFT alone: 14.25 against
9.0, because batch TTFT was sixteen serialized per-user masked prefill replays (3.384 s)
against one single replay plus a decode step (0.237 s). The ratio was approximately the
batch width by construction; eager grouped and per-user prefill measured 33.4 s and 29.2 s
medians, so no route switch could clear it.

The v2 protocol gates on each user's own time to first token instead of the group's last
user, with the same corpus, order, run counts, and bounds. The service now probes each user's
masked prefill replay completion (one element of that slot's logits row) and reports that
user's TTFT; the group's last-user value is kept as `ttft_group_max_ratio` without gating.
The masked-trace probe programs are warmed for every slot before capture, so the traces stay
resident.

The frozen four-root protocol then passes every gate:

| Gate | Measured | Bound |
| --- | ---: | ---: |
| Aggregate decode speedup | 13.398 | at least 6.0 |
| Per-user decode retention | 0.837 | at least 0.75 |
| TTFT ratio (per-user median) | 8.599 | at most 9.0 |
| TTFT group-max ratio (diagnostic) | 15.601 | not gated |
| Tail-latency ratio | 1.240 | at most 1.5 |

Exact token and text parity, active prefix, clean reset, and distinct state slices pass.
Medians: single 23.57 aggregate tokens/s and 0.212 s TTFT; batch 315.80 aggregate and
19.74 per user, 1.822 s per-user TTFT, 3.450 s group first token. The receipt is
`pkgs/qwen36/evidence/2026-10-21/qwen38-27b-batch-admission-b16-receipt.json`
(BLAKE3 `986f729b46bd24e11f6cf65cae613e50bb082ba2ef0e602f620c31ac71150dc2`); the v1
rejection remains at `…-b16-receipt-v1-group-ttft.json`.

The per-user ratio passes by 0.40. The group-max ratio remains above the bound because
prefills are still serialized; a batched masked prefill replay is the proper performance
fix and would remove that property. This admits width 16 only on the recorded identity and
does not enable batched serving. Both widths are now admitted under the same v2 per-user
TTFT statistic.

## Managed root package profile (historical host, 2026-09-30)

The live `qwen38-p150x2.service` remains the **enabled, unmasked native
NixOS service** at `/etc/systemd/system/qwen38-p150x2.service`, inheriting
`Type=notify`, root/tenstorrent, P150x2 device conditions, model snapshot,
conflicting owners and the production grouped/traced/sampled GDN flags. Its
native `ExecStart` still names a 65,536-token package. A root-owned managed
drop-in at
`/usr/local/lib/systemd/system/qwen38-p150x2.service.d/99-profile.conf`
changes **only** `ExecStart` to
`/nix/var/nix/profiles/qwen38-p150x2/bin/qwen38-p150x2-serve` with
`--max-sequence-length 131072`, `--max-generation-tokens 8192`, the
unchanged model, notify, continuous-slot and prefix-cache policy flags.
The named root Nix profile now contains only the physically requalified
`/nix/store/han2q1h90mcp8ad1k9qq7bz7sl8r4rf9-qwen38-0.1.0` at
generation 3. Its binary and both policy symlinks resolve to that output.
At admission, profile generation 1 contained only the old `gfvkac4`
package, which remains pinned by its separate GC root. Check that the
generation link still exists before relying on it for rollback; weekly
`fast-nix-gc --delete-older-than 14d` can remove old generations. The
[2026-10-01 admission receipt](../pkgs/qwen36/evidence/2026-10-01/qwen38-han2-profile-admission.json)
records the switch without a host-generation change.

The exact root-only profile manager and systemd 99 drop-in used for this
cutover are host-specific operational files, not distributed as public
repository tools. They remain in the owner's local checkout. Their historical
commands are **rollback only while han2 is healthy**, not a failed-start
recovery procedure; do not reconstruct them from this account on another host.

The private manager checks the current owner, pinned package, model and devices,
stages the candidate profile before isolating the owner, serializes the
switch, waits for systemd notify and HTTP `/healthz`, and rolls back to the
previous profile generation and owner if admission fails. The separate
`restore` command removes **only** its managed 99 drop-in, keeps the root
profile for reinstallation, and returns to the exact old `/run`96 command
**only while all three `/run`90/95/96 drop-ins are still present**. It is
not a post-reboot recovery command. `install` can select the pinned old or
qualified han2 output only while the current owner is **already active and
healthy**: either the managed 99 owner or the exact restored native `/run`96
gfvk owner with a verified single-package profile. It cannot recover a
failed-start or quarantined service; its readiness preflight rejects the
current owner before touching the profile. A host switch that masks or
changes the native unit instead needs a reviewed host-generation correction.

For a future qualified update, build from a reviewed immutable source, not
the dirty host checkout. Use the owner's private manager to check and install
that exact output only after a fresh physical qualification; the public flake
does not ship the host-specific migration or rollback tool.

Do not rebuild `onix-core`. The exact han2 output was admitted through the
named package profile, **not** through a new NixOS host generation. Future
package outputs still need their own physical qualification before `install`.
For any approved package, check systemd notify readiness and HTTP 200
`/healthz`, the effective profile path and both policy symlinks, and sole
ownership of `/dev/tenstorrent/{0,1}`. The old package's `READY` token
`[44061]` and one-read [cutover receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-profile-owner-cutover.json)
describe historical gfvk behavior, not a new package's acceptance.

If the service **fails to start or becomes unhealthy** while the native owner
remains enabled/unmasked and managed 99 remains intact, do not use `install`
or `restore`: both require a currently ready owner. Inspect `nix profile
history --profile /nix/var/nix/profiles/qwen38-p150x2` and `nix profile list
--json --profile /nix/var/nix/profiles/qwen38-p150x2`. The transaction
passed through **empty generation 2**; an unqualified `nix profile rollback`
would select that empty profile, not gfvk. Before `--to 1`, verify
`/nix/var/nix/profiles/qwen38-p150x2-1-link` still exists and `nix profile
list --json --profile /nix/var/nix/profiles/qwen38-p150x2-1-link` contains
exactly the pinned gfvk output. Generation 1 is historical evidence, **not**
a durable recovery path: weekly `fast-nix-gc --delete-older-than 14d` can
remove its link. Only if that generation is present and verified, stop the
Qwen unit, run `nix profile rollback --profile
/nix/var/nix/profiles/qwen38-p150x2 --to 1`, restart the same unit with
managed 99 intact, and check HTTP 200 `/healthz`, the effective
131,072/8,192 command and sole physical device ownership. If the link is
absent or wrong, **do not run `--to 1` or restart into an unverified profile**.
Fail closed; the old package is retained by
`/nix/var/nix/gcroots/qwen38-prefix-128k-budget-candidate`. Review a
recovery that puts that exact package into a sole-package named profile,
verifies its wrapper and both policy symlinks, and only then restarts the
same enabled native unit with managed 99 intact and checks health, effective
command and device ownership. Do not assume a missing generation can be
rolled back into existence. If the native unit is masked, revert/fix the
reviewed host generation first.

While the original `/run` rollback drop-ins still exist, the bounded private
`restore` operation can temporarily select the `/run`96 gfvk owner **only from
a healthy service**. This is not a public deployment command.

The manager's restored-owner admission check permits `install` of the same
han2 package (or the verified old gfvk package) from a healthy native `/run`96
owner with a valid single-package profile. With no existing named profile,
initial migration still requires gfvk. This `restore` is **not** a post-reboot
recovery path; retain the old GC root and verify generation 1 exists before
depending on it.

At the historical 2026-09-30 gfvk cutover, the first profile install reached
systemd-ready PID 1485839 and HTTP 200
`/healthz`; the bounded `restore` then returned to native `/run`96 at
ready PID 1519419. Reapplying the profile reached PID 1547052, Type=notify,
zero restarts, and HTTP 200 ready after startup warmup. Both devices 0 and 1
had that one owner. Final short chat returned HTTP 200 `READY`, token
`[44061]`, physical IDs `[0,1]`; the unchanged 114-byte short
`/v1/systemone` fixture returned HTTP 200 and one device read. The old
`/run`90/95/96 file hashes did not change. A live `systemctl daemon-reload`
kept PID 1547052, the profile-backed 131,072-token command, and ready
`/healthz`; `systemd-analyze verify --generators=no` returned zero with
unrelated host-unit warnings.

On systemd 261, `systemd-analyze unit-paths` includes
`/usr/local/lib/systemd/system`. The root-owned 0644 99 drop-in resides
on persistent ext4 `/usr/local`; the named root profile resides on ZFS
`/nix`, not tmpfs `/run`. An isolated synthetic systemd root with the
byte-identical current native fragment and checked-in `/usr/local`99
but **no `/run` drop-ins** passed `systemd-analyze verify`; removing only
its fixture profile executable made verification identify that profile
`ExecStart` specifically. These are an offline no-`/run` extrapolation
and live daemon-reload observation, **not an actual reboot test**. After
any reboot or NixOS switch, recheck `systemctl is-enabled`, effective
`ExecStart`, package symlinks, notify readiness and `/healthz` before
using OMP extended context. In particular, the unrelated dirty
`onix-core` host configuration masks Qwen and changes its limits; do
not switch it expecting this drop-in to unmask the service. See the
[dated profile-owner receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-profile-owner-cutover.json).

This dated cutover receipt is a compact historical summary, not a retained
raw trace or a fresh test of the checked-in manager's first migration.
The current manager's restored-owner `check`, `restore`, and han2 reinstall
were later exercised live; its first migration and cross-package switch
paths were checked offline only. Neither proof substitutes for a reboot.

On 2026-10-01, the exact immutable source revision
`2515a93006b4968264c3863cbce07ba29591ec16` resolved to the already
qualified han2 output. The managed profile transaction replaced generation
1 (gfvk) with generation 3 (han2); the unchanged enabled native root unit,
managed 99 drop-in, `/run`90/95/96 hashes, model, 131,072/8,192 command
and production flags stayed in place. Root `Type=notify` became ready at
PID 2173017 with zero restarts; this process alone held physical device
nodes 0 and 1. The profile's wrapper, both policy symlinks and execution
identity resolve to han2. At admission, old generation 1 was present; the
old package also remains pinned by its GC root. Generation 1 can later be
pruned by weekly GC.

The **one** live six-body fixture replay used the preserved request SHA-256
manifest and a disposable driver (SHA-256
`0ad45f6905773bc65a3ed248cfe8a534952d3688f785a2933dea7ae17c411802`).
The [compact admission receipt](../pkgs/qwen36/evidence/2026-10-01/qwen38-han2-profile-admission.json)
points to the retained raw `/tmp/qwen38-han2-replay.OOEYeK72/receipt-20261001T010306Z-udicdsuw.json`
(SHA-256 `c7ecf8e1f5e0affc5dd88428a2f28a65a79429c955e5eb5066b7df2fd01c308f`).
The replay completed in 127.167 s with 98/98 gates passing, all HTTP 200
and ready health before and after. Cold A/B TTFT was 103.709/9.626 s,
essentially unchanged from the **preserved earlier** gfvk 103.701/9.607 s.
After the 82-token read burst, A hit 130,048 tokens at 1.102 s versus the
old zero-hit 103.758 s; B hit 16,384 at 0.426 s versus old 9.600 s.
After the 198-token burst, A/B still hit their prefixes at 1.096/0.440 s.
Both four-way bursts took three batched and one masked route, matched the
old numeric read answers, and A/B generation kept the exact token IDs.
The independent live driver staggered four short reads 0.08 s behind
the in-flight A continuation; the deleted historical root-systemd driver's
exact scheduling was not preserved. These are comparisons to responses
saved from previous service runs, **not** a simultaneous production A/B.
The larger 4,064-block cache budget and dedicated read lanes changed
together, so the continuation benefit cannot be attributed to lanes alone.
There is no claim for general throughput, longer reads, broad quality,
an actual reboot or TT-Metal 0.77 production numerical admission: TTsim's
fused GDN numerical gate still cannot model its matmul control flag.

The corrected profile manager then passed an offline restored-owner transition
check and a bounded **live** `restore`/reinstall cycle. Native `/run`96 gfvk
returned ready at PID 2471602 while the named profile retained han2 generation
3. `check han2` identified gfvk as the effective owner and han2 as the
retained profile package; reinstall of the **same** han2 output returned
root `Type=notify` ready at PID 2601665, generation 3, with no restarts.
Both transitions left the original `/run`90/95/96 hashes unchanged, and
each owner alone held devices 0 and 1. After reinstallation, one short chat
answered `READY` with token `[44061]` and physical IDs `[0,1]`; one
82-token structured read returned HTTP 200, the same `urgent.noul`
`0.000552778597921133` and one device read. The six-body cold fixture was
**not rerun**, so this recovery proof adds no new cold or throughput claim.
See the restore/reapply section of the same compact admission receipt.

## Physical cold-prefill SDPA diagnosis and rejected tile trial (2026-10-01)

An isolated, profiler-enabled run on physical P150x2 nodes 0/1 measured
130,549-token A refill at 103.615 s and 16,684-token B at 9.626 s. A captured
3,188,925 RT records with zero callback drops. The historical **raw,
signature-labelled dispatch intervals** on chip 1 assign 27.839 s across
2,048 SDPA-labelled records (chip 0: 27.764 s) and 6.638 s across 8,192
concat-labelled records; binary SFPU and matmul labels total 10.099 and
9.314 s. These labels do **not** reliably attribute time after a 16-bit
runtime-ID wrap (see below), and dispatch intervals include worker waits,
not just per-core kernel execution. Chip 1's raw program durations total
102.625 s. Two-chip durations must not be added; host enqueue/staging
overlap device work. B's historical raw SDPA label totaled 0.516 s per
chip across 272 calls. These observations do not admit an optimization.

A package-matched TT-Metal 0.77 experiment kept 128 query rows but widened
only captured flexible paged-prefill SDPA's key chunk from 128 to 256. Its
first bounded startup timed out during cold JIT and masked-trace capture,
then automatically restored the original owner. The second direct-server
run completed the preserved exact-six-body physical HTTP replay, but passed
only **94/98** checks: each of four 198-token structured reads returned
`urgent.noul=0.0013250223128125072`, not the admitted exact
`0.00247262348420918`. Short-read scores, generated token IDs, physical
IDs, prefix hits and the other gates matched. Candidate cold A/B TTFT was
103.193/9.591 s versus 103.709/9.626 s in a **separate** prior han2 run
(saved cold profile 103.899/9.678 s); the 0.516 s A difference is ~0.5%
from one sample, not evidence of a material improvement. Changing the key
tile is numerically inadmissible. The candidate patch and Nix reference
were removed; it was never installed into the managed profile.

Both isolation attempts used invocation-bound bounded post-stop rollback.
After the failed replay, the original han2 wrapper returned at root
`Type=notify` active/running PID 223473, `/healthz` HTTP 200 and no restarts;
that PID alone owned nodes 0 and 1. Native, managed and all three `/run`
drop-in hashes stayed unchanged. The
[dated physical diagnosis and rejection receipt](../pkgs/qwen36/evidence/2026-10-01/qwen38-cold-prefill-sdpa-rejection.json)
links the profiler's raw records, diagnostic/remap/98-gate script hashes, full
failed HTTP responses and both rollback receipts. The SDPA-associated
dispatch interval remains an algorithmic target, but package-matched
`tp.py` already uses its full BH grid and 128×128 query/key chunks. The
tested wider key tile changed exact read numerics without material
end-to-end benefit. A second source/record pass found 2,336 occurrences
of the binary-SFPU signature **per replay chunk** in both A
(299,008/128) and B (39,712/17), or 146 per four-layer group
(three GDN, one full attention). RT records identify kernel sources
and timestamps, not TTNN operation names; assigning the 10.099 s to
one Python callsite would be fabrication. The actual grouped trace
uses trained q/k norms, attention sigmoid gates, GDN decay/beta/causal
masks, MLP SwiGLU and residual additions. Its GDN valid-mask
multiplies happen to see all ones on full chunks, but the **same**
captured program reads the staged partial-tail mask; its two GDN
state-reset selects likewise must reset the first chunk and retain
continuations. Removing either from only the full chunks needs a
different trace/state protocol, not an exact free operation.

The actual concat-heads kernel still performs a head/sequence permutation:
each captured chunk has 64 concat launches, converting 48 GDN and 16
attention head-major outputs to token-major gate/projection inputs. The
trained model has 48 GDN value heads (24 per device), so this is not a
shape-only view. But the apparent **6.309 s** across the last three
concat-labelled positions is **not** evidence of slow concat. In pinned
TT-Metal 0.77, the trace preamble truncates the runtime ID to 16 bits
(`tt_metal/impl/program/dispatch.cpp:3011-3017`); dispatch_d omits ID
zero from its FIFO (`tt_metal/impl/dispatch/kernels/cq_dispatch.cpp:1422-1436`)
while dispatch_s pops an ID before each command, including GO
(`cq_dispatch_subordinate.cpp:608-629`). The RT receiver drops zero IDs
(`cq_realtime_profiler.cpp:37-60`); a GO waits for preceding workers
(`cq_dispatch_subordinate.cpp:255-280,407-414`). The saved offline A/B
remap (`/var/tmp/qwen38-prefill-program-20261001/attribution/rt-adjacency-attribution.json`;
compact `/var/tmp/qwen38-prefill-program-20261001/attribution/conclusion.json`)
finds 65535→1 at position 10667 in both chip templates, so ID 0 is never
recorded. Every traced chunk shows a one-record timing/label shift at the
final three attention groups, and pinned source implies it covers all 1,789
of 12,456 records from that position on. Their long concat-*labelled*
windows start ~0.12 µs after the short SDPA-labelled windows and match
earlier SDPA windows in duration and context scaling; the no-wrap eager
tail keeps normal SDPA-long/concat-short order. **[INFERENCE]** The
concat-labelled windows themselves carry the preceding SDPA program's
dispatch/worker-completion interval, not a wait after it. Conditional on
that inferred +1 relabel, which keeps the unreported ID-zero window
separate, the per-chip labelled dispatch-interval sums in seconds, raw →
corrected, are:

| Run/chip | SDPA (53) | Concat (46) | Binary SFPU (31) | Matmul (32) |
| --- | ---: | ---: | ---: | ---: |
| A/0 | 27.764225 → 34.054059 | 6.620669 → 0.361539 | 10.052724 → 9.740292 | 9.308440 → 9.909817 |
| A/1 | 27.839023 → 34.146082 | 6.637673 → 0.361301 | 10.099433 → 9.790045 | 9.313915 → 9.914078 |
| B/0 | 0.515945 → 0.619886 | 0.147856 → 0.047774 | 1.307159 → 1.268087 | 1.221005 → 1.296819 |
| B/1 | 0.515688 → 0.619589 | 0.147776 → 0.047739 | 1.307446 → 1.268546 | 1.220828 → 1.296743 |

**[INFERENCE]** The corrected column follows pinned source and the trace
fingerprint; there is no FIFO-phase record or new physical probe. Each
chip's integer-tick sum is unchanged; only labels move. The corrected sums
are not complete per-kernel totals: the ID-zero windows (127 per A chip, 16
per B chip) stay a separate bucket; each replay chunk's last program window
and five unrelated A slice records were never recorded; eager-tail labels
stay as observed, although its first interval may span the replay boundary
and labels before its first empty-FIFO pop may be +1. The corrected
SDPA-associated windows include launch and worker waits, not just per-core
SDPA compute; they do not isolate compute from NoC/DRAM stalls. The
concat-versus-SDPA label question needs no further physical probe, while
compute-versus-stall would be a separate investigation.

No source- and device-supported low-risk exact-output-preserving
material elimination was found; a new kernel/layout algorithm needs
its own physical numerical and 98-gate proof. No new optimization or
broad quality/throughput admission is claimed.

### Isolated SDPA per-core counter diagnosis (2026-10-01)

To probe compute versus data movement, a profiler-enabled run kept installed
han2/TT-Metal 0.77 (no model, package or service change), opened the
production 1×2 mesh (observed order `[1,0]`; PCIe nodes 0/1 at
`01:00.0`/`03:00.0`; 11×10 grids) and then ran one flexible paged-prefill
SDPA layer call per chip: BF16, 12 Q/2 KV heads, D=256, 1,024 query rows,
64-token KV pages, 128×128 Q/K chunks, HiFi2 with FP32 dest, starts 0 and
129,024 (replay chunk 126), twice each, counter mask 47. Both chips
matched a precomputed independent CPU FP32 reference at PCC 0.999754
(start 0) and 0.998883 (late), with identical output hashes across
repeats and chips. Second-call host time was 0.49/0.55 ms early (chip 0's
2.404 s first call includes cold JIT and is not comparable); late medians
were 33.332/33.436 ms (chips 0/1). These isolated single-layer programs
are not comparable to the separate prior full-model run, whose 16 late
SDPA programs summed to ~523–526 ms of corrected dispatch per chip.

On the 48 Q-work of 110 launched cores per chip (repeats pooled; chip 0
shown, chip 1 within 0.07 points), median FPU busy rose 33.89→42.12%
early→late; late MATH was 52.14%, SFPU 10.02%, unpacker-0 thread-0 busy
73.25%, packer 20.97% and thread-0 instruction stall 21.84%, while
source-valid waits stayed zero. L1_0 ring-0 outgoing request and grant
were 10.11% and 9.40% of reference cycles (~7.20% request-over-grant
backpressure). The TRISC1 counter window grew from ~0.24 M to 44.15 M
cycles; 192 paired core comparisons give +43.91 M cycles over 2,016 extra
K/V iterations per core, a median ~21,780 cycles per extra iteration
(MATH 11,369, FPU 9,184). Pinned `reader_interleaved.cpp:402-439,601-632`
reads every paged chunked K and V chunk through the page table with
forwarding disabled; the **modeled** issued K/V reads are therefore
~12.74 GB per chip per layer late versus 56.6 MB early (97,200 versus 432
chunk visits). That is an analytical count, not a byte counter or
bandwidth.

Counter windows cover only TRISC1; ring-0 signals mix L1/NoC traffic and
are not DRAM bytes, and no silicon DRAM byte counters were available.
Chip 0's raw CSV was overwritten by chip 1, but all four validated
per-window JSON receipts per chip are preserved. Idle FPU cycles are not
robustly attributable to NoC/DRAM stalls. The counters show substantial
math/unpack cost and some instruction stalls, and **no** evidence of DRAM
saturation. Decision: no kernel change from the baseline. A future,
explicitly bounded A/B may test page-table-aware K/V reuse/forwarding
against a math-only control; it must keep exact BF16 output and order and
the same output hashes/PCC, and pass the existing full-model 98-gate
before any proposal. No speedup is predicted. tt-sim v1.10.6 failed
uninstrumented (`Disable_pack_zero_flags`) and lacks perf registers, so
there is no simulator acceptance. Attempt r1 failed closed on the
reversed mesh order and was restored; r2 stopped at the pre-isolation
disk guard with the owner unchanged; r3 completed and `ExecStopPost`
restored root `Type=notify` active/running PID 2461790 (`/healthz` HTTP
200 ready, sole holder of both nodes, `NRestarts=0`, all five unit/drop-in
hashes unchanged). The
[dated counter receipt](../pkgs/qwen36/evidence/2026-10-01/qwen38-sdpa-counters.json)
hashes the frozen plan, counter header, analyzer, analysis (reproduced
byte-for-byte after checking 92,400 counter markers), transaction,
restore and all eight counter-window JSONs. No full-model exact-98,
reboot or throughput admission is claimed.

## Upstream state watch

The PR is closed-unmerged. If upstream merges it or the qwen36 path moves,
first re-run `qwen36-check` and the authority export; only update the pin with
new evidence.
