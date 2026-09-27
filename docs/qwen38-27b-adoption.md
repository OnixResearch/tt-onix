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

One request may hold 65,536 tokens (prompt and generation) and generate 8,192;
the module options `maximumSequenceLength` and `maximumGenerationTokens` default
to these ceilings. KV comes from one pool of 4,096 blocks of 64 tokens (8 GiB of
bf16 per chip, `kv_pool_blocks` in `decode-profile.ncl`). Each batch group gives
every row a contiguous run sized to its request: whole 1,024-token prefill
buckets plus every decoded position. The worker adds requests to a group only
while the pool holds them all, so one long request leaves fewer rows for short
ones. The pool's first 16 blocks keep a structured read's saved prefix, and its
last block backs unused page-table entries. Startup logs the allocator's DRAM
totals as the `device-dram` stage.

On britton-desktop the pooled build became ready in 700 s with 9.2 GiB of DRAM
free per chip. A needle placed at the start of the prompt was recalled at 6K,
16K, 32K and 60K tokens. TTFT was 3.3, 9.2, 19.2 and 39.2 s, and decode was
23.2, 22.7, 22.0 and 21.1 tokens/s. A group of one 30K row and three short rows
kept the short outputs equal to their solo runs. Prefill still runs one user
after another within a group, and a group runs until its last request finishes,
so a long prompt or generation delays the requests queued behind it.

A continuous-slot scheduler is available as a device-free candidate. The flag
`--continuous-slot-policy $out/share/qwen36/continuous-slot-policy.json`, in
candidate decode mode, replaces the group coordinator at construction. It keeps
one B4 decode trace resident and refills a slot at the next window boundary once
its request ends. Queue, refill, fairness and stall bounds come from
`continuous-slot-policy.ncl`. Drift or a stalled device effect quarantines the
engine, and `/healthz` then answers 503. The candidate is not deployed and has no
hardware evidence yet; the change
`.cairn/changes/qwen38-continuous-slot-scheduler` records its tasks and evidence.

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

This host runs the final package as the lingering per-user
`qwen38-p150x2.service` unit on `127.0.0.1:8000`. Both previous inference
services remain stopped while Qwen owns the devices. A live 64-token request
also passed on the final deployed package.

## Upstream state watch

The PR is closed-unmerged. If upstream merges it or the qwen36 path moves,
first re-run `qwen36-check` and the authority export; only update the pin with
new evidence.
