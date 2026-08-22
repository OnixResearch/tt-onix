# Qwen3.8-27B adoption (tt-metal PR 53543)

This document records the repository adoption of
[tenstorrent/tt-metal pull 53543](https://github.com/tenstorrent/tt-metal/pull/53543),
"Enable Qwen3.8-27B on the existing qwen36 Blackhole path".

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

- `nix build .#qwen36` — the deployable; `qwen36`, `qwen36-check`,
  `qwen36-serve` binaries, plus the installed `models/` tree.
- `qwen36-check` — device-free adoption gate (no hardware).
- `qwen36-serve` — text-only vLLM 0.10 entrypoint; fails closed when vLLM is
  not importable.
- `qwen36-p150x2-mtp-probe` — diagnostic target-verification probe; it cannot
  admit MTP or change production behavior.
- `qwen36-p150x2-decode-component-probe` — diagnostic component replay probe;
  it cannot select production behavior.
- `qwen36-p150x2-ttft-control` — diagnostic same-process TTFT control for
  request-cache capacity and parked-trace effects; it cannot admit traced decode.
- `nix build .#checks.x86_64-linux.qwen36` and
  `.#checks.x86_64-linux.qwen36-authority-negative` — CI-shaped gates.

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
state carry, and serialized short-prompt paged serving. It does not prove full
64-layer logits PCC against a CPU model. It also makes no long-context,
streaming, parallel-request, batch-serving, thermal, or profiler claim. The
TTFT and throughput result applies only to the recorded three-run scope.

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
8. long context, streaming, parallel requests, and batch serving — pending.

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
`/v1/models`, `/v1/completions`, and `/v1/chat/completions`. Requests are
serialized because one process owns both P150 devices. Only greedy,
non-streaming responses are admitted.

This host runs the final package as the lingering per-user
`qwen38-p150x2.service` unit on `127.0.0.1:8000`. Both previous inference
services remain stopped while Qwen owns the devices. A live 64-token request
also passed on the final deployed package.

## Upstream state watch

The PR is closed-unmerged. If upstream merges it or the qwen36 path moves,
first re-run `qwen36-check` and the authority export; only update the pin with
new evidence.
