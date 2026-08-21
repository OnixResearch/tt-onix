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
  architecture `Qwen3_5ForConditionalGeneration`, 18 checkpoint shards,
  weights excluded;
- hashed metadata artifacts (license, config, tokenizers, preprocessors,
  checkpoint index) with exact Nix and BLAKE3 identities.

Drift in any recorded fact fails the package evaluation. A separate
`qwen36-authority-negative` check proves a drifted commit is rejected.

## Deployable surface

- `nix build .#qwen36` — the deployable; `qwen36`, `qwen36-check`,
  `qwen36-serve` binaries, plus the installed `models/` tree.
- `qwen36-check` — device-free adoption gate (no hardware).
- `qwen36-serve` — text-only vLLM 0.10 entrypoint; fails closed when vLLM is
  not importable.
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
5. eight-layer paged-contract logits — passed against the bespoke path;
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

Evidence:

- `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-traced-decode-candidate-receipt.json`
- `cairn/archive/2026-08-20-accelerate-qwen38-p150x2-decode/evidence/physical-result.md`
- `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-traced-decode-ttft-candidate-receipt.json`
- `cairn/archive/2026-08-20-reduce-qwen38-traced-decode-ttft/evidence/physical-result.md`

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
